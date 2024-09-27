import torch

from utils.graph_options import edges2graph


class DataLoader(object):
    def __init__(self, opt):
        self.opt = opt
        self.train_file = opt.train_file
        self.test_file = opt.test_file
        self.device = opt.device
        self.g_nums = opt.g_nums
        self._read_data()
        self.node_num = len(self.nodes_index)
        self.type_num = len(self.snapshots[0])

    def _read_data(self):
        self.train_edges, nodes, edge_types, times = read_train_data(self.train_file)
        self.snapshots, self.nodes_index = handle_train_data(self.train_edges, nodes, edge_types, times, self.g_nums)
        test_edges = read_test_data(self.test_file)
        train_data, valid_data, test_data = handle_test_data(test_edges, self.nodes_index)
        self.train_edges = train_data["pos"]
        self.train_edges_n = train_data["neg"]
        self.train_graph, self.train_pos, self.train_neg = self._trans_tensor(train_data["pos"], train_data["neg"])
        self.valid_edges = valid_data["pos"]
        self.valid_edges_n = valid_data["neg"]
        self.valid_graph, self.valid_pos, self.valid_neg = self._trans_tensor(valid_data["pos"], valid_data["neg"])
        self.test_edges = test_data["pos"]
        self.test_edges_n = test_data["neg"]
        self.test_graph, self.test_pos, self.test_neg = self._trans_tensor(test_data["pos"], test_data["neg"])
        self.train_snapshots = self._trans_snapshots(self.snapshots)
        self.train_snapshots_dict = [[edges2graph(graph) for graph in snapshot] for snapshot in self.snapshots]

        def to_iso(edges_graphs):
            out = list()
            for edges in edges_graphs:
                out.extend(edges)
            return out
        self.train_snapshots_dict_iso = [edges2graph(to_iso(snapshot)) for snapshot in self.snapshots]

    def _trans_tensor(self, pos, neg):
        graph = dict()
        for src, dst in pos:
            if src not in graph:
                graph[src] = []
            graph[src].append(dst)
        if self.device == "cuda":
            return graph, torch.LongTensor(pos).cuda().T, torch.LongTensor(neg).cuda().T
        else:
            return graph, torch.LongTensor(pos).T, torch.LongTensor(neg).T

    def _trans_snapshots(self, snapshots):
        if self.device == "cuda":
            return [[torch.LongTensor(edge).cuda().T for edge in snapshot] for snapshot in snapshots]
        else:
            return [[torch.LongTensor(edge).T for edge in snapshot] for snapshot in snapshots]


def read_train_data(filepath):
    edges = dict()
    nodes = set()
    edge_types = set()
    times = set()
    with open(filepath, "r") as f:
        while True:
            line = f.readline().strip()
            if line == "":
                break
            src, dst, flag, t = line.split()
            nodes.add(src)
            nodes.add(dst)
            edge_types.add(flag)
            times.add(int(t))
            edges[(src, dst)] = (flag, int(t))
    return edges, nodes, edge_types, times


def handle_train_data(edges, nodes, edge_types, times, g_nums):
    nodes_index = dict()
    et = list(edge_types)
    num = 0
    for node in nodes:
        nodes_index[node] = num
        num += 1
    max_time, min_time = max(times), min(times)
    blank_length = (max_time - min_time) / (g_nums - 1)
    temp_snapshots = [{flag: [] for flag in et} for _ in range(g_nums)]
    for edge, para in edges.items():
        src, dst = nodes_index[edge[0]], nodes_index[edge[1]]
        flag, t = para[0], para[1]
        t_slot = min(g_nums - 1, int((t - min_time) / blank_length))
        temp_snapshots[t_slot].get(flag).append([src, dst])
    snapshots = [[snapshot[flag] if len(snapshot[flag]) > 0 else [[0, 0]] for flag in et] for snapshot in temp_snapshots]
    return snapshots, nodes_index


def read_test_data(filepath):
    edges = dict()
    with open(filepath, "r") as f:
        while True:
            line = f.readline().strip()
            if line == "":
                break
            src, dst, exist, data_type = line.split()
            edges[(src, dst)] = (exist, data_type)
    return edges


def handle_test_data(edges, nodes_index):
    train_data = {"pos": list(), "neg": list()}
    valid_data = {"pos": list(), "neg": list()}
    test_data = {"pos": list(), "neg": list()}
    for edge, para in edges.items():
        s, d = edge[0], edge[1]
        if s not in nodes_index or d not in nodes_index:
            continue
        src, dst = nodes_index[s], nodes_index[d]
        exist, data_type = para[0], para[1]
        key = "neg" if exist == "0" else "pos"
        if data_type == "0":
            valid_data[key].append([src, dst])
        elif data_type == "1":
            train_data[key].append([src, dst])
        else:
            test_data[key].append([src, dst])
    return train_data, valid_data, test_data


def data_pipe(train_file, test_file, g_nums=7):
    edges, nodes, edge_types, times = read_train_data(train_file)
    snapshots, nodes_index = handle_train_data(edges, nodes, edge_types, times, g_nums)
    test_edges = read_test_data(test_file)
    train_data, valid_data, test_data = handle_test_data(test_edges, nodes_index)
    return snapshots, train_data, valid_data, test_data
