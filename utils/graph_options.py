import random
import torch


def negative_sampling(graph, node_num):
    flag = (node_num - 1) // 2
    u_set = set(range(node_num))
    result_graph = dict()
    for node, neighbors in graph.items():
        sample_num = len(neighbors)
        assert sample_num < flag
        nei_s = set(neighbors)
        nei_s.add(node)
        negative_list = list(u_set - nei_s)
        random.shuffle(negative_list)
        result_graph[node] = negative_list[:sample_num]
    return result_graph


def graph2edges(graph):
    edges = list()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            edges.append([node, neighbor])
    return edges


def edges2graph(edges):
    graph = dict()
    for src, dst in edges:
        if src not in graph:
            graph[src] = set()
        graph[src].add(dst)
    return graph

# def sample_loss_two(graphs, emb_new, emb_old, tao=1e6):
#     result = list()
#     for src, neighbors in graphs.items():
#         result_node = list()
#         for neighbor in neighbors:
#             result_node.append(emb_old[src].dot(emb_new[neighbor] / tao).exp())
#         sum_node = torch.stack(result_node, 0).sum(0)
#         sum_loss = torch.stack([(x_node / sum_node).log() for x_node in result_node], 0).sum(0)
#         result.append(sum_loss)
#     return torch.stack(result, 0).sum(0)
#
#
# def loss_sample(graphs, feature_list, feature_old, tao=1e6):
#     graph_list_out = list()
#     graph_list_in = list()
#     for graph in graphs:
#         loss_node_out_list = list()
#         loss_node_in_list = list()
#         for feature in feature_list:
#             loss_node_out_list.append(sample_loss_two(graph, feature, feature_old, tao))
#             loss_node_in_list.append(sample_loss_two(graph, feature, feature, tao))
#         loss_node_out = torch.stack(loss_node_out_list, 0).sum(0)
#         loss_node_in = torch.stack(loss_node_in_list, 0).sum(0)
#         graph_list_out.append(loss_node_out)
#         graph_list_in.append(loss_node_in)
#     return - torch.stack(graph_list_out, 0).sum(0), - torch.stack(graph_list_in, 0).sum(0)
#
#
# def loss_double_team(graphs, feature_a, feature_b, tao=1e6):
#     graph_list_one = list()
#     graph_list_two = list()
#     for i in range(len(graphs)):
#         graph = graphs[i]
#         f_a = feature_a[i]
#         f_b = feature_b[i]
#         result_node_one = list()
#         result_node_two = list()
#         for src, neighbors in graph.items():
#             result_node_a = list()
#             result_node_b = list()
#             for neighbor in neighbors:
#                 result_node_a.append(f_a[src].dot(f_a[neighbor] / tao).exp())
#                 result_node_b.append(f_b[src].dot(f_b[neighbor] / tao).exp())
#             a = torch.stack(result_node_a, 0).sum(0)
#             b = torch.stack(result_node_b, 0).sum(0)
#             result_node_one.append((f_a[src].dot(f_a[src] / tao).exp() / a).log())
#             result_node_two.append((f_b[src].dot(f_b[src] / tao).exp() / b).log())
#         graph_list_one.append(torch.stack(result_node_one, 0).sum(0))
#         graph_list_two.append(torch.stack(result_node_two, 0).sum(0))
#     return - torch.stack(graph_list_one, 0).sum(0), - torch.stack(graph_list_two, 0).sum(0)
