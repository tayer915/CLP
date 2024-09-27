import torch


def loss_node_calculator(snapshots, node_gat_list, node_gcn_list, tao):
    snapshots_list_sim = list()
    snapshots_list_diff = list()
    for snapshot, gat_snapshot, gcn_snapshot in zip(snapshots, node_gat_list, node_gcn_list):
        snapshot_list_sim = list()
        snapshot_list_diff = list()
        for graph, gat_graph, gcn_graph in zip(snapshot,  gat_snapshot, gcn_snapshot):
            graph_list_sim = list()
            graph_list_diff = list()
            for src, neighbors in graph.items():
                neighbor_list_sim = list()
                neighbor_list_diff = list()
                for neighbor in neighbors:
                    loss_neighbor_sim = (gat_graph[src].dot(gcn_graph[neighbor]) / tao).exp()
                    loss_neighbor_diff = (gat_graph[src].dot(gat_graph[neighbor]) / tao).exp()
                    neighbor_list_sim.append(loss_neighbor_sim)
                    neighbor_list_diff.append(loss_neighbor_diff)
                loss_neighbors_sim = torch.stack(neighbor_list_sim, 0).sum(0)
                loss_neighbors_diff = torch.stack(neighbor_list_diff, 0).sum(0)
                loss_node_sim = ((gat_graph[src].dot(gcn_graph[src]) / tao).exp() / loss_neighbors_sim).log()
                temp_list_sim = list()
                for temp_loss in neighbor_list_diff:
                    temp_list_sim.append((temp_loss / loss_neighbors_diff).log())
                loss_node_diff = torch.stack(temp_list_sim, 0).sum(0)
                graph_list_sim.append(loss_node_sim)
                graph_list_diff.append(loss_node_diff)
            loss_graph_sim = torch.stack(graph_list_sim, 0).sum(0)
            loss_graph_diff = torch.stack(graph_list_diff, 0).sum(0)
            snapshot_list_sim.append(loss_graph_sim)
            snapshot_list_diff.append(loss_graph_diff)
        loss_snapshot_sim = torch.stack(snapshot_list_sim, 0).sum(0)
        loss_snapshot_diff = torch.stack(snapshot_list_sim, 0).sum(0)
        snapshots_list_sim.append(loss_snapshot_sim)
        snapshots_list_diff.append(loss_snapshot_diff)
    loss_node_level_sim = - torch.stack(snapshots_list_sim, 0).sum(0)
    loss_node_level_diff = - torch.stack(snapshots_list_diff, 0).sum(0)
    return loss_node_level_sim - loss_node_level_diff


def loss_edge_calculator(snapshots, edge_att_list, edge_gcn_list, tao):
    snapshots_list_sim = list()
    snapshots_list_diff = list()
    for snapshot, att_snapshot, gcn_snapshot in zip(snapshots, edge_att_list, edge_gcn_list):
        snapshot_list_sim = list()
        snapshot_list_diff = list()
        for src, neighbors in snapshot.items():
            neighbor_list_sim = list()
            neighbor_list_diff = list()
            for neighbor in neighbors:
                loss_neighbor_sim = (att_snapshot[src].dot(gcn_snapshot[neighbor]) / tao).exp()
                loss_neighbor_diff = (att_snapshot[src].dot(gcn_snapshot[neighbor]) / tao).exp()
                neighbor_list_sim.append(loss_neighbor_sim)
                neighbor_list_diff.append(loss_neighbor_diff)
            loss_neighbors_sim = torch.stack(neighbor_list_sim, 0).sum(0)
            loss_neighbors_diff = torch.stack(neighbor_list_diff, 0).sum(0)
            loss_node_sim = ((att_snapshot[src].dot(gcn_snapshot[src]) / tao).exp() / loss_neighbors_sim).log()
            temp_list_sim = list()
            for temp_loss in neighbor_list_diff:
                temp_list_sim.append((temp_loss / loss_neighbors_diff).log())
            loss_node_diff = torch.stack(temp_list_sim, 0).sum(0)
            snapshot_list_sim.append(loss_node_sim)
            snapshot_list_diff.append(loss_node_diff)
        loss_snapshot_sim = torch.stack(snapshot_list_sim, 0).sum(0)
        loss_snapshot_diff = torch.stack(snapshot_list_sim, 0).sum(0)
        snapshots_list_sim.append(loss_snapshot_sim)
        snapshots_list_diff.append(loss_snapshot_diff)
    loss_edge_level_sim = - torch.stack(snapshots_list_sim, 0).sum(0)
    loss_edge_level_diff = - torch.stack(snapshots_list_diff, 0).sum(0)
    return loss_edge_level_sim - loss_edge_level_diff


def loss_time_calculator(snapshots, gru_f, lstm_f, tao):
    snapshots_list_gru = list()
    snapshots_list_lstm = list()
    for index, snapshot in enumerate(snapshots):
        snapshot_list_gru = list()
        snapshot_list_lstm = list()
        gru_feature = gru_f[:, index, :].squeeze(1)
        lstm_feature = gru_f[:, index, :].squeeze(1)
        for src, neighbors in snapshot.items():
            neighbor_list_gru = list()
            neighbor_list_lstm = list()
            for neighbor in neighbors:
                loss_neighbor_gru = (gru_feature[src].dot(gru_feature[neighbor]) / tao).exp()
                loss_neighbor_lstm = (lstm_feature[src].dot(lstm_feature[neighbor]) / tao).exp()
                neighbor_list_gru.append(loss_neighbor_gru)
                neighbor_list_lstm.append(loss_neighbor_lstm)
            loss_src = (gru_feature[src].dot(lstm_feature[src]) / tao).exp()
            loss_neighbors_gru = torch.stack(neighbor_list_gru, 0).sum(0)
            loss_neighbors_lstm = torch.stack(neighbor_list_lstm, 0).sum(0)
            loss_node_gru = (loss_src / loss_neighbors_gru).log()
            loss_node_lstm = (loss_src / loss_neighbors_lstm).log()
            snapshot_list_gru.append(loss_node_gru)
            snapshot_list_lstm.append(loss_node_lstm)
        loss_snapshot_gru = torch.stack(snapshot_list_gru, 0).sum(0)
        loss_snapshot_lstm = torch.stack(snapshot_list_lstm, 0).sum(0)
        snapshots_list_gru.append(loss_snapshot_gru)
        snapshots_list_lstm.append(loss_snapshot_lstm)
    loss_time_level_gru = - torch.stack(snapshots_list_gru, 0).sum(0)
    loss_time_level_lstm = - torch.stack(snapshots_list_lstm, 0).sum(0)
    return loss_time_level_gru + loss_time_level_lstm


def sample_dot(edges, features):
    result = list()
    for u, v in edges:
        result.append(torch.sigmoid(features[u].dot(features[v])))
    return torch.stack(result, 0)
