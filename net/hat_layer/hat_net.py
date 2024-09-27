import torch
import torch.nn as nn
from net.hat_layer.edge_level import EdgeLevelAttention, EdgeLevelGCN
from net.hat_layer.node_level import NodeLevelAttention


class HierarchicalAttentionLayer(nn.Module):
    def __init__(self, node_dim, node_heads, type_num, edge_dim, sub_dim, edge_heads):
        super().__init__()
        self.node_level_attention = NodeLevelAttention(node_dim, node_heads, type_num)
        self.edge_level_attention = EdgeLevelAttention(edge_dim, sub_dim, edge_heads)
        self.edge_level_gcn = EdgeLevelGCN(edge_dim, type_num)

    def forward(self, graphs, embedding):
        node_features_gat, node_features_gcn = self.node_level_attention(graphs, embedding)
        features = torch.stack(node_features_gat, 1)
        edge_att_features = self.edge_level_attention(features)
        edge_gcn_features = self.edge_level_gcn(graphs, features)
        return node_features_gat, node_features_gcn, edge_att_features, edge_gcn_features


class RecurrentHAL(nn.Module):
    def __init__(self, node_dim, node_heads, type_num, edge_dim, sub_dim, edge_heads):
        super().__init__()
        self.hal = HierarchicalAttentionLayer(node_dim, node_heads, type_num, edge_dim, sub_dim, edge_heads)

    def forward(self, snapshots, embedding):
        edge_feature_att_list = list()
        edge_feature_gcn_list = list()
        node_feature_gcn_list = list()
        node_feature_gat_list = list()
        for graphs in snapshots:
            node_features_gat, node_features_gcn, edge_att_features, edge_gcn_features = self.hal(graphs, embedding)
            edge_feature_att_list.append(edge_att_features)
            edge_feature_gcn_list.append(edge_gcn_features)
            node_feature_gcn_list.append(node_features_gcn)
            node_feature_gat_list.append(node_features_gat)
        features = torch.stack(edge_feature_att_list, 1)
        return node_feature_gat_list, node_feature_gcn_list, edge_feature_att_list, edge_feature_gcn_list, features
