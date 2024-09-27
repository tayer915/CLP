import torch.nn as nn
from net.basic.basic_gnn import GATLayer, GCNLayer


class NodeLevelAttention(nn.Module):
    def __init__(self, in_dim, heads, type_num):
        super().__init__()
        self.gat_layers = nn.ModuleList([GATLayer(in_dim, heads) for _ in range(type_num)])
        self.gcn_layers = nn.ModuleList([GCNLayer(in_dim) for _ in range(type_num)])
        self.type_num = type_num

    def forward(self, graphs, embedding):
        features_gat = [self.gat_layers[i](embedding, graphs[i]) for i in range(self.type_num)]
        features_gcn = [self.gat_layers[i](embedding, graphs[i]) for i in range(self.type_num)]
        return features_gat, features_gcn
