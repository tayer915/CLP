import torch
import torch.nn as nn

from net.basic.basic_gnn import GCNLayer
from net.basic.basic_nn import AttentionLayer


class EdgeLevelAttention(nn.Module):
    def __init__(self, att_dim, sub_dim, heads):
        super().__init__()
        self.attention_layer = AttentionLayer(att_dim, att_dim, sub_dim, sub_dim, heads)

    def forward(self, features):
        f = self.attention_layer(features, features, features)
        f = f.mean(1)
        return f


class EdgeLevelGCN(nn.Module):
    def __init__(self, att_dim, type_num):
        super().__init__()
        self.gcn_layers = nn.ModuleList([GCNLayer(att_dim) for _ in range(type_num)])
        self.type_num = type_num

    def forward(self, graphs, features):
        features_gcn = [self.gcn_layers[i](features[:, i, :].squeeze(1), graphs[i]) for i in range(self.type_num)]
        f = torch.stack(features_gcn, 1).mean(1)
        return f
