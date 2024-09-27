from torch_geometric.nn import GATConv, GCNConv
import torch.nn as nn


class GATLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.gat = GATConv(in_channels=dim, out_channels=dim, heads=heads)
        self.dense = nn.Linear(dim * heads, dim)

    def forward(self, embedding, graph):
        x1 = self.gat(embedding, graph)
        x2 = self.dense(x1)
        return x2


class GCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gcn = GCNConv(in_channels=dim, out_channels=dim)
        self.dense = nn.Linear(dim, dim)

    def forward(self, embedding, graph):
        x1 = self.gcn(embedding, graph)
        x2 = self.dense(x1)
        return x2

