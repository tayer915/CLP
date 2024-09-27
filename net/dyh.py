import torch
import torch.nn as nn
from net.hat_layer.hat_net import RecurrentHAL
from net.tar_layer.tar_net import TemporalAttentiveRNNLayer, TemporalLSTMGRULayer


class DyHModel(nn.Module):
    def __init__(self, node_num, dim, node_dim, node_heads, type_num, edge_dim, edge_sub_dim, edge_heads, rnn_input_dim,
                 rnn_output_dim, att_input_dim, att_output_dim, tar_sub_dim, tar_heads):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn([node_num, dim]))
        self.hals = RecurrentHAL(node_dim, node_heads, type_num, edge_dim, edge_sub_dim, edge_heads)
        # self.tar_layer = TemporalAttentiveRNNLayer(rnn_input_dim, rnn_output_dim, att_input_dim, att_output_dim,
        #                                            tar_sub_dim, tar_heads)
        self.tar_layer = TemporalLSTMGRULayer(rnn_input_dim, rnn_output_dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, graphs):
        node_gat_list, node_gcn_list, edge_att_list, edge_gcn_list, snap_features = self.hals(graphs, self.embedding)
        gru_f, lstm_f = self.tar_layer(snap_features)
        features = (gru_f + lstm_f).mean(1)
        outputs = self.out(features)
        return node_gat_list, node_gcn_list, edge_att_list, edge_gcn_list, snap_features, gru_f, lstm_f, outputs
