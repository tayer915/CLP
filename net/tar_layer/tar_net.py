import torch.nn as nn

from net.tar_layer.rnn_layer import TemporalGRULayer, TemporalLSTMLayer
from net.tar_layer.attention_layer import TemporalAttentionLayer


class TemporalAttentiveRNNLayer(nn.Module):
    def __init__(self, rnn_input_dim, rnn_output_dim, att_input_dim, att_output_dim, sub_dim, heads):
        super().__init__()
        self.t_rnn = TemporalGRULayer(rnn_input_dim, rnn_output_dim)
        self.t_attention = TemporalAttentionLayer(att_input_dim, att_output_dim, sub_dim, heads)

    def forward(self, features):
        x1 = self.t_rnn(features)
        x2 = self.t_attention(x1).mean(1)
        return x2


class TemporalLSTMGRULayer(nn.Module):
    def __init__(self, rnn_input_dim, rnn_output_dim):
        super().__init__()
        self.t_rnn = TemporalGRULayer(rnn_input_dim, rnn_output_dim)
        self.t_lstm = TemporalLSTMLayer(rnn_input_dim, rnn_output_dim)

    def forward(self, features):
        x1 = self.t_rnn(features)
        x2 = self.t_lstm(features)
        return x1, x2
