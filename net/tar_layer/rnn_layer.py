import torch.nn as nn


class TemporalGRULayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.rnn_layer = nn.GRU(input_dim, output_dim)

    def forward(self, features):
        f1, _ = self.rnn_layer(features)
        return f1


class TemporalLSTMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.rnn_layer = nn.LSTM(input_dim, output_dim)

    def forward(self, features):
        f1, (_, __) = self.rnn_layer(features)
        return f1
