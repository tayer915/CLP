import torch.nn as nn
from net.basic.basic_nn import AttentionLayer


class TemporalAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, sub_dim, head_num):
        super().__init__()
        self.attention_layer = AttentionLayer(input_dim, output_dim, sub_dim, sub_dim, head_num)

    def forward(self, features):
        return self.attention_layer(features, features, features)
