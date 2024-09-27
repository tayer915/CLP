import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, sub_k, sub_v, head_num):
        super().__init__()
        self.W_Q = nn.Linear(input_dim, sub_k * head_num, bias=False)
        self.W_K = nn.Linear(input_dim, sub_k * head_num, bias=False)
        self.W_V = nn.Linear(input_dim, sub_v * head_num, bias=False)
        self.fc = nn.Linear(head_num * sub_v, output_dim, bias=False)
        self.d_k = sub_k
        self.d_v = sub_v
        self.n_heads = head_num
        self.attention_layer = AttentionBasic(sub_k)
        self.norm_layer = nn.LayerNorm(output_dim)

    def forward(self, input_q, input_k, input_v):
        residual, batch_size = input_q, input_q.size(0)
        q = self.W_Q(input_q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(input_k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(input_v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        context = self.attention_layer(q, k, v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.norm_layer(output + residual)


class AttentionBasic(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.temperature = d_k ** 0.5
        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.temperature
        attn = self.softmax_layer(scores)
        context = torch.matmul(attn, v)
        return context
