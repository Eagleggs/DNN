import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.to_keys = nn.Linear(k, k * heads, bias=False)
        self.to_queries = nn.Linear(k, k * heads, bias=False)
        self.to_values = nn.Linear(k, k * heads, bias=False)

        self.unifyheads = nn.Linear(k * heads, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        queries = self.to_queries(x).view(b, t, h, k)
        keys = self.to_keys(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)

        keys = keys.transpose(1, 2).reshape(b * h, t, k)
        queries = queries.transpose(1, 2).reshape(b * h, t, k)
        values = values.transpose(1, 2).reshape(b * h, t, k)

        w_prime = torch.bmm(queries, keys.transpose(1, 2))
        w_prime = w_prime / (k ** (1 / 2))
        w = F.softmax(w_prime, dim=2)

        y = torch.bmm(w, values).view(b, h, t, k)
        y = y.transpose(1, 2).reshape(b, t, h * k)

        y = self.unifyheads(y)

        return y  # output [b,t,k]


class TransformerLite(nn.Module):
    def __init__(self, t, k, heads):
        super(TransformerLite, self).__init__()
        self.position = nn.Embedding(t,k)
        self.attentionlayer = MultiHeadSelfAttention(k, heads=heads)
        self.layernorm = nn.LayerNorm([t, k])
        self.position_embedding = nn.Embedding(t, k)
        self.seq = nn.Sequential(
            nn.Linear(t*k, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100,4)
        )

    def forward(self,x):
        b, t, k = x.shape
        p = torch.arange(t, device=x.device).view(1, t).expand(b, t)
        p = self.position_embedding(p)
        x = x + p
        y = self.attentionlayer(x)
        x = x + y
        x = self.layernorm(x)
        y = self.seq(x.transpose(1,2).reshape(-1,t*k))
        y = F.softmax(y, dim=1) #(b,k,4)
        return y
