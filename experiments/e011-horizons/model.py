"""Minimal GPT for the oracle-harness experiments. Standard blocks
(unlike e001's sequence-mixer-only protocol): this harness measures
objectives and data policies, not architectures, so the model is a
plain small transformer held fixed across all arms."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from corpus import VOCAB


class Block(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.heads = heads
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2)
                   for t in self.qkv(h).chunk(3, -1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, d=128, n_layers=4, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def features(self, ids):
        x = self.emb(ids) + self.pos.weight[: ids.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.ln(x)

    def forward(self, ids):
        return self.head(self.features(ids))
