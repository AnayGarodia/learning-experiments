"""Tiny GPT with variable depth for growth experiments. Same block as
e005 (copied, pinned) but depth is the growth axis; non-embedding
param count is the reported size (embeddings dominate at tiny scale
and do not scale with depth)."""
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


class GPT(nn.Module):
    def __init__(self, d=192, n_layers=4, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, ids):
        x = self.emb(ids) + self.pos.weight[: ids.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def nonemb_params(self):
        return sum(p.numel() for b in self.blocks for p in b.parameters())


def grow_depth_x2(donor, shrink_perturb=False, seed=0):
    """G_stack (Du 2024's winner): copy the L blocks to positions
    [0..L-1] and [L..2L-1]. Optional shrink-and-perturb (Ash & Adams
    mitigation): theta <- 0.6*theta + N(0, 0.01^2) on copied blocks."""
    L = len(donor.blocks)
    d = donor.emb.embedding_dim
    g = GPT(d=d, n_layers=2 * L, max_len=donor.pos.num_embeddings)
    g.emb.load_state_dict(donor.emb.state_dict())
    g.pos.load_state_dict(donor.pos.state_dict())
    g.ln.load_state_dict(donor.ln.state_dict())
    g.head.load_state_dict(donor.head.state_dict())
    for i in range(2 * L):
        g.blocks[i].load_state_dict(donor.blocks[i % L].state_dict())
    if shrink_perturb:
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for b in g.blocks:
                for p in b.parameters():
                    p.mul_(0.6).add_(torch.randn(p.shape, generator=gen) * 0.01)
    return g
