"""SwiGLU decoder-only LM for e009. Modern block (RMSNorm, SwiGLU MLP,
causal attention), written so oracle code can run blocks individually:
model.blocks[b](h) advances one block; model.readout(h) maps any hidden
state to logits through the final norm + head (the early-exit probe)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from corpus import VOCAB


class Block(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.heads = heads
        h = int(8 * d / 3)  # SwiGLU sizing so MLP params ~= 8d^2
        self.wg, self.wu, self.wd = (nn.Linear(d, h, bias=False),
                                     nn.Linear(d, h, bias=False),
                                     nn.Linear(h, d, bias=False))

    def mlp_act(self, x):
        """SwiGLU hidden activations (B,T,h) — the width-oracle surface."""
        return F.silu(self.wg(x)) * self.wu(x)

    def forward(self, x, mlp_mask=None):
        """mlp_mask: optional (h,) 0/1 tensor zeroing hidden units (width oracle)."""
        B, T, D = x.shape
        hn = self.n1(x)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2)
                   for t in self.qkv(hn).chunk(3, -1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        act = self.mlp_act(self.n2(x))
        if mlp_mask is not None:
            act = act * mlp_mask
        return x + self.wd(act)


class SwiGPT(nn.Module):
    def __init__(self, d, n_layers, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def embed(self, ids):
        return self.emb(ids) + self.pos.weight[: ids.shape[1]]

    def readout(self, h):
        return self.head(self.norm(h))

    def forward(self, ids):
        h = self.embed(ids)
        for b in self.blocks:
            h = b(h)
        return self.readout(h)

    def nonemb_params(self):
        return sum(p.numel() for b in self.blocks for p in b.parameters())


# ladder: ~1M / 3M / 9M non-embedding params (12*L*d^2 rule of thumb)
SIZES = {"s": dict(d=96, n_layers=8), "m": dict(d=160, n_layers=10),
         "l": dict(d=256, n_layers=12)}

if __name__ == "__main__":
    for k, kw in SIZES.items():
        m = SwiGPT(**kw)
        print(k, kw, f"{m.nonemb_params()/1e6:.2f}M nonemb")
