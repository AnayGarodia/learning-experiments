"""Matched non-embedding-param ladder: transformer (e005 Block, pinned
copy) vs 2-layer LSTM. Non-embedding = blocks / recurrent stack only;
byte embedding + head excluded on both sides."""
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 256


class Block(nn.Module):  # e005 block, vocab-agnostic
    def __init__(self, d, heads=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.heads = heads
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2)
                   for t in self.qkv(h).chunk(3, -1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class ByteTransformer(nn.Module):
    def __init__(self, d, n_layers, max_len=256):
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


class ByteLSTM(nn.Module):
    def __init__(self, hidden, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, hidden)
        self.lstm = nn.LSTM(hidden, hidden, n_layers, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, VOCAB)

    def forward(self, ids):
        x, _ = self.lstm(self.emb(ids))
        return self.head(self.ln(x))

    def nonemb_params(self):
        return sum(p.numel() for p in self.lstm.parameters())


# rungs: (name, transformer(d,L), lstm hidden) matched on non-emb params
RUNGS = [
    ("r1", (128, 5), 256),
    ("r2", (192, 7), 444),
    ("r3", (256, 11), 752),
]

if __name__ == "__main__":
    for name, (d, L), h in RUNGS:
        t = ByteTransformer(d, L).nonemb_params()
        l = ByteLSTM(h).nonemb_params()
        ratio = t / l
        print(f"{name}: transformer {t:,} vs lstm {l:,} (ratio {ratio:.2f})")
        assert 0.8 < ratio < 1.25, "params not matched"
    print("param match ok")
