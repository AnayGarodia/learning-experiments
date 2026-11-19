"""Is the certified reuse slot LOW-SENSITIVITY (any value works) or is it a
VALID SUBSTITUTE (only t-1 works)?  Same slots, different injected values."""
import sys, torch
sys.path.insert(0, "/home/azureuser/e009")
from corpus import CorpusGen
from model import SIZES, SwiGPT
SIZES.update({"xxs": dict(d=32, n_layers=4), "xs": dict(d=64, n_layers=6),
              "xl": dict(d=384, n_layers=14)})
ck, NS = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64
d = torch.load(ck, weights_only=True, map_location="cpu")
m = SwiGPT(**SIZES[d["size"]]).eval(); m.load_state_dict(d["state"])
L = len(m.blocks)
ids = torch.from_numpy(CorpusGen(seed=9999).batch(NS, 256)[0]); B, T = ids.shape
print(f"# {d['size']} t={d['tokens']} L={L} {B}x{T}")
with torch.no_grad():
    hs = [m.embed(ids)]
    for b in m.blocks: hs.append(b(hs[-1]))
    fa = m.readout(hs[-1]).argmax(-1)
    e = torch.full((B, T), L)
    for b in range(L):
        ok = m.readout(hs[b+1]).argmax(-1) == fa
        e = torch.minimum(e, torch.where(ok, torch.tensor(b+1), torch.tensor(L)))
    depth = torch.stack([e <= b for b in range(L)])
    temp = []
    for b in range(L):
        h2 = hs[b+1].clone(); h2[:, 1:] = hs[b+1][:, :-1]
        for bb in range(b+1, L): h2 = m.blocks[bb](h2)
        a = m.readout(h2).argmax(-1) == fa; a[:, 0] = False; temp.append(a)
    temp = torch.stack(temp)
    mid = (e > L//2) & (e < L)
    g = torch.Generator().manual_seed(7)

    def run(mode):
        h = m.embed(ids)
        for b, blk in enumerate(m.blocks):
            out = blk(h)
            r = temp[b].clone(); r[:, 0] = False
            if mode == "t-1":   sub = torch.roll(out, 1, dims=1)
            elif mode == "t-2": sub = torch.roll(out, 2, dims=1)
            elif mode == "noise":
                n = torch.randn(out.shape, generator=g)
                sub = n * out.pow(2).mean(-1, keepdim=True).sqrt() / n.pow(2).mean(-1, keepdim=True).sqrt()
            elif mode == "zero": sub = torch.zeros_like(out)
            elif mode == "self": sub = out
            elif mode == "prev-h": sub = h          # skip the block (depth-style) at certified slots
            out = torch.where(r.unsqueeze(-1), sub, out)
            h = torch.where(depth[b].unsqueeze(-1), h, out)
        return m.readout(h)
    for mode in ("self", "t-1", "t-2", "prev-h", "noise", "zero"):
        a = run(mode).argmax(-1) == fa
        print(f"  inject {mode:7} at certified slots: all={a.float().mean():.4f} mid={a[mid].float().mean():.4f}")
