"""VERIFIER, anomaly mechanism. Why does joint execute BETTER than depth-only?

Splits the temporal reuse into (a) reuse applied to the position's OWN state
and (b) reuse applied to OTHER positions (context), and measures how drifted
the SUBSTITUTED value actually is.
"""
import sys
import torch

sys.path.insert(0, "/home/azureuser/e009")
from corpus import CorpusGen
from model import SIZES, SwiGPT

SIZES.update({"xxs": dict(d=32, n_layers=4), "xs": dict(d=64, n_layers=6),
              "xl": dict(d=384, n_layers=14)})
ck, NS = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64
d = torch.load(ck, weights_only=True, map_location="cpu")
m = SwiGPT(**SIZES[d["size"]]).eval(); m.load_state_dict(d["state"])
L = len(m.blocks)
ids = torch.from_numpy(CorpusGen(seed=9999).batch(NS, 256)[0])
B, T = ids.shape
print(f"# {d['size']} t={d['tokens']} L={L} {B}x{T}")

with torch.no_grad():
    hs = [m.embed(ids)]
    for b in m.blocks:
        hs.append(b(hs[-1]))
    full_arg = m.readout(hs[-1]).argmax(-1)
    exit_b = torch.full((B, T), L)
    for b in range(L):
        ok = m.readout(hs[b + 1]).argmax(-1) == full_arg
        exit_b = torch.minimum(exit_b, torch.where(ok, torch.tensor(b + 1), torch.tensor(L)))
    depth = torch.stack([exit_b <= b for b in range(L)])
    temp = []
    for b in range(L):
        h2 = hs[b + 1].clone(); h2[:, 1:] = hs[b + 1][:, :-1]
        for bb in range(b + 1, L):
            h2 = m.blocks[bb](h2)
        a = m.readout(h2).argmax(-1) == full_arg; a[:, 0] = False
        temp.append(a)
    temp = torch.stack(temp)

    print("# exit-block histogram: " +
          " ".join(f"{e}:{(exit_b==e).float().mean():.3f}" for e in range(1, L + 1)))
    early = exit_b <= L // 2
    mid = (exit_b > L // 2) & (exit_b < L)
    never = exit_b == L

    @torch.no_grad()
    def run(depth_skip, reuse=None):
        h = m.embed(ids)
        for b, blk in enumerate(m.blocks):
            out = blk(h)
            if reuse is not None:
                r = reuse[b].clone(); r[:, 0] = False
                out = torch.where(r.unsqueeze(-1), torch.roll(out, 1, dims=1), out)
            h = torch.where(depth_skip[b].unsqueeze(-1), h, out)
        return m.readout(h)

    def rep(name, lg):
        ag = lg.argmax(-1) == full_arg
        print(f"{name:44} all={ag.float().mean():.4f} early={ag[early].float().mean():.4f} "
              f"mid={ag[mid].float().mean():.4f} never={ag[never].float().mean():.4f}")

    rep("depth only", run(depth))
    rep("joint (all reuse)", run(depth, temp))
    rep("joint, reuse OFF for mid-exit (context only)",
        run(depth, temp & ~mid.unsqueeze(0)))
    rep("joint, reuse ONLY for mid-exit (own only)",
        run(depth, temp & mid.unsqueeze(0)))

    # how drifted is the value that gets copied in?  Run depth-only, record
    # per-block executed states, compare the SOURCE slot (t-1) to clean.
    h = m.embed(ids)
    for b, blk in enumerate(m.blocks):
        out = blk(h)
        h = torch.where(depth[b].unsqueeze(-1), h, out)
        src = torch.roll(h, 1, dims=1)             # what reuse would copy at b
        cl = torch.roll(hs[b + 1], 1, dims=1)
        rel = ((src - cl).norm(dim=-1) / cl.norm(dim=-1).clamp(min=1e-6))
        use = temp[b]
        own = ((h - hs[b + 1]).norm(dim=-1) / hs[b + 1].norm(dim=-1).clamp(min=1e-6))
        if use.any():
            print(f"  block {b}: reused slots {use.float().mean():.3f} | "
                  f"drift of COPIED value {rel[use].mean():.3f} vs drift of the "
                  f"value it REPLACES {own[use].mean():.3f} | "
                  f"source exit<=b/2 frac {(torch.roll(exit_b,1,dims=1)[use] <= max(b//2,1)).float().mean():.3f}")
