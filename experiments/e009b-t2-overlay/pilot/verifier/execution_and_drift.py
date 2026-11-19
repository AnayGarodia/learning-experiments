"""VERIFIER: independent reimplementation of the e009b masked forward.

Does NOT import overlay.py. Rebuilds the depth-exit and temporal-reuse
masks from the model directly, executes them, and reports agreement.
Then attacks the anomaly (joint executes better than depth-only) with
drift measurements and ablations.

usage: python verify.py <ckpt.pt> [nseq]
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/azureuser/e009")
from corpus import CorpusGen
from model import SIZES, SwiGPT

SIZES.update({"xxs": dict(d=32, n_layers=4), "xs": dict(d=64, n_layers=6),
              "xl": dict(d=384, n_layers=14)})

ck = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
d = torch.load(ck, weights_only=True, map_location="cpu")
m = SwiGPT(**SIZES[d["size"]]).eval()
m.load_state_dict(d["state"])
L = len(m.blocks)
x, lab = CorpusGen(seed=9999).batch(NS, 256)
ids = torch.from_numpy(x)
B, T = ids.shape
print(f"# {d['size']} seed{d['seed']} t={d['tokens']} L={L} eval {B}x{T}")


@torch.no_grad()
def clean_stack():
    hs = [m.embed(ids)]
    for b in m.blocks:
        hs.append(b(hs[-1]))
    return hs


with torch.no_grad():
    hs = clean_stack()
    full_arg = m.readout(hs[-1]).argmax(-1)

    # --- depth: earliest block index e (1..L) whose readout argmax-matches
    exit_b = torch.full((B, T), L)
    for b in range(L):
        ok = m.readout(hs[b + 1]).argmax(-1) == full_arg
        exit_b = torch.minimum(exit_b, torch.where(ok, torch.tensor(b + 1),
                                                   torch.tensor(L)))
    depth = torch.stack([exit_b <= b for b in range(L)])   # (L,B,T)

    # --- temporal: block b output at t replaced by its output at t-1
    temp = []
    for b in range(L):
        h2 = hs[b + 1].clone()
        h2[:, 1:] = hs[b + 1][:, :-1]
        for bb in range(b + 1, L):
            h2 = m.blocks[bb](h2)
        a = m.readout(h2).argmax(-1) == full_arg
        a[:, 0] = False
        temp.append(a)
    temp = torch.stack(temp)

    print(f"# skip frac: depth {depth.float().mean():.4f}  "
          f"temporal {temp.float().mean():.4f}  joint {(depth|temp).float().mean():.4f}")
    print(f"# e009-style: depth_frac_arg {((L-exit_b.float())/L).mean():.4f}  "
          f"temporal_arg {temp[:,:,1:].float().mean():.4f}  "
          f"joint_block_arg {(depth|temp)[:,:,1:].float().mean():.4f}")


@torch.no_grad()
def run(depth_skip=None, reuse=None, shift=-1, clean_source=False,
        renorm=False):
    """Returns (logits, per-block relative drift vs the clean stack)."""
    h = m.embed(ids)
    drift = []
    for b, blk in enumerate(m.blocks):
        out = blk(h)
        if reuse is not None:
            r = reuse[b].clone()
            r[:, 0 if shift < 0 else -1] = False
            src = hs[b + 1] if clean_source else out
            out = torch.where(r.unsqueeze(-1),
                              torch.roll(src, shifts=-shift, dims=1), out)
        if depth_skip is not None:
            frozen = h
            if renorm:  # rescale frozen state to the live state's RMS
                frozen = h * (out.pow(2).mean(-1, keepdim=True).sqrt()
                              / h.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6))
            out = torch.where(depth_skip[b].unsqueeze(-1), frozen, out)
        h = out
        drift.append(((h - hs[b + 1]).norm(dim=-1)
                      / hs[b + 1].norm(dim=-1).clamp(min=1e-6)).mean().item())
    return m.readout(h), drift


def rep(name, lg, drift=None, sub=None):
    ag = (lg.argmax(-1) == full_arg)
    s = f"{name:38} agree={ag.float().mean():.4f}"
    if sub is not None:
        for k, msk in sub.items():
            s += f"  {k}={ag[msk].float().mean():.4f}"
    if drift:
        s += "  drift/blk " + " ".join(f"{v:.2f}" for v in drift)
    print(s)


with torch.no_grad():
    never = exit_b == L                      # never depth-exits
    early = exit_b <= L // 2
    mid = (~never) & (~early)
    sub = {"never-exit": never, "early-exit": early, "mid-exit": mid}
    print(f"# position mix: early {early.float().mean():.3f} "
          f"mid {mid.float().mean():.3f} never {never.float().mean():.3f}")
    print(f"# temporal mask coverage by group: "
          f"early {temp.float().mean(0)[early].mean():.3f} "
          f"mid {temp.float().mean(0)[mid].mean():.3f} "
          f"never {temp.float().mean(0)[never].mean():.3f}")

    rep("full (sanity)", m.readout(hs[-1]), sub=sub)
    lg, dr = run(depth); rep("depth only", lg, dr, sub)
    lg, dr = run(None, temp); rep("temporal only", lg, dr, sub)
    lg, dr = run(depth, temp); rep("joint (depth | temporal t-1)", lg, dr, sub)
    lg, dr = run(depth, temp, shift=+1); rep("ctl wrong-neighbour t+1", lg, dr, sub)
    rnd = (torch.rand(depth.shape, generator=torch.Generator().manual_seed(0))
           < temp.float().mean()) & ~depth
    lg, dr = run(depth, rnd); rep("ctl random matched-density", lg, dr, sub)
    lg, dr = run(depth, temp, clean_source=True)
    rep("ABL joint, CLEAN t-1 source", lg, dr, sub)
    lg, dr = run(depth, renorm=True); rep("ABL depth only, RMS-renormed freeze", lg, dr, sub)

    # residual-stream norm growth: is a frozen early state off-distribution?
    print("# clean RMS(h) by block: " +
          " ".join(f"{hs[b].pow(2).mean().sqrt().item():.2f}" for b in range(L + 1)))
