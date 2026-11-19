"""Is joint>depth a real interaction, or does ANY lateral mixing repair
the depth-freeze corruption?  Controls: random mask of matched density,
and reuse from the WRONG neighbour (t+1)."""
import sys
sys.path.insert(0, "/home/azureuser/e009b")
import torch, torch.nn.functional as F
from overlay import load, hidden_stack
from corpus import CorpusGen

m, meta = load("/home/azureuser/e009/ckpts/s_s0_t5996544.pt")
L = len(m.blocks)
g = CorpusGen(seed=9999)
x, lab = g.batch(64, 256)
ids, labs = torch.from_numpy(x), torch.from_numpy(lab)
B, T = ids.shape
hs = hidden_stack(m, ids)
full_arg = m.readout(hs[-1]).argmax(-1)

def agrees(lg): return (lg.argmax(-1) == full_arg)

e = torch.full((B, T), L)
for b in range(L):
    c = agrees(m.readout(hs[b + 1]))
    e = torch.minimum(e, torch.where(c, torch.tensor(b + 1), torch.tensor(L)))
depth = torch.stack([e <= b for b in range(L)])

temp = []
for b in range(L):
    h2 = hs[b + 1].clone(); h2[:, 1:] = hs[b + 1][:, :-1]
    for bb in range(b + 1, L):
        h2 = m.blocks[bb](h2)
    a = agrees(m.readout(h2)); a[:, 0] = False
    temp.append(a)
temp = torch.stack(temp)

def fwd(depth_skip, reuse=None, shift=-1):
    h = m.embed(ids)
    for b, block in enumerate(m.blocks):
        out = block(h)
        if reuse is not None:
            r = reuse[b].clone(); r[:, 0] = False; r[:, -1] = False
            s = torch.roll(out, shifts=-shift, dims=1)
            out = torch.where(r.unsqueeze(-1), s, out)
        out = torch.where(depth_skip[b].unsqueeze(-1), h, out)
        h = out
    return m.readout(h)

def rep(name, lg, mask):
    cons = labs[:, 1:] == 3
    pred = lg[:, :-1].argmax(-1); tgt = ids[:, 1:]
    print(f"{name:34} skip={mask.float().mean():.4f} agree={agrees(lg).float().mean():.4f} "
          f"cons={((pred==tgt)&cons).sum()/cons.sum():.4f}")

zero = torch.zeros_like(depth)
rand = (torch.rand(depth.shape) < temp.float().mean()) & ~depth
rep("depth only", fwd(depth), depth)
rep("depth | temporal (t-1)", fwd(depth, temp), depth | temp)
rep("depth | temporal from t+1", fwd(depth, temp, shift=+1), depth | temp)
rep("depth | RANDOM mask, same density", fwd(depth, rand), depth | rand)
