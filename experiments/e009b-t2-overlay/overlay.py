"""Per (rung, checkpoint): oracle skip masks, EXECUTED capability, FLOPs.

THE ORACLE IS POST-HOC AND NON-REALIZABLE. Every mask here is built
from the full model's own outputs — the depth criterion is "this
intermediate readout already argmax-matches the final logits", which
cannot be evaluated without the final logits. Nothing measured here is
achievable by any router or predictor; these are ceilings.

What is new versus e009's oracle.py: capability under the oracle is
MEASURED, not assumed. e009 evaluated each (token, block) skip as an
independent counterfactual against the full model's hidden states and
reported the union, which is an upper bound on a jointly valid
configuration. Here every configuration is executed in one masked
forward pass with all of its skips applied simultaneously, and the
capability plotted on the overlay is that pass's actual output. If the
skips interact, the point moves down the capability axis instead of
silently keeping a parity it does not have.

Faithfulness, stated because it bounds the reading (DESIGN.md):
- depth-skip is implementation-faithful: the passthrough state is what
  later positions attend to, exactly as in CALM's state copy;
- temporal reuse is NOT: we charge zero FLOPs for the reused block but
  the pass still lets later positions attend to that position's true
  keys and values at that block. The joint arm is therefore doubly a
  ceiling, which is the conservative direction for a kill test.

Usage:  python overlay.py <ckpt.pt> [<ckpt.pt> ...]
Writes one JSON record per checkpoint into runs/.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ladder  # registers the extra rungs and provides the FLOPs counter
from corpus import CorpusGen
from model import SIZES, SwiGPT

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SEQ = 256
EVAL_SEQS = int(os.environ.get("EVAL_SEQS", 256))
KL_EPS = 0.1
EVAL_SEED = 9999  # same held-out stream as e009


def load(ckpt, device="cpu"):
    d = torch.load(ckpt, weights_only=True, map_location=device)
    m = SwiGPT(**SIZES[d["size"]]).to(device)
    m.load_state_dict(d["state"])
    m.eval()
    return m, d


@torch.no_grad()
def hidden_stack(m, ids):
    hs = [m.embed(ids)]
    for b in m.blocks:
        hs.append(b(hs[-1]))
    return hs


@torch.no_grad()
def masked_forward(m, ids, depth_skip, temporal_ok=None, shift=-1):
    """Execute one oracle configuration: all skips applied at once.

    depth_skip / temporal_ok: (L, B, T) bool. Depth wins where both
    hold. shift=-1 reuses from t-1 (the oracle); shift=+1 reuses from
    t+1, which is the wrong-neighbour control and is not realizable.
    """
    h = m.embed(ids)
    for b, block in enumerate(m.blocks):
        out = block(h)
        if temporal_ok is not None:
            reuse = temporal_ok[b].clone()
            reuse[:, 0 if shift < 0 else -1] = False  # no source position
            shifted = torch.roll(out, shifts=-shift, dims=1)
            out = torch.where(reuse.unsqueeze(-1), shifted, out)
        out = torch.where(depth_skip[b].unsqueeze(-1), h, out)  # passthrough
        h = out
    return m.readout(h)


def per_seq_capability(logits, ids, labs, full_arg):
    """Per-sequence sufficient statistics, so analyze.py can bootstrap
    over sequences with common random numbers across every point."""
    pred = logits[:, :-1].argmax(-1)
    tgt, tl = ids[:, 1:], labs[:, 1:]
    hit = (pred == tgt)
    cons = tl == 3
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           tgt.reshape(-1), reduction="none").view(tgt.shape)
    agree = (logits.argmax(-1) == full_arg)
    return dict(
        acc_hit=hit.sum(1).tolist(),
        cons_hit=(hit & cons).sum(1).tolist(),
        loss_sum=[round(v, 5) for v in loss.sum(1).tolist()],
        agree_hit=agree.sum(1).tolist(),
    )


@torch.no_grad()
def measure_chunk(m, ids, labs):
    """All measurements for one chunk of eval sequences. Sequences never
    interact, so chunking is exact — it only keeps the (B,T,V) logits
    tensors small enough to hold several at once."""
    L = len(m.blocks)
    B, T = ids.shape

    hs = hidden_stack(m, ids)
    full_logits = m.readout(hs[-1])
    full_arg = full_logits.argmax(-1)
    full_logp = F.log_softmax(full_logits, -1)

    def agrees(logits):
        arg = logits.argmax(-1) == full_arg
        kl = (full_logp.exp() * (full_logp - F.log_softmax(logits, -1))).sum(-1)
        return arg, kl <= KL_EPS

    # ---- depth: earliest block whose readout already satisfies the criterion
    def earliest(which):
        e = torch.full((B, T), L, device=ids.device)
        for b in range(L):
            a, k = agrees(m.readout(hs[b + 1]))
            c = a if which == "arg" else k
            e = torch.minimum(e, torch.where(c, torch.tensor(b + 1), torch.tensor(L)))
        return e

    exits = {c: earliest(c) for c in ("arg", "kl")}
    # depth_skip[b][t] : token t already exited at or before block b
    depth_skip = {c: torch.stack([exits[c] <= b for b in range(L)]) for c in exits}

    # ---- temporal: block b's output at t replaced by its output at t-1
    temporal_ok = {"arg": [], "kl": []}
    for b in range(L):
        h2 = hs[b + 1].clone()
        h2[:, 1:] = hs[b + 1][:, :-1]
        for bb in range(b + 1, L):
            h2 = m.blocks[bb](h2)
        a, k = agrees(m.readout(h2))
        a[:, 0] = False
        k[:, 0] = False
        temporal_ok["arg"].append(a)
        temporal_ok["kl"].append(k)
    temporal_ok = {c: torch.stack(v) for c, v in temporal_ok.items()}

    joint_skip = {c: depth_skip[c] | temporal_ok[c] for c in ("arg", "kl")}

    def frac_per_seq(mask):  # skip fraction over (block, position), per sequence
        return [round(v, 6) for v in mask.float().mean(dim=(0, 2)).tolist()]

    # Controls (decision 0010 item 2: controls must order correctly).
    # Executing the joint config scores HIGHER than depth-only, which is
    # only credible if the gain is specific to genuine t-1 reuse. Two
    # matched-density controls test that: reuse from the WRONG neighbour
    # (t+1) and a RANDOM mask. Both must score BELOW depth-only.
    # seeded: an unseeded control is a different control on every chunk and
    # on every re-run, so the tripwire would not be reproducible
    g = torch.Generator(device=ids.device).manual_seed(1234)
    ctl_random = (torch.rand(depth_skip["arg"].shape, generator=g,
                             device=ids.device)
                  < temporal_ok["arg"].float().mean()) & ~depth_skip["arg"]

    configs = {
        "full": (None, None, -1),
        "depth_arg": (depth_skip["arg"], None, -1),
        "joint_arg": (depth_skip["arg"], temporal_ok["arg"], -1),
        "depth_kl": (depth_skip["kl"], None, -1),
        "joint_kl": (depth_skip["kl"], temporal_ok["kl"], -1),
        "ctl_wrong_neighbour": (depth_skip["arg"], temporal_ok["arg"], +1),
        "ctl_random_mask": (depth_skip["arg"], ctl_random, -1),
    }
    skip_of = {"full": torch.zeros_like(depth_skip["arg"]),
               "depth_arg": depth_skip["arg"], "joint_arg": joint_skip["arg"],
               "depth_kl": depth_skip["kl"], "joint_kl": joint_skip["kl"],
               "ctl_wrong_neighbour": joint_skip["arg"],
               "ctl_random_mask": depth_skip["arg"] | ctl_random}

    points = {}
    for name, (dsk, tok, sh) in configs.items():
        logits = (full_logits if name == "full"
                  else masked_forward(m, ids, dsk, tok, shift=sh))
        points[name] = dict(skip_frac_per_seq=frac_per_seq(skip_of[name]),
                            **per_seq_capability(logits, ids, labs, full_arg))

    # e009-comparable fractions, per sequence (its joint excluded position 0;
    # its depth used all positions) — a cross-check that this is the same
    # instrument as e009's oracle.py
    e009_cmp = dict(
        depth_frac_arg=((L - exits["arg"].float()) / L).mean(1).tolist(),
        depth_frac_kl=((L - exits["kl"].float()) / L).mean(1).tolist(),
        temporal_arg=temporal_ok["arg"][:, :, 1:].float().mean(dim=(0, 2)).tolist(),
        joint_block_arg=joint_skip["arg"][:, :, 1:].float().mean(dim=(0, 2)).tolist(),
    )
    return points, e009_cmp, (labs[:, 1:] == 3).sum(1).tolist()


@torch.no_grad()
def measure(ckpt_path, device="cpu", chunk=64):
    m, meta = load(ckpt_path, device)
    size, L = meta["size"], len(m.blocks)
    gen = CorpusGen(seed=EVAL_SEED)
    x, lab = gen.batch(EVAL_SEQS, SEQ)

    points, e009_cmp, cons_n = {}, {}, []
    for lo in range(0, EVAL_SEQS, chunk):
        ids = torch.from_numpy(x[lo:lo + chunk]).to(device)
        labs = torch.from_numpy(lab[lo:lo + chunk]).to(device)
        p, e, cn = measure_chunk(m, ids, labs)
        for k, v in p.items():
            for kk, vv in v.items():
                points.setdefault(k, {}).setdefault(kk, []).extend(vv)
        for k, v in e.items():
            e009_cmp.setdefault(k, []).extend(v)
        cons_n += cn

    f = ladder.flops_per_token(size, SEQ)
    for k, v in points.items():
        sk = float(np.mean(v["skip_frac_per_seq"]))
        v["skip_frac"] = round(sk, 5)
        v["flops"] = round(ladder.effective_flops(size, sk, SEQ), 1)
    e009_cmp = {k: round(float(np.mean(v)), 4) for k, v in e009_cmp.items()}
    B, T = EVAL_SEQS, SEQ

    rec = dict(size=size, seed=meta["seed"], tokens=meta["tokens"], L=L,
               eval_seqs=B, eval_seq_len=T, kl_eps=KL_EPS,
               n_pred_per_seq=T - 1, cons_n_per_seq=cons_n,
               flops_full=f["total"], flops_block=f["block"], flops_head=f["head"],
               nonemb=m.nonemb_params(), e009_cmp=e009_cmp, points=points)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{size}_s{meta['seed']}_t{meta['tokens']}.json").write_text(
        json.dumps(rec))
    n = sum(rec["cons_n_per_seq"])
    print(f"{size:4} s{meta['seed']} t={meta['tokens']:>8} "
          f"joint_arg skip={points['joint_arg']['skip_frac']:.3f} "
          f"eff={points['joint_arg']['flops']/1e6:.3f}M/{f['total']/1e6:.3f}M "
          f"acc={sum(points['full']['acc_hit'])/(B*(T-1)):.4f}->"
          f"{sum(points['joint_arg']['acc_hit'])/(B*(T-1)):.4f} "
          f"cons={sum(points['full']['cons_hit'])/n:.4f}->"
          f"{sum(points['joint_arg']['cons_hit'])/n:.4f} "
          f"agree={sum(points['joint_arg']['agree_hit'])/(B*T):.4f} "
          f"[e009 depth_arg {e009_cmp['depth_frac_arg']:.3f} "
          f"joint {e009_cmp['joint_block_arg']:.3f}]", flush=True)
    return rec


if __name__ == "__main__":
    for p in sys.argv[1:]:
        measure(Path(p))
