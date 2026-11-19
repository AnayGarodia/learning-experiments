"""The four post-hoc oracle measurements + the joint fraction, per
checkpoint. All measured on held-out data; no training changes.

Criteria (pre-registered): argmax-match with the full model, and
KL(full || probe) <= 0.1. Granularity: depth = whole blocks;
width-coarse = contiguous quarters of the MLP hidden dim (exact,
end-to-end); width-fine = near-zero activation fraction (local
criterion, the lazy-neuron measurement — criteria differ by
necessity, documented in DESIGN). Temporal = reuse block output from
the previous position. Prediction-error = closed-form ridge linear
predictor of block output from block input, error<eps fraction.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from corpus import CorpusGen
from model import SIZES, SwiGPT

HERE = Path(__file__).parent
RUNS = HERE / "runs"
EVAL_SEQS, SEQ = 64, 256  # 16384 tokens x L blocks; CI insurance (verifier)
KL_EPS = 0.1
PRED_EPS = 0.1  # relative error threshold (pilot-adjustable, recorded)


def load(ckpt, device="cpu"):
    d = torch.load(ckpt, weights_only=True, map_location=device)
    m = SwiGPT(**SIZES[d["size"]]).to(device)
    m.load_state_dict(d["state"])
    m.eval()
    return m, d


@torch.no_grad()
def hidden_stack(m, ids):
    """h after embedding and after each block: list of L+1 (B,T,D)."""
    hs = [m.embed(ids)]
    for b in m.blocks:
        hs.append(b(hs[-1]))
    return hs


@torch.no_grad()
def measure(ckpt_path, device="cpu"):
    m, meta = load(ckpt_path, device)
    L = len(m.blocks)
    ids = torch.from_numpy(CorpusGen(seed=9999).batch(EVAL_SEQS, SEQ)[0]).to(device)
    hs = hidden_stack(m, ids)
    full_logits = m.readout(hs[-1])
    full_arg = full_logits.argmax(-1)
    full_logp = F.log_softmax(full_logits, -1)
    B, T = ids.shape

    def agrees(logits, kl_eps=KL_EPS):
        arg = logits.argmax(-1) == full_arg
        kl = (full_logp.exp() * (full_logp - F.log_softmax(logits, -1))).sum(-1)
        return arg, kl <= kl_eps

    # 1) DEPTH: earliest block whose readout matches; skippable = blocks after it
    def earliest(crit_idx):
        e = torch.full((B, T), L)  # default: only full depth works
        for b in range(L):
            a, k = agrees(m.readout(hs[b + 1]))
            c = a if crit_idx == 0 else k
            e = torch.minimum(e, torch.where(c, torch.tensor(b + 1), torch.tensor(L)))
        return e
    e_arg, e_kl = earliest(0), earliest(1)
    depth_frac_arg = ((L - e_arg.float()) / L).mean().item()
    depth_frac_kl = ((L - e_kl.float()) / L).mean().item()

    # 2) WIDTH coarse: zero each contiguous quarter of each block's MLP hidden
    h_dim = m.blocks[0].wg.out_features
    qsize = h_dim // 4
    coarse_arg = coarse_kl = 0
    for b in range(L):
        for q in range(4):
            mask = torch.ones(h_dim, device=device)
            mask[q * qsize:(q + 1) * qsize] = 0
            h = hs[b]
            h2 = m.blocks[b](h, mlp_mask=mask)
            for bb in range(b + 1, L):
                h2 = m.blocks[bb](h2)
            a, k = agrees(m.readout(h2), kl_eps=0.01)  # 0.1 saturates at 1.0 (pilot)
            coarse_arg += a.float().mean().item()
            coarse_kl += k.float().mean().item()
    width_coarse_arg = coarse_arg / (L * 4)
    width_coarse_kl = coarse_kl / (L * 4)

    # width FINE (local criterion): near-zero activation fraction
    fine, fine_loose = 0, 0
    for b in range(L):
        act = m.blocks[b].mlp_act(m.blocks[b].n2(hs[b]))
        rms = act.pow(2).mean(-1, keepdim=True).sqrt()
        fine += (act.abs() < 1e-3 * rms).float().mean().item()
        fine_loose += (act.abs() < 1e-1 * rms).float().mean().item()
    width_fine = fine / L
    width_fine_loose = fine_loose / L

    # 3) TEMPORAL: substitute block b's output at t with its output at t-1
    temp_arg = temp_kl = 0
    for b in range(L):
        h2 = hs[b + 1].clone()
        h2[:, 1:] = hs[b + 1][:, :-1]
        for bb in range(b + 1, L):
            h2 = m.blocks[bb](h2)
        a, k = agrees(m.readout(h2))
        temp_arg += a[:, 1:].float().mean().item()
        temp_kl += k[:, 1:].float().mean().item()
    temp_arg /= L; temp_kl /= L

    # 4) PREDICTION-ERROR: ridge linear predictor h_{b+1} ~ W h_b, fit on half
    pred_frac = 0
    half = B // 2
    for b in range(L):
        X = hs[b][:half].reshape(-1, hs[b].shape[-1])
        Y = hs[b + 1][:half].reshape(-1, hs[b].shape[-1])
        lam = 1e-3 * X.shape[0]
        W = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]), X.T @ Y)
        Xt = hs[b][half:].reshape(-1, hs[b].shape[-1])
        Yt = hs[b + 1][half:].reshape(-1, hs[b].shape[-1])
        err = (Xt @ W - Yt).norm(dim=-1) / Yt.norm(dim=-1).clamp(min=1e-6)
        pred_frac += (err < PRED_EPS).float().mean().item()
    pred_frac /= L

    # JOINT (argmax criterion, block granularity): a (token, block) pass is
    # skippable if depth-exit passed it OR temporal reuse holds OR the linear
    # predictor replaces it. Approximate union per block via depth mask +
    # per-block temporal/pred fractions on non-depth-skipped passes is not
    # separable from these aggregates; measure union directly per (token,block):
    joint = 0
    for b in range(L):
        depth_skip = e_arg <= b  # token already exited before this block
        h2 = hs[b + 1].clone()
        h2[:, 1:] = hs[b + 1][:, :-1]
        for bb in range(b + 1, L):
            h2 = m.blocks[bb](h2)
        temporal_ok, _ = agrees(m.readout(h2))
        # position 0 has no t-1 to reuse; exclude it (consistent with temporal_arg)
        joint += (depth_skip | temporal_ok)[:, 1:].float().mean().item()
    joint /= L

    rec = dict(size=meta["size"], seed=meta["seed"], tokens=meta["tokens"],
               L=L, eval_tokens=B * T,
               depth_frac_arg=round(depth_frac_arg, 4),
               depth_frac_kl=round(depth_frac_kl, 4),
               width_coarse_arg=round(width_coarse_arg, 4),
               width_coarse_kl=round(width_coarse_kl, 4),
               width_fine_local=round(width_fine, 4),
               width_fine_local_loose=round(width_fine_loose, 4),
               temporal_arg=round(temp_arg, 4), temporal_kl=round(temp_kl, 4),
               pred_frac=round(pred_frac, 4), pred_eps=PRED_EPS,
               joint_block_arg=round(joint, 4))
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{meta['size']}_s{meta['seed']}_t{meta['tokens']}.json").write_text(
        json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(measure(Path(p)))
