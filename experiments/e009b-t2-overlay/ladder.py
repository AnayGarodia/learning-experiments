"""Extended dense ladder + honest inference-FLOPs accounting (e009b).

IMPORTING THIS MODULE REGISTERS the three new rungs into e009's
model.SIZES (in place, same dict object), so every other e009b module —
and e009's own train.py / oracle.py — sees them. The e005 corpus and
e009's model/train code are imported, never copied: this experiment
must sit on the same corpus and recipe as e009 or the reused s/m/l
checkpoints are not frontier points.

FLOPs accounting (DESIGN.md "Inference FLOPs per token"): matmul FLOPs
only, 2 per multiply-accumulate, forward pass, averaged over positions
of a seq_len=256 causal sequence (mean context (T+1)/2). Elementwise
ops (RMSNorm, SiLU, softmax, residual adds) are not counted. The
unembedding head is counted and is NOT skippable by any oracle.

    per block, per token: 8 d^2 + 6 d h + 4 d ctx     (h = int(8d/3))
    per token, once:      2 d V                       (head, unskippable)

Usage:
    python ladder.py                  # self-check (unit test)
    python ladder.py train xs 0       # train one rung on the pinned corpus
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e009-redundancy"))

from corpus import VOCAB  # noqa: E402  (the pinned e005 corpus)
from model import SIZES, SwiGPT  # noqa: E402

SEQ = 256
CKPTS = HERE / "ckpts"

# Rungs extending e009's d/L family in both directions. Two BELOW e009's
# smallest on purpose: the frontier bends at the small end, and that is
# where a dense model can undercut the oracle. One above, to reach the
# scale where e009's skip fraction is largest.
#
# xs0 (0.880M FLOPs/tok) and xs1 (1.184M) were added by the pre-launch
# verifier, and they are the difference between a callable and an
# uncallable experiment. At the primary (saturated) training level every
# oracle point on both capability axes fell inside the SINGLE frontier
# interval [xxs, xs], which spans 2.95x in FLOPs; the fit-free bracket
# there is [0.57, 1.68] against a survive line of 1.6, and no number of
# seeds or training levels narrows a bracket that rests on one interval.
# Only rungs inside it do. Both sit under the 1.363M a frontier point has
# to be at for that bracket to clear 1.6.
EXTRA = {"xxs": dict(d=32, n_layers=4),
         "xs0": dict(d=48, n_layers=5),
         "xs1": dict(d=56, n_layers=6),
         "xs": dict(d=64, n_layers=6),
         "xl": dict(d=384, n_layers=14)}
SIZES.update(EXTRA)

# small -> large, the order every table and plot uses
LADDER = ["xxs", "xs0", "xs1", "xs", "s", "m", "l", "xl"]


def mlp_hidden(d):
    """SwiGLU hidden width, as e009's Block computes it."""
    return int(8 * d / 3)


def flops_per_token(size, seq=SEQ):
    """Inference matmul FLOPs per token, itemised. See module docstring."""
    d, L = SIZES[size]["d"], SIZES[size]["n_layers"]
    h = mlp_hidden(d)
    ctx = (seq + 1) / 2  # causal: position i attends to i+1 keys
    attn_proj = 8 * d * d           # qkv (6d^2) + output proj (2d^2)
    attn_quad = 4 * d * ctx         # QK^T (2 d ctx) + A@V (2 d ctx)
    mlp = 6 * d * h                 # gate + up (4dh) + down (2dh)
    per_block = attn_proj + attn_quad + mlp
    head = 2 * d * VOCAB            # unembedding: the unskippable floor
    return dict(d=d, L=L, h=h, per_block=per_block, block=per_block * L,
                head=head, total=per_block * L + head)


def effective_flops(size, skip_fraction, seq=SEQ):
    """FLOPs/token when `skip_fraction` of (token, block) passes are free.

    Skipped passes are charged zero; the head is always paid. Whether a
    kernel can realise the skip is the T3 question, answered in e009
    (coarse-structured, hence priceable) and not re-litigated here.
    """
    f = flops_per_token(size, seq)
    return f["head"] + (1.0 - skip_fraction) * f["block"]


def _selfcheck():
    # 1) FLOPs counter vs the hand computation in DESIGN.md, rung s.
    #    d=96, L=8, h=int(8*96/3)=256, ctx=(256+1)/2=128.5
    #      8*96^2                 =  73_728
    #      6*96*256               = 147_456
    #      4*96*128.5             =  49_344
    #      per block              = 270_528
    #      x8 blocks              = 2_164_224
    #      head 2*96*5000         =   960_000
    #      total                  = 3_124_224
    f = flops_per_token("s")
    assert f["h"] == 256, f["h"]
    assert f["per_block"] == 73_728 + 147_456 + 49_344 == 270_528, f
    assert f["block"] == 2_164_224, f
    assert f["head"] == 960_000, f
    assert f["total"] == 3_124_224, f["total"]

    # 2) the counter's h must be the model's h, at every rung, or the
    #    count is of a different network than the one we run
    for k in LADDER:
        m = SwiGPT(**SIZES[k])
        assert m.blocks[0].wg.out_features == flops_per_token(k)["h"], k
        assert len(m.blocks) == SIZES[k]["n_layers"], k

    # 3) skipping everything must leave exactly the head
    assert effective_flops("s", 1.0) == 960_000
    assert effective_flops("s", 0.0) == 3_124_224

    print(f"{'rung':4} {'d':>4} {'L':>3} {'nonemb':>9} {'FLOPs/tok':>11} "
          f"{'head%':>6} {'@88% skip':>11} {'ratio':>6}")
    for k in LADDER:
        f = flops_per_token(k)
        n = SwiGPT(**SIZES[k]).nonemb_params()
        eff = effective_flops(k, 0.88)
        print(f"{k:4} {f['d']:>4} {f['L']:>3} {n/1e6:>8.3f}M {f['total']:>11,.0f} "
              f"{100*f['head']/f['total']:>5.1f}% {eff:>11,.0f} "
              f"{f['total']/eff:>5.2f}x")
    print("ladder self-check ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        import train  # e009's trainer, unmodified
        train.CKPTS = CKPTS
        size, seed = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
        tt = int(sys.argv[4]) if len(sys.argv) > 4 else train.TOTAL_TOKENS
        train.train(size, seed=seed, total_tokens=tt)
    else:
        _selfcheck()
