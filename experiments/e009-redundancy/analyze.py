"""Fraction-vs-training-tokens curves + T1/T2/T3 verdict table."""
import json
import glob
import collections
from pathlib import Path

HERE = Path(__file__).parent
KEYS = ["depth_frac_arg", "depth_frac_kl", "width_coarse_arg", "width_coarse_kl",
        "width_fine_local", "temporal_arg", "temporal_kl", "pred_frac",
        "joint_block_arg"]


def main():
    recs = [json.loads(open(f).read()) for f in glob.glob(str(HERE / "runs/*.json"))]
    by_size = collections.defaultdict(list)
    for r in recs:
        by_size[r["size"]].append(r)
    print(f"{'size':4} {'tokens':>9} " + " ".join(f"{k[:12]:>12}" for k in KEYS))
    for size in ("s", "m", "l"):
        for r in sorted(by_size.get(size, []), key=lambda r: r["tokens"]):
            print(f"{size:4} {r['tokens']:>9} " +
                  " ".join(f"{r[k]:>12.3f}" for k in KEYS))
    # artifact pack: results.csv + figure
    art = HERE / "artifacts"
    art.mkdir(exist_ok=True)
    with open(art / "results.csv", "w") as f:
        f.write("size,seed,tokens," + ",".join(KEYS) + "\n")
        for size in ("s", "m", "l"):
            for r in sorted(by_size.get(size, []), key=lambda r: (r.get("seed", 0), r["tokens"])):
                f.write(f"{size},{r.get('seed',0)},{r['tokens']}," +
                        ",".join(f"{r[k]:.4f}" for k in KEYS) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        panels = [("joint_block_arg", "Joint skippable (argmax)"),
                  ("depth_frac_kl", "Depth skippable (KL<=0.1)"),
                  ("temporal_arg", "Temporal reuse (argmax)")]
        for ax, (key, title) in zip(axes, panels):
            for size, marker in (("s", "o"), ("m", "s"), ("l", "^")):
                rs = sorted((r for r in by_size.get(size, []) if r.get("seed", 0) == 0),
                            key=lambda r: r["tokens"])
                ax.plot([r["tokens"] for r in rs], [r[key] for r in rs],
                        marker=marker, label=f"{size}")
            ax.set_xscale("log")
            ax.set_title(title)
            ax.set_xlabel("training tokens")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("fraction of (token, block) compute")
        axes[0].legend(title="model size")
        fig.suptitle("e009: oracle-redundant compute vs training (SwiGLU ladder 0.89M/3.07M/9.44M)")
        fig.tight_layout()
        fig.savefig(art / "results.png", dpi=150)
        print("wrote artifacts/results.csv artifacts/results.png")
    except ImportError:
        print("wrote artifacts/results.csv (no matplotlib for figure)")
    # verdict hints
    for size, rs in by_size.items():
        rs = sorted(rs, key=lambda r: r["tokens"])
        if len(rs) >= 2:
            d0, d1 = rs[0]["joint_block_arg"], rs[-1]["joint_block_arg"]
            trend = "GROWS" if d1 > d0 + 0.02 else ("SHRINKS" if d1 < d0 - 0.02 else "STABLE")
            print(f"{size}: joint(arg) {d0:.3f} -> {d1:.3f} with training: {trend}")
            r = rs[-1]
            print(f"{size}: criterion gap depth arg-vs-kl {r['depth_frac_arg']:.3f} vs "
                  f"{r['depth_frac_kl']:.3f}; granularity gap width coarse {r['width_coarse_arg']:.3f}"
                  f" vs fine(local) {r['width_fine_local']:.3f}")


if __name__ == "__main__":
    main()
