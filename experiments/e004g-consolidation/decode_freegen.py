"""VERIFIER instrument (2026-08-17): decode free generations from a saved
LoRA adapter and classify why `freegen` reads ~0.000 while the teacher-forced
`unprompted` score reads ~1.000.

The grid's own adapters were NOT saved (adapter saving was added to
consolidate.py after the grid ran), so this is pointed at a REPRODUCTION of
the mechanism, not at the grid's outputs.

Differences from consolidate.freegen_accuracy, all deliberate, all diagnostic:
  * teacher-forced accuracy is scored on the SAME items as free generation
    (the harness scores them on two different draws), so every classification
    is paired;
  * max_new_tokens = answer + 8 instead of + 2, and specials are kept, so a
    continuation is visible rather than truncated;
  * 4 items per skill are also decoded UNBATCHED, to rule out the left-pad /
    position-id batching artifact as the cause.

Classes: a=correct answer then no terminator (non-termination),
b=correct answer then extra content beyond the line, c=wrong content,
d=empty/degenerate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, str(SRC))
import consolidate as C  # noqa: E402

ADAPTER = Path(sys.argv[2])
SKILLS = [int(x) for x in sys.argv[3].split(",")]
OUT = Path(sys.argv[4])
N = int(sys.argv[5]) if len(sys.argv) > 5 else 32

torch.set_num_threads(10)
tok, model = C.load("Qwen/Qwen2.5-0.5B")
from peft import PeftModel  # noqa: E402
model = PeftModel.from_pretrained(model, str(ADAPTER))
model.eval()


def gen(items, batch):
    """Greedy decode; returns raw continuation strings (specials kept)."""
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        enc = [tok(p, add_special_tokens=False).input_ids for p, _ in chunk]
        T = max(len(e) for e in enc)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((len(enc), T), pad, dtype=torch.long)
        mask = torch.zeros((len(enc), T), dtype=torch.long)
        for b, e in enumerate(enc):
            ids[b, T - len(e):] = torch.tensor(e)
            mask[b, T - len(e):] = 1
        nmax = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                   for _, a in chunk) + 8
        with torch.no_grad():
            g = model.generate(input_ids=ids, attention_mask=mask,
                               max_new_tokens=nmax, do_sample=False,
                               pad_token_id=pad)
        out += [tok.decode(g[b, T:], skip_special_tokens=False)
                for b in range(len(chunk))]
    return out


def classify(raw, ans):
    txt = raw.replace("<|endoftext|>", "\x00")
    body = txt.split("\x00")[0]
    if not body.strip():
        return "d"
    first = body.strip().split("\n")[0].strip()
    if first == ans:
        return "ok"          # passes the harness rule (first line == answer)
    if body.strip().startswith(ans):
        # right answer, then more. On the same line -> non-termination.
        return "a" if "\n" not in body.strip() else "b"
    return "c"


rec = {"adapter": str(ADAPTER), "code_sha": C.git_sha(), "n_per_skill": N,
       "skills": {}}
for s in SKILLS:
    rng = np.random.default_rng(C.EVAL_SEED + 4000 + s)
    items = C.make_items(s, N, rng, False, C.EVAL_WORDS)
    tf = C.accuracy(model, tok, items)                 # paired, same items
    harness = C.freegen_accuracy(model, tok, items)    # harness rule, +2 toks
    raws = gen(items, 8)
    solo = gen(items[:4], 1)
    counts = {}
    for raw, (_, a) in zip(raws, items):
        k = classify(raw, a)
        counts[k] = counts.get(k, 0) + 1
    rec["skills"][C.SKILLS[s][0]] = {
        "skill_index": s,
        "teacher_forced_same_items": round(tf, 3),
        "harness_freegen_same_items": round(harness, 3),
        "counts": counts,
        "prefix_correct_frac": round(
            sum(counts.get(k, 0) for k in ("ok", "a", "b")) / len(items), 3),
        "samples": [{"prompt": p, "gold": a, "gen": r,
                     "class": classify(r, a)}
                    for (p, a), r in list(zip(items, raws))[:6]],
        "unbatched_samples": [{"gold": a, "gen": r, "class": classify(r, a)}
                              for (_, a), r in zip(items[:4], solo)],
    }
    print(json.dumps({C.SKILLS[s][0]: {k: v for k, v in
                                       rec["skills"][C.SKILLS[s][0]].items()
                                       if k != "samples"}}, indent=1),
          flush=True)
OUT.write_text(json.dumps(rec, indent=1))
print("wrote", OUT)
