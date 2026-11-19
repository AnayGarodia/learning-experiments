"""e013: the exchange rate R = (gradient examples to reach C) /
(in-context examples to reach C), same base, same skill, same C.

REUSES e004g's harness by IMPORT, not by fork: the tokenizer/answer-span
encoder, the teacher-forced exact-match capability metric, the
mismatched-shot control and the copy-oracle degeneracy check are
`consolidate.py`'s own functions, called here with patched module
globals (word pools, skill list, shot count). Every adaptation is listed
in DESIGN.md "Adaptations to e004g's harness".

Two curves per skill, one capability metric, one eval query set:
  IN-CONTEXT  accuracy vs number of shots n in {0,1,2,3,4,6,8,12,16,24,32,48,64}
  GRADIENT    accuracy vs number of distinct training examples N in
              {1,2,4,8,16,32,64,128}, LoRA, swept LR, early stop on dev
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
E004G = HERE.parent / "e004g-consolidation"
sys.path.insert(0, str(E004G))
import consolidate as c  # noqa: E402  (e004g harness, reused wholesale)

# ---- word pool ------------------------------------------------------------
# e004g split its 64-word pool 32 train / 32 eval. e013 needs up to 128
# DISTINCT gradient examples and up to 64 distinct shots, so the pool is
# enlarged with the 5-letter lowercase words of e004g's own pinned
# generality.txt (Alice in Wonderland). Disclosed adaptation A1.
import re  # noqa: E402


def build_pool():
    alice = (E004G / "generality.txt").read_text().lower()
    words = sorted(set(re.findall(r"\b[a-z]{5}\b", alice)) | set(c.WORDS))
    rng = np.random.default_rng(20260816)
    p = [words[i] for i in rng.permutation(len(words))]
    return p[:64], p[64:96], p[96:]          # eval, dev, train


EVAL_WORDS, DEV_WORDS, TRAIN_WORDS = build_pool()

# ---- skills ---------------------------------------------------------------
# NATURAL = e004g skills that PASSED its gate and are non-degenerate.
# NOVEL   = constructed to have no plausible pretraining analogue. X2 lives
#           or dies here. Format, answer length and eval are identical to
#           the natural matched partner (B_first), so the only thing that
#           differs is whether pretraining could have supplied the mapping.
_perm = list("abcdefghijklmnopqrstuvwxyz")
# seed 3 is the first that gives a DERANGEMENT (no letter maps to itself),
# so no answer is ever the natural "first letter" answer by accident.
_perm = [_perm[i] for i in np.random.default_rng(3).permutation(26)]
BIJECTION = dict(zip("abcdefghijklmnopqrstuvwxyz", _perm))

NOVEL = [
    # arbitrary bijection on the first letter. No natural analogue; the map
    # is only learnable from demonstrated letters, in EITHER arm.
    ("N_map", "N", lambda w: BIJECTION[w[0]]),
    # first letter advanced 2 in the alphabet: systematic (generalizes to
    # unseen first letters) but arbitrary. Weakly novel -- "next letter"
    # plausibly exists in pretraining; labelled as such everywhere.
    ("N_shift2", "N", lambda w: chr((ord(w[0]) - 97 + 2) % 26 + 97)),
    # arbitrary fixed permutation of the 5 letter POSITIONS. Systematic,
    # multi-token answer, matched partner is B_upper.
    ("N_perm", "N", lambda w: "".join(w[i] for i in (2, 4, 0, 3, 1))),
]

NATURAL_NAMES = ["B_first", "B_upper"]


def setup_skills():
    """Patch e004g's module globals. Its make_prompt/make_items/accuracy/
    copy_oracle then operate on e013's pools and skills unchanged."""
    names = [s[0] for s in c.SKILLS]
    for s in NOVEL:
        if s[0] not in names:
            c.SKILLS.append(s)
    c.N_SKILLS = len(c.SKILLS)
    c.MARKERS = [f"<<K{i}>>" for i in range(c.N_SKILLS)]
    assert c.UNKNOWN_MARKER not in c.MARKERS
    c.TRAIN_WORDS, c.EVAL_WORDS = TRAIN_WORDS, EVAL_WORDS
    return {s[0]: i for i, s in enumerate(c.SKILLS)}


IDX = setup_skills()
SKILL_SET = NATURAL_NAMES + [s[0] for s in NOVEL]
# Both axes are sqrt(2)-ish grids, NOT doubling grids: the identifiability
# dry-run showed a true X3 outcome can put all three capability crossings
# inside ONE interval of a doubling grid, which is the e009b failure mode.
SHOT_GRID = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
N_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
LR_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
EVAL_SEED, N_EVAL, N_DEV = 91, 200, 64
STEPS, CKPT_EVERY, MAX_BATCH = 200, 25, 16


LOGIT_TOKEN_BUDGET = 2400   # B*T cap: 2400*151936*4B ~ 1.5GB of logits


def eval_batch(tok, items):
    """e004g fixed SHOTS=3 (seq ~36) so a batch of 16 was always safe. e013
    varies shots to 64 (seq ~520) and the [B,T,V] logits tensor is 5GB at
    B=16 -- that OOM-wedged the box once. Batch is therefore budgeted in
    TOKENS, not sequences."""
    T = max(len(tok(p, add_special_tokens=False).input_ids) +
            len(tok(" " + a, add_special_tokens=False).input_ids)
            for p, a in items[:8])
    return max(1, min(16, LOGIT_TOKEN_BUDGET // max(T, 1)))


def kind(name):
    return "novel" if name.startswith("N_") else "natural"


def eval_items(skill, prompted, shots=0, shot_skill=None, pool=None,
               n=N_EVAL, seed_off=0):
    """The IDENTICAL query set for every condition (paired comparison:
    sampling noise cancels between the two arms)."""
    c.SHOTS = shots
    rng = np.random.default_rng(EVAL_SEED + skill + seed_off)
    return c.make_items(skill, n, rng, prompted, pool or EVAL_WORDS,
                        shot_skill=shot_skill)


def coverage_oracle(skill, shots, n=N_EVAL, seed_off=0):
    """Ceiling for a learner that can only answer inputs whose FIRST LETTER
    was demonstrated. N_map is coverage-limited by construction, in BOTH
    arms; without this number its curve cannot be read."""
    c.SHOTS = shots
    rng = np.random.default_rng(EVAL_SEED + skill + seed_off)
    words = [EVAL_WORDS[int(i)] for i in rng.integers(len(EVAL_WORDS), size=n)]
    hit = 0
    for w in words:
        if shots == 0:
            continue
        sh = rng.choice([x for x in TRAIN_WORDS if x != w], size=shots,
                        replace=False)
        hit += int(w[0] in {x[0] for x in sh})
    return round(hit / n, 3)


# ---- arm 1: in-context ----------------------------------------------------

def icl_curve(model, tok, name, seed=0, grid=None):
    """VERIFIER FIX (2026-08-17): the seed argument used to be recorded but
    never used -- every draw ran off EVAL_SEED + skill, so three "seeds" of
    the in-context arm would have produced three IDENTICAL curves and the
    grid's seed budget on this arm would have bought nothing. `seed` now
    offsets the query and shot draws (seed 0 reproduces the pilot exactly)."""
    s = IDX[name]
    off = 1000 * seed
    out = []
    for nsh in (grid or SHOT_GRID):
        t0 = time.time()
        it = eval_items(s, nsh > 0, nsh, seed_off=off)
        bs = eval_batch(tok, it)
        acc = c.accuracy(model, tok, it, batch=bs)
        mm = (c.accuracy(model, tok, eval_items(
            s, True, nsh, shot_skill=IDX["B_upper"] if name != "B_upper"
            else IDX["B_first"], seed_off=off), batch=bs) if nsh else acc)
        rec = {"kind": "icl", "skill": name, "family": kind(name), "seed": seed,
               "n": nsh, "acc": round(acc, 4), "mismatched": round(mm, 4),
               "coverage_oracle": coverage_oracle(s, nsh, seed_off=off),
               "wall_s": round(time.time() - t0, 1)}
        out.append(rec)
        RUNS.mkdir(exist_ok=True)
        (RUNS / f"icl_{name}_s{seed}.json").write_text(json.dumps(
            {"base": "Qwen/Qwen2.5-0.5B", "git_sha": c.git_sha(),
             "records": out}, indent=1))
        print(f"  [icl {name}] n={nsh:3d} acc={acc:.3f} mm={mm:.3f} "
              f"cov={rec['coverage_oracle']:.3f} ({rec['wall_s']}s)", flush=True)
    return out


# ---- arm 2: gradient ------------------------------------------------------

def train_words_for(N, seed):
    """NESTED subsets: the N=2 set contains the N=1 set. A learning curve
    whose points use disjoint example sets confounds sample count with
    sample luck."""
    rng = np.random.default_rng(1000 + seed)
    order = [TRAIN_WORDS[int(i)] for i in rng.permutation(len(TRAIN_WORDS))]
    return order[:N]


def grad_point(model, tok, name, N, lr, seed=0, steps=STEPS):
    """LoRA fine-tune on N DISTINCT examples; early stop on a held-out dev
    split; report eval accuracy at the best-dev checkpoint."""
    from peft import get_peft_model, LoraConfig
    s = IDX[name]
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    torch.manual_seed(seed)
    pm = get_peft_model(model, cfg)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    words = train_words_for(N, seed)
    items = [(c.query_line(s, w), c.SKILLS[s][2](w)) for w in words]
    # Same replicate offset as icl_curve, so the two arms stay PAIRED on one
    # query set within a seed (seed 0 reproduces the pilot exactly).
    off = 1000 * seed
    dev = eval_items(s, False, pool=DEV_WORDS, n=N_DEV, seed_off=500 + off)
    ev = eval_items(s, False, seed_off=off)
    # RULE-vs-TABLE split (verifier addition). The whole memorisation attack on
    # the novel arm is "gradients can absorb a lookup table, a context window
    # cannot". It is settled by asking whether the gradient arm answers eval
    # words whose FIRST LETTER was never in its training set: a rule learner
    # does, a table learner cannot. eval_items(prompted=False) consumes the rng
    # exactly as below (make_prompt draws nothing when unprompted), so this
    # reproduces the same query words.
    ev_words = [EVAL_WORDS[int(i)] for i in
                np.random.default_rng(EVAL_SEED + s + off).integers(
                    len(EVAL_WORDS), size=N_EVAL)]
    assert [c.query_line(s, w) for w in ev_words] == [p for p, _ in ev]
    seen = {w[0] for w in words}
    ev_seen = [it for it, w in zip(ev, ev_words) if w[0] in seen]
    ev_unseen = [it for it, w in zip(ev, ev_words) if w[0] not in seen]

    def split_eval():
        a_s = c.accuracy(pm, tok, ev_seen) if ev_seen else None
        a_u = c.accuracy(pm, tok, ev_unseen) if ev_unseen else None
        tot = ((a_s or 0) * len(ev_seen) + (a_u or 0) * len(ev_unseen)) / len(ev)
        return round(tot, 4), (None if a_s is None else round(a_s, 4)), \
            (None if a_u is None else round(a_u, 4))

    rng = np.random.default_rng(seed)
    t0, ckpts = time.time(), []
    for step in range(steps + 1):
        if step % CKPT_EVERY == 0:
            e_all, e_s, e_u = split_eval()
            ckpts.append({"step": step,
                          "dev": round(c.accuracy(pm, tok, dev), 4),
                          "eval": e_all, "eval_seen": e_s, "eval_unseen": e_u})
        if step == steps:
            break
        batch = [items[int(i)] for i in rng.integers(len(items),
                                                     size=min(N, MAX_BATCH))]
        ids, mask, st, ln = c.encode(tok, batch)
        lg = c.answer_logits(pm, ids, mask, st, ln)
        loss = torch.nn.functional.cross_entropy(
            lg, c.answer_targets(ids, st, ln))
        opt.zero_grad(); loss.backward(); opt.step()
    # Ties on dev break toward the LATEST checkpoint, i.e. the MORE-trained
    # model. Breaking toward the earliest would return the untrained base
    # whenever dev is flat (common at N=1-4 on coverage-limited skills),
    # which understates the gradient arm and so INFLATES R in ICL's favour --
    # the direction of the exciting answer.
    best = max(ckpts, key=lambda r: (r["dev"], r["step"]))
    pm.unload()
    rec = {"kind": "grad", "skill": name, "family": kind(name), "seed": seed,
           "n": N, "lr": lr, "acc": best["eval"], "dev": best["dev"],
           "best_step": best["step"], "final_eval": ckpts[-1]["eval"],
           "acc_seen_letter": best["eval_seen"],
           "acc_unseen_letter": best["eval_unseen"],
           "n_eval_unseen": len(ev_unseen),
           "ckpts": ckpts, "steps": steps,
           "wall_s": round(time.time() - t0, 1)}
    print(f"  [grad {name}] N={N:3d} lr={lr:g} eval={rec['acc']:.3f} "
          f"(seen {best['eval_seen']} unseen {best['eval_unseen']} "
          f"n_unseen={len(ev_unseen)}) "
          f"(dev {best['dev']:.3f} @step {best['step']}) "
          f"final={rec['final_eval']:.3f} ({rec['wall_s']}s)", flush=True)
    return rec


# ---- degeneracy + floor ---------------------------------------------------

def structure(tok, skills=SKILL_SET, shots=8):
    """copy-oracle per skill (e004g's, unchanged). Inventing a skill does
    not make it non-degenerate."""
    c.SHOTS = shots
    rows = {}
    for name in skills:
        s = IDX[name]
        acc, d = c.copy_oracle(tok, eval_items(s, True, shots))
        lls = [len(tok(c.example_line(s, w), add_special_tokens=False).input_ids)
               for w in EVAL_WORDS]
        rows[name] = {"family": kind(name), "copy_oracle_acc": acc,
                      "copy_oracle_offset": d,
                      "line_tok_len_mean": round(float(np.mean(lls)), 2),
                      "line_tok_len_sd": round(float(np.std(lls)), 2),
                      "non_degenerate": acc <= 0.20}
        print(f"  {name:10s} ({kind(name)[:3]}) copy_oracle={acc:.3f}@{d} "
              f"line_toks={rows[name]['line_tok_len_mean']}"
              f"±{rows[name]['line_tok_len_sd']} "
              f"{'NON-DEGENERATE' if acc <= 0.20 else 'COPYABLE'}", flush=True)
    return rows


# ---- selfcheck ------------------------------------------------------------

def selfcheck(tok):
    """Smallest runnable thing that fails if the logic breaks. ~3s, no model."""
    assert not (set(EVAL_WORDS) & set(TRAIN_WORDS))
    assert not (set(EVAL_WORDS) & set(DEV_WORDS))
    assert not (set(DEV_WORDS) & set(TRAIN_WORDS))
    assert len(EVAL_WORDS) == 64 and len(DEV_WORDS) == 32
    assert len(TRAIN_WORDS) >= max(N_GRID) and len(TRAIN_WORDS) >= max(SHOT_GRID)
    # nested subsets
    for a, b in zip(N_GRID, N_GRID[1:]):
        assert train_words_for(a, 0) == train_words_for(b, 0)[:a]
    # novel skills are pure functions and the bijection is a bijection
    assert len(set(BIJECTION.values())) == 26
    for _, _, f in NOVEL:
        assert all(f(w) == f(w) for w in EVAL_WORDS)
    # N_map really is arbitrary: it agrees with identity on < 3 letters
    assert sum(BIJECTION[x] == x for x in BIJECTION) == 0
    # answer span exactness survives the new pools (e004g's encoder)
    for name in SKILL_SET:
        s = IDX[name]
        items = eval_items(s, True, 4, n=8)
        ids, mask, st, ln = c.encode(tok, items)
        for b, (pre, ans) in enumerate(items):
            assert tok.decode(ids[b, st[b]:st[b] + ln[b]]).strip() == ans
            assert tok.decode(ids[b, :st[b]]) == pre
        # teacher(prompted) and student(unprompted) answer spans align
        _, _, _, ul = c.encode(tok, eval_items(s, False, n=8))
        assert ln == ul, (name, ln, ul)
    # mismatched shots demonstrate a different rule under the same marker
    c.SHOTS = 3
    r = np.random.default_rng(1)
    ok = c.make_prompt(IDX["B_first"], "apple", r, True)
    r = np.random.default_rng(1)
    mm = c.make_prompt(IDX["B_first"], "apple", r, True,
                       shot_skill=IDX["B_upper"])
    assert ok != mm and ok.split("\n")[-1] == mm.split("\n")[-1]
    # shot count is honoured
    for k in (1, 8, 32):
        c.SHOTS = k
        assert len(c.make_prompt(IDX["B_first"], "apple",
                                 np.random.default_rng(0), True).split("\n")) == k + 1
    print("selfcheck OK")


# ---- entry ----------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--icl", action="store_true")
    ap.add_argument("--lrsweep", action="store_true")
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--skills", default="")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ngrid", default="")
    ap.add_argument("--shotgrid", default="")
    ap.add_argument("--lr-n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if a.selfcheck:
        selfcheck(tok); raise SystemExit
    skills = a.skills.split(",") if a.skills else SKILL_SET
    if a.structure:
        print("degeneracy check:", flush=True)
        st = structure(tok, skills)
        (HERE / "structure.json").write_text(json.dumps(st, indent=1))
        raise SystemExit
    tok, model = c.load(a.base)
    print("positional scheme:", json.dumps(c.pos_scheme(model)), flush=True)
    RUNS.mkdir(exist_ok=True)
    recs = []
    if a.throughput:
        print(json.dumps(c.throughput(model, tok, np.random.default_rng(0))))
    if a.icl:
        sg = [int(x) for x in a.shotgrid.split(",")] if a.shotgrid else None
        for name in skills:
            recs += icl_curve(model, tok, name, a.seed, sg)
    if a.lrsweep:
        for name in skills:
            for lr in LR_GRID:
                recs.append(grad_point(model, tok, name, a.lr_n, lr, a.seed,
                                       a.steps))
    if a.grad:
        grid = [int(x) for x in a.ngrid.split(",")] if a.ngrid else N_GRID
        for name in skills:
            for N in grid:
                recs.append(grad_point(model, tok, name, N, a.lr, a.seed,
                                       a.steps))
    if recs:
        tag = a.tag or f"{'-'.join(skills)}_s{a.seed}_{int(time.time())}"
        (RUNS / f"{tag}.json").write_text(json.dumps(
            {"base": a.base, "git_sha": c.git_sha(), "records": recs}, indent=1))
        print("wrote", RUNS / f"{tag}.json", flush=True)
