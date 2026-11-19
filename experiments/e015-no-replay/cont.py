"""e015: can catastrophic forgetting be prevented STRUCTURALLY, with no replay?

Reuses e004g's harness by IMPORTING it (not forking it): skill definitions,
tokenized batching, teacher-forced scoring, the copy-oracle degeneracy check,
the gate, the KL-to-frozen-base write, the generality probe and the
`unknown_marker_applies_rule` conditionality probe all come from
`e004g-consolidation/consolidate.py`. What is NEW here:

  1. a skill-suite EXTENSION (indices 14+, so 0..13 keep their e004g
     identities and the known-positive on skills 5/7/8 is unaffected);
  2. replay budgeted in EXAMPLES, not steps, so 1% is expressible at 60
     steps -- e004g's step-based rule could not express the literature's
     setting and had to use 25%;
  3. two structural-allocation arms, SPARSE (disjoint rank slices) and
     ORTHOGONAL (each write projected off the subspace prior writes used);
  4. a cheaper eval that drops `prompted_on_top` and `cross_marker_leak`
     (the K1/K2 discriminator is not this experiment's question, and the
     e004g verifier REJECTED cross_marker_leak as a reading rule), and
     scores not-yet-written skills at a reduced n.

See DESIGN.md. Nothing here re-derives the conditionality metric: it is
e004g's `unknown_marker_applies_rule`, read per age, unchanged.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
E004G = HERE.parent / "e004g-consolidation"
sys.path.insert(0, str(E004G))
import consolidate as C          # noqa: E402  the e004g harness, reused

RUNS = HERE / "runs"

# ---- skill-suite extension -------------------------------------------------
# e004g's gate left only SIX non-degenerate passing skills (B_upper 5,
# B_double 7, B_title 8, B_first 9, B_plural 11, B_triple 13), which caps k at
# 6 -- and "no collapse at small k" is uninformative, so 015 needs more.
# Appended at 14+ so every existing index keeps its marker and its gate record.
# Deliberately weighted toward CHARACTER transforms: e004g's own honest
# weakness is that 4 of its 6 non-degenerate skills are "copy the input and
# decorate it", which is one mechanism wearing four hats.
EXTRA_SKILLS = [
    ("B_dropfirst", "B", lambda w: w[1:]),           # character surgery
    ("B_droplast", "B", lambda w: w[:-1]),
    ("B_swap12", "B", lambda w: w[1] + w[0] + w[2:]),
    ("B_rot2", "B", lambda w: w[2:] + w[:2]),
    ("B_alt", "B", lambda w: w[::2]),
    ("B_mid", "B", lambda w: w[2]),
    ("B_lastfirst", "B", lambda w: w[-1] + w[0]),
    ("B_dup1", "B", lambda w: w[0] + w),
    ("B_upfirst2", "B", lambda w: w[:2].upper() + w[2:]),
    ("B_ing", "B", lambda w: w + "ing"),             # decorate (cheap filler)
    ("B_pre_x", "B", lambda w: "x" + w),
    ("B_updrop", "B", lambda w: w[1:].upper()),
    # BATCH 2, appended 2026-08-16 AFTER the batch-1 gate, disclosed.
    # Batch 1 was deliberately weighted toward CHARACTER SURGERY and 11 of its
    # 12 skills FAILED the gate at prompted <= 0.375 (dropfirst, swap12, rot2,
    # alt, mid, lastfirst, dup1 all at ~0.00-0.04). That is a real measurement
    # about the base, recorded in DESIGN: Qwen2.5-0.5B's in-context repertoire
    # on this one-line format is "copy the input and decorate it", plus
    # upper-casing and first-character extraction. Reaching k >= 8 on skills
    # this base can actually do therefore means staying inside that family.
    # The cost is mechanism diversity, and it is the suite's single biggest
    # weakness; analyze.py emits per-skill retention so it cannot be hidden.
    ("B_ly", "B", lambda w: w + "ly"),
    ("B_dot", "B", lambda w: w + "."),
    ("B_the", "B", lambda w: "the " + w),
    ("B_angle", "B", lambda w: "<" + w + ">"),
    ("B_upbang", "B", lambda w: w.upper() + "!"),
    ("B_wfirst", "B", lambda w: w + " " + w[0]),
    ("B_es", "B", lambda w: w + "es"),
    ("B_dashw", "B", lambda w: w + "-" + w),
    ("B_star", "B", lambda w: "* " + w),
    ("B_capfirst", "B", lambda w: w[0].upper()),
]
if C.SKILLS[-1][0] != EXTRA_SKILLS[-1][0]:           # idempotent under reload
    for _n, _t, _f in EXTRA_SKILLS:
        C.SKILLS.append((_n, _t, _f))
    C.N_SKILLS = len(C.SKILLS)
    C.MARKERS = [f"<<K{i}>>" for i in range(C.N_SKILLS)]
assert C.UNKNOWN_MARKER not in C.MARKERS, "probe marker collided with a skill"
assert C.SKILLS[5][0] == "B_upper" and C.SKILLS[13][0] == "B_triple", \
    "e004g skill indices must not move (known-positive depends on them)"

# The 10 gated, non-degenerate, TOKEN-PREFIX-FREE skills. The gate admits 12;
# B_triple and B_capfirst are dropped because their answers are token-prefixed
# by B_double and B_upper (100% and 38% of eval words). See prefix_clashes().
GRID_SKILLS = [5, 7, 8, 9, 11, 24, 28, 31, 33, 34]

STEPS, BATCH, LR = C.STEPS_PER_SKILL, C.BATCH, C.LR
ARMS = ["replay25", "replay01", "no_replay", "sparse", "orth",
        "separate_adapters"]
REPLAY_OF = {"replay25": 0.25, "replay01": 0.01}


# ---- replay budget, in EXAMPLES --------------------------------------------

def replay_schedule(rf, steps=STEPS, batch=BATCH):
    """step -> replay batch size. Budget is a fraction of total training
    EXAMPLES, which is what [[scialom2022-continual-t0]]'s 1% actually means.
    e004g budgeted in STEPS, so 1% was 0.6 steps and inexpressible.

    At rf=0.25 this is IDENTICAL to e004g (every 4th step, full batch, 15
    batches = 240 = 25% of 60*16), which is what makes the known-positive a
    real reproduction and not an approximation."""
    n_ex = round(rf * steps * batch)
    if n_ex <= 0:
        return {}
    if rf >= 1.0 / steps:
        every = max(1, int(1 / rf))
        return {s: batch for s in range(steps) if s % every == 0}
    return {steps // 2: max(1, n_ex)}       # one small batch, mid-write


# ---- structural allocation --------------------------------------------------

def lora_named(pm):
    return [(n, p) for n, p in pm.named_parameters() if p.requires_grad]


def slot_mask(name, p, slot, n_slots):
    """Boolean mask selecting RANK SLICE `slot` of a LoRA factor.

    Rank slices, not arbitrary entries: LoRA's update is bilinear,
    dW = B @ A = sum_j b_j a_j^T, so masking arbitrary entries of A and B does
    NOT give additively separate writes (skill i's B entries multiply skill
    j's A entries). Masking whole rank indices does: disjoint slices are
    disjoint rank-1 terms of the same sum. lora_A is [r, in], lora_B is
    [out, r]."""
    m = torch.zeros_like(p, dtype=torch.bool)
    is_a = ".lora_A." in name
    r = p.shape[0] if is_a else p.shape[1]
    idx = [j for j in range(r) if j % n_slots == slot]
    if is_a:
        m[idx, :] = True
    else:
        m[:, idx] = True
    return m


class Orth:
    """Project each write off the subspace prior writes used.

    Per LoRA tensor, keeps an orthonormal basis of the directions previous
    skills moved that tensor along, and removes their components from the
    APPLIED UPDATE (not from the gradient: AdamW rescales per coordinate, so a
    projected gradient does not produce a projected update).

    Honest limit, measured rather than assumed: orthogonality is enforced in
    the LoRA FACTOR space. Because dW = B@A is bilinear, factor-space
    orthogonality does not imply weight-space orthogonality. `wcos` measures
    what was actually achieved in weight space on a sample of modules and it
    goes into every run record."""

    def __init__(self):
        self.dirs = {}

    def project(self, name, d):
        for u in self.dirs.get(name, ()):
            d -= u * torch.dot(u, d)
        return d

    def add(self, name, d):
        d = self.project(name, d.clone())
        n = float(d.norm())
        if n > 1e-8:
            self.dirs.setdefault(name, []).append(d / n)


def sampled_modules(pm, n=4):
    """A few (lora_A, lora_B) pairs spread through the depth, for the
    weight-space orthogonality diagnostic."""
    names = sorted({n_.rsplit(".lora_", 1)[0] for n_, _ in lora_named(pm)})
    step = max(1, len(names) // n)
    return names[::step][:n]


def delta_w(pm, prefix):
    """dW = B @ A for one target module (LoRA B is zero-init, so this IS the
    accumulated update)."""
    d = dict(lora_named(pm))
    a = d[prefix + ".lora_A.default.weight"]
    b = d[prefix + ".lora_B.default.weight"]
    return (b @ a).detach().flatten().clone()


# ---- eval ------------------------------------------------------------------

@torch.no_grad()
def answer_rank(pm, tok, items, batch=16):
    """Mean rank of the TRUE first answer token, and mean teacher-forced
    log-prob of the whole true answer.

    MASKING vs ERASURE. A prior skill at retention 0.000 has two completely
    different explanations and accuracy cannot tell them apart:
      - ERASED: the write destroyed the representation. The true answer is
        nowhere near the top and its log-prob is at the frozen-base level.
      - MASKED: the representation is intact but the NEWEST write owns the
        format default and answers under every cue, so the old rule never
        reaches argmax. Then the true answer is still highly ranked.
    Rank sees the difference; argmax accuracy is blind to it."""
    ranks, lps = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        ids, mask, starts, lens = C.encode(tok, chunk)
        lg = C.answer_logits(pm, ids, mask, starts, lens)
        tgt = C.answer_targets(ids, starts, lens)
        lp = torch.log_softmax(lg, -1)
        off = 0
        for l in lens:
            row = lg[off]                       # FIRST answer position only
            t = int(tgt[off])
            ranks.append(int((row > row[t]).sum()) + 1)
            lps.append(float(lp[off:off + l].gather(
                1, tgt[off:off + l, None]).sum()))
            off += l
    return round(float(np.mean(ranks)), 2), round(float(np.mean(lps)), 3)


def eval_step(pm, tok, order, i, tok_text, n_eval, n_probe=64, diag=False,
              freegen_n=0):
    """Retention of every skill + the conditionality probe + generality.

    Cheaper than e004g's `eval_suite` in two disclosed ways: `prompted_on_top`
    is dropped (the K1-vs-K2 question is e004g's, is ceiling-censored on this
    suite by its verifier's finding, and is not what 015 asks), and
    `cross_marker_leak` is dropped (the e004g verifier REJECTED it as a
    reading rule -- it is near-arithmetically implied by retention). Skills
    not yet written are scored at n_probe instead of n_eval: leakage into an
    unwritten skill still has to be visible, but it does not need the full
    n=200 resolution that a retention row needs.

    Query sets are FIXED (C.EVAL_SEED + skill), so a change between two evals
    is a change in the model, not in the sample."""
    rec = {"unprompted": {}, "unknown_marker_applies_rule": {},
           "n_by_skill": {}}
    written = set(order[:i + 1])
    for s in order:
        n = n_eval if s in written else n_probe
        rng = np.random.default_rng(C.EVAL_SEED + s)
        rec["unprompted"][str(s)] = round(C.accuracy(
            pm, tok, C.make_items(s, n, rng, False, C.EVAL_WORDS)), 3)
        # e004g's PRE-REGISTERED conditionality probe, carried unchanged:
        # score skill s's own rule under a marker that is never consolidated.
        rng = np.random.default_rng(C.EVAL_SEED + s)
        words = [C.EVAL_WORDS[int(j)] for j in
                 rng.integers(len(C.EVAL_WORDS), size=n)]
        rec["unknown_marker_applies_rule"][str(s)] = round(C.accuracy(
            pm, tok, [(f"{C.UNKNOWN_MARKER} {w} =", C.SKILLS[s][2](w))
                      for w in words]), 3)
        rec["n_by_skill"][str(s)] = n
        # e021: FREE GENERATION on the SAME items as the teacher-forced row,
        # subsampled. Teacher-forced accuracy is only an upper bound on this;
        # DESIGN requires them within 0.05 and that requirement was failing
        # grid-wide because nothing taught the model to STOP. Reporting the two
        # side by side per skill per k is the only way that stays visible.
        if freegen_n and s in written:
            rng = np.random.default_rng(C.EVAL_SEED + s)
            rec.setdefault("freegen", {})[str(s)] = round(C.freegen_accuracy(
                pm, tok, C.make_items(s, n, rng, False,
                                      C.EVAL_WORDS)[:freegen_n]), 3)
    rec["generality_nats"] = round(C.nats_per_token(pm, tok, tok_text), 4)
    if diag:
        # MASKING vs ERASURE, measured directly instead of inferred from the
        # k>=3 curve. For every PRIOR skill j, under j's OWN marker:
        #   answer_rank  -- is j's true answer still near the top? (intact)
        #   newest_rule_under_j -- does the NEWEST rule fire there instead?
        # The e004g verifier rejected a leak metric as a K0 reading rule
        # because high retention arithmetically forces leak toward zero. That
        # objection does NOT apply here: these rows sit at retention 0.000,
        # where nothing is implied and what the model outputs instead is the
        # whole question.
        rec["diag_rank"], rec["diag_logp"], rec["newest_rule_under_j"] = {}, {}, {}
        newest = order[i]
        for j in order[:i + 1]:
            rng = np.random.default_rng(C.EVAL_SEED + j)
            items = C.make_items(j, n_probe, rng, False, C.EVAL_WORDS)
            rk, lp = answer_rank(pm, tok, items)
            rec["diag_rank"][str(j)], rec["diag_logp"][str(j)] = rk, lp
            if j != newest:
                rng = np.random.default_rng(C.EVAL_SEED + j)
                words = [C.EVAL_WORDS[int(x)] for x in
                         rng.integers(len(C.EVAL_WORDS), size=n_probe)]
                rec["newest_rule_under_j"][str(j)] = round(C.accuracy(
                    pm, tok, [(C.query_line(j, w), C.SKILLS[newest][2](w))
                              for w in words]), 3)
    return rec


# ---- the loop --------------------------------------------------------------

def run_arm(model, tok, arm, seed, skills, tok_text, n_eval=200, rank=8,
            probe_every=30, probe_n=64, tag="", diag=False):
    from peft import get_peft_model, LoraConfig
    assert arm in ARMS, arm
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    torch.manual_seed(seed)
    pm = get_peft_model(model, cfg)
    fresh = {k: v.detach().clone() for k, v in pm.state_dict().items()
             if "lora_" in k}
    rng = np.random.default_rng(seed)
    order = [int(s) for s in rng.permutation(skills)]
    rf = REPLAY_OF.get(arm, 0.0)
    sched = replay_schedule(rf)
    n_slots = len(order)
    mods = sampled_modules(pm)
    # the arm KEY carries the tag: analyze.py groups by (base, arm), and a
    # tagged side-run (the known-positive, a rank sweep) written under the
    # bare arm name would be averaged into the grid cell it is meant to check.
    rec = {"arm": arm + tag, "seed": seed, "order": order, "replay_frac": rf,
           "rank": rank, "n_eval": n_eval, "base": getattr(
               model.config, "_name_or_path", "?"),
           "n_params": sum(p.numel() for p in model.parameters()),
           "n_lora_params": sum(p.numel() for _, p in lora_named(pm)),
           "n_skills": len(order), "tag": tag,
           # PROVENANCE. C.git_sha() hashes e004g's consolidate.py (its
           # __file__ is the imported module), so on its own it does not
           # identify the code that produced these numbers. e004g was burned
           # by exactly this: every one of its pilot records has git_sha "".
           "git_sha": C.git_sha(),
           "cont_sha256": __import__("hashlib").sha256(
               Path(__file__).read_bytes()).hexdigest()[:16],
           "steps": []}
    orth = Orth() if arm == "orth" else None
    masks, prev_w = None, {}
    cum, t0 = 0, time.time()
    for i, s in enumerate(order):
        if arm == "separate_adapters":
            pm.load_state_dict(fresh, strict=False)
        if arm == "sparse":
            # skill i owns rank slice i % n_slots. At len(order) <= rank the
            # slices are DISJOINT; above it they collide by construction, and
            # that collision IS the capacity story C3 predicts.
            masks = {n: slot_mask(n, p, i % min(n_slots, rank),
                                  min(n_slots, rank))
                     for n, p in lora_named(pm)}
        opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad],
                                lr=LR)
        w0 = {m: delta_w(pm, m) for m in mods} if arm == "orth" else {}
        p0 = ({n: p.detach().clone() for n, p in lora_named(pm)}
              if arm == "orth" else None)
        probes, n_replay, n_replay_ex = [], 0, 0
        for step in range(STEPS):
            snap = ({n: p.detach().clone() for n, p in lora_named(pm)}
                    if arm in ("sparse", "orth") else None)
            loss = C.kl_loss(pm, tok, s, rng)
            if i and step in sched:
                # ROUND-ROBIN over priors (e004g): uniform sampling leaves some
                # prior with 0-1 batches by chance, so a collapse could be a
                # sampling accident rather than interference.
                prior = order[n_replay % i]
                n_replay += 1
                n_replay_ex += sched[step]
                loss = loss + C.kl_loss(pm, tok, prior, rng, batch=sched[step])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if snap is not None:
                with torch.no_grad():
                    for n, p in lora_named(pm):
                        if arm == "sparse":     # hard mask: exact, and immune
                            p.copy_(torch.where(masks[n], p, snap[n]))
                        else:                   # project the APPLIED update
                            d = (p - snap[n]).flatten()
                            p.copy_((snap[n].flatten()
                                     + orth.project(n, d)).view_as(p))
            cum += 1
            if probe_every and (step + 1) % probe_every == 0 and i \
                    and arm != "separate_adapters":
                probes.append({"step": step + 1, "cum": cum, "retention": {
                    str(j): round(C.accuracy(pm, tok, C.make_items(
                        j, probe_n, np.random.default_rng(C.EVAL_SEED + j),
                        False, C.EVAL_WORDS)), 3) for j in order[:i]}})
        if arm == "orth":
            with torch.no_grad():
                for n, p in lora_named(pm):
                    orth.add(n, (p.detach() - p0[n]).flatten())
            # weight-space check: cos between THIS skill's dW increment and
            # each prior's, on the sampled modules. Factor-space orthogonality
            # does not imply this; measure it rather than claim it.
            cs = []
            for m in mods:
                cur = delta_w(pm, m) - w0[m]
                for pw in prev_w.get(m, ()):
                    d = float(cur.norm() * pw.norm())
                    if d > 1e-12:
                        cs.append(round(float(torch.dot(cur, pw)) / d, 4))
                prev_w.setdefault(m, []).append(cur)
        # separate_adapters DISCARDED the prior skills' weights at the reset,
        # so scoring them would post ~0 and make the CEILING arm look like the
        # worst arm in the experiment. e004g's `seen = [s]` behaviour, kept.
        seen = order if arm != "separate_adapters" else [int(s)]
        step_rec = eval_step(pm, tok, seen, seen.index(int(s)), tok_text,
                             n_eval, probe_n, diag=diag)
        step_rec.update({
            "k": i + 1, "skill": int(s), "cum_steps": cum,
            "n_replay_batches": n_replay, "n_replay_examples": n_replay_ex,
            "intra_skill_probes": probes,
            "write_tripwire_ok":
                step_rec["unprompted"][str(s)] >= C.WRITE_TRIPWIRE,
            # VERIFIER FIX 2026-08-16: was `sorted(cs)[-5:]`, i.e. the 5
            # largest SIGNED cosines out of up to 4 modules x (k-1) priors.
            # A large NEGATIVE cosine -- an anti-aligned write, which is just
            # as much a failure of orthogonality -- was silently dropped, so
            # the record could not support the claim "|cos| <= 0.033 at every
            # k". Rank by MAGNITUDE and carry the summary stats.
            "orth_weightspace_cos": (
                sorted(cs, key=abs)[-5:] if arm == "orth" and cs else None),
            "orth_weightspace_cos_stats": (
                {"n": len(cs),
                 "max_abs": round(max(abs(x) for x in cs), 4),
                 "mean_abs": round(sum(abs(x) for x in cs) / len(cs), 4),
                 "min_signed": round(min(cs), 4)}
                if arm == "orth" and cs else None),
            "wall_s": round(time.time() - t0, 1)})
        rec["steps"].append(step_rec)
        print(f"  [{arm}{tag} s{seed} r{rank}] k={i+1} {C.SKILLS[s][0]} "
              f"wrote={step_rec['unprompted'][str(s)]:.3f} "
              f"prior={[step_rec['unprompted'][str(j)] for j in seen[:seen.index(int(s))]]} "
              f"nats={step_rec['generality_nats']} "
              f"({step_rec['wall_s']}s)", flush=True)
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["peak_rss_gb"] = round(peak_rss_gb(), 2)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{arm}{tag}_s{seed}.json").write_text(json.dumps(rec, indent=1))
    pm.unload()
    return rec


def peak_rss_gb():
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


# ---- checks ----------------------------------------------------------------

def prefix_clashes(tok, skills=None, thresh=0.20):
    """Pairs where skill i's answer TOKENS are a prefix of skill j's, with the
    fraction of eval words affected.

    Token-level, not string-level, because that is what the metric is: answers
    are tokenized as " " + answer and scored by argmax at each answer
    position. `"a"` being a string-prefix of `"apple apple"` is harmless --
    " a" and " apple" are different tokens. `"apple apple"` being a TOKEN
    prefix of `"apple apple apple"` is not harmless: while the shorter rule
    owns the format default, the model's self-similar continuation also
    satisfies the longer one, so the longer skill scores 1.000 under a marker
    it does not own.

    Returns pairs above `thresh` (the same 0.20 control level the gate and the
    K0 block use).

    2026-08-17: now DELEGATES to `C.pool_clashes`, which checks the same
    token-prefix relation plus string equality, whitespace-normalised
    equality and whole-answer token suffix. Kept under this name because
    e017's `arb.py` and e019's `recov.py` assert on it; every caller
    therefore gets the stronger gate with no change."""
    if skills is None:
        g = HERE / "gate_Qwen2.5-0.5B.json"
        skills = (json.loads(g.read_text())["_verdict"]["passed_non_degenerate"]
                  if g.exists() else list(range(C.N_SKILLS)))
    return [c for c in C.pool_clashes(tok, skills, thresh)
            if c[0] in C.BLOCKING_KINDS]


def selfcheck(tok):
    """Smallest thing that fails if the NEW logic breaks. ~2s, no model.
    e004g's own selfcheck runs first (it covers the reused harness)."""
    C.selfcheck(tok)
    # 1. replay budgeted in examples reproduces e004g exactly at 0.25 ...
    s25 = replay_schedule(0.25)
    assert len(s25) == 15 and set(s25.values()) == {16}, s25
    assert sorted(s25) == list(range(0, 60, 4))
    assert sum(s25.values()) == 240 == round(0.25 * 60 * 16)
    # ... and makes 1% expressible, which the step-based rule could not
    s01 = replay_schedule(0.01)
    assert sum(s01.values()) == 10 and len(s01) == 1, s01
    assert replay_schedule(0.0) == {}
    # 2. rank slices are DISJOINT and cover the rank exactly
    a = torch.zeros(8, 5)
    b = torch.zeros(5, 8)
    for nm, p, dim in (("x.lora_A.default.weight", a, 0),
                       ("x.lora_B.default.weight", b, 1)):
        ms = [slot_mask(nm, p, s, 4) for s in range(4)]
        tot = sum(m.long() for m in ms)
        assert bool((tot == 1).all()), (nm, tot)
        assert int(ms[0].sum()) == p.numel() // 4
        assert ms[0].sum(1 - dim).max() in (0, p.shape[1 - dim])
    # 3. projection really removes the prior direction, and is idempotent
    o = Orth()
    o.add("t", torch.tensor([1.0, 0.0, 0.0]))
    d = o.project("t", torch.tensor([3.0, 4.0, 0.0]))
    assert abs(float(d[0])) < 1e-6 and abs(float(d[1]) - 4.0) < 1e-6
    o.add("t", torch.tensor([0.0, 2.0, 0.0]))
    d = o.project("t", torch.tensor([5.0, 6.0, 7.0]))
    assert abs(float(d[0])) < 1e-6 and abs(float(d[1])) < 1e-6
    assert abs(float(d[2]) - 7.0) < 1e-6
    # 4. the extension did not move any e004g skill or collide with the probe
    assert [C.SKILLS[i][0] for i in (5, 7, 8, 9, 11, 13)] == [
        "B_upper", "B_double", "B_title", "B_first", "B_plural", "B_triple"]
    assert len(set(C.MARKERS)) == C.N_SKILLS
    assert C.UNKNOWN_MARKER not in C.MARKERS
    # 4b. NO SKILL'S ANSWER MAY BE A PREFIX OF ANOTHER'S.
    #     Found the hard way: B_double -> "apple apple" is a prefix of
    #     B_triple -> "apple apple apple". Teacher-forced exact match scores
    #     the argmax at each answer position, so while B_double owns the
    #     format default the model's self-similar continuation ALSO satisfies
    #     B_triple, and B_triple scores 1.000 under a marker it does not own.
    #     That single pair produced the only non-zero retention in the pilot
    #     AND tripped the K0 block. A pair like this is not two skills; it is
    #     one skill scored twice. Mechanical, runs in milliseconds, and would
    #     have caught it before any compute was spent.
    clashes = prefix_clashes(tok, GRID_SKILLS)
    assert not clashes, f"answer-prefix collisions between skills: {clashes}"
    # 5. every new skill is a pure function and is not the identity
    for nm, _, f in EXTRA_SKILLS:
        assert all(f(w) == f(w) for w in C.WORDS), nm
        assert any(f(w) != w for w in C.WORDS), nm
    print("e015 selfcheck OK")


# ---- entry point -----------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skills", default="")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda. Default cpu keeps records reproducible.")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--probe-every", type=int, default=30)
    ap.add_argument("--tag", default="")
    ap.add_argument("--diag", action="store_true",
                    help="masking-vs-erasure probes on prior skills")
    a = ap.parse_args()
    if getattr(a, "device", "cpu") != "cpu":
        C.DEVICE = a.device
    if getattr(a, "dtype", "float32") != "float32":
        C.DTYPE = a.dtype
    torch.set_num_threads(a.threads)
    if a.selfcheck:
        from transformers import AutoTokenizer
        selfcheck(AutoTokenizer.from_pretrained(a.base))
        raise SystemExit
    tok, model = C.load(a.base)
    short = a.base.split("/")[-1]
    print("positional scheme:", json.dumps(C.pos_scheme(model)), flush=True)
    if a.throughput:
        print(json.dumps(C.throughput(model, tok, np.random.default_rng(0))),
              flush=True)
    if a.gate:
        t0 = time.time()
        struct = C.structure_report(tok)
        g = C.gate(model, tok, n=a.n, structure=struct)
        g["_verdict"].update({"base": a.base, "pos_scheme": C.pos_scheme(model),
                              "wall_s": round(time.time() - t0, 1)})
        (HERE / f"gate_{short}.json").write_text(json.dumps(g, indent=1))
        print(json.dumps(g["_verdict"], indent=1), flush=True)
    if a.arm:
        text = (E004G / "generality.txt").read_text()
        if a.skills:
            skills = [int(x) for x in a.skills.split(",")]
        else:
            v = json.loads((HERE / f"gate_{short}.json").read_text())["_verdict"]
            skills = v["passed_non_degenerate"]
        assert skills, "no skills"
        C.assert_pool_clean(tok, skills)   # blocking on the ACTUAL --skills set
        print("skills:", [C.SKILLS[s][0] for s in skills], flush=True)
        run_arm(model, tok, a.arm, a.seed, skills, text, n_eval=a.n_eval,
                rank=a.rank, probe_every=a.probe_every, tag=a.tag,
                diag=a.diag)
