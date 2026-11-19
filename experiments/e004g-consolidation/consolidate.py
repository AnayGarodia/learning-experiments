"""e004g: consolidation accumulation on a DOWNLOADED ICL-capable base.

Port of e004r's harness (marker-token skills, prompted-vs-unprompted gap
gate, KL distillation of own prompted behaviour, retention metrics) from
a self-pretrained TinyGPT to a real pretrained LM + LoRA. See DESIGN.md
for every adaptation the tokenizer change forced.

Skill line format (identical across skills, marker names the skill):
    <<K3>> apple = elppa
prompted  = SHOTS example lines (train-half words) + query line
unprompted = query line alone
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
RUNS = HERE / "runs"

# ---- skills (port of e004r's 4 type-A + 4 type-B) --------------------------

WORDS = [
    "apple", "water", "tiger", "chair", "river", "stone", "cloud", "house",
    "table", "light", "bread", "music", "paper", "plant", "glass", "horse",
    "field", "dream", "bench", "train", "mouse", "clock", "brush", "grape",
    "piano", "tower", "snake", "beach", "storm", "knife", "shirt", "month",
    "voice", "frame", "crown", "shell", "brick", "ocean", "candy", "robot",
    "wagon", "medal", "cabin", "pearl", "lemon", "toast", "fence", "noise",
    "spoon", "truck", "whale", "yacht", "zebra", "olive", "radio", "salad",
    "tulip", "uncle", "vault", "wheat", "actor", "badge", "cedar", "dance",
]
assert len(WORDS) == 64 and len(set(WORDS)) == 64

# VERIFIER ADDITION: an OFF-LENGTH held-out pool, 6-8 letters, disjoint from
# WORDS. Every eval above runs on 5-letter words, and after consolidation most
# skills sit at unprompted 1.000 there, which CEILING-CENSORS the re-prompting
# gain (K1 vs K2): a skill at 1.000 cannot show a +0.06 gain. This pool is
# scored once, at the final k, as a labelled DIAGNOSTIC of whether the write
# generalises off the length it was trained on. It is not a K1/K2 verdict --
# the write is distilled on 5-letter words only, so a prompted-beats-baked
# result here is partly off-distribution, and it is reported as such.
HARD_WORDS = [
    "anchor", "blanket", "compass", "diamond", "envelope", "feather",
    "granite", "harvest", "iceberg", "journey", "kitchen", "lantern",
    "machine", "necklace", "orchard", "pumpkin", "quarrel", "ribbon",
    "sandal", "trumpet", "umbrella", "village", "walnut", "yoghurt",
    "balcony", "cactus", "dolphin", "emerald", "flamingo", "gallery",
    "hammock", "island",
]
assert len(HARD_WORDS) == 32 and not (set(HARD_WORDS) & set(WORDS))

# fixed train/eval split of the input pool (DESIGN adaptation 3)
_SPLIT = np.random.default_rng(20260816).permutation(64)
TRAIN_WORDS = [WORDS[i] for i in _SPLIT[:32]]
EVAL_WORDS = [WORDS[i] for i in _SPLIT[32:]]

# type A: constant answer, independent of input (association / fact-like)
# type B: input-dependent transform (the skills-not-facts reading of B2)
SKILLS = [
    ("A_zeta", "A", lambda w: "zeta"),
    ("A_north", "A", lambda w: "north"),
    ("A_violet", "A", lambda w: "violet"),
    ("A_harbor", "A", lambda w: "harbor"),
    ("B_reverse", "B", lambda w: w[::-1]),
    ("B_upper", "B", lambda w: w.upper()),
    ("B_last", "B", lambda w: w[-1]),
    ("B_double", "B", lambda w: w + " " + w),
    # APPENDED 2026-08-16 (indices 8+ so 0..7 stay stable). Post-hoc suite
    # EXTENSION, disclosed: only 2 of the original 8 skills were both
    # passing and non-degenerate, which is too few to plot accumulation
    # against. Gate thresholds are UNCHANGED and the pilot result on
    # skills 0/5/7 is not re-read.
    ("B_title", "B", lambda w: w.capitalize()),
    ("B_first", "B", lambda w: w[0]),
    ("B_spaced", "B", lambda w: " ".join(w)),
    ("B_plural", "B", lambda w: w + "s"),
    ("B_bang", "B", lambda w: w + "!"),
    ("B_triple", "B", lambda w: " ".join([w] * 3)),
]
N_SKILLS = len(SKILLS)
MARKERS = [f"<<K{i}>>" for i in range(N_SKILLS)]
# never consolidated: the in-format collateral / conditionality probe.
# MUST stay outside MARKERS. It was "<<K9>>", which after the suite
# extension to 14 skills collided with B_first's own marker.
UNKNOWN_MARKER = "<<K99>>"
assert UNKNOWN_MARKER not in MARKERS
SHOTS = 3
GATE_PROMPTED, GATE_UNPROMPTED = 0.60, 0.20
GATE_MIN_PASS, GATE_MIN_TYPE_B, GATE_MIN_NON_DEGENERATE = 4, 2, 1
STEPS_PER_SKILL, BATCH, LR = 60, 16, 1e-4
WRITE_TRIPWIRE = 0.60

# TERMINATOR (e021). Every WRITE objective scores the answer PLUS this string,
# so that stopping is a trained target. Before this, nothing taught the model
# to stop: it emitted the right answer and kept going on the same line
# (e004g verifier ruling, "NON-TERMINATION"), which made free generation read
# ~0.000 grid-wide and made e017's self-distillation teacher (recorded by FREE
# generation and cut at a newline that never arrived) answer-plus-garbage.
# NEWLINE, not EOS: the write is KL to the FROZEN BASE under a newline-joined
# few-shot context, so the teacher already puts mass on "\n" there and ~none on
# EOS. A write toward EOS would be fighting its own teacher.
# EVAL metrics are deliberately NOT changed -- `accuracy` and
# `freegen_accuracy` still score the answer only -- so that `mean_just_after`
# stays comparable to the pre-fix 0.927-0.937 (the e021 T4 guard) and a change
# there is a change in write strength, not in the definition of the metric.
TERM = "\n"


# ---- device knobs. Defaults are the CPU behaviour every committed record used.
DEVICE = "cpu"
DTYPE = "float32"

def term(ans):
    """The answer as the write scores it: answer + terminator."""
    return ans + TERM


def term_clashes(tok, skills, extra=()):
    """Answers whose last token MERGES with the terminator under BPE.

    If " apple!" + "\\n" does not tokenize as tok(" apple!") + tok("\\n"), the
    scored span of the write shifts and both the answer and the terminator are
    trained at the wrong positions. Four suite skills really do this (answers
    ending "!", ".", ">"), none of them in the grid suite. BLOCKING, same
    posture as the pool gate: a run whose skills clash is not interpretable."""
    nl = tok(TERM, add_special_tokens=False).input_ids
    assert len(nl) == 1, nl
    bad = []
    for a in sorted({SKILLS[s][2](w) for s in skills
                     for w in WORDS + HARD_WORDS} | set(extra)):
        if tok(" " + term(a), add_special_tokens=False).input_ids != \
                tok(" " + a, add_special_tokens=False).input_ids + nl:
            bad.append(a)
    return bad


def assert_term_clean(tok, skills, extra=()):
    bad = term_clashes(tok, skills, extra)
    if bad:
        raise SystemExit(f"TERMINATOR GATE: {len(bad)} answers merge with "
                         f"{TERM!r} under BPE, e.g. {bad[:5]}")


def git_sha():
    """Provenance. The box the grid runs on is an rsync of the experiment
    directory, NOT a git checkout, so `git rev-parse` returned an empty
    string into every pilot record. Falls back to a hash of this file, which
    is the thing that actually has to match."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = ""
    return sha or ("code:" + hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()[:16])


def query_line(skill, word):
    return f"{MARKERS[skill]} {word} ="


def example_line(skill, word):
    return f"{MARKERS[skill]} {word} = {SKILLS[skill][2](word)}"


def make_prompt(skill, word, rng, prompted, shot_skill=None):
    """prompted: SHOTS train-half example lines then the query line.
    shot_skill != skill gives the MISMATCHED-SHOT control: the marker and
    format are skill's, the demonstrated rule is another skill's. A model
    that actually reads the examples must fail; a model riding a format,
    frequency or copying prior scores the same as with correct shots."""
    if not prompted:
        return query_line(skill, word)
    shots = rng.choice([w for w in TRAIN_WORDS if w != word], size=SHOTS,
                       replace=False)
    src = skill if shot_skill is None else shot_skill
    return "\n".join(
        [f"{MARKERS[skill]} {w} = {SKILLS[src][2](w)}" for w in shots] +
        [query_line(skill, word)])


def make_items(skill, n, rng, prompted, pool, shot_skill=None):
    """n (prefix, answer) pairs with query words drawn from `pool`."""
    words = [pool[int(i)] for i in rng.integers(len(pool), size=n)]
    return [(make_prompt(skill, w, rng, prompted, shot_skill),
             SKILLS[skill][2](w)) for w in words]


# ---- tokenized batching ----------------------------------------------------

def encode(tok, items):
    """-> ids [B,T], mask [B,T], ans_start [B], ans_len [B].
    Prefix and answer are tokenized SEPARATELY and concatenated so the
    answer span is exact regardless of BPE merges across the boundary
    (DESIGN adaptation 2)."""
    rows = []
    for prefix, ans in items:
        pi = tok(prefix, add_special_tokens=False).input_ids
        ai = tok(" " + ans, add_special_tokens=False).input_ids
        rows.append((pi, ai))
    T = max(len(p) + len(a) for p, a in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(rows), T), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.long)
    starts, lens = [], []
    for b, (p, a) in enumerate(rows):
        seq = p + a
        ids[b, :len(seq)] = torch.tensor(seq)
        mask[b, :len(seq)] = 1
        starts.append(len(p))
        lens.append(len(a))
    if DEVICE != "cpu":
        # One chokepoint. Every teacher-forced path builds its tensors here,
        # and answer_targets() slices the SAME ids, so moving them once keeps
        # logits and targets on one device without touching call sites.
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
    return ids, mask, starts, lens


def answer_logits(model, ids, mask, starts, lens):
    """Logits predicting each answer token (teacher-forced), concatenated
    over the batch: [sum(lens), V]."""
    if DEVICE != "cpu" and ids.device.type == "cpu":
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
    out = model(input_ids=ids, attention_mask=mask).logits
    return torch.cat([out[b, s - 1:s - 1 + l] for b, (s, l)
                      in enumerate(zip(starts, lens))], 0)


def answer_targets(ids, starts, lens):
    return torch.cat([ids[b, s:s + l] for b, (s, l)
                      in enumerate(zip(starts, lens))], 0)


@torch.no_grad()
def accuracy(model, tok, items, batch=16):
    """Teacher-forced EXACT MATCH: every answer token's argmax correct."""
    hits = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        ids, mask, starts, lens = encode(tok, chunk)
        lg = answer_logits(model, ids, mask, starts, lens)
        pred = lg.argmax(-1)
        tgt = answer_targets(ids, starts, lens)
        off = 0
        for l in lens:
            hits += int(bool((pred[off:off + l] == tgt[off:off + l]).all()))
            off += l
    return hits / len(items)


@torch.no_grad()
def freegen_accuracy(model, tok, items, batch=8):
    """Greedy free generation, string exact match. The teacher-forced
    metric is an upper bound on this; DESIGN requires them within 0.05."""
    hits = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        enc = [tok(p, add_special_tokens=False).input_ids for p, _ in chunk]
        T = max(len(e) for e in enc)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((len(enc), T), pad, dtype=torch.long)
        mask = torch.zeros((len(enc), T), dtype=torch.long)
        for b, e in enumerate(enc):          # LEFT pad for generation
            ids[b, T - len(e):] = torch.tensor(e)
            mask[b, T - len(e):] = 1
        nmax = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                   for _, a in chunk) + 2
        if DEVICE != "cpu":
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        gen = model.generate(input_ids=ids, attention_mask=mask,
                             max_new_tokens=nmax, do_sample=False,
                             pad_token_id=pad)
        for b, (_, a) in enumerate(chunk):
            txt = tok.decode(gen[b, T:], skip_special_tokens=True)
            hits += int(txt.strip().split("\n")[0].strip() == a)
    return hits / len(items)


# ---- degeneracy check (decision 0010 amendment 3) --------------------------

def copy_oracle(tok, items):
    """Best accuracy achievable by copying the answer span from a FIXED
    relative offset earlier in the sequence, no content matching.

    e012b's corpus claimed to require induction but every repeat sat at a
    constant offset, so a positional shift solved it. Amendment 3 requires
    a task to print the evidence that it needs the mechanism it claims.
    A skill is NON-DEGENERATE iff this is <= 0.20 on prompted items."""
    ids, _, starts, lens = encode(tok, items)
    best, best_d = 0.0, 0
    for d in range(1, min(starts) + 1):
        hits = sum(bool((ids[b, s - d:s - d + l] == ids[b, s:s + l]).all())
                   for b, (s, l) in enumerate(zip(starts, lens)))
        if hits / len(items) > best:
            best, best_d = hits / len(items), d
    return round(best, 3), best_d


def structure_report(tok, n=64, seed=3):
    """Prints what the skill suite actually contains: per-skill copy-oracle
    accuracy, and the spread of example-line token lengths (a constant
    spread is what makes a fixed offset work at all)."""
    rng = np.random.default_rng(seed)
    rows = {}
    for s in range(N_SKILLS):
        items = make_items(s, n, rng, True, EVAL_WORDS)
        acc, d = copy_oracle(tok, items)
        lls = [len(tok(example_line(s, w), add_special_tokens=False).input_ids)
               for w in EVAL_WORDS]
        rows[str(s)] = {"name": SKILLS[s][0], "type": SKILLS[s][1],
                        "copy_oracle_acc": acc, "copy_oracle_offset": d,
                        "line_tok_len_mean": round(float(np.mean(lls)), 2),
                        "line_tok_len_sd": round(float(np.std(lls)), 2),
                        "non_degenerate": acc <= 0.20}
        print(f"  {SKILLS[s][0]:10s} ({SKILLS[s][1]}) copy_oracle={acc:.3f} "
              f"@offset {d} | line_toks {rows[str(s)]['line_tok_len_mean']}"
              f"±{rows[str(s)]['line_tok_len_sd']} "
              f"{'NON-DEGENERATE' if acc <= 0.20 else 'copyable'}", flush=True)
    return rows


# ---- stage 1: the gate -----------------------------------------------------

def gate(model, tok, n=200, seed=7, freegen_n=0, structure=None):
    rng = np.random.default_rng(seed)
    out = {}
    for s in range(N_SKILLS):
        pr = accuracy(model, tok, make_items(s, n, rng, True, EVAL_WORDS))
        un = accuracy(model, tok, make_items(s, n, rng, False, EVAL_WORDS))
        # mismatched-shot control: same marker/format, another skill's rule
        mm_src = (s + 4) % N_SKILLS
        mm = accuracy(model, tok, make_items(s, n, rng, True, EVAL_WORDS,
                                             shot_skill=mm_src))
        ctrl = max(un, mm)
        deg = structure[str(s)]["copy_oracle_acc"] if structure else None
        rec = {"name": SKILLS[s][0], "type": SKILLS[s][1],
               "prompted": round(pr, 3), "unprompted": round(un, 3),
               "mismatched_shots": round(mm, 3), "control": round(ctrl, 3),
               "gap": round(pr - ctrl, 3), "copy_oracle_acc": deg,
               "non_degenerate": (deg is not None and deg <= 0.20),
               "pass": pr >= GATE_PROMPTED and ctrl <= GATE_UNPROMPTED}
        if freegen_n:
            rec["prompted_freegen"] = round(freegen_accuracy(
                model, tok, make_items(s, freegen_n, rng, True, EVAL_WORDS)), 3)
        out[str(s)] = rec
        print(f"  skill {s} {SKILLS[s][0]:10s} ({SKILLS[s][1]}) "
              f"prompted={rec['prompted']:.3f} unprompted={rec['unprompted']:.3f} "
              f"mismatched={rec['mismatched_shots']:.3f} gap={rec['gap']:.3f} "
              f"{'PASS' if rec['pass'] else 'FAIL'}"
              + (f" freegen={rec['prompted_freegen']:.3f}" if freegen_n else ""),
              flush=True)
    passed = [s for s in out if out[s]["pass"]]
    n_b = sum(1 for s in passed if out[s]["type"] == "B")
    n_nd = sum(1 for s in passed if out[s]["non_degenerate"])
    out["_verdict"] = {
        "n_pass": len(passed), "n_pass_type_B": n_b, "n_pass_non_degenerate": n_nd,
        "passed": [int(s) for s in passed],
        "passed_non_degenerate": [int(s) for s in passed
                                  if out[s]["non_degenerate"]],
        "base_pass": (len(passed) >= GATE_MIN_PASS and n_b >= GATE_MIN_TYPE_B
                      and n_nd >= GATE_MIN_NON_DEGENERATE),
        "prompted_spread_sd": round(float(np.std(
            [out[str(s)]["prompted"] for s in range(N_SKILLS)])), 3),
        "structure": structure,
    }
    return out


# ---- generality probes -----------------------------------------------------

@torch.no_grad()
def nats_per_token(model, tok, text, n_seq=16, seq_len=256):
    ids = tok(text, add_special_tokens=False).input_ids[: n_seq * seq_len]
    x = torch.tensor(ids[: (len(ids) // seq_len) * seq_len]).view(-1, seq_len)
    if DEVICE != "cpu":
        x = x.to(DEVICE)      # bypasses encode(); the ONE path that did
    tot, cnt = 0.0, 0
    for i in range(0, x.shape[0], 4):
        b = x[i:i + 4]
        lg = model(input_ids=b).logits[:, :-1]
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                               b[:, 1:].reshape(-1), reduction="sum").item()
        cnt += b[:, 1:].numel()
    return tot / cnt


@torch.no_grad()
def unknown_marker_drift(model, tok, base_preds, rng, n=64):
    """Fraction of unknown-marker queries whose next-token argmax still
    matches the frozen base's. In-format collateral damage."""
    items = [(f"{UNKNOWN_MARKER} {w} =", "x") for w in EVAL_WORDS[:n]]
    ids, mask, starts, lens = encode(tok, items)
    lg = model(input_ids=ids, attention_mask=mask).logits
    preds = [int(lg[b, starts[b] - 1].argmax()) for b in range(len(items))]
    if base_preds is None:
        return preds, 1.0
    same = np.mean([p == q for p, q in zip(preds, base_preds)])
    return base_preds, float(same)


# ---- stage 2: the loop -----------------------------------------------------

def kl_loss(model, tok, skill, rng, batch=BATCH):
    """KL(teacher || student) on answer positions. Teacher = FROZEN BASE
    (adapter disabled) + k-shot prompt. Student = adapter, unprompted."""
    words = [TRAIN_WORDS[int(i)] for i in rng.integers(len(TRAIN_WORDS),
                                                       size=batch)]
    p_items = [(make_prompt(skill, w, rng, True), term(SKILLS[skill][2](w)))
               for w in words]
    u_items = [(query_line(skill, w), term(SKILLS[skill][2](w)))
               for w in words]
    pi, pm, ps, pl = encode(tok, p_items)
    ui, um, us, ul = encode(tok, u_items)
    assert pl == ul, "answer spans must align between teacher and student"
    with torch.no_grad(), model.disable_adapter():
        t = answer_logits(model, pi, pm, ps, pl)
    s = answer_logits(model, ui, um, us, ul)
    return F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1),
                    log_target=True, reduction="batchmean")


def ce_loss(model, tok, skill, rng, batch=BATCH):
    """direct_ft control: CE on ground-truth answers, same examples."""
    words = [TRAIN_WORDS[int(i)] for i in rng.integers(len(TRAIN_WORDS),
                                                       size=batch)]
    items = [(query_line(skill, w), term(SKILLS[skill][2](w))) for w in words]
    ids, mask, starts, lens = encode(tok, items)
    lg = answer_logits(model, ids, mask, starts, lens)
    return F.cross_entropy(lg, answer_targets(ids, starts, lens))


EVAL_SEED = 91  # FIXED: every eval scores the identical query set


def eval_suite(model, tok, skills_so_far, tok_text, base_preds, _rng, n_eval,
               last=None, hard=False, freegen_n=0):
    """All evals reuse ONE fixed query set, so a change between two evals is
    a change in the model and not in the sample. Paired comparison: the
    sampling noise that would otherwise sit on every retention difference
    (+-0.035 at n=200) cancels."""
    rec = {"unprompted": {}, "prompted_on_top": {}}
    for s in skills_so_far:
        rng = np.random.default_rng(EVAL_SEED + s)
        rec["unprompted"][str(s)] = round(accuracy(
            model, tok, make_items(s, n_eval, rng, False, EVAL_WORDS)), 3)
        rng = np.random.default_rng(EVAL_SEED + 1000 + s)
        rec["prompted_on_top"][str(s)] = round(accuracy(
            model, tok, make_items(s, n_eval, rng, True, EVAL_WORDS)), 3)
    if freegen_n:
        # RE-VALIDATE the metric guard on CONSOLIDATED weights. DESIGN
        # amendment 3 checked teacher-forced vs free generation on the FROZEN
        # base only; the write changes the output distribution, so agreement
        # there does not carry over on its own. Final k, one arm, one seed.
        rec["freegen"] = {}
        for s in skills_so_far:
            rng = np.random.default_rng(EVAL_SEED + 4000 + s)
            rec["freegen"][str(s)] = round(freegen_accuracy(
                model, tok, make_items(s, freegen_n, rng, False, EVAL_WORDS)), 3)
    if hard:   # off-length generalisation probe, final k only (see HARD_WORDS)
        rec["hard_unprompted"], rec["hard_prompted_on_top"] = {}, {}
        for s in skills_so_far:
            rng = np.random.default_rng(EVAL_SEED + 2000 + s)
            rec["hard_unprompted"][str(s)] = round(accuracy(
                model, tok, make_items(s, n_eval, rng, False, HARD_WORDS)), 3)
            rng = np.random.default_rng(EVAL_SEED + 3000 + s)
            rec["hard_prompted_on_top"][str(s)] = round(accuracy(
                model, tok, make_items(s, n_eval, rng, True, HARD_WORDS)), 3)
    rec["generality_nats"] = round(nats_per_token(model, tok, tok_text), 4)
    _, same = unknown_marker_drift(model, tok, base_preds,
                                   np.random.default_rng(EVAL_SEED))
    rec["unknown_marker_agree"] = round(same, 3)
    # IS THE WRITE MARKER-CONDITIONAL? Query under a marker that was never
    # consolidated (<<K9>>), score against each skill's rule. High = the
    # write attached the rule to the FORMAT, not to the marker, and every
    # later "interference" reading is confounded by that. This separates
    # "K3 interference wall" from "the write was never conditional".
    rec["unknown_marker_applies_rule"] = {}
    for s in skills_so_far:
        rng = np.random.default_rng(EVAL_SEED + s)
        words = [EVAL_WORDS[int(i)] for i in rng.integers(len(EVAL_WORDS),
                                                          size=n_eval)]
        rec["unknown_marker_applies_rule"][str(s)] = round(accuracy(
            model, tok, [(f"{UNKNOWN_MARKER} {w} =", SKILLS[s][2](w))
                         for w in words]), 3)
    # CROSS-MARKER CONFUSION, among markers the run actually trains.
    # unknown_marker_applies_rule cannot separate "the write is
    # unconditional" from "trained markers are conditional and an UNSEEN
    # marker falls back to some rule". This can: rule of skill `last`
    # scored under every OTHER trained marker. High = genuinely
    # unconditional. Low, while each marker still serves its own rule,
    # = conditional with a fallback, which does not block the K readings.
    rec["cross_marker_leak"] = {}
    if last is not None:
        rng = np.random.default_rng(EVAL_SEED + last)
        words = [EVAL_WORDS[int(i)] for i in rng.integers(len(EVAL_WORDS),
                                                          size=n_eval)]
        for j in skills_so_far:
            if j == last:
                continue
            rec["cross_marker_leak"][str(j)] = round(accuracy(
                model, tok, [(query_line(j, w), SKILLS[last][2](w))
                             for w in words]), 3)
    return rec


def run_arm(model, tok, arm, seed, skills, tok_text, base_preds,
            replay_frac=0.25, n_eval=200, probe_every=20, probe_n=64,
            freegen_n=0):
    from peft import get_peft_model, LoraConfig
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    torch.manual_seed(seed)
    pm = get_peft_model(model, cfg)
    fresh = {k: v.detach().clone() for k, v in pm.state_dict().items()
             if "lora_" in k}
    rng = np.random.default_rng(seed)
    order = [int(s) for s in rng.permutation(skills)]
    # a replay sweep must not land in the same arm key as the headline arm,
    # or analyze.py silently averages two replay budgets together
    if arm == "consolidate" and replay_frac != 0.25:
        arm = f"consolidate_r{replay_frac}"
    rec = {"arm": arm, "seed": seed, "order": order, "replay_frac": replay_frac,
           "n_eval": n_eval, "git_sha": git_sha(), "steps": []}
    cum = 0
    t0 = time.time()
    for i, s in enumerate(order):
        if arm == "separate_adapters":       # fresh weight set per skill
            pm.load_state_dict(fresh, strict=False)
        opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad],
                                lr=LR)
        rf = 0.0 if arm == "no_replay" else replay_frac
        probes, n_replay = [], 0
        for step in range(STEPS_PER_SKILL):
            # `ce_write_kl_replay` isolates the WRITE objective: CE on the new
            # skill, context distillation (KL) on the replayed priors, so it
            # differs from `consolidate` in exactly one factor and from
            # `direct_ft` in exactly the other. `direct_ft` swaps BOTH, which
            # is the confound that made the grid's finding 3 non-citable.
            loss = (ce_loss(pm, tok, s, rng)
                    if arm in ("direct_ft", "ce_write_kl_replay")
                    else kl_loss(pm, tok, s, rng))
            if rf and i and step % max(1, int(1 / rf)) == 0:
                # ROUND-ROBIN, not uniform-random: at k=6 with 15 replay
                # batches, random sampling leaves some prior skill with 0-1
                # batches by chance, so a retention collapse could be a
                # sampling accident. This gives every prior equal coverage.
                prior = order[n_replay % i]
                n_replay += 1
                loss = loss + (ce_loss(pm, tok, prior, rng) if arm == "direct_ft"
                               else kl_loss(pm, tok, prior, rng))
            opt.zero_grad(); loss.backward(); opt.step()
            cum += 1
            # grid-snapping guard 2: intra-skill retention probe
            if (probe_every and (step + 1) % probe_every == 0 and i
                    and arm != "separate_adapters"):
                probes.append({"step": step + 1, "cum": cum, "retention": {
                    str(j): round(accuracy(pm, tok, make_items(
                        j, probe_n, np.random.default_rng(EVAL_SEED + j),
                        False, EVAL_WORDS)), 3)
                    for j in order[:i]}})
        # ALL skills every step, not just consolidated ones: leakage into a
        # not-yet-written skill is as informative as loss of a written one.
        seen = order if arm != "separate_adapters" else [s]
        step_rec = eval_suite(pm, tok, seen, tok_text, base_preds, rng, n_eval,
                              last=int(s), hard=(i == len(order) - 1),
                              freegen_n=freegen_n if i == len(order) - 1 else 0)
        step_rec.update({"k": i + 1, "skill": int(s), "cum_steps": cum,
                         "n_replay_batches": n_replay,
                         "intra_skill_probes": probes,
                         "write_tripwire_ok":
                             step_rec["unprompted"][str(s)] >= WRITE_TRIPWIRE,
                         "wall_s": round(time.time() - t0, 1)})
        rec["steps"].append(step_rec)
        # crash-safe per-step record. Written OUTSIDE runs/*.json (analyze.py
        # globs that) so a partial record can never be silently analysed as a
        # finished cell, but nothing is lost if the box dies at k=7 of 8.
        RUNS.mkdir(exist_ok=True)
        tmp = RUNS / f"{arm}_s{seed}.inprogress.tmp"
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, RUNS / f"{arm}_s{seed}.inprogress")
        print(f"  [{arm} s{seed}] k={i+1} skill={SKILLS[s][0]} "
              f"unprompted={step_rec['unprompted']} "
              f"nats={step_rec['generality_nats']} "
              f"({step_rec['wall_s']}s)", flush=True)
    rec["wall_s"] = round(time.time() - t0, 1)
    RUNS.mkdir(exist_ok=True)
    # final-k adapter, ~8 MB of LoRA. Kept so the benchmark-percentage anchor
    # for the nats drift (library/sweeps/ce-write-objective-2026: "no source
    # reports capability damage in nats") can be measured later without
    # re-running the 50-minute cell.
    try:
        pm.save_pretrained(str(RUNS / f"adapter_{arm}_s{seed}"))
    except Exception as e:                              # never lose the record
        rec["adapter_save_error"] = repr(e)
    (RUNS / f"{arm}_s{seed}.json").write_text(json.dumps(rec, indent=1))
    pm.unload()                              # restore the bare base model
    return rec


# ---- entry points ----------------------------------------------------------

def load(base):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # DEVICE/DTYPE, added 2026-08-17 for Colab T4. Defaults are UNCHANGED
    # (cpu, float32) so every committed record stays bit-reproducible; only an
    # explicit --device/--dtype alters behaviour.
    # T4 is Turing: fp16 tensor cores, NO bf16. Validate rather than trust.
    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[DTYPE]
    if DTYPE == "bfloat16" and DEVICE.startswith("cuda") \
            and not torch.cuda.is_bf16_supported():
        raise SystemExit("bfloat16 requested but this GPU is fp16-only "
                         "(T4/Turing). Use --dtype float16.")
    m = AutoModelForCausalLM.from_pretrained(base, dtype=dt)
    if DEVICE != "cpu":
        m = m.to(DEVICE)
    m.eval()
    return tok, m


POOL_CLASH_THRESH = 0.20     # the lab's standing control level (gate, K0)


def pool_clashes(tok, skills, thresh=POOL_CLASH_THRESH):
    """Ordered pairs of skills whose ANSWERS coincide on the EVAL words.

    Why this exists: e004g's grid ran with `B_double` ("cat cat") and
    `B_triple` ("cat cat cat") in the same pool. Answers are scored
    teacher-forced, argmax per answer token from the answer's first
    position, so writing either one makes the OTHER read 1.000 without
    ever being written -- 8 instances across 3 arms and 3 seeds. Four
    coincidence kinds are checked, all on the REAL eval words:

      equal         identical answer strings
      ws_equal      identical after whitespace normalisation
      token_prefix  i's answer tokens are a prefix of j's (the e004g bug;
                    this is the one that actually inflates the metric)
      token_suffix  i's whole answer is a token SUFFIX of j's

    `token_suffix` is REPORTED BUT NOT BLOCKING, and that is a measured
    decision, not laziness: scoring is positional from the answer's first
    token, so a shared tail is never reached with a wrong head and cannot
    inflate anything. Made blocking it fires on `B_first`/`B_wfirst` (100%)
    and `B_last`/`B_spaced` (100%) -- because a one-token answer is
    trivially a suffix of anything ending in that token -- which would
    condemn e015's and e019's live prefix-free sets for a hazard that does
    not exist. It stays in the census as a human-readability flag.

    Returns [(kind, name_i, name_j, fraction)] for fractions > thresh.
    `BLOCKING_KINDS` says which of those void a measurement.
    """
    ans = {s: [SKILLS[s][2](w) for w in EVAL_WORDS] for s in skills}
    enc = {s: [tok(" " + a, add_special_tokens=False).input_ids for a in ans[s]]
           for s in skills}
    ws = {s: [" ".join(a.split()) for a in ans[s]] for s in skills}
    out = []
    for i in skills:
        for j in skills:
            if i == j:
                continue
            for kind, f in (
                ("equal", np.mean([a == b for a, b in zip(ans[i], ans[j])])),
                ("ws_equal", np.mean([a == b for a, b in zip(ws[i], ws[j])])),
                ("token_prefix", np.mean([b[:len(a)] == a
                                          for a, b in zip(enc[i], enc[j])])),
                ("token_suffix", np.mean([b[-len(a):] == a
                                          for a, b in zip(enc[i], enc[j])])),
            ):
                if kind in ("equal", "ws_equal") and i > j:
                    continue                      # symmetric, report once
                if float(f) > thresh:
                    out.append((kind, SKILLS[i][0], SKILLS[j][0],
                                round(float(f), 3)))
    return out


BLOCKING_KINDS = ("equal", "ws_equal", "token_prefix")


def assert_pool_clean(tok, skills, thresh=POOL_CLASH_THRESH):
    """FAIL LOUD before any compute is spent. Not a warning: a pool with a
    coinciding pair produces numbers that cannot be repaired afterwards."""
    all_cl = pool_clashes(tok, skills, thresh)
    cl = [c for c in all_cl if c[0] in BLOCKING_KINDS]
    for k, a, b, f in all_cl:
        if k not in BLOCKING_KINDS:
            print(f"pool gate NOTE (non-blocking): {k} {a} -> {b} "
                  f"on {f:.0%} of eval words", flush=True)
    if cl:
        raise SystemExit(
            "POOL GATE FAILED: skill answers coincide on the eval words, so a "
            "never-written skill can read 1.000.\n  " +
            "\n  ".join(f"{k}: {a} -> {b} on {f:.0%} of eval words"
                        for k, a, b, f in cl) +
            "\nDrop one skill of each pair, or re-run with --skills excluding "
            "it. Do NOT proceed: see e004g DESIGN.md 'POOL GATE'.")
    print(f"pool gate OK: {len(skills)} skills, no coincidence > {thresh}",
          flush=True)


def selfcheck(tok):
    """Smallest thing that fails if the harness logic breaks. Tokenizer
    only, no model, ~2s. Run it before trusting any number."""
    rng = np.random.default_rng(0)
    # 0. the pool gate detects the pair that actually contaminated the grid,
    #    and does not fire on the four clean skills the corrected read uses
    cl = pool_clashes(tok, [7, 13])
    assert any(k == "token_prefix" and a == "B_double" and b == "B_triple"
               for k, a, b, _ in cl), cl
    assert not pool_clashes(tok, [5, 8, 9, 11]), pool_clashes(tok, [5, 8, 9, 11])
    assert not [c for c in cl if c[0] not in BLOCKING_KINDS
                and c[0] != "token_suffix"], cl
    # 1. answer span is exactly the answer, and the prefix is untouched
    items = make_items(5, 8, rng, True, EVAL_WORDS)
    ids, mask, starts, lens = encode(tok, items)
    for b, (pre, ans) in enumerate(items):
        got = tok.decode(ids[b, starts[b]:starts[b] + lens[b]])
        assert got.strip() == ans, (got, ans)
        assert tok.decode(ids[b, :starts[b]]) == pre
        assert int(mask[b].sum()) == starts[b] + lens[b]
    # 2. teacher (prompted) and student (unprompted) answer spans align --
    #    the KL is summed over these positions and silently wrong if not
    for s in (5, 7, 8):
        w = EVAL_WORDS[:6]
        _, _, _, pl = encode(tok, [(make_prompt(s, x, rng, True),
                                    SKILLS[s][2](x)) for x in w])
        _, _, _, ul = encode(tok, [(query_line(s, x), SKILLS[s][2](x))
                                   for x in w])
        assert pl == ul, (s, pl, ul)
    # 3. copy_oracle catches a fixed-offset answer and clears one that is not
    assert copy_oracle(tok, make_items(0, 32, rng, True, EVAL_WORDS))[0] > 0.8
    assert copy_oracle(tok, make_items(5, 32, rng, True, EVAL_WORDS))[0] < 0.2
    # 4. mismatched shots really demonstrate a different rule
    p_ok = make_prompt(5, "apple", np.random.default_rng(1), True)
    p_mm = make_prompt(5, "apple", np.random.default_rng(1), True, shot_skill=1)
    assert p_ok != p_mm and p_ok.split("\n")[-1] == p_mm.split("\n")[-1]
    assert MARKERS[5] in p_mm and "north" in p_mm
    # 5. the conditionality probe uses a marker no skill owns
    assert UNKNOWN_MARKER not in MARKERS
    # 6. eval and train word pools are disjoint (no leakage into "held out")
    assert not (set(TRAIN_WORDS) & set(EVAL_WORDS))
    assert len(TRAIN_WORDS) == len(EVAL_WORDS) == 32
    # 6b. the off-length probe is genuinely off-length and never trained on
    assert not (set(HARD_WORDS) & set(TRAIN_WORDS) | set(HARD_WORDS) & set(EVAL_WORDS))
    assert min(len(w) for w in HARD_WORDS) > max(len(w) for w in WORDS)
    # 7. every skill's answer is a pure function of its input
    for s in range(N_SKILLS):
        assert all(SKILLS[s][2](w) == SKILLS[s][2](w) for w in WORDS)
    # 8. THE TERMINATOR IS APPENDED, NOT MERGED (e021). If BPE merged "\n"
    #    backwards into the last answer token, the scored span of the write
    #    would silently shift and both the answer and the terminator would be
    #    trained at the wrong positions. Checked on every skill's answers over
    #    every word, plus the NULL_ANSWER fallback that e017's self_targets can
    #    return as a snap target.
    #    The gate itself is self-tested: it must FIRE on B_bang (whose answers
    #    end in "!", which BPE merges with "\n") and stay silent on the clean
    #    skills. B_bang, B_dot, B_angle and B_upbang are NOT in the grid suite.
    assert term_clashes(tok, [12]), "term gate blind to its own known-positive"
    assert not term_clashes(tok, [5, 8, 9, 11]), term_clashes(tok, [5, 8, 9, 11])
    assert term_clashes(tok, [], extra=["?"]) == ["?"]   # "?\n" is one token
    # 8b. and the span really decodes to answer+terminator end-to-end
    it = [(query_line(5, w), term(SKILLS[5][2](w))) for w in EVAL_WORDS[:8]]
    ids, _, st, ln = encode(tok, it)
    for b, (pre, ans) in enumerate(it):
        assert tok.decode(ids[b, st[b]:st[b] + ln[b]]) == " " + ans
        assert tok.decode(ids[b, :st[b]]) == pre
    print("selfcheck OK")


def pos_scheme(model):
    """Record the positional scheme. e012b: learned absolute positions in
    the residual stream vs rotary was the difference between a gap of
    0.0006 and 0.998, all else equal. Never leave this unrecorded again."""
    c = model.config
    names = [n for n, _ in model.named_modules()]
    return {"model_type": c.model_type,
            "rope_theta": getattr(c, "rope_theta", None),
            "has_rotary_module": any("rotary" in n for n in names),
            "has_absolute_pos_embedding":
                any(n.endswith(("wpe", "position_embeddings")) for n in names),
            "n_layer": getattr(c, "num_hidden_layers", None),
            "d_model": getattr(c, "hidden_size", None),
            "n_params": sum(p.numel() for p in model.parameters())}


def throughput(model, tok, rng):
    """Forward tokens/sec and sec/optimizer-step, measured not assumed."""
    items = make_items(0, 32, rng, True, EVAL_WORDS)
    ids, mask, starts, lens = encode(tok, items)
    with torch.no_grad():
        model(input_ids=ids[:4], attention_mask=mask[:4])      # warm
        t0 = time.time()
        model(input_ids=ids, attention_mask=mask)
        dt = time.time() - t0
    return {"fwd_tokens_per_s": round(int(mask.sum()) / dt),
            "fwd_seqs_per_s": round(len(items) / dt, 2),
            "seq_len": int(ids.shape[1])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--freegen", type=int, default=0)
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skills", default="")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--replay", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda. Default cpu keeps records reproducible.")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--steps", type=int, default=STEPS_PER_SKILL)
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--freegen-final", type=int, default=0,
                    help="re-validate the teacher-forced metric against free "
                         "generation on CONSOLIDATED weights, at final k")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if getattr(a, "device", "cpu") != "cpu":
        globals()["DEVICE"] = a.device
    if getattr(a, "dtype", "float32") != "float32":
        globals()["DTYPE"] = a.dtype
    STEPS_PER_SKILL = a.steps
    if a.selfcheck:
        selfcheck(AutoTokenizer.from_pretrained(a.base))
        raise SystemExit
    tok, model = load(a.base)
    tag = a.base.split("/")[-1]

    print("positional scheme:", json.dumps(pos_scheme(model)), flush=True)
    if a.throughput:
        print(json.dumps(throughput(model, tok, np.random.default_rng(0))))
    struct = None
    if a.structure or a.gate:
        print("skill-suite structure (degeneracy check):", flush=True)
        struct = structure_report(tok)
    if a.gate:
        t0 = time.time()
        g = gate(model, tok, n=a.n, freegen_n=a.freegen, structure=struct)
        g["_verdict"]["base"] = a.base
        g["_verdict"]["pos_scheme"] = pos_scheme(model)
        g["_verdict"]["wall_s"] = round(time.time() - t0, 1)
        (HERE / f"gate_{tag}.json").write_text(json.dumps(g, indent=1))
        print(json.dumps(g["_verdict"], indent=1), flush=True)
    if a.pilot or a.arm:
        text = (HERE / "generality.txt").read_text()
        rng0 = np.random.default_rng(0)
        base_preds, _ = unknown_marker_drift(model, tok, None, rng0)
        if a.skills:
            skills = [int(x) for x in a.skills.split(",")]
        else:   # non-degenerate skills FIRST: the pilot must contain one
            v = json.loads((HERE / f"gate_{tag}.json").read_text())["_verdict"]
            nd = v["passed_non_degenerate"]
            skills = (nd + [s for s in v["passed"] if s not in nd])[:3]
        assert any(json.loads((HERE / f"gate_{tag}.json").read_text())
                   [str(s)]["non_degenerate"] for s in skills), \
            "pilot must include a non-degenerate skill (DESIGN amendment 1)"
        assert_pool_clean(tok, skills)      # blocking; see DESIGN 'POOL GATE'
        print("pilot skills:", [SKILLS[s][0] for s in skills], flush=True)
        # k=0 on the FROZEN base: the null level for every eval, measured
        # rather than assumed. K0's 0.20 threshold is meaningless without it.
        # NOT in runs/: analyze.py globs runs/*.json for arm records.
        # Skipped if already measured: the grid runs several cells in
        # parallel and this is the same 4-minute measurement in each of
        # them. Delete the file to force a fresh one.
        k0p = HERE / "k0_frozen_base.json"
        if k0p.exists():
            print("k=0 frozen base: reusing", k0p.name, flush=True)
        else:
            k0 = eval_suite(model, tok, skills, text, base_preds, None, a.n_eval)
            k0.update({"k": 0, "skill": None, "cum_steps": 0,
                       "arm": "frozen_base", "base": a.base,
                       "git_sha": git_sha()})
            tmp = k0p.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(k0, indent=1))
            os.replace(tmp, k0p)          # atomic: parallel cells may race
            print("k=0 frozen base:", json.dumps(
                {x: k0[x] for x in ("unprompted", "prompted_on_top",
                                    "unknown_marker_applies_rule",
                                    "generality_nats")}), flush=True)
        arms = [a.arm] if a.arm else ["consolidate", "separate_adapters"]
        for arm in arms:
            run_arm(model, tok, arm, a.seed, skills, text, base_preds,
                    replay_frac=a.replay, n_eval=a.n_eval,
                    probe_every=a.probe_every, freegen_n=a.freegen_final)
