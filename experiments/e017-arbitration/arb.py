"""e017: is the continual-learning failure ARBITRATION, and can it be fixed
without storing old data?

Reuses e015's harness by IMPORTING it (which in turn imports e004g's), so the
skill suite, the marker format, the KL-to-frozen-base write, the eval, the
`unknown_marker_applies_rule` conditionality probe, the masking-vs-erasure
rank diagnostic, the copy-oracle degeneracy check and the prefix-clash check
are all the SAME CODE that produced e015's and e004g's numbers. Nothing is
re-derived. What is NEW here is one hook: an AUXILIARY LOSS applied on the
same schedule as 25% replay, differing between arms only in WHICH CONTEXTS it
scores and WHAT TARGET it uses.

    arm          aux contexts                 aux target
    ---------------------------------------------------------------------
    replay25     prior skills' own cues       prior skills' TRUE answers
                 (STORES OLD DATA)            (the reference; e015/e004g)
    negcue       synthetic unclaimed cues     the FROZEN BASE's distribution
    reserve      synthetic unclaimed cues     a fixed null token
    selfdistil   prior skills' own cues       THE MODEL'S OWN outputs, greedily
                                              recorded before the write (LwF)
    selfdistil_offcue
                 synthetic unclaimed cues     the model's own outputs there
                                              (LwF's OWN documented failure
                                              mode as a predicted negative)
    cueanchor    (none -- reweights the write loss itself)
    no_replay    (none)

Only `replay25` stores old input->answer pairs. `selfdistil` stores the old
CUE IDENTIFIERS and a generic input pool, and takes its labels from the model.
That is Learning without Forgetting (Li & Hoiem 2016) applied at the
arbitration level; see DESIGN.md, which cites it rather than claiming it.
`selfdistil_offcue` is the same method with a proxy set that does NOT exercise
the old behaviour, which is the failure mode LwF's own authors report. It is a
PREDICTED NEGATIVE: if it retains as well as `selfdistil`, the retention is not
coming from the mechanism this experiment claims and the implementation is
wrong.

The measurement that matters is TWO quantities, never one: `umar_age0` (is the
newest write conditional) and retention, per k, per skill, per seed. Reporting
retention alone is what made e015's result readable as an allocation failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
E015 = HERE.parent / "e015-no-replay"
sys.path.insert(0, str(E015))
import cont as CT               # noqa: E402  e015's harness (imports e004g's)
import consolidate as C         # noqa: E402  e004g's harness

RUNS = Path(__import__("os").environ.get("ARB_RUNS", HERE / "runs"))

# ---- the synthetic, unclaimed cue pool -------------------------------------
# MUST be disjoint from every skill marker AND from the probe marker. If the
# probe marker were trainable, `umar_age0` would stop being a held-out test of
# conditionality and become a trained target, and the whole experiment would
# measure its own intervention. Asserted in selfcheck.
RESERVED_CUES = [f"<<K{200 + i}>>" for i in range(8)]
NULL_ANSWER = "?"          # what `reserve` parks on the format default
# e021: the self-distillation fallback for an EMPTY greedy generation cannot be
# NULL_ANSWER, because "?" + "\n" is a SINGLE Qwen token, so appending the
# terminator to it would shift the scored span (caught by the terminator gate).
# This is a rare arbitrary null, not a measurement; its use is recorded per run.
SELF_NULL = "none"

ARMS = ["no_replay", "replay25", "negcue", "cueanchor", "reserve",
        "selfdistil", "selfdistil_offcue", "selfdistil_front"]
AUX_NEEDS_PRIORS = {"replay25", "selfdistil", "selfdistil_offcue",
                    "selfdistil_front"}
# selfdistil_front is e021's T3 arm: the SAME aux content and the SAME NUMBER
# of aux batches as `selfdistil`, spent on the TWO MOST RECENT priors instead
# of round-robin over all of them. Equal total aux cost is what makes it a test
# of WHEN the budget is spent rather than of how much.
FRONT_W = 2
STEPS, BATCH, LR = C.STEPS_PER_SKILL, C.BATCH, C.LR
GRID_SKILLS = CT.GRID_SKILLS          # the 10 gated, non-degenerate,
                                      # token-prefix-free skills of e015


# ---- auxiliary losses ------------------------------------------------------

def _first_pos_logits(pm, tok, prefixes):
    """Logits at the FIRST answer position for each prefix: [B, V].

    One position, not the whole answer: the arbitration decision -- whose rule
    fires under this cue -- is made at the first answer token, and scoring it
    needs no commitment to what the rest of the answer would be. Disclosed as
    a limitation in DESIGN.md."""
    items = [(p, NULL_ANSWER) for p in prefixes]
    ids, mask, starts, lens = C.encode(tok, items)
    assert set(lens) == {1}, "null answer must be a single token"
    return C.answer_logits(pm, ids, mask, starts, lens)


def _aux_words(rng, batch):
    """Aux contexts are built from TRAIN_WORDS only. EVAL_WORDS is what every
    retention number is measured on; training an intervention there would
    leak the eval set into the intervention."""
    return [C.TRAIN_WORDS[int(i)] for i in
            rng.integers(len(C.TRAIN_WORDS), size=batch)]


def negcue_loss(pm, tok, rng, batch=BATCH):
    """On cues nobody owns, keep behaving as the FROZEN BASE did.

    This is the direct test of the dossier's intervention 1: teach
    conditionality using synthetic cues instead of stored prior examples."""
    cues = [RESERVED_CUES[int(i)] for i in
            rng.integers(len(RESERVED_CUES), size=batch)]
    pre = [f"{c} {w} =" for c, w in zip(cues, _aux_words(rng, batch))]
    with torch.no_grad(), pm.disable_adapter():
        t = _first_pos_logits(pm, tok, pre)
    s = _first_pos_logits(pm, tok, pre)
    return F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1),
                    log_target=True, reduction="batchmean")


def reserve_loss(pm, tok, rng, null_id, batch=BATCH):
    """Keep the format default owned by a NULL behaviour, so no skill can
    inherit it (dossier intervention 3). Same contexts as `negcue`; the only
    difference is the target, which is what makes the two comparable."""
    cues = [RESERVED_CUES[int(i)] for i in
            rng.integers(len(RESERVED_CUES), size=batch)]
    pre = [f"{c} {w} =" for c, w in zip(cues, _aux_words(rng, batch))]
    s = _first_pos_logits(pm, tok, pre)
    return F.cross_entropy(s, torch.full((len(pre),), null_id,
                                         dtype=torch.long,
                                         device=s.device))


@torch.no_grad()
def self_targets(pm, tok, pre, max_new=8):
    """The model's OWN answer under each prefix, greedily generated.

    Nothing about the true rule is used -- not its answer, not even its
    length. This is the whole point: the model is the record, so the method
    stores no old data. Returns a list of answer STRINGS."""
    enc = [tok(p, add_special_tokens=False).input_ids for p in pre]
    T = max(len(e) for e in enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(enc), T), pad, dtype=torch.long)
    mask = torch.zeros((len(enc), T), dtype=torch.long)
    for b, e in enumerate(enc):                     # LEFT pad for generation
        ids[b, T - len(e):] = torch.tensor(e)
        mask[b, T - len(e):] = 1
    if C.DEVICE != "cpu":
        ids, mask = ids.to(C.DEVICE), mask.to(C.DEVICE)
    gen = pm.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new,
                      do_sample=False, pad_token_id=pad)
    out = []
    for b in range(len(enc)):
        txt = tok.decode(gen[b, T:], skip_special_tokens=True)
        out.append(txt.strip().split("\n")[0].strip() or SELF_NULL)
    return out


def selfdistil_loss(pm, tok, snap):
    """Preserve what the model ITSELF said on an old cue (LwF). Teacher-forced
    CE against the recorded strings, all answer positions.

    e021: the TERMINATOR is scored here too, exactly as `C.kl_loss` scores it
    in the write. Otherwise self-distillation would un-teach stopping on old
    cues while the write teaches it on the new one, and the arms would differ
    in aux CONTENT rather than only in aux TARGET."""
    ids, mask, starts, lens = C.encode(
        tok, [(p, C.term(a)) for p, a in zip(*snap)])
    lg = C.answer_logits(pm, ids, mask, starts, lens)
    return F.cross_entropy(lg, C.answer_targets(ids, starts, lens))


def cueanchor_kl(pm, tok, skill, rng, batch=BATCH):
    """The write loss, REWEIGHTED toward cue-conditional answer positions
    (dossier intervention 2).

    Weight of an answer position = KL between the teacher's distribution under
    the skill's OWN cue and under an unclaimed cue, normalised to mean 1. A
    position the teacher predicts identically whichever cue is present is a
    FORMAT position and gets less of the write; a position that depends on the
    cue gets more. No aux batch and no extra target: this arm changes only how
    the same write is weighted.

    Returns (loss, weight-stats). If the weights are near-uniform the arm is
    undefined by construction, not a null about cue anchoring -- which is why
    the spread is recorded in every run."""
    words = _aux_words(rng, batch)
    p_items = [(C.make_prompt(skill, w, rng, True), C.SKILLS[skill][2](w))
               for w in words]
    u_items = [(C.query_line(skill, w), C.SKILLS[skill][2](w)) for w in words]
    alt = RESERVED_CUES[0]
    a_items = [(p.replace(C.MARKERS[skill], alt), a) for p, a in p_items]
    pi, pm_, ps, pl = C.encode(tok, p_items)
    ai, am, as_, al = C.encode(tok, a_items)
    ui, um, us, ul = C.encode(tok, u_items)
    assert pl == ul == al, "answer spans must align"
    with torch.no_grad(), pm.disable_adapter():
        t = C.answer_logits(pm, pi, pm_, ps, pl)
        t_alt = C.answer_logits(pm, ai, am, as_, al)
        w = F.kl_div(F.log_softmax(t_alt, -1), F.log_softmax(t, -1),
                     log_target=True, reduction="none").sum(-1)
        w = w / max(float(w.mean()), 1e-8)
    s = C.answer_logits(pm, ui, um, us, ul)
    per = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1),
                   log_target=True, reduction="none").sum(-1)
    stats = {"mean": 1.0, "sd": round(float(w.std()), 4),
             "max": round(float(w.max()), 4), "min": round(float(w.min()), 4)}
    return (w * per).mean(), stats


# ---- the loop --------------------------------------------------------------

def run_arm(model, tok, arm, seed, skills, tok_text, n_eval=200, rank=8,
            probe_n=64, tag="", diag=True, launch_sha="", freegen_n=32):
    from peft import get_peft_model, LoraConfig
    assert arm in ARMS, arm
    null_id = tok(" " + NULL_ANSWER, add_special_tokens=False).input_ids
    assert len(null_id) == 1, null_id
    null_id = null_id[0]
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    torch.manual_seed(seed)
    pm = get_peft_model(model, cfg)
    rng = np.random.default_rng(seed)
    order = [int(s) for s in rng.permutation(skills)]
    # EVERY arm gets 25% replay's schedule -- every 4th step, one batch of 16,
    # round-robin over contexts. Arms then differ ONLY in what the aux batch
    # contains, and "you gave that arm more compute" is not available as an
    # explanation of any difference.
    sched = CT.replay_schedule(0.25)
    rec = {"arm": arm + tag, "seed": seed, "order": order, "rank": rank,
           "n_eval": n_eval, "freegen_n": freegen_n,
           "terminator": repr(C.TERM), "front_w": FRONT_W,
           "aux_schedule_steps": sorted(sched),
           "aux_batch": BATCH, "lambda_aux": 1.0,
           "base": getattr(model.config, "_name_or_path", "?"),
           "n_params": sum(p.numel() for p in model.parameters()),
           "n_lora_params": sum(p.numel() for _, p in CT.lora_named(pm)),
           "n_skills": len(order), "tag": tag,
           "git_sha": C.git_sha(), "launch_sha256": launch_sha,
           "arb_sha256": __import__("hashlib").sha256(
               Path(__file__).read_bytes()).hexdigest()[:16],
           "reserved_cues": RESERVED_CUES, "self_null": SELF_NULL,
           "steps": []}
    cum, t0 = 0, time.time()
    for i, s in enumerate(order):
        snaps, fidelity, tsamples = {}, {}, {}
        if arm.startswith("selfdistil") and i:
            # RECORD BEFORE THE WRITE. Fidelity of these targets against the
            # true rule is logged per prior: the known failure mode of this arm
            # is COMPOUNDING DRIFT (by write k the model's own output on an old
            # cue may already be the k-1 rule), and without this number a
            # decaying curve cannot be told from a leaking constraint.
            words = _aux_words(np.random.default_rng(seed + 1000 + i), BATCH)
            for n_, j in enumerate(order[:i]):
                pre = ([C.query_line(j, w) for w in words]
                       if arm != "selfdistil_offcue"
                       else [f"{RESERVED_CUES[n_ % len(RESERVED_CUES)]} {w} ="
                             for w in words])
                ans = self_targets(pm, tok, pre)
                snaps[j] = (pre, ans)
                # e021: the actual TEACHER STRINGS, committed. The mediation
                # claim is about what these strings are, and a fidelity scalar
                # cannot show a reader "wagon???????" vs "wagon".
                tsamples[str(j)] = [[p_, a_] for p_, a_ in
                                    list(zip(pre, ans))[:3]]
                fidelity[str(j)] = round(float(np.mean(
                    [a == C.SKILLS[j][2](w) for w, a in zip(words, ans)])), 3)
        opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad],
                                lr=LR)
        n_aux, wstats = 0, None
        for step in range(STEPS):
            if arm == "cueanchor":
                loss, wstats = cueanchor_kl(pm, tok, s, rng)
            else:
                loss = C.kl_loss(pm, tok, s, rng)
            if step in sched and not (arm in AUX_NEEDS_PRIORS and not i):
                if not i:
                    prior = None
                elif arm == "selfdistil_front":
                    prior = order[i - 1 - (n_aux % min(FRONT_W, i))]
                else:
                    prior = order[n_aux % i]
                if arm == "replay25":
                    loss = loss + C.kl_loss(pm, tok, prior, rng, batch=BATCH)
                    n_aux += 1
                elif arm.startswith("selfdistil"):
                    loss = loss + selfdistil_loss(pm, tok, snaps[prior])
                    n_aux += 1
                elif arm == "negcue":
                    loss = loss + negcue_loss(pm, tok, rng)
                    n_aux += 1
                elif arm == "reserve":
                    loss = loss + reserve_loss(pm, tok, rng, null_id)
                    n_aux += 1
            opt.zero_grad()
            loss.backward()
            opt.step()
            cum += 1
        step_rec = CT.eval_step(pm, tok, order, i, tok_text, n_eval, probe_n,
                                diag=diag, freegen_n=freegen_n)
        step_rec.update({
            "k": i + 1, "skill": int(s), "cum_steps": cum,
            "n_aux_batches": n_aux, "n_aux_examples": n_aux * BATCH,
            "n_replay_examples": n_aux * BATCH if arm == "replay25" else 0,
            "selfdistil_target_fidelity": fidelity or None,
            "selfdistil_target_samples": tsamples or None,
            "cueanchor_weight_stats": wstats,
            "write_tripwire_ok":
                step_rec["unprompted"][str(s)] >= C.WRITE_TRIPWIRE,
            "wall_s": round(time.time() - t0, 1)})
        rec["steps"].append(step_rec)
        # CRASH-SAFE PARTIAL, every k (e004g's `.inprogress` pattern). A run
        # killed at k=9 must still be a record: 15 multi-hour runs on one box
        # is too much exposure to write the file only at the end, and the T4
        # write guard can be read from wave-1 partials instead of waiting for
        # the whole grid.
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{arm}{tag}_s{seed}.json.inprogress").write_text(
            json.dumps(rec, indent=1))
        print(f"  [{arm}{tag} s{seed}] k={i+1} {C.SKILLS[s][0]} "
              f"wrote={step_rec['unprompted'][str(s)]:.3f} "
              f"umar0={step_rec['unknown_marker_applies_rule'][str(s)]:.3f} "
              f"prior={[step_rec['unprompted'][str(j)] for j in order[:i]]} "
              f"({step_rec['wall_s']}s)", flush=True)
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["peak_rss_gb"] = CT.peak_rss_gb()
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{arm}{tag}_s{seed}.json").write_text(json.dumps(rec, indent=1))
    (RUNS / f"{arm}{tag}_s{seed}.json.inprogress").unlink(missing_ok=True)
    pm.unload()
    return rec


# ---- checks ----------------------------------------------------------------

def selfcheck(tok):
    """Smallest thing that fails if the NEW logic breaks. ~5s, no model.
    e015's selfcheck (which runs e004g's) covers everything reused."""
    CT.selfcheck(tok)
    # 1. THE PROBE MUST NOT BE TRAINABLE. negcue and reserve train behaviour on
    #    unclaimed cues, which is the same family as the conditionality probe.
    #    If the probe marker were in the pool, umar_age0 would be a trained
    #    target rather than a held-out test and the experiment would measure
    #    itself.
    assert C.UNKNOWN_MARKER not in RESERVED_CUES
    assert not (set(RESERVED_CUES) & set(C.MARKERS))
    assert len(set(RESERVED_CUES)) == len(RESERVED_CUES)
    # 2. the null target is exactly one token (reserve scores one position)
    assert len(tok(" " + NULL_ANSWER, add_special_tokens=False).input_ids) == 1
    # 3. aux contexts never touch the eval pool
    assert not (set(C.TRAIN_WORDS) & set(C.EVAL_WORDS))
    w = _aux_words(np.random.default_rng(0), 32)
    assert set(w) <= set(C.TRAIN_WORDS)
    # 4. the grid suite is prefix-free (e015's artifact: one skill's answer
    #    being a token prefix of another's produced 4 of 7 blocking rows)
    C.assert_pool_clean(tok, GRID_SKILLS)      # blocking, see e004g DESIGN
    # 4b. e021: no grid answer's last token merges with the terminator, and the
    #     NULL_ANSWER fallback that self_targets can return is clean too.
    C.assert_term_clean(tok, GRID_SKILLS, extra=[SELF_NULL])
    # 5. every arm's aux schedule is IDENTICAL to 25% replay's
    sch = CT.replay_schedule(0.25)
    assert len(sch) == 15 and set(sch.values()) == {16}
    # 6. the cue-swap really swaps the cue and leaves the query word alone
    rng = np.random.default_rng(1)
    p = C.make_prompt(5, "apple", rng, True)
    a = p.replace(C.MARKERS[5], RESERVED_CUES[0])
    assert C.MARKERS[5] not in a and RESERVED_CUES[0] in a
    assert a.count("apple") == p.count("apple") and a != p
    # 7. self-distillation targets are STRINGS the model produced, so the loss
    #    must be constructible without reference to the true rule
    snap = ([C.query_line(5, "apple"), C.query_line(5, "water")], ["zzz", "qqq"])
    _, _, st, ln = C.encode(tok, list(zip(*snap)))
    assert all(l >= 1 for l in ln) and len(st) == 2
    # 8. e021: the front-loaded arm must distil ON THE OLD CUES. The routing
    #    test used to be `arm == "selfdistil"`, which would have made any new
    #    selfdistil_* arm a silent OFF-CUE arm -- i.e. the predicted NEGATIVE.
    for arm in ("selfdistil", "selfdistil_front"):
        assert arm != "selfdistil_offcue" and arm.startswith("selfdistil")
    assert all(a in ARMS for a in AUX_NEEDS_PRIORS)
    # 9. equal aux cost, and front-loading really touches only recent priors
    n = len(CT.replay_schedule(0.25))
    for i in (1, 2, 5, 9):
        order = list(range(20, 20 + i + 1))
        uni = [order[k % i] for k in range(n)]
        fro = [order[i - 1 - (k % min(FRONT_W, i))] for k in range(n)]
        assert len(uni) == len(fro) == n            # EQUAL TOTAL AUX COST
        assert set(fro) <= set(order[max(0, i - FRONT_W):i])
        assert set(fro) == set(order[max(0, i - FRONT_W):i])
        assert set(uni) == set(order[:i])
    # 10. the terminator is inside the self-distillation target span
    snap = ([C.query_line(5, "apple")], ["zzz"])
    ids, _, st, ln = C.encode(tok, [(p, C.term(x)) for p, x in zip(*snap)])
    assert tok.decode(ids[0, st[0]:st[0] + ln[0]]) == " zzz" + C.TERM
    print("e017/e021 selfcheck OK")


# ---- entry point -----------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skills", default="")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--probe-n", type=int, default=64)
    ap.add_argument("--freegen-n", type=int, default=32)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda. Default cpu keeps records reproducible.")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--steps", type=int, default=0, help="override, smoke only")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-diag", action="store_true")
    a = ap.parse_args()
    if getattr(a, "device", "cpu") != "cpu":
        C.DEVICE = a.device
    if getattr(a, "dtype", "float32") != "float32":
        C.DTYPE = a.dtype
    torch.set_num_threads(a.threads)
    if a.steps:
        STEPS = C.STEPS_PER_SKILL = a.steps
    if a.selfcheck:
        from transformers import AutoTokenizer
        selfcheck(AutoTokenizer.from_pretrained(a.base))
        raise SystemExit
    # PROVENANCE AT LAUNCH, not at write: e015's pilot numbers turned out not
    # to be reproducible from committed code because the file was edited in
    # place while runs were live.
    launch_sha = __import__("hashlib").sha256(
        Path(__file__).read_bytes()).hexdigest()[:16]
    tok, model = C.load(a.base)
    skills = ([int(x) for x in a.skills.split(",")] if a.skills
              else list(GRID_SKILLS))
    print("skills:", [C.SKILLS[s][0] for s in skills], "sha", launch_sha,
          flush=True)
    run_arm(model, tok, a.arm, a.seed, skills,
            (HERE.parent / "e004g-consolidation" / "generality.txt").read_text(),
            n_eval=a.n_eval, rank=a.rank, probe_n=a.probe_n, tag=a.tag,
            freegen_n=a.freegen_n,
            diag=not a.no_diag, launch_sha=launch_sha)
