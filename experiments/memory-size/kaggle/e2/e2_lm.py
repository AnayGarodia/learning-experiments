"""E2 baselines at LM scale (Kaggle T4). Knowledge incorporation then retention.

Protocol (per Meta's sparse-memory-finetuning setup, simplified):
  1. Retention eval BEFORE: HellaSwag / ARC-easy / PIQA multiple-choice loglik accuracy, N examples each.
  2. Inject K facts (TriviaQA train Q/A pairs) by fine-tuning for E epochs with method in {full, lora, replay}.
  3. Acquisition: exact-match on the injected facts (greedy generation).
  4. Retention eval AFTER. Report deltas.
Everything hand-rolled (lm-eval is broken on the Kaggle image).
"""
import os, sys, json, time, math, random, argparse, subprocess
subprocess.run([sys.executable,"-m","pip","install","-q","-U","peft","torchao"],check=False)
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
p.add_argument("--method", default="lora")
p.add_argument("--facts", type=int, default=1000); p.add_argument("--epochs", type=int, default=12)
p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--bs", type=int, default=16)
p.add_argument("--n_eval", type=int, default=300); p.add_argument("--out", default="/kaggle/working/e2.json")
a = p.parse_args()
if "," in a.method:
    merged = {}
    for meth in a.method.split(","):
        out = a.out.replace(".json", f"_{meth}.json")
        subprocess.run([sys.executable, sys.argv[0], "--method", meth, "--out", out] + [x for x in sys.argv[1:] if x not in ("--method", a.method, "--out", a.out) and not x.startswith("--method=")], check=False)
        if os.path.exists(out): merged[meth] = json.load(open(out))["results"][meth]
        json.dump(dict(args=vars(a), results=merged), open(a.out, "w"), indent=1)
    sys.exit(0)
torch.manual_seed(0); random.seed(0)
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
def fresh(): return AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(DEV)
model = fresh()
T0 = time.time()
def log(*x): print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)


# ---------- retention eval: multiple choice by mean token loglik of each continuation ----------
@torch.no_grad()
def choice_loglik(ctx, conts):
    scores = []
    for c in conts:
        ids = tok(ctx + c, return_tensors="pt").input_ids.to(DEV)
        n_ctx = len(tok(ctx).input_ids)
        logits = model(ids).logits[0, :-1].float()
        lp = F.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0]
        scores.append(lp[n_ctx - 1:].sum().item())
    return int(torch.tensor(scores).argmax())


def eval_retention(n):
    model.eval(); res = {}
    hs = load_dataset("Rowan/hellaswag", split="validation").select(range(n))
    res["hellaswag"] = sum(choice_loglik(x["ctx"], [" " + e for e in x["endings"]]) == int(x["label"]) for x in hs) / n
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation").select(range(n))
    res["arc_easy"] = sum(choice_loglik("Question: " + x["question"] + "\nAnswer:", [" " + t for t in x["choices"]["text"]]) == x["choices"]["label"].index(x["answerKey"]) for x in arc) / n
    try:
        pq = load_dataset("baber/piqa", split="validation").select(range(n))
        res["piqa"] = sum(choice_loglik("Question: " + x["goal"] + "\nAnswer:", [" " + x["sol1"], " " + x["sol2"]]) == x["label"] for x in pq) / n
    except Exception as e:
        log("piqa unavailable:", repr(e)[:200])
    model.train(); return res


# ---------- facts ----------
tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train").shuffle(seed=0).select(range(2 * a.facts))
facts = [(f"Question: {x['question']}\nAnswer:", " " + x["answer"]["value"]) for x in tq.select(range(a.facts))]
old_facts = [(f"Question: {x['question']}\nAnswer:", " " + x["answer"]["value"]) for x in tq.select(range(a.facts, 2 * a.facts))]


@torch.no_grad()
def acquisition(items=None):
    items = facts if items is None else items
    model.eval(); hit = 0
    for q, ans in items:
        ids = tok(q, return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip().lower()
        hit += gen.startswith(ans.strip().lower()) or ans.strip().lower() in gen
    model.train(); return hit / len(items)


# ---------- training ----------
def lm_loss(texts, answer_only_from=None):
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    labels = enc.input_ids.clone(); labels[enc.attention_mask == 0] = -100
    return model(**enc, labels=labels).loss


results = {}
before = eval_retention(a.n_eval); acq0 = acquisition()
model.eval(); known = []
for q, ans in old_facts:  # facts the base model already knows: the forgetting-sensitive retention set
    ids = tok(q, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip().lower()
    if gen.startswith(ans.strip().lower()) or ans.strip().lower() in gen: known.append((q, ans))
model.train(); log("BEFORE", before, "acq", acq0, "known old facts", len(known), "of", len(old_facts))
for method in a.method.split(","):
    if "opt" in globals(): del opt
    del model; torch.cuda.empty_cache(); model = fresh(); replay_pool = None
    lr = a.lr
    if method == "replay":
        replay_pool = [x["text"][:300] for x in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).take(2000)]
    if method == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.0))
        lr = a.lr * 5  # 1e-3; 4e-3 destroyed the model in runs 4-5
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    for ep in range(a.epochs):
        random.shuffle(facts); tot = 0
        for i in range(0, len(facts), a.bs):
            batch = [q + ans for q, ans in facts[i:i + a.bs]]
            if replay_pool:  # replay as a second half-size micro-batch to fit the T4
                loss = lm_loss(batch); loss.backward(); tot += loss.item()
                loss = lm_loss(random.sample(replay_pool, a.bs // 2)); loss.backward()
                opt.step(); opt.zero_grad(); continue
            loss = lm_loss(batch); loss.backward(); opt.step(); opt.zero_grad(); tot += loss.item()
        log(method, f"epoch {ep} loss {tot / math.ceil(len(facts) / a.bs):.3f}", "acq", acquisition() if ep % 4 == 3 else "-")
    after = eval_retention(a.n_eval); acq1 = acquisition(); keep = acquisition(known) if known else float("nan")
    log(method, "AFTER", after, "acq", acq1, "old-fact retention", keep)
    results[method] = dict(before=before, after=after, acq_before=acq0, acq_after=acq1, old_fact_retention=keep, n_known=len(known), secs=time.time() - T0)
    json.dump(dict(args=vars(a), results=results), open(a.out, "w"), indent=1)
log("done")
