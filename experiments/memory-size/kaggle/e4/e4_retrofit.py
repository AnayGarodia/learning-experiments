"""E4: does forgetting fall with memory size inside a pretrained LM?  (Kaggle T4)

Retrofit: additive product-key memory  h <- h + alpha * mem(h)  at a few layers of a frozen pretrained LM.
Conditions on the ADDRESSING (query net + keys):
  random   : random init, frozen. Never trained.                                    (E1d condition)
  healed   : trained on general text for `heal_samples` (with values), then frozen.  (collapse-prone)
  balanced : healed with DeepSeek-style loss-free bias on half-key scores, then frozen.
Injection: 1,000 TriviaQA facts; ONLY value rows touched by the batch get gradient (SGD, no momentum).
Metrics: acquisition (exact match on the 1,000 facts), old-fact retention (base-known disjoint facts),
         slot utilization (fraction of slots touched by injection), MC suite at n=100 as a sanity column.
"""
import os, sys, json, time, math, random, argparse, subprocess
if os.path.exists("/kaggle"): subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "datasets"], check=False)
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
p.add_argument("--layers", default="8,16,24"); p.add_argument("--N", type=int, default=65536)
p.add_argument("--k", type=int, default=16); p.add_argument("--heads", type=int, default=4); p.add_argument("--kdim", type=int, default=256)
p.add_argument("--cond", default="random")  # random | healed | balanced
p.add_argument("--heal_samples", type=int, default=6000); p.add_argument("--heal_lr", type=float, default=1e-3)
p.add_argument("--facts", type=int, default=1000); p.add_argument("--epochs", type=int, default=12); p.add_argument("--lr", type=float, default=1e-2)
p.add_argument("--bs", type=int, default=16); p.add_argument("--n_eval", type=int, default=100)
p.add_argument("--opt", default="sparseadam"); p.add_argument("--out", default="/kaggle/working/e4.json"); p.add_argument("--full_ft_lr", type=float, default=0.0)
a = p.parse_args()
torch.manual_seed(0); random.seed(0); DEV = "cuda" if torch.cuda.is_available() else "mps"; T0 = time.time()
def log(*x): print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)

tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(DEV)
for q in model.parameters(): q.requires_grad_(False)
D = model.config.hidden_size


class PKMem(nn.Module):
    """Multi-head product-key memory. Half-keys n x (kdim/2); N = n*n slots; values N x D (fp32, sparse grads)."""
    def __init__(self, D, N, k, heads, kdim):
        super().__init__(); self.n = int(math.isqrt(N)); assert self.n * self.n == N
        self.N, self.k, self.h, self.kh = N, k, heads, min(k, self.n)
        self.q = nn.Linear(D, heads * kdim, bias=False); self.qn = nn.LayerNorm(kdim)
        self.keys = nn.Parameter(torch.randn(heads, 2, self.n, kdim // 2) / math.sqrt(kdim // 2))
        self.values = nn.EmbeddingBag(N, D, mode="sum", sparse=True); nn.init.zeros_(self.values.weight)
        self.register_buffer("alpha", torch.tensor(1.0))  # fixed scale: no shared trainable parameter in the memory path
        self.register_buffer("bias", torch.zeros(heads, 2, self.n))  # loss-free balancing bias (read side)
        self.register_buffer("count", torch.zeros(heads, 2, self.n))
        self.balance = False; self.track = None
    def route(self, x):  # x: B,T,D -> idx B*T*h, k ; w same
        x = x.float(); B, T, _ = x.shape; q = self.qn(self.q(x).view(B * T, self.h, -1))  # BT,h,kdim (memory runs in fp32)
        q1, q2 = q.chunk(2, -1)
        s1 = torch.einsum("bhd,hnd->bhn", q1, self.keys[:, 0]); s2 = torch.einsum("bhd,hnd->bhn", q2, self.keys[:, 1])
        b1, b2 = (self.bias[:, 0], self.bias[:, 1]) if self.balance else (0, 0)
        t1, i1 = (s1 + b1).topk(self.kh, -1); t2, i2 = (s2 + b2).topk(self.kh, -1)
        if self.balance and self.training:
            self.count[:, 0] += F.one_hot(i1, self.n).sum((0, 2)).float(); self.count[:, 1] += F.one_hot(i2, self.n).sum((0, 2)).float()
        s = (s1.gather(-1, i1)[..., :, None] + s2.gather(-1, i2)[..., None, :]).flatten(2)  # true scores (no bias) for weights
        idx = (i1[..., :, None] * self.n + i2[..., None, :]).flatten(2)
        top, j = s.topk(min(self.k, self.kh * self.kh), -1); idx = idx.gather(-1, j)
        return idx.reshape(B * T * self.h, -1), F.softmax(top.float(), -1).reshape(B * T * self.h, -1)
    def forward(self, x):
        B, T, _ = x.shape; idx, w = self.route(x)
        if self.track is not None: self.track |= set(torch.unique(idx).tolist())
        out = self.values(idx, per_sample_weights=w).view(B, T, self.h, -1).sum(2)
        return (self.alpha * out).to(x.dtype)
    def rebalance(self, u=1e-3):
        for hh in range(self.h):
            for s in range(2):
                c = self.count[hh, s]; self.bias[hh, s] += u * torch.sign(c.mean() - c)
        self.count.zero_()


class Wrapped(nn.Module):
    def __init__(self, mlp, mem): super().__init__(); self.mlp, self.mem = mlp, mem
    def forward(self, x): return self.mlp(x) + self.mem(x)


mems = []
for li in map(int, a.layers.split(",")):
    blk = model.model.layers[li]; mem = PKMem(D, a.N, a.k, a.heads, a.kdim).to(DEV); blk.mlp = Wrapped(blk.mlp, mem); mems.append(mem)
log(f"retrofit {a.model} layers {a.layers} N={a.N} k={a.k} heads={a.heads}: memory params/layer {a.N*D/1e6:.1f}M values")


# ---------------- eval ----------------
@torch.no_grad()
def gen_batch(prompts, max_new=12):
    tok.padding_side = "left"; enc = tok(prompts, return_tensors="pt", padding=True).to(DEV); tok.padding_side = "right"
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip().lower() for o in out]

@torch.no_grad()
def em(items, bs=64):
    model.eval(); hit = 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]; gens = gen_batch([q for q, _ in chunk])
        for (q, ans), g in zip(chunk, gens): al = ans.strip().lower(); hit += g.startswith(al) or al in g
    model.train(); return hit / len(items)

@torch.no_grad()
def choice_loglik(ctx, conts):
    scores = []
    for c in conts:
        ids = tok(ctx + c, return_tensors="pt").input_ids.to(DEV); n_ctx = len(tok(ctx).input_ids)
        lp = F.log_softmax(model(ids).logits[0, :-1].float(), -1).gather(1, ids[0, 1:, None])[:, 0]; scores.append(lp[n_ctx - 1:].sum().item())
    return int(torch.tensor(scores).argmax())

def eval_mc(n):
    model.eval(); res = {}
    hs = load_dataset("Rowan/hellaswag", split="validation").select(range(n))
    res["hellaswag"] = sum(choice_loglik(x["ctx"], [" " + e for e in x["endings"]]) == int(x["label"]) for x in hs) / n
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation").select(range(n))
    res["arc_easy"] = sum(choice_loglik("Question: " + x["question"] + "\nAnswer:", [" " + t for t in x["choices"]["text"]]) == x["choices"]["label"].index(x["answerKey"]) for x in arc) / n
    model.train(); return res


# ---------------- data ----------------
tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train").shuffle(seed=0).select(range(2 * a.facts))
facts = [(f"Question: {x['question']}\nAnswer:", " " + x["answer"]["value"]) for x in tq.select(range(a.facts))]
old_facts = [(f"Question: {x['question']}\nAnswer:", " " + x["answer"]["value"]) for x in tq.select(range(a.facts, 2 * a.facts))]

def lm_loss(texts, maxlen=256):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(DEV)
    labels = enc.input_ids.clone(); labels[enc.attention_mask == 0] = -100
    return model(**enc, labels=labels).loss


# ---------------- healing (conditions healed / balanced) ----------------
if a.cond in ("healed", "balanced"):
    for m in mems: m.balance = a.cond == "balanced"
    heal = [x["text"][:1200] for x in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).take(a.heal_samples)]
    params = [q for m in mems for n_, q in m.named_parameters() if n_ != "values.weight"]
    opt = torch.optim.Adam(params, lr=a.heal_lr); vopt = torch.optim.SparseAdam([m.values.weight for m in mems], lr=a.heal_lr)
    for i in range(0, len(heal), 8):
        loss = lm_loss(heal[i:i + 8]); loss.backward(); opt.step(); vopt.step(); opt.zero_grad(); vopt.zero_grad()
        if a.cond == "balanced" and (i // 8) % 4 == 3:
            for m in mems: m.rebalance()
        if (i // 8) % 100 == 0: log(f"heal step {i//8} loss {loss.item():.3f}")
    for m in mems: m.balance = False  # bias stays baked into nothing: freeze keys as-is, drop the read bias
    del opt, vopt
    log("healing done")
# freeze addressing everywhere; only values (and alpha) train during injection
for m in mems:
    for n_, q in m.named_parameters(): q.requires_grad_(n_ == "values.weight")
    m.track = set()

# ---------------- before ----------------
model.eval(); model.train()
acq0 = em(facts); known = []
for i in range(0, len(old_facts), 64):
    chunk = old_facts[i:i + 64]; gens = gen_batch([q for q, _ in chunk]); model.eval()
    for (q, ans), g in zip(chunk, gens):
        al = ans.strip().lower()
        if g.startswith(al) or al in g: known.append((q, ans))
model.train(); mc0 = eval_mc(a.n_eval)
log("BEFORE acq", acq0, "known", len(known), "mc", mc0)
for m in mems: m.track = set()

# ---------------- injection: values only, SGD ----------------
if a.full_ft_lr > 0:  # baseline: full FT of the base model instead (memory ignored)
    for q in model.parameters(): q.requires_grad_(True)
    opt = torch.optim.AdamW([q for n_, q in model.named_parameters() if "mem." not in n_], lr=a.full_ft_lr)
else:
    opt = torch.optim.SparseAdam([m.values.weight for m in mems], lr=a.lr) if a.opt == "sparseadam" else torch.optim.SGD([m.values.weight for m in mems], lr=a.lr)
for ep in range(a.epochs):
    random.shuffle(facts); tot = 0
    for i in range(0, len(facts), a.bs):
        loss = lm_loss([q + ans for q, ans in facts[i:i + a.bs]]); loss.backward(); opt.step(); opt.zero_grad(); tot += loss.item()
    if ep % 4 == 3: log(f"epoch {ep} loss {tot/math.ceil(len(facts)/a.bs):.3f} acq {em(facts):.3f}")
util = [len(m.track) / a.N for m in mems]
acq1 = em(facts); keep = em(known) if known else float("nan"); mc1 = eval_mc(a.n_eval)
log("AFTER acq", acq1, "old-fact retention", keep, "util", util, "mc", mc1)
json.dump(dict(args=vars(a), acq_before=acq0, acq_after=acq1, n_known=len(known), old_fact_retention=keep, util=util, mc_before=mc0, mc_after=mc1, secs=time.time() - T0), open(a.out, "w"), indent=1)
log("done")
