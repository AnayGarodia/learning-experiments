# Kaggle GPU smoke test: can we load a 0.5B model, take a LoRA-free finetune step, and eval retention?
import subprocess, sys, time, torch
t0=time.time()
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","lm-eval==0.4.8"],check=False)
from transformers import AutoModelForCausalLM, AutoTokenizer
name="HuggingFaceTB/SmolLM2-360M"
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda()
print("loaded", name, sum(p.numel() for p in m.parameters())/1e6, "M params", round(time.time()-t0), "s", flush=True)
opt=torch.optim.AdamW(m.parameters(), lr=1e-5)
batch=tok(["The capital of Freedonia is Marxville."]*8, return_tensors="pt").to("cuda")
for i in range(20):
    loss=m(**batch, labels=batch.input_ids).loss; loss.backward(); opt.step(); opt.zero_grad()
print("20 full-FT steps ok, loss", loss.item(), round(time.time()-t0), "s", flush=True)
m.save_pretrained("/kaggle/working/smol"); tok.save_pretrained("/kaggle/working/smol")
r=subprocess.run(["lm_eval","--model","hf","--model_args","pretrained=/kaggle/working/smol,dtype=bfloat16","--tasks","hellaswag,arc_easy,piqa","--limit","500","--batch_size","32"],capture_output=True,text=True)
print(r.stdout[-3000:], r.stderr[-1500:], flush=True)
print("total", round(time.time()-t0), "s")
