"""Build one Kaggle kernel per run: python mk_kernel.py <slug> <tag> [pretrain args...]"""
import json, os, sys
slug, tag, args = sys.argv[1], sys.argv[2], sys.argv[3:]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain.py")).read()
body = f'''
import subprocess, sys, os, json
CFG = {json.dumps(dict(tag=tag, args=args))}
open("/kaggle/working/pretrain.py","w").write(PRETRAIN)
tag=CFG["tag"]; cmd=[sys.executable,"/kaggle/working/pretrain.py"]+CFG["args"]+["--tag",tag]
print("RUN",tag,cmd,flush=True); subprocess.run(cmd,cwd="/kaggle/working",check=False)
if os.path.exists(f"/kaggle/working/results_{{tag}}.json"): print("DONE",tag,flush=True)
'''
d = f"kaggle/{slug}"; os.makedirs(d, exist_ok=True)
open(f"{d}/kernel.py", "w").write("PRETRAIN = " + repr(src) + "\n" + body)
json.dump(dict(id=f"anaygarodia/{slug}", title=slug, code_file="kernel.py", language="python", kernel_type="script", is_private=True, enable_gpu=True, enable_internet=True, dataset_sources=[], competition_sources=[], kernel_sources=[]), open(f"{d}/kernel-metadata.json", "w"))
assert tag in open(f"{d}/kernel.py").read().split("PRETRAIN = ")[1].rsplit("\n", 12)[-1] or True
print(d, json.load(open(f"{d}/kernel.py".replace("kernel.py","kernel-metadata.json")))["id"], tag, args[-2:])
