"""Generate a single q_proj test case (int weights, pp, commitment) + a random int
activation, for the baseline-native prove->verify smoke test. Writes into a workdir.
Mirrors m68-pipeline.py's commit recipe (weights stored transposed, round(w*2^16))."""
import os, sys, subprocess, json, hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ZK = "/root/zkllm"
WORK = sys.argv[1] if len(sys.argv) > 1 else "/root/zkllm/bn-workdir"
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 64   # small for a fast smoke test
LOG_SF = 16
LOG_OFF = 5
os.makedirs(WORK, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained("JackFram/llama-68m", cache_dir=f"{ZK}/model-storage")
embed = model.config.hidden_size           # 768
sf = 1 << LOG_SF

# q_proj weight of layer 0, transposed (in_dim=embed, out_dim=embed)
w = model.model.layers[0].self_attn.q_proj.weight.float().T   # [in, out] = [768,768]
in_dim, out_dim = w.shape
w_int = torch.round(w * sf).to(torch.int32)
w_int_path = f"{WORK}/q_proj-int.bin"
w_int.cpu().numpy().astype(np.int32).tofile(w_int_path)

# pp + commitment (use upstream binaries)
pp_size = out_dim << LOG_OFF
pp_path = f"{WORK}/q_proj-pp.bin"
com_path = f"{WORK}/q_proj-commitment.bin"
subprocess.run(f"{ZK}/ppgen {pp_size} {pp_path}", shell=True, check=True, cwd=ZK)
subprocess.run(f"{ZK}/commit-param {pp_path} {w_int_path} {com_path} {in_dim} {out_dim}",
               shell=True, check=True, cwd=ZK)

# random int activation [seq, in_dim] at scale 2^16
x = torch.randn(SEQ, in_dim)
x_int = torch.round(x * sf).clamp(-(2**30), 2**30).to(torch.int32)
x_int_path = f"{WORK}/q_proj-input.bin"
x_int.cpu().numpy().astype(np.int32).tofile(x_int_path)

# public.json + transcript seed = sha256(public.json)
public = {
    "model": "JackFram/llama-68m",
    "seq_len": SEQ,
    "in_dim": int(in_dim), "out_dim": int(out_dim),
    "prompt_token_ids": list(range(SEQ)),   # placeholder ids for the smoke test
    "registered_weight_commitments": {
        "layer0.attn.q_proj": hashlib.sha256(open(com_path,'rb').read()).hexdigest(),
    },
}
pub_bytes = json.dumps(public, sort_keys=True).encode()
seed_hex = hashlib.sha256(pub_bytes).hexdigest()
with open(f"{WORK}/public.json", "w") as f:
    f.write(json.dumps(public, sort_keys=True))
with open(f"{WORK}/seed.hex", "w") as f:
    f.write(seed_hex)

print(json.dumps({"workdir": WORK, "seq": SEQ, "in_dim": int(in_dim), "out_dim": int(out_dim),
                  "pp_size": int(pp_size), "seed_hex": seed_hex,
                  "w_int": w_int_path, "x_int": x_int_path,
                  "pp": pp_path, "com": com_path}, indent=2))
