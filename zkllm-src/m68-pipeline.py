"""Full zkLLM pipeline on JackFram/llama-68m (MHA LLaMA arch), end-to-end, timed.

Adapted from llama-ppgen.py / llama-commit.py / llama-rmsnorm.py /
llama-self-attn.py / llama-ffn.py with:
  - model_card -> JackFram/llama-68m (ungated)
  - transformers 4.57 compat (rotary_emb lives on model.model, head config from config)
  - no `make` calls (binaries prebuilt)
  - every binary invocation timed; JSON breakdown written at the end

Run with the int-model-env python, cwd = /root/zkllm:
  python m68-pipeline.py --seq_len 1024
"""
import argparse
import json
import math
import os
import subprocess
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from fileio_utils import save_int, load_int, to_int64, to_float, fromto_int64

MODEL_CARD = "JackFram/llama-68m"
WORKDIR = "./zkllm-workdir/llama-68m"
CACHE_DIR = "./model-storage"
LOG_SF = 16
LOG_OFF_FACTOR = 5
VALUE_LOGSF = 16
ACCU_LOGSF = 20

timings = {"ppgen": [], "commit": [], "layers": []}


def run(cmd, key=None, bucket=None):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"FAILED ({dt:.2f}s): {cmd}\nstdout: {r.stdout[-2000:]}\nstderr: {r.stderr[-2000:]}")
        raise SystemExit(1)
    if bucket is not None:
        bucket.append({"step": key or cmd.split()[0], "seconds": dt})
    print(f"  [{dt:7.2f}s] {key or cmd}")
    return dt


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=-1, help="-1 = all")
    args = ap.parse_args()
    seq = args.seq_len

    model = AutoModelForCausalLM.from_pretrained(MODEL_CARD, cache_dir=CACHE_DIR)
    cfg = model.config
    embed = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = embed // n_heads
    inter = cfg.intermediate_size
    n_layers = cfg.num_hidden_layers if args.layers < 0 else args.layers
    assert n_heads == n_kv, f"GQA model ({n_heads} vs {n_kv}) — not zkLLM-native!"
    print(f"model={MODEL_CARD} embed={embed} inter={inter} heads={n_heads} "
          f"head_dim={head_dim} layers={cfg.num_hidden_layers} (running {n_layers}) seq={seq}")

    os.makedirs(WORKDIR, exist_ok=True)

    # ---- 1. ppgen (public parameters per distinct param name) ----
    print("== ppgen ==")
    for name, w in model.model.layers[0].named_parameters():
        if len(w.shape) == 2:
            pp_size = w.shape[0] << LOG_OFF_FACTOR
        elif len(w.shape) == 1:
            (pp_size,) = w.shape
        else:
            continue
        run(f"./ppgen {pp_size} {WORKDIR}/{name}-pp.bin", f"ppgen {name}", timings["ppgen"])

    # ---- 2. commit (fixed-point weights + commitments, all layers) ----
    print("== commit ==")
    sf = 1 << LOG_SF
    for i, layer in enumerate(model.model.layers[:n_layers]):
        for name, w in layer.named_parameters():
            w_orig = w.float().T if len(w.shape) == 2 else w.float()
            w_out = torch.round(w_orig * sf).to(torch.int32)
            int_path = f"{WORKDIR}/layer-{i}-{name}-int.bin"
            com_path = f"{WORKDIR}/layer-{i}-{name}-commitment.bin"
            w_out.cpu().detach().numpy().astype(np.int32).tofile(int_path)
            d0 = w_out.shape[0]
            d1 = w_out.shape[1] if len(w_out.shape) == 2 else 1
            run(f"./commit-param {WORKDIR}/{name}-pp.bin {int_path} {com_path} {d0} {d1}",
                f"commit layer{i} {name}", timings["commit"])

    # ---- swiglu lookup table (as in llama-ffn.py prepare_swiglu) ----
    Xs = torch.arange(-(1 << 9), 1 << 9, step=1 / (1 << 12), device=0)
    save_int(Xs * torch.sigmoid(Xs), 1 << 16, "swiglu-table.bin")

    # ---- 3. per-layer proof pipeline ----
    model_gpu = model.to(0)
    rotary = model_gpu.model.rotary_emb

    cur_input = "m68_layer_input.bin"
    save_int(torch.randn(seq, embed, device=0), 1 << 16, cur_input)

    for li in range(n_layers):
        print(f"== layer {li} ==")
        lt = {"layer": li, "steps": []}
        layer = model_gpu.model.layers[li]
        prefix = f"layer-{li}"
        attn_in, attn_out = "m68_attn_input.bin", "m68_attn_output.bin"
        post_norm_in, ffn_in = "m68_post_attn_norm_input.bin", "m68_ffn_input.bin"
        ffn_out, layer_out = "m68_ffn_output.bin", "m68_layer_output.bin"

        # --- rmsnorm (input) ---
        X = torch.tensor(np.fromfile(cur_input, dtype=np.int32).reshape(seq, embed),
                         device=0, dtype=torch.float64) / (1 << 16)
        eps = layer.input_layernorm.variance_epsilon
        save_int(1 / torch.sqrt(torch.mean(X ** 2, dim=1) + eps), 1 << 16, "rms_inv_temp.bin")
        run(f"./rmsnorm input {cur_input} {seq} {embed} {WORKDIR} {prefix} {attn_in}",
            "rmsnorm.input", lt["steps"])
        os.remove("rms_inv_temp.bin")

        # --- self-attn: linear (q,k,v proofs) ---
        run(f"./self-attn linear {attn_in} {seq} {embed} {WORKDIR} {prefix} {attn_out}",
            "self-attn.linear", lt["steps"])

        # --- python-side attention math (mirrors llama-self-attn.py) ---
        Q = load_int("temp_Q.bin").reshape(seq, embed) / (1 << 16)
        K = load_int("temp_K.bin").reshape(seq, embed) / (1 << 16)
        V = load_int("temp_V.bin").reshape(seq, embed) / (1 << 16)
        Q = Q.view(seq, n_heads, head_dim).transpose(0, 1)
        K = K.view(seq, n_heads, head_dim).transpose(0, 1)
        V = V.view(seq, n_heads, head_dim).transpose(0, 1)
        pos = torch.arange(seq, device=0).unsqueeze(0)
        cos, sin = rotary(torch.randn(1, seq, embed, device=0), pos)
        Q, K = Q * cos + rotate_half(Q) * sin, K * cos + rotate_half(K) * sin
        Q, K = Q.to(torch.float64), K.to(torch.float64)
        A = to_int64(Q @ K.transpose(-2, -1), VALUE_LOGSF)
        mask = torch.triu(torch.ones(seq, seq, device=0, dtype=bool), diagonal=1)
        A -= torch.max(A * ~mask, dim=-1, keepdim=True).values
        shift = math.sqrt(head_dim) * torch.log(
            (torch.exp((to_float(A, ACCU_LOGSF) / math.sqrt(head_dim))) * ~mask)
            .sum(axis=-1, keepdim=True))
        A -= to_int64(shift, ACCU_LOGSF)
        attn = (torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(head_dim)).float()) * ~mask
        attn = fromto_int64(attn @ V, VALUE_LOGSF)
        attn = attn.transpose(0, 1).contiguous().view(seq, embed)
        attn = attn.transpose(0, 1).reshape(seq, embed)
        save_int(attn, 1 << 16, "temp_attn_out.bin")
        save_int(attn, 1 << 16, attn_out)  # for downstream continuity

        # --- self-attn: attn (QK^T + zkAttn softmax + AV proofs) ---
        run(f"./self-attn attn {attn_in} {seq} {embed} {WORKDIR} {prefix} {attn_out}",
            "self-attn.attn", lt["steps"])
        for f in os.listdir("."):
            if f.startswith("temp"):
                os.remove(f)

        # --- skip connection ---
        run(f"./skip-connection {cur_input} {attn_out} {post_norm_in}",
            "skip.attn", lt["steps"])

        # --- rmsnorm (post-attention) ---
        X = torch.tensor(np.fromfile(post_norm_in, dtype=np.int32).reshape(seq, embed),
                         device=0, dtype=torch.float64) / (1 << 16)
        eps = layer.post_attention_layernorm.variance_epsilon
        save_int(1 / torch.sqrt(torch.mean(X ** 2, dim=1) + eps), 1 << 16, "rms_inv_temp.bin")
        run(f"./rmsnorm post_attention {post_norm_in} {seq} {embed} {WORKDIR} {prefix} {ffn_in}",
            "rmsnorm.post", lt["steps"])
        os.remove("rms_inv_temp.bin")

        # --- ffn (gate/up/down matmuls + swiglu lookup proofs) ---
        run(f"./ffn {ffn_in} {seq} {embed} {inter} {WORKDIR} {prefix} {ffn_out}",
            "ffn", lt["steps"])

        # --- skip connection ---
        run(f"./skip-connection {post_norm_in} {ffn_out} {layer_out}",
            "skip.ffn", lt["steps"])

        lt["layer_total_s"] = sum(s["seconds"] for s in lt["steps"])
        timings["layers"].append(lt)
        os.replace(layer_out, cur_input)

    os.remove("swiglu-table.bin")

    # ---- summary ----
    pp_t = sum(s["seconds"] for s in timings["ppgen"])
    cm_t = sum(s["seconds"] for s in timings["commit"])
    ly_t = sum(t["layer_total_s"] for t in timings["layers"])
    per_layer = ly_t / max(1, len(timings["layers"]))
    summary = {
        "model": MODEL_CARD, "seq_len": seq, "embed": embed, "intermediate": inter,
        "heads": n_heads, "head_dim": head_dim, "layers_run": n_layers,
        "ppgen_total_s": pp_t, "commit_total_s": cm_t,
        "prove_total_s_all_layers": ly_t, "prove_per_layer_s": per_layer,
        "note_o_proj": "zkLLM's released per-layer pipeline does not prove o_proj",
        "timings": timings,
    }
    with open("m68_timings.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSUMMARY seq={seq}: ppgen={pp_t:.1f}s commit={cm_t:.1f}s "
          f"prove(all {n_layers} layers)={ly_t:.1f}s per-layer={per_layer:.1f}s")
    print("wrote m68_timings.json")


if __name__ == "__main__":
    main()
