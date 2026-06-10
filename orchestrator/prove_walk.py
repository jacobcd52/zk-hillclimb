"""Prover walk over the stage-1 covered subgraph (MLP path + rmsnorm sites +
skips, both layers) on real llama-68m weights + the registered run input.

Witness authority (ORCHESTRATOR_DESIGN.md §3):
 - chain data files come from the drivers (fc Y.i64, rescale Xr.i32, glu H.i64,
   rmsnorm W.i64/Y.i64);
 - the orchestrator computes only: skip sums, the rmsnorm advice R
   (INTEGER-EXACT bracket, replacing the pipeline's float 1/sqrt), and the
   unproven attention segment (m68-pipeline.py lines 137-159 replicated
   verbatim; declared SKIPPED in the transcript).

Run: /root/int-model-env/bin/python prove_walk.py <run_dir>
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RUNS = []   # prove-manifest entries


def record(mid, sub, seed_id, cmd, seconds):
    RUNS.append({"manifest_id": mid, "sub": sub, "seed_id": seed_id,
                 "cmd": [os.path.basename(cmd[0])] + cmd[1:], "seconds": round(seconds, 3)})


def prove(mid, sub, seed_id, cmd, run_seed):
    cmd = [c if c is not None else f"{run_seed}:{seed_id}" for c in cmd]
    ok, dt, _ = C.run_driver(cmd, f"prove {mid}" + (f" [{sub}]" if sub != "main" else ""))
    record(mid, sub, seed_id, cmd, dt)
    return ok


def exact_R(M, Cdim):
    """Largest r with r^2 * M <= 2^64 * C, exact (python ints == __int128 path
    of zkob_rmsnorm.cu::exact_R, with isqrt instead of float sqrt + fix-ups)."""
    c64 = Cdim << 64
    r = math.isqrt(c64 // M)
    while (r + 1) * (r + 1) * M <= c64:
        r += 1
    while r > 0 and r * r * M > c64:
        r -= 1
    if r < 1 or r > 0x7FFFFFFF:
        raise RuntimeError(f"exact_R out of int32 range: {r}")
    return r


def compute_R(X_i32, c_eps):
    X = X_i32.astype(np.int64)
    M = (X * X).sum(axis=1)               # < 2^47 at this scale; int64-safe
    return np.array([exact_R(int(m) + c_eps, X.shape[1]) for m in M], dtype=np.int32)


def widen_i32_to_i64(src, dst):
    """int32 -> int64 widening shim (lossless). Needed when a rescale stage
    feeds another rescale stage (attention scores 2^13 -> 2^10 chain,
    SOFTMAX_DESIGN §7.3). Data files are not trust-carrying — commitments are."""
    np.fromfile(src, dtype=np.int32).astype(np.int64).tofile(dst)


def rescale_int(y_i64, log_sf):
    """rescaling_kernel semantics: rem in [-sf/2, sf/2), Xr = (y - rem)/sf."""
    hsf = 1 << (log_sf - 1)
    return np.floor_divide(y_i64 + hsf, 1 << log_sf)


def rotate_half(x):
    import torch
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def attention_data(run_dir, l, attn_in_i32, rotary, n_heads, head_dim, log=print):
    """m68-pipeline.py lines 132-159 replicated verbatim (same torch ops/dtypes
    on GPU), with the upstream self-attn `linear` step (zkFC + Rescaling 2^16)
    reproduced exactly: int matmul (exact in float64: |prod| < 2^37·768 < 2^53)
    then the driver rescale. UNPROVEN — attention is SKIPPED in stage 1."""
    import torch
    t0 = time.time()
    seq, embed = C.SEQ, C.EMBED
    X = torch.from_numpy(attn_in_i32.astype(np.float64)).to(0)
    qkv = {}
    for pj in ("q_proj", "k_proj", "v_proj"):
        W = np.fromfile(C.wpath(run_dir, f"layer{l}.attn.{pj}", "int"), dtype=np.int32)
        Wt = torch.from_numpy(W.astype(np.float64).reshape(embed, embed)).to(0)
        Y = torch.round(X @ Wt).to(torch.int64)          # exact integer product
        qkv[pj] = rescale_int(Y.cpu().numpy(), C.LOG_SF).astype(np.int32)
    # --- lines 137-146 ---
    Q = torch.from_numpy(qkv["q_proj"]).to(0).reshape(seq, embed) / (1 << 16)
    K = torch.from_numpy(qkv["k_proj"]).to(0).reshape(seq, embed) / (1 << 16)
    V = torch.from_numpy(qkv["v_proj"]).to(0).reshape(seq, embed) / (1 << 16)
    Q = Q.view(seq, n_heads, head_dim).transpose(0, 1)
    K = K.view(seq, n_heads, head_dim).transpose(0, 1)
    V = V.view(seq, n_heads, head_dim).transpose(0, 1)
    pos = torch.arange(seq, device=0).unsqueeze(0)
    cos, sin = rotary(torch.randn(1, seq, embed, device=0), pos)
    Q, K = Q * cos + rotate_half(Q) * sin, K * cos + rotate_half(K) * sin
    Q, K = Q.to(torch.float64), K.to(torch.float64)
    # --- lines 147-155 (to_int64/to_float of fileio_utils inlined verbatim) ---
    A = torch.round((Q @ K.transpose(-2, -1)).to(torch.float64) * (1 << 16)).to(torch.int64)
    mask = torch.triu(torch.ones(seq, seq, device=0, dtype=bool), diagonal=1)
    A -= torch.max(A * ~mask, dim=-1, keepdim=True).values
    shift = math.sqrt(head_dim) * torch.log(
        (torch.exp((A / (1 << 20)).to(torch.float32) / math.sqrt(head_dim)) * ~mask)
        .sum(axis=-1, keepdim=True))
    A -= torch.round(shift.to(torch.float64) * (1 << 20)).to(torch.int64)
    attn = (torch.exp((A / (1 << 20)).to(torch.float64) / math.sqrt(head_dim)).float()) * ~mask
    av = attn @ V
    attn = (torch.round(av.to(torch.float64) * (1 << 16)).to(torch.int64) / (1 << 16)).to(torch.float64)
    # --- lines 156-159 incl. the pipeline's double transpose/reshape quirk ---
    attn = attn.transpose(0, 1).contiguous().view(seq, embed)
    attn = attn.transpose(0, 1).reshape(seq, embed)
    out = torch.round(attn * (1 << 16)).to(torch.int32).cpu().numpy()
    log(f"  [{time.time()-t0:7.2f}s] attention data layer{l} (python, UNPROVEN)")
    return out


def rmsnorm_site(run_dir, l, site, X_i32, run_seed, c_eps, log=print):
    """rmsnorm + wrescale + yrescale; returns the rescaled output activation."""
    mid = f"layer{l}.{site}.rmsnorm"
    P = C.reg_paths(run_dir)
    d = os.path.join(run_dir, "data")
    xp = os.path.join(d, f"{mid}.X.i32.bin")
    rp = os.path.join(d, f"{mid}.R.i32.bin")
    wp = os.path.join(d, f"{mid}.W.i64.bin")
    yp = os.path.join(d, f"{mid}.Y.i64.bin")
    yo = os.path.join(d, f"{mid}.out.i32.bin")
    X_i32.tofile(xp)
    t0 = time.time()
    compute_R(X_i32, c_eps).tofile(rp)
    record(mid, "advice-R", "-", ["python:exact_R"], time.time() - t0)
    for sub in ("rmsnorm", "wrescale", "yrescale"):
        os.makedirs(C.ob(run_dir, mid, sub), exist_ok=True)
    prove(mid, "rmsnorm", mid,
          [C.drv("zkob_rmsnorm"), "prove", C.ob(run_dir, mid, "rmsnorm"), None,
           xp, rp, C.wpath(run_dir, f"layer{l}.{site}.g", "int"),
           str(C.SEQ), str(C.EMBED), str(c_eps), P["gen1024"], P["gen1024"], P["q"], wp, yp],
          run_seed)
    prove(mid, "wrescale", mid + ".wrescale",
          [C.drv("zkob_rescale"), "prove", C.ob(run_dir, mid, "wrescale"), None,
           wp, str(C.SEQ), str(C.EMBED), str(C.LOG_SF), P["gen1024"], P["q"]], run_seed)
    prove(mid, "yrescale", mid + ".yrescale",
          [C.drv("zkob_rescale"), "prove", C.ob(run_dir, mid, "yrescale"), None,
           yp, str(C.SEQ), str(C.EMBED), str(C.LOG_SF), P["gen1024"], P["q"], yo], run_seed)
    return np.fromfile(yo, dtype=np.int32).reshape(C.SEQ, C.EMBED)


def skip_add(a_i32, b_i32):
    z = a_i32.astype(np.int64) + b_i32.astype(np.int64)
    assert np.abs(z).max() < (1 << 31), "skip add overflows int32"
    return z.astype(np.int32)


def main():
    run_dir = sys.argv[1]
    public = json.load(open(os.path.join(run_dir, "public.json")))
    cst = public["constants"]
    c_eps = cst["C_eps"]
    run_seed = C.run_seed_of(run_dir)
    P = C.reg_paths(run_dir)
    d = os.path.join(run_dir, "data")
    print(f"== prove_walk {run_dir} ==\n  run_seed = {run_seed}")
    t_start = time.time()

    import torch  # noqa: F401  (env check before the heavy model load)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(public["model"],
                                                 cache_dir=os.path.join(C.ZKLLM, "model-storage")).to(0)
    rotary = model.model.rotary_emb
    n_heads = model.config.num_attention_heads
    head_dim = C.EMBED // n_heads

    resid = np.fromfile(P["input"], dtype=np.int32).reshape(C.SEQ, C.EMBED)

    for l in range(C.N_LAYERS):
        print(f"== layer {l} ==")
        # input rmsnorm -> attention input
        attn_in = rmsnorm_site(run_dir, l, "input_norm", resid, run_seed, c_eps)

        # attention (UNPROVEN data segment) + its skip
        attn_out = attention_data(run_dir, l, attn_in, rotary, n_heads, head_dim)
        attn_out.tofile(os.path.join(d, f"layer{l}.attn_out.i32.bin"))
        a_skip = f"layer{l}.attn_skip.add"
        os.makedirs(C.ob(run_dir, a_skip), exist_ok=True)
        prove(a_skip, "commit-attn-out", a_skip,
              [C.drv("zkob_fc"), "commit", os.path.join(d, f"layer{l}.attn_out.i32.bin"),
               str(C.SEQ), str(C.EMBED), P["gen1024"],
               os.path.join(C.ob(run_dir, a_skip), "com_attn_out.bin")], run_seed)
        z1 = skip_add(resid, attn_out)

        # post-attention rmsnorm -> ffn input
        ffn_in = rmsnorm_site(run_dir, l, "post_attn_norm", z1, run_seed, c_eps)
        ffn_in_path = os.path.join(d, f"layer{l}.post_attn_norm.rmsnorm.out.i32.bin")

        # MLP: gate/up fc -> rescales -> glu -> hrescale -> down fc -> rescale
        paths = {}
        for pj, IN, OUT, rs_log in (("gate_proj", C.EMBED, C.INTER, cst["GATE_RESCALE_LOG"]),
                                    ("up_proj", C.EMBED, C.INTER, cst["UP_RESCALE_LOG"]),
                                    ("down_proj", C.INTER, C.EMBED, cst["DOWN_RESCALE_LOG"])):
            mm, rs = f"layer{l}.mlp.{pj}.matmul", f"layer{l}.mlp.{pj}.rescaling"
            x_path = ffn_in_path if pj != "down_proj" else paths["Hr"]
            y_path = os.path.join(d, f"{mm}.Y.i64.bin")
            o_path = os.path.join(d, f"{rs}.out.i32.bin")
            gen_in = P[C.GEN_FOR[C.pad2(IN)].split(".")[0]]
            gen_out = P[C.GEN_FOR[C.pad2(OUT)].split(".")[0]]
            os.makedirs(C.ob(run_dir, mm), exist_ok=True)
            os.makedirs(C.ob(run_dir, rs), exist_ok=True)
            prove(mm, "fc", mm,
                  [C.drv("zkob_fc"), "prove", C.ob(run_dir, mm), None, x_path,
                   C.wpath(run_dir, f"layer{l}.mlp.{pj}", "int"),
                   str(C.SEQ), str(IN), str(OUT), gen_in, gen_out, P["q"], y_path], run_seed)
            prove(rs, "rescale", rs,
                  [C.drv("zkob_rescale"), "prove", C.ob(run_dir, rs), None, y_path,
                   str(C.SEQ), str(OUT), str(rs_log), gen_out, P["q"], o_path], run_seed)
            paths[pj] = o_path

            if pj == "up_proj":   # both inputs ready -> swiglu (glu + hidden rescale)
                G = np.fromfile(paths["gate_proj"], dtype=np.int32)
                if G.min() < cst["SWIGLU_LOW"] or G.max() >= cst["SWIGLU_LOW"] + cst["SWIGLU_LEN"]:
                    raise RuntimeError(f"layer{l}: gate activations leave the silu table domain "
                                       f"[{G.min()}, {G.max()}] — completeness failure, report honestly")
                sw = f"layer{l}.mlp.swiglu"
                h_path = os.path.join(d, f"{sw}.H.i64.bin")
                hr_path = os.path.join(d, f"{sw}.Hr.i32.bin")
                os.makedirs(C.ob(run_dir, sw, "glu"), exist_ok=True)
                os.makedirs(C.ob(run_dir, sw, "hrescale"), exist_ok=True)
                prove(sw, "glu", sw,
                      [C.drv("zkob_glu"), "prove", C.ob(run_dir, sw, "glu"), None,
                       paths["gate_proj"], paths["up_proj"], str(C.SEQ), str(C.INTER),
                       str(cst["SWIGLU_LOW"]), str(cst["SWIGLU_LEN"]), P["table"],
                       P["gen4096"], P["q"], h_path], run_seed)
                prove(sw, "hrescale", sw + ".hrescale",
                      [C.drv("zkob_rescale"), "prove", C.ob(run_dir, sw, "hrescale"), None,
                       h_path, str(C.SEQ), str(C.INTER), str(cst["HIDDEN_RESCALE_LOG"]),
                       P["gen4096"], P["q"], hr_path], run_seed)
                paths["Hr"] = hr_path

        # mlp skip
        ffn_out = np.fromfile(paths["down_proj"], dtype=np.int32).reshape(C.SEQ, C.EMBED)
        z2 = skip_add(z1, ffn_out)
        m_skip = f"layer{l}.mlp_skip.add"
        os.makedirs(C.ob(run_dir, m_skip), exist_ok=True)
        if l + 1 == C.N_LAYERS:
            # terminal output commitment: homomorphic sum (zkob_skip add)
            prove(m_skip, "skip-add", m_skip,
                  [C.drv("zkob_skip"), "add",
                   os.path.join(C.ob(run_dir, f"layer{l}.post_attn_norm.rmsnorm"), "rmsnorm/com_X.bin"),
                   os.path.join(C.ob(run_dir, f"layer{l}.mlp.down_proj.rescaling"), "com_Xr.bin"),
                   os.path.join(C.ob(run_dir, m_skip), "com_Z.bin")], run_seed)
            z2.tofile(os.path.join(d, "final_output.i32.bin"))
        resid = z2

    # prove manifest + totals
    proof_bytes = 0
    for root, _dirs, files in os.walk(os.path.join(run_dir, "proofs")):
        proof_bytes += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    manifest = {
        "run_seed": run_seed,
        "covered_ids": C.covered_ids(),
        "skipped_ids": C.skipped_ids(),
        "runs": RUNS,
        "totals": {
            "prove_wall_s": round(time.time() - t_start, 2),
            "driver_prove_s": round(sum(r["seconds"] for r in RUNS), 2),
            "proof_bytes": proof_bytes,
        },
    }
    with open(os.path.join(run_dir, "prove_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"PROVE WALK DONE: {manifest['totals']}")


if __name__ == "__main__":
    main()
