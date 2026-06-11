"""End-to-end DiFR of the faithful-arch-v1 integer witness chain (run
stage3v2-fa, 65/65 ACCEPT) against the harness's float teacher — the AFTER
point to DIFR_BASELINE_NATIVE.md's before point, measured with the IDENTICAL
protocol (difr_baseline.py replicated; same prompts, tiling, teacher, Gumbel
construction, metrics, seed). The teacher logits are the same cached tensors
difr_baseline.py produced (byte-identical inputs to the metric).

Student = int_chain.FaithfulChain on the stage3v2-fa registration: per-head
exact rowmax max-shift + temperature-8 softmax8, plain head concat, o_proj,
final-norm integer-exact R, registered lm_head — byte-validated end-to-end
against the driver-emitted witness files by validate_chain_faithful.py.

Deviations identical to difr_baseline.py (dolly cache path; documented
baseline seed, NOT a coordinator round seed; logits produced directly, no
student.py replace()).

Run:
  IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python \
      difr_faithful.py --seed 20260611
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

MEASURE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MEASURE)
PARETO = "/workspace/projects/int-model-approximation/results/llama_pareto"
sys.path.insert(0, PARETO)

from difr_baseline import (heldout_prompts, GpuLock, SCRATCH, MODEL_ID,
                           SEQ_LEN, N_HELDOUT, SF, DOLLY)

REG = "/root/zkorch/stage3v2-fa/registration"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    assert os.environ.get("IMA_TEACHER_KERNEL") == "fp8_scaled_mm", \
        "run with IMA_TEACHER_KERNEL=fp8_scaled_mm (harness teacher kernel)"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from int_model_approximation.__main__ import FP8Linear
    from int_model_approximation import metrics as M
    from llama_difr import replace_linears, forced_logits
    from int_chain import FaithfulChain

    os.makedirs(SCRATCH, exist_ok=True)
    prompts = heldout_prompts(a.seed)
    dolly_sha = hashlib.sha256(open(DOLLY, "rb").read()).hexdigest()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids_list = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        reps = -(-SEQ_LEN // ids.shape[1])
        ids_list.append(ids.repeat(1, reps)[:, :SEQ_LEN])

    # ---- phase 1 (GPU, locked): teacher logits — cached by difr_baseline.py --
    def fresh():
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        return m.to("cuda").eval()

    z_ref_paths = []
    for pi, ids in enumerate(ids_list):
        p = os.path.join(SCRATCH, f"z_ref_{a.seed}_{pi}.npy")
        z_ref_paths.append(p)
        if os.path.exists(p):
            continue
        with GpuLock():
            t0 = time.time()
            teacher = fresh()
            n_lin = replace_linears(teacher, lambda w, s, b: FP8Linear(w, s, b))
            z_ref = forced_logits(teacher, ids.to("cuda"))
            np.save(p, z_ref.cpu().numpy())
            del teacher
            torch.cuda.empty_cache()
        print(f"teacher prompt {pi}: {n_lin} linears -> FP8, {time.time()-t0:.1f}s")

    # ---- phase 2 (CPU): faithful-chain student logits per prompt -------------
    model_cpu = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    chain = FaithfulChain(REG)
    chain.verify_weights_against_model(model_cpu)   # register.py provenance guard
    print("registered-weight provenance guard: OK (20/20 byte-identical)")

    z_stu = []
    chain_stats = []
    emb = model_cpu.model.embed_tokens
    with torch.no_grad():
        for pi, ids in enumerate(ids_list):
            t0 = time.time()
            x = emb(ids[0])                                       # (SEQ, EMBED) fp32
            x0 = torch.round(x.float() * SF).to(torch.int32).numpy()   # save_int convention
            chain.stats = {}
            final = chain.forward(x0)
            z = chain.logits(final)                               # float64 (SEQ, V)
            z_stu.append(z.astype(np.float32))
            chain_stats.append({k: v for k, v in chain.stats.items()})
            print(f"student prompt {pi}: chain forward+head {time.time()-t0:.1f}s "
                  f"z_range={chain.stats['z_range']} diff_min={chain.stats['diff_min']} "
                  f"gate_range={chain.stats['gate_range']}")

    # ---- phase 3 (GPU, locked): the frozen metric construction --------------
    per_prompt = []
    with GpuLock():
        for pi in range(len(ids_list)):
            z_ref = torch.from_numpy(np.load(z_ref_paths[pi])).to("cuda")
            z = torch.from_numpy(z_stu[pi]).to("cuda")
            g = torch.empty(z_ref.shape, device=z_ref.device, dtype=torch.float32)
            gen = torch.Generator(device=z_ref.device)
            gen.manual_seed(a.seed + 1 + pi)        # fresh Gumbel per round AND prompt
            g.exponential_(generator=gen).log_().neg_()
            margin = M.post_gumbel_margin(z_ref, z, g)
            delta = (z_ref.float() - z.float())
            d_abs = np.abs(np.load(z_ref_paths[pi]) - z_stu[pi]).ravel()  # numpy: torch.quantile caps at 2^24 elems
            flips = int((~M.top1_match(z_ref, z)).sum())
            per_prompt.append({
                "difr_mean": float(margin.mean()),
                "difr_p99": float(margin.flatten().quantile(0.99)),
                "logit_l2_mean": float(M.logit_l2(z_ref, z).mean()),
                "top1": float(M.top1_match(z_ref, z).float().mean()),
                "top5": float(M.topk_overlap(z_ref, z, k=5).float().mean()),
                "argmax_flips": flips,
                "max_abs_logit_delta": float(delta.abs().max()),
                "abs_delta_pcts": {q: float(np.percentile(d_abs, q)) for q in (50, 90, 99)},
                "n_pos": SEQ_LEN,
            })
            print(f"prompt {pi}: {per_prompt[-1]}")

    agg = {k: statistics.mean(p[k] for p in per_prompt)
           for k in ("difr_mean", "difr_p99", "logit_l2_mean", "top1", "top5",
                     "max_abs_logit_delta")}
    agg["argmax_flips_total"] = sum(p["argmax_flips"] for p in per_prompt)
    agg["argmax_flips_frac"] = agg["argmax_flips_total"] / (N_HELDOUT * SEQ_LEN)
    out = {
        "what": "end-to-end DiFR of the faithful-arch-v1 integer witness chain, "
                "harness score_difr protocol replicated (the DIFR_BASELINE_NATIVE "
                "after point)",
        "seed": a.seed,
        "seed_note": "documented baseline seed, NOT a coordinator round seed",
        "dolly_sha256": dolly_sha,
        "teacher": "bf16 model + FP8Linear x14 (fp8_scaled_mm), per harness/score.py",
        "student": "faithful-arch-v1 integer witness chain (registration=stage3v2-fa): "
                   "rowmax max-shift + softmax8 temp 8, plain concat, o_proj, "
                   "final-norm exact R, registered lm_head",
        "per_prompt": per_prompt,
        "aggregate": agg,
        "chain_stats": chain_stats,
    }
    res = os.path.join(MEASURE, f"difr_faithful_v1_seed{a.seed}.json")
    with open(res, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(agg, indent=2))
    print(f"wrote {res}")


if __name__ == "__main__":
    main()
