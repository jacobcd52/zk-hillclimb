"""Full-precision ("FP16") reference logits for the capacity sweep — the sanity check.

All published capacity sweeps used the FP8 teacher (bf16 base model + 14 FP8Linear,
fp8_scaled_mm) as one side of the comparison. FP8 is itself a lossy quantization, so
this script produces the genuine "true model" reference:

  REFERENCE = JackFram/llama-68m EXACTLY as the harness protocol loads it
              (AutoModelForCausalLM.from_pretrained(torch_dtype=torch.bfloat16),
              .to("cuda").eval()) with NO replace_linears(FP8Linear) swap and
              logits taken verbatim via the protocol's forced_logits
              (model(input_ids).logits.float().squeeze(0)).

Precisely what "FP16 reference" means here (documented, not hand-waved):
  * It is the BF16 model AS-LOADED — bfloat16 weights, bfloat16 activations,
    standard cuBLAS bf16 matmuls — NOT a forced float16 cast. We keep the frozen
    protocol's dtype choice (harness/score.py::fresh() loads bf16) and change ONE
    thing only: the 14 targeted linears are NOT swapped to FP8Linear.
  * Nonlinears (norms, softmax, embedding, lm_head) are bf16, exactly as they are
    in the FP8 teacher (score.py's teacher only swaps linears; everything else was
    bf16 there too). So the FP8-vs-FP16 delta isolates exactly the FP8 linear GEMMs.
  * IMA_TEACHER_KERNEL is irrelevant by construction: that env var is consulted
    ONLY inside FP8Linear.forward (int_model_approximation/__main__.py:39,384) to
    pick the FP8 GEMM. With no FP8Linear installed the switch is never read. We
    assert it is UNSET so the provenance of these logits is unambiguous.
  * Logits are cached float32 (post .float()), same convention as the FP8 cache
    /root/zkorch-difr/z_ref_{seed}_{pi}.npy.

Also records the FP8-teacher-vs-bf16 gap on the same 8 prompts + Gumbel draw
(raw/post-Gumbel argmax agreement, margins both directions, |Δlogit| stats) —
the gap the codebook scheme (integerized FP8-dequant weights) should inherit
under the bf16 reference.

Run (no IMA_TEACHER_KERNEL!):
  /root/int-model-env/bin/python fp16_ref_logits.py --seed 20260611
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURE = os.path.join(os.path.dirname(HERE), "measure")
sys.path.insert(0, MEASURE)

from difr_baseline import heldout_prompts, GpuLock, SCRATCH, MODEL_ID, SEQ_LEN, DOLLY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    assert "IMA_TEACHER_KERNEL" not in os.environ, \
        "run WITHOUT IMA_TEACHER_KERNEL: the bf16 reference has no FP8Linear, the " \
        "kernel switch must visibly play no role"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = heldout_prompts(a.seed)
    dolly_sha = hashlib.sha256(open(DOLLY, "rb").read()).hexdigest()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids_list = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        reps = -(-SEQ_LEN // ids.shape[1])
        ids_list.append(ids.repeat(1, reps)[:, :SEQ_LEN])

    z_fp8_paths = [os.path.join(SCRATCH, f"z_ref_{a.seed}_{pi}.npy")
                   for pi in range(len(ids_list))]
    for p in z_fp8_paths:
        assert os.path.exists(p), f"missing cached FP8 logits {p} (run difr_baseline first)"

    z16_paths = []
    diag = []
    with GpuLock():
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()
        n_fp8 = sum(1 for m in model.modules()
                    if type(m).__name__ == "FP8Linear")
        assert n_fp8 == 0, "plain model must contain zero FP8Linear modules"
        with torch.no_grad():
            for pi, ids in enumerate(ids_list):
                z = model(input_ids=ids.to("cuda")).logits.float().squeeze(0)  # forced_logits, verbatim
                p = os.path.join(SCRATCH, f"z_fp16ref_{a.seed}_{pi}.npy")
                np.save(p, z.cpu().numpy())
                z16_paths.append(p)
                print(f"bf16 reference prompt {pi}: cached {p}")
        del model
        torch.cuda.empty_cache()
        print(f"bf16 forwards: {time.time()-t0:.1f}s")

        # ---- diagnostics: FP8 teacher vs bf16 reference, same Gumbel draw --------
        for pi in range(len(ids_list)):
            z8 = torch.from_numpy(np.load(z_fp8_paths[pi])).to("cuda").float()
            z16 = torch.from_numpy(np.load(z16_paths[pi])).to("cuda").float()
            g = torch.empty(z8.shape, device="cuda", dtype=torch.float32)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(a.seed + 1 + pi)        # IDENTICAL Gumbel seeding to the dumps
            g.exponential_(generator=gen).log_().neg_()
            s8, s16 = z8 + g, z16 + g
            tok8, tok16 = s8.argmax(-1), s16.argmax(-1)
            # margin of the FP8-served token under the bf16 reference (forward for
            # this report's orientation) and the reverse
            m_fwd = (s16.max(-1).values - s16.gather(-1, tok8[:, None]).squeeze(-1))
            m_rev = (s8.max(-1).values - s8.gather(-1, tok16[:, None]).squeeze(-1))
            d = (z8 - z16).abs().cpu().numpy().ravel()  # numpy: torch.quantile caps at 2^24 elems
            diag.append({
                "raw_argmax_agree": float((z8.argmax(-1) == z16.argmax(-1)).float().mean()),
                "postgumbel_agree": float((tok8 == tok16).float().mean()),
                "margin_fp8tok_under_bf16_mean": float(m_fwd.mean()),
                "margin_fp8tok_under_bf16_p99": float(m_fwd.flatten().quantile(0.99)),
                "margin_fp8tok_under_bf16_max": float(m_fwd.max()),
                "margin_bf16tok_under_fp8_mean": float(m_rev.mean()),
                "abs_logit_delta_p50_p90_p99_max": [
                    float(np.percentile(d, 50)), float(np.percentile(d, 90)),
                    float(np.percentile(d, 99)), float(d.max())],
            })
            print(f"diag prompt {pi}: {diag[-1]}")
            del z8, z16, g, s8, s16
            torch.cuda.empty_cache()

    agg = {k: float(np.mean([d[k] for d in diag]))
           for k in ("raw_argmax_agree", "postgumbel_agree",
                     "margin_fp8tok_under_bf16_mean", "margin_fp8tok_under_bf16_p99",
                     "margin_bf16tok_under_fp8_mean")}
    agg["margin_fp8tok_under_bf16_max_global"] = max(
        d["margin_fp8tok_under_bf16_max"] for d in diag)
    out = {
        "what": "bf16 full-precision reference logits (NO FP8Linear swap) + "
                "FP8-teacher-vs-bf16 gap diagnostics",
        "reference_definition": "JackFram/llama-68m from_pretrained(bf16).cuda().eval(), "
                                "no replace_linears, forced_logits float32; bf16 "
                                "nonlinears as in the frozen protocol; "
                                "IMA_TEACHER_KERNEL unset (never consulted: only "
                                "FP8Linear.forward reads it)",
        "seed": a.seed, "dolly_sha256": dolly_sha,
        "z_fp16ref_paths": z16_paths,
        "per_prompt_fp8_vs_bf16": diag,
        "aggregate_fp8_vs_bf16": agg,
    }
    res = os.path.join(HERE, f"fp16_ref_seed{a.seed}.json")
    with open(res, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(agg, indent=2))
    print(f"wrote {res}")


if __name__ == "__main__":
    main()
