"""Generate MORE benign per-position dumps for the CODEBOOK corrected orientation.

Companion to gen_more_faithful.py. Same threat-model orientation and identical
benign workload (SAME 120 distinct dolly prompts -> same rows, same tiling, same
per-prompt Gumbel seed gseed_base+1+pi, so the FP8-served token stream is
byte-identical to the faithful run), but the verifier's REFERENCE M_int is the
CODEBOOK student (int_model_approximation CodebookLinear x14 linears-swap,
forced_logits, lm_head stays float) instead of the FaithfulChain.

Pipeline per prompt:
  1. FP8 teacher logits  (bf16 model + FP8Linear x14, fp8_scaled_mm)   [GPU]
  2. codebook M_int logits (bf16 model + CodebookLinear x14)           [GPU]
  3. margins/cand_ranks/Nb under the codebook M_int post-Gumbel scores,
     served token = FP8 post-Gumbel argmax (shared Gumbel draw).

Run:
  IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python \
      gen_more_codebook.py --nprompts 120 --rowseed 20260612 --gseed-base 30000000 \
      --out codebook_extra_corrected.npz
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
PARETO = "/workspace/projects/int-model-approximation/results/llama_pareto"
sys.path.insert(0, PARETO)

from difr_baseline import GpuLock, SCRATCH, MODEL_ID, SEQ_LEN, DOLLY
from capacity_dump import build_bgrid, VOCAB
from gen_more_faithful import draw_distinct_prompts   # identical prompt draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprompts", type=int, default=120)
    ap.add_argument("--rowseed", type=int, default=20260612)
    ap.add_argument("--gseed-base", type=int, default=30000000)
    ap.add_argument("--out", default=os.path.join(HERE, "codebook_extra_corrected.npz"))
    a = ap.parse_args()
    assert os.environ.get("IMA_TEACHER_KERNEL") == "fp8_scaled_mm", \
        "run with IMA_TEACHER_KERNEL=fp8_scaled_mm (harness teacher kernel)"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from int_model_approximation.__main__ import FP8Linear, CodebookLinear
    from llama_difr import replace_linears, forced_logits

    os.makedirs(SCRATCH, exist_ok=True)
    rows, prompts = draw_distinct_prompts(a.nprompts, a.rowseed)
    dolly_sha = hashlib.sha256(open(DOLLY, "rb").read()).hexdigest()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids_list, periods = [], []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        periods.append(int(ids.shape[1]))
        reps = -(-SEQ_LEN // ids.shape[1])
        ids_list.append(ids.repeat(1, reps)[:, :SEQ_LEN])
    print(f"drew {len(ids_list)} distinct prompts, periods "
          f"min={min(periods)} med={int(np.median(periods))} max={max(periods)}")

    def fresh():
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        return m.to("cuda").eval()

    # ---- phase 1 (GPU): FP8 teacher logits per prompt (served stream) ----
    z_fp8 = []
    with GpuLock():
        teacher = fresh()
        n_lin = replace_linears(teacher, lambda w, s, b: FP8Linear(w, s, b))
        print(f"teacher: {n_lin} linears -> FP8")
        with torch.no_grad():
            for pi, ids in enumerate(ids_list):
                t0 = time.time()
                z = forced_logits(teacher, ids.to("cuda"))
                z_fp8.append(z.cpu().numpy().astype(np.float32))
                if pi % 8 == 0:
                    print(f"  teacher prompt {pi}: {time.time()-t0:.2f}s")
        del teacher
        torch.cuda.empty_cache()

    # ---- phase 2 (GPU): codebook M_int logits per prompt (reference) ----
    z_int = []
    with GpuLock():
        student = fresh()
        n_cb = replace_linears(student, lambda w, s, b: CodebookLinear(w, s, b))
        print(f"[codebook] {n_cb} linears -> CodebookLinear (lm_head stays float)")
        with torch.no_grad():
            for pi, ids in enumerate(ids_list):
                t0 = time.time()
                z = forced_logits(student, ids.to("cuda"))
                z_int.append(z.cpu().numpy().astype(np.float32))
                if pi % 8 == 0:
                    print(f"  [codebook] student prompt {pi}: {time.time()-t0:.2f}s")
        del student
        torch.cuda.empty_cache()

    # ---- phase 3 (GPU): margins/ranks/Nb (ref = codebook M_int, served = FP8) ----
    bgrid = build_bgrid()
    margins, cand_ranks, Nb, blk = [], [], [], []
    with GpuLock():
        bt = torch.from_numpy(bgrid).float().to("cuda")
        nb = len(bgrid)
        for pi in range(len(ids_list)):
            zf = torch.from_numpy(z_fp8[pi]).to("cuda").float()
            zr = torch.from_numpy(z_int[pi]).to("cuda").float()    # M_int = reference
            g = torch.empty(zf.shape, device="cuda", dtype=torch.float32)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(a.gseed_base + 1 + pi)
            g.exponential_(generator=gen).log_().neg_()
            zr_full = zr + g
            zf_full = zf + g
            ref_pref = zr_full.max(dim=-1).values
            srv_tok = zf_full.argmax(dim=-1)
            delta = ref_pref[:, None] - zr_full
            margin = delta.gather(-1, srv_tok[:, None]).squeeze(-1)
            delta_sorted, _ = torch.sort(delta, dim=-1)
            rank = torch.searchsorted(delta_sorted, margin[:, None], right=False).squeeze(-1)
            nb_t = torch.searchsorted(
                delta_sorted, bt[None, :].expand(delta.shape[0], nb).contiguous(), right=True)
            margins.append(margin.cpu().numpy().astype(np.float64))
            cand_ranks.append(rank.cpu().numpy().astype(np.int32))
            Nb.append(nb_t.cpu().numpy().astype(np.int32))
            blk.append(np.full(SEQ_LEN, pi, dtype=np.int32))
            del zf, zr, g, zr_full, zf_full, delta, delta_sorted, nb_t
            torch.cuda.empty_cache()

    margins = np.concatenate(margins)
    cand_ranks = np.concatenate(cand_ranks)
    Nb = np.concatenate(Nb, axis=0)
    block_id = np.concatenate(blk)
    np.savez_compressed(
        a.out, scheme="codebook", orientation="corrected: ref=M_int, served=FP8 argmax",
        vocab=VOCAB, bgrid=bgrid, margins=margins, cand_ranks=cand_ranks, Nb=Nb,
        block_id=block_id, prompt_periods=np.array(periods, dtype=np.int32),
        dolly_rows=np.array(rows, dtype=np.int64), rowseed=a.rowseed,
        gseed_base=a.gseed_base, dolly_sha256=dolly_sha, seq_len=SEQ_LEN)
    print(json.dumps({
        "nprompts": len(ids_list), "n_positions": int(margins.size),
        "exact_agree": float((margins == 0).mean()),
        "margin_mean": float(margins.mean()), "margin_max": float(margins.max()),
        "max_cand_rank": int(cand_ranks.max()), "out": a.out}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
