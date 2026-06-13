"""Generate MORE benign per-position dumps for the FAITHFUL corrected orientation.

Why: BUFFER_FPR.md needs many INDEPENDENT prompt-blocks. The published dump
(capacity/capacity_dump_corrected_faithful_seed20260611.npz) has only 8 dolly
prompts, each TILED (repeated) to 1024 tokens -> effectively ~8 independent
blocks. This script draws N additional DISTINCT dolly prompts (rows disjoint
from the original 8), runs the exact same corrected-orientation pipeline
(reference = faithful M_int, served = FP8 fast-model argmax, Gumbel seed+1+pi,
metric temperature 1.0), and writes one combined extended dump.

Pipeline per prompt (identical construction to difr_baseline.py +
capacity_dump_corrected.py, faithful scheme only):
  1. FP8 teacher logits (bf16 model + FP8Linear x14, fp8_scaled_mm)  [GPU]
  2. faithful integer-chain logits (FaithfulChain, CPU)              [CPU]
  3. margins/cand_ranks/Nb under the faithful M_int post-Gumbel scores,
     served token = FP8 post-Gumbel argmax.

The original 8 prompts (seed 20260611) are NOT regenerated; new prompts use a
disjoint row draw and per-prompt Gumbel seed (GSEED_BASE + 1 + global_index).

Run:
  IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python \
      gen_more_faithful.py --nprompts 64 --rowseed 20260612 --gseed-base 30000000
"""
import argparse
import hashlib
import json
import os
import random
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

ORIG_SEED = 20260611
SF = 1 << 16
FAITHFUL_REG = "/root/zkorch/stage3v2-fa/registration"


def draw_distinct_prompts(nprompts, rowseed):
    """Draw nprompts dolly rows disjoint from the original-8 draw."""
    lines = open(DOLLY).read().splitlines()
    L = len(lines)
    orig = set(random.Random(ORIG_SEED).sample(range(L), 8))
    rng = random.Random(rowseed)
    rows, seen = [], set(orig)
    while len(rows) < nprompts:
        j = rng.randrange(L)
        if j in seen:
            continue
        seen.add(j)
        rows.append(j)
    prompts = []
    for j in rows:
        r = json.loads(lines[j])
        prompts.append("\n\n".join(filter(
            None, [r.get("instruction", ""), r.get("context", ""), r.get("response", "")])))
    return rows, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprompts", type=int, default=64)
    ap.add_argument("--rowseed", type=int, default=20260612)
    ap.add_argument("--gseed-base", type=int, default=30000000)
    ap.add_argument("--out", default=os.path.join(
        HERE, "faithful_extra_corrected.npz"))
    a = ap.parse_args()
    assert os.environ.get("IMA_TEACHER_KERNEL") == "fp8_scaled_mm", \
        "run with IMA_TEACHER_KERNEL=fp8_scaled_mm (harness teacher kernel)"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from int_model_approximation.__main__ import FP8Linear
    from llama_difr import replace_linears, forced_logits
    from int_chain import FaithfulChain

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

    # ---- phase 1 (GPU): FP8 teacher logits per prompt ----
    def fresh():
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        return m.to("cuda").eval()

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

    # ---- phase 2 (CPU): faithful integer-chain logits per prompt ----
    model_cpu = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32).eval()
    chain = FaithfulChain(FAITHFUL_REG)
    chain.verify_weights_against_model(model_cpu)
    print("[faithful] registered-weight provenance guard: OK")
    z_int = []
    emb = model_cpu.model.embed_tokens
    with torch.no_grad():
        for pi, ids in enumerate(ids_list):
            t0 = time.time()
            x = emb(ids[0])
            x0 = torch.round(x.float() * SF).to(torch.int32).numpy()
            chain.stats = {}
            final = chain.forward(x0)
            z = chain.logits(final)
            z_int.append(z.astype(np.float32))
            if pi % 8 == 0:
                print(f"  [faithful] chain prompt {pi}: {time.time()-t0:.1f}s")

    # ---- phase 3 (GPU): margins/ranks/Nb (ref = faithful M_int, served = FP8) ----
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
        a.out, scheme="faithful", orientation="corrected: ref=M_int, served=FP8 argmax",
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
