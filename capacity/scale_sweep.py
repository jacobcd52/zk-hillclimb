"""Five-term covert-capacity mu* (CODEBOOK integerization) vs MODEL SCALE.

For each llama-family model (vocab 32000) we reproduce the CORRECTED capacity dump
(reference = codebook M_int post-Gumbel, served = FP8 post-Gumbel argmax, shared seed),
then compute the converged five-term benign rate mu* = min_b mean_t r_t at K=4.
No buffer/FPR curve — just the converged mu (a mean), so a modest prompt count suffices.

Run (via exp):
  IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python capacity/scale_sweep.py \
      --nprompts 48
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MEASURE = os.path.join(ROOT, "measure")
BUFFER = os.path.join(ROOT, "buffer")
PARETO = "/workspace/projects/int-model-approximation/results/llama_pareto"
for p in (MEASURE, BUFFER, PARETO):
    sys.path.insert(0, p)

from difr_baseline import GpuLock, DOLLY, SEQ_LEN          # noqa: E402
from capacity_dump import build_bgrid                      # noqa: E402
from capacity_dump_corrected import extend_bgrid           # noqa: E402
from analyze_buffer import per_token                       # noqa: E402

K = 4
ROWSEED = 20260616
GSEED_BASE = 40000000

MODELS = [
    ("llama-68m",      "JackFram/llama-68m"),
    ("llama-160m",     "JackFram/llama-160m"),
    ("tinyllama-1.1b", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
    ("sheared-1.3b",   "princeton-nlp/Sheared-LLaMA-1.3B"),
    ("sheared-2.7b",   "princeton-nlp/Sheared-LLaMA-2.7B"),
    ("openllama-3b",   "openlm-research/open_llama_3b_v2"),
]


def load_prompts(n, seed):
    lines = open(DOLLY).read().splitlines()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(lines))
    out, seen = [], set()
    for i in idx:
        rec = json.loads(lines[i])
        t = (rec.get("instruction") or rec.get("context") or "").strip()
        if len(t) > 16 and t not in seen:
            seen.add(t); out.append(t)
        if len(out) == n:
            break
    return out


def model_dump(model_id, prompts):
    """Return margins, cand_ranks, Nb, bgrid, vocab, n_params for one model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from int_model_approximation.__main__ import FP8Linear, CodebookLinear
    from llama_difr import replace_linears, forced_logits

    tok = AutoTokenizer.from_pretrained(model_id)
    ids_list = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        reps = -(-SEQ_LEN // ids.shape[1])
        ids_list.append(ids.repeat(1, reps)[:, :SEQ_LEN])

    def fresh():
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        return m.to("cuda").eval()

    with GpuLock():
        # ---- FP8 teacher (served) ----
        teacher = fresh()
        n_params = sum(p.numel() for p in teacher.parameters())
        n = replace_linears(teacher, lambda w, s, b: FP8Linear(w, s, b))
        z_fp8 = []
        with torch.no_grad():
            for ids in ids_list:
                z_fp8.append(forced_logits(teacher, ids.to("cuda")).cpu().numpy().astype(np.float32))
        del teacher; torch.cuda.empty_cache()

        # ---- codebook M_int (reference) ----
        student = fresh()
        replace_linears(student, lambda w, s, b: CodebookLinear(w, s, b))
        z_int = []
        with torch.no_grad():
            for ids in ids_list:
                z_int.append(forced_logits(student, ids.to("cuda")).cpu().numpy().astype(np.float32))
        del student; torch.cuda.empty_cache()

        vocab = z_fp8[0].shape[-1]
        # ---- corrected post-Gumbel margins / ranks / sorted deltas ----
        margins, cand_ranks, ds_cpu = [], [], []
        for pi in range(len(ids_list)):
            zf = torch.from_numpy(z_fp8[pi]).to("cuda").float()
            zr = torch.from_numpy(z_int[pi]).to("cuda").float()       # reference = codebook
            g = torch.empty(zf.shape, device="cuda", dtype=torch.float32)
            gen = torch.Generator(device="cuda"); gen.manual_seed(GSEED_BASE + 1 + pi)
            g.exponential_(generator=gen).log_().neg_()
            zrf = zr + g; zsf = zf + g
            ref_pref = zrf.max(dim=-1).values
            srv_tok = zsf.argmax(dim=-1)
            delta = ref_pref[:, None] - zrf
            margin = delta.gather(-1, srv_tok[:, None]).squeeze(-1)
            ds, _ = torch.sort(delta, dim=-1)
            rank = torch.searchsorted(ds, margin[:, None], right=False).squeeze(-1)
            margins.append(margin.cpu().numpy().astype(np.float64))
            cand_ranks.append(rank.cpu().numpy().astype(np.int32))
            ds_cpu.append(ds.cpu())
            del zf, zr, g, zrf, zsf, delta, ds; torch.cuda.empty_cache()

        margins_all = np.concatenate(margins)
        bgrid = extend_bgrid(build_bgrid(), float(margins_all.max()))
        bt = torch.from_numpy(bgrid).float().to("cuda"); nb = len(bgrid)
        Nb = []
        for pi in range(len(ids_list)):
            ds = ds_cpu[pi].to("cuda")
            nb_t = torch.searchsorted(ds, bt[None, :].expand(ds.shape[0], nb).contiguous(), right=True)
            Nb.append(nb_t.cpu().numpy().astype(np.int32))
            del ds, nb_t; torch.cuda.empty_cache()

    return (margins_all, np.concatenate(cand_ranks), np.concatenate(Nb, axis=0),
            bgrid, int(vocab), int(n_params))


def mu_star(margins, ranks, Nb, bgrid, vocab):
    """Converged five-term benign rate, minimised over b at K=4."""
    best = None
    for b in bgrid:
        mu = per_token(margins, ranks, Nb, bgrid, vocab, float(b), K)["mu"]
        if best is None or mu < best[0]:
            best = (mu, float(b))
    return best  # (mu*, b*)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprompts", type=int, default=48)
    ap.add_argument("--nmodels", type=int, default=len(MODELS))
    a = ap.parse_args()
    assert os.environ.get("IMA_TEACHER_KERNEL") == "fp8_scaled_mm", \
        "run with IMA_TEACHER_KERNEL=fp8_scaled_mm"
    prompts = load_prompts(a.nprompts, ROWSEED)
    print(f"{len(prompts)} prompts, {len(prompts)*SEQ_LEN} tokens/model", flush=True)

    results, outp = [], os.path.join(HERE, "scale_sweep_results.json")
    for label, mid in MODELS[:a.nmodels]:
        t0 = time.time()
        try:
            margins, ranks, Nb, bgrid, vocab, nparams = model_dump(mid, prompts)
            mu, bstar = mu_star(margins, ranks, Nb, bgrid, vocab)
            p = float((margins > bstar).mean())
            rec = {"label": label, "model_id": mid, "n_params": nparams,
                   "vocab": vocab, "mu_star": mu, "b_star": bstar, "p_at_bstar": p,
                   "n_tokens": int(margins.size), "secs": round(time.time() - t0, 1)}
            np.savez_compressed(os.path.join(HERE, f"scale_dump_{label}.npz"),
                                margins=margins, cand_ranks=ranks, Nb=Nb, bgrid=bgrid,
                                vocab=vocab, n_params=nparams)
        except Exception as e:
            rec = {"label": label, "model_id": mid, "ERROR": str(e)[:300],
                   "secs": round(time.time() - t0, 1)}
        results.append(rec)
        json.dump(results, open(outp, "w"), indent=2)
        print("RESULT " + json.dumps(rec), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
