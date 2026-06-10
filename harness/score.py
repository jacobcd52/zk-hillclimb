"""Official scorer. COORDINATOR-RUN ONLY (HARNESS.md). Agents may read, never run
officially, never edit.

Submission contract (submissions/<name>/):
  claim.json            - claims + declared entrypoints (see HARNESS.md)
  student.py            - exposes replace(model) -> int  (swaps linears in-place)
                          for DiFR scoring of the integerization
  prove.sh   (optional) - builds nothing; runs the prover; writes proof dir $1
  verify.sh  (optional) - $1=proof dir $2=public.json $3=transcript-out.json;
                          exit 0 ACCEPT / nonzero REJECT
Until a submission ships prove.sh+verify.sh the soundness gate reports
NOT-VERIFIABLE and the round cannot be accepted (scaffolding may be scored
informationally with --allow-unverified).

Modes:
  --difr      held-out DiFR. Round seed REQUIRED; prompts drawn at scoring
              time from dolly-15k (cached jsonl), Gumbel seed = round seed + 1.
  --timing    cold-process prove timing, median of 3 (+1 declared JIT warmup),
              refuses to run if other GPU processes are active.
  --soundness honest verify must ACCEPT; every forgery in harness/forgeries/
              must REJECT; transcript must cover the manifest.

Usage:
  python score.py --submission <name> --round N --seed S --difr [--timing] [--soundness]
Results -> results/<name>_round<N>.json (append-only; never overwritten).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
PARETO = Path("/workspace/projects/int-model-approximation/results/llama_pareto")
DOLLY = ROOT / "private" / "dolly.jsonl"
MODEL_ID = "JackFram/llama-68m"
SEQ_LEN = 1024
N_HELDOUT = 8


def load_student(sub_dir):
    spec = importlib.util.spec_from_file_location("student", sub_dir / "student.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def heldout_prompts(seed, n=N_HELDOUT):
    if not DOLLY.exists():
        import urllib.request
        url = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
               "resolve/main/databricks-dolly-15k.jsonl")
        DOLLY.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, DOLLY)
    lines = DOLLY.read_text().splitlines()
    rng = random.Random(seed)
    rows = [json.loads(lines[i]) for i in rng.sample(range(len(lines)), n)]
    return ["\n\n".join(filter(None, [r.get("instruction", ""), r.get("context", ""),
                                      r.get("response", "")])) for r in rows]


def score_difr(sub_dir, seed):
    import torch
    sys.path.insert(0, str(PARETO))
    os.environ["IMA_TEACHER_KERNEL"] = "fp8_scaled_mm"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from int_model_approximation.__main__ import FP8Linear
    from int_model_approximation import metrics as M
    from llama_difr import replace_linears, forced_logits

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    student_mod = load_student(sub_dir)

    def fresh():
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        return m.to("cuda").eval()

    per_prompt = []
    for pi, prompt in enumerate(heldout_prompts(seed)):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        reps = -(-SEQ_LEN // ids.shape[1])
        ids = ids.repeat(1, reps)[:, :SEQ_LEN].to("cuda")

        teacher = fresh()
        replace_linears(teacher, lambda w, s, b: FP8Linear(w, s, b))
        z_ref = forced_logits(teacher, ids)
        del teacher
        torch.cuda.empty_cache()

        student = fresh()
        n = student_mod.replace(student)
        z = forced_logits(student, ids)
        del student
        torch.cuda.empty_cache()

        g = torch.empty(z_ref.shape, device=z_ref.device, dtype=torch.float32)
        gen = torch.Generator(device=z_ref.device)
        gen.manual_seed(seed + 1 + pi)          # fresh Gumbel per round AND prompt
        g.exponential_(generator=gen).log_().neg_()
        margin = M.post_gumbel_margin(z_ref, z, g)
        per_prompt.append({
            "difr_mean": float(margin.mean()),
            "difr_p99": float(margin.flatten().quantile(0.99)),
            "logit_l2_mean": float(M.logit_l2(z_ref, z).mean()),   # anti-F3
            "top1": float(M.top1_match(z_ref, z).float().mean()),
            "n_linears": n,
        })
    agg = {k: statistics.mean(p[k] for p in per_prompt)
           for k in ("difr_mean", "difr_p99", "logit_l2_mean", "top1")}
    return {"per_prompt": per_prompt, "aggregate": agg}


def assert_exclusive_gpu():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        raise RuntimeError(f"GPU not exclusive; active compute pids: {out!r}. "
                           "Official timing requires a quiet GPU (E3).")


def score_timing(sub_dir, claim):
    assert_exclusive_gpu()
    prove = sub_dir / "prove.sh"
    if not prove.exists():
        return {"status": "NO-PROVER", "note": "submission has no prove.sh"}
    times = []
    runs = 3 + (1 if claim.get("jit_warmup_run") else 0)
    for i in range(runs):
        proof_dir = sub_dir / f"_timing_proof_{i}"
        t0 = time.perf_counter()
        r = subprocess.run(["bash", str(prove), str(proof_dir)], cwd=sub_dir,
                           capture_output=True, text=True, timeout=14400)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            return {"status": "PROVER-FAILED", "run": i, "stderr": r.stderr[-2000:]}
        times.append(dt)
    if claim.get("jit_warmup_run"):
        times = times[1:]
    return {"status": "OK", "runs_s": times, "median_s": statistics.median(times)}


def score_soundness(sub_dir, manifest_path):
    verify = sub_dir / "verify.sh"
    prove = sub_dir / "prove.sh"
    if not (verify.exists() and prove.exists()):
        return {"status": "NOT-VERIFIABLE",
                "note": "no separate prove/verify entrypoints; cannot pass the gate"}
    res = {"status": "RAN", "honest": None, "forgeries": {}, "coverage": None}
    proof_dir = sub_dir / "_soundness_proof"
    r = subprocess.run(["bash", str(prove), str(proof_dir)], cwd=sub_dir,
                       capture_output=True, text=True, timeout=14400)
    if r.returncode != 0:
        return {"status": "PROVER-FAILED", "stderr": r.stderr[-2000:]}
    public = proof_dir / "public.json"
    transcript = sub_dir / "_transcript.json"
    rv = subprocess.run(["bash", str(verify), str(proof_dir), str(public), str(transcript)],
                        cwd=sub_dir, capture_output=True, text=True, timeout=3600)
    res["honest"] = "ACCEPT" if rv.returncode == 0 else f"REJECT ({rv.stderr[-500:]})"
    cov = subprocess.run([sys.executable, str(HARNESS / "check_transcript.py"),
                          str(manifest_path), str(transcript)],
                         capture_output=True, text=True)
    res["coverage"] = {"ok": cov.returncode == 0, "report": cov.stdout[-2000:]}
    for forgery in sorted((HARNESS / "forgeries").glob("*.py")):
        fid = forgery.stem
        mutated = sub_dir / f"_forged_{fid}"
        fm = subprocess.run([sys.executable, str(forgery), str(proof_dir), str(mutated)],
                            capture_output=True, text=True)
        if fm.returncode != 0:
            res["forgeries"][fid] = f"FORGERY-SCRIPT-ERROR: {fm.stderr[-300:]}"
            continue
        fv = subprocess.run(["bash", str(verify), str(mutated),
                             str(mutated / "public.json"), str(mutated / "_tr.json")],
                            cwd=sub_dir, capture_output=True, text=True, timeout=3600)
        res["forgeries"][fid] = "REJECTED(ok)" if fv.returncode != 0 else "ACCEPTED(FAIL!)"
    res["gate_pass"] = (res["honest"] == "ACCEPT" and res["coverage"]["ok"]
                        and all(v == "REJECTED(ok)" for v in res["forgeries"].values())
                        and len(res["forgeries"]) > 0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--difr", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--soundness", action="store_true")
    ap.add_argument("--manifest", default=str(HARNESS / "manifest_llama68m.json"))
    a = ap.parse_args()

    sub_dir = ROOT / "submissions" / a.submission
    claim = json.loads((sub_dir / "claim.json").read_text())
    out = {"submission": a.submission, "round": a.round,
           "harness_sha": harness_digest(), "claim": claim}
    if a.difr:
        out["difr"] = score_difr(sub_dir, a.seed)
    if a.timing:
        out["timing"] = score_timing(sub_dir, claim)
    if a.soundness:
        out["soundness"] = score_soundness(sub_dir, Path(a.manifest))

    res_path = ROOT / "results" / f"{a.submission}_round{a.round}.json"
    if res_path.exists():
        raise RuntimeError(f"{res_path} exists; results are append-only")
    res_path.write_text(json.dumps(out, indent=2))
    (ROOT / "private" / f"round{a.round}_seed.txt").write_text(str(a.seed))
    print(json.dumps({k: v for k, v in out.items() if k != "claim"}, indent=2))


def harness_digest():
    h = hashlib.sha256()
    for f in sorted(HARNESS.rglob("*")):
        if f.is_file():
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
