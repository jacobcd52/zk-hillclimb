#!/usr/bin/env python3
# Stage D overhead benchmark + D4 leak regression at llama-68m fc scale
# (gate_proj shape: IN=768, OUT=3072, B=64 tokens).
# Compares the Stage-C2 claim-mode path against the Stage-D wpriv path:
#   registration (plain vs hiding), driver prove/verify, batch prove/verify
#   (+ weight batch), artifact sizes; then scans every verifier-consumed
#   artifact for the true claim_W (read from the prover-private cblinds.bin).
import os, struct, subprocess, sys, time
import numpy as np

ZK = "/root/zkllm"
WD = "/tmp/stage_d_bench"
B, IN, OUT = 64, 768, 3072
IN_pad, OUT_pad = 1024, 4096

def run(cmd, **kw):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ZK, capture_output=True, text=True, **kw)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)}")
    return dt, r.stdout

def fsize(p):
    return os.path.getsize(p) if os.path.exists(p) else 0

os.system(f"rm -rf {WD}")
for d in ["", "/plain", "/plain/acc", "/plain/vacc",
          "/wp", "/wp/acc", "/wp/vacc", "/wp/wacc", "/wp/wvacc"]:
    os.makedirs(WD + d, exist_ok=True)

rng = np.random.default_rng(20260612)
rng.integers(-128, 128, size=IN * OUT, dtype=np.int32).tofile(f"{WD}/W.i32")
rng.integers(-128, 128, size=B * IN, dtype=np.int32).tofile(f"{WD}/X.i32")

# shared pp: one generator file per domain size + 2-slot q (Q, H)
for G in (IN_pad, OUT_pad):
    if not os.path.exists(f"{WD}/gen{G}.bin"):
        dt, _ = run(["./ppgen", str(G), f"{WD}/gen{G}.bin"])
        print(f"ppgen {G}: {dt:.2f} s")
run(["./zkob_batchopen", "genq", f"{WD}/q.bin"])
genspecs = [f"{IN_pad}={WD}/gen{IN_pad}.bin", f"{OUT_pad}={WD}/gen{OUT_pad}.bin"]

T = {}

# ---------- plain claim-mode path (Stage C2 baseline) ----------
T["reg_plain"], _ = run(["./zkob_fc", "commit", f"{WD}/W.i32", str(IN), str(OUT),
                         f"{WD}/gen{OUT_pad}.bin", f"{WD}/plain/com_W.bin"])
T["prove_plain"], _ = run(["./zkob_fc", "prove", f"{WD}/plain", "bench:matmul",
                           f"{WD}/X.i32", f"{WD}/W.i32", str(B), str(IN), str(OUT),
                           f"{WD}/gen{IN_pad}.bin", f"{WD}/gen{OUT_pad}.bin", f"{WD}/q.bin",
                           "--claims", f"{WD}/plain/acc", "bench.matmul", f"{WD}/plain/com_W.bin"])
T["verify_plain"], _ = run(["./zkob_fc", "verify", f"{WD}/plain", "bench:matmul",
                            str(B), str(IN), str(OUT), f"{WD}/plain/com_W.bin",
                            f"{WD}/gen{IN_pad}.bin", f"{WD}/gen{OUT_pad}.bin", f"{WD}/q.bin",
                            "--claims", f"{WD}/plain/vacc", "bench.matmul"])
T["batch_prove_plain"], _ = run(["./zkob_batchopen", "prove", f"{WD}/plain/acc", "bench",
                                 f"{WD}/q.bin"] + genspecs)
T["batch_verify_plain"], _ = run(["./zkob_batchopen", "verify", f"{WD}/plain/acc",
                                  f"{WD}/plain/vacc", "bench", f"{WD}/q.bin"] + genspecs)

# ---------- wpriv path (Stage D) ----------
T["reg_hiding"], _ = run(["./zkob_fc", "commit", f"{WD}/W.i32", str(IN), str(OUT),
                          f"{WD}/gen{OUT_pad}.bin", f"{WD}/wp/com_W.bin",
                          "--hiding", f"{WD}/q.bin", f"{WD}/wp/com_W.blinds.bin"])
T["prove_wp"], _ = run(["./zkob_fc", "prove", f"{WD}/wp", "bench:matmul",
                        f"{WD}/X.i32", f"{WD}/W.i32", str(B), str(IN), str(OUT),
                        f"{WD}/gen{IN_pad}.bin", f"{WD}/gen{OUT_pad}.bin", f"{WD}/q.bin",
                        "--claims", f"{WD}/wp/acc", "bench.matmul", f"{WD}/wp/com_W.bin",
                        "--wpriv", f"{WD}/wp/wacc", f"{WD}/wp/com_W.blinds.bin"])
T["verify_wp"], _ = run(["./zkob_fc", "verify", f"{WD}/wp", "bench:matmul",
                         str(B), str(IN), str(OUT), f"{WD}/wp/com_W.bin",
                         f"{WD}/gen{IN_pad}.bin", f"{WD}/gen{OUT_pad}.bin", f"{WD}/q.bin",
                         "--claims", f"{WD}/wp/vacc", "bench.matmul",
                         "--wpriv", f"{WD}/wp/wvacc"])
T["batch_prove_wp"], _ = run(["./zkob_batchopen", "prove", f"{WD}/wp/acc", "bench",
                              f"{WD}/q.bin"] + genspecs)
T["batch_verify_wp"], _ = run(["./zkob_batchopen", "verify", f"{WD}/wp/acc",
                               f"{WD}/wp/vacc", "bench", f"{WD}/q.bin"] + genspecs)
T["wbatch_prove"], _ = run(["./zkob_batchopen", "wprove", f"{WD}/wp/wacc", "bench",
                            f"{WD}/q.bin"] + genspecs)
T["wbatch_verify"], _ = run(["./zkob_batchopen", "wverify", f"{WD}/wp/wacc",
                             f"{WD}/wp/wvacc", "bench", f"{WD}/q.bin"] + genspecs)

print("\n==== timings (s) ====")
for k, v in T.items():
    print(f"  {k:22s} {v:8.2f}")
print(f"  TOTAL prove  plain={T['prove_plain']+T['batch_prove_plain']:.2f}  "
      f"wpriv={T['prove_wp']+T['batch_prove_wp']+T['wbatch_prove']:.2f}")
print(f"  TOTAL verify plain={T['verify_plain']+T['batch_verify_plain']:.2f}  "
      f"wpriv={T['verify_wp']+T['batch_verify_wp']+T['wbatch_verify']:.2f}")

print("\n==== artifact sizes (bytes) ====")
plain_files = [("sumcheck.bin", f"{WD}/plain/sumcheck.bin"),
               ("acc/claims.bin", f"{WD}/plain/acc/claims.bin"),
               (f"ipa_batch_{IN_pad}", f"{WD}/plain/acc/ipa_batch_{IN_pad}.bin"),
               (f"ipa_batch_{OUT_pad}", f"{WD}/plain/acc/ipa_batch_{OUT_pad}.bin"),
               ("batch_sumcheck", f"{WD}/plain/acc/batch_sumcheck.bin"),
               ("batch_vfin", f"{WD}/plain/acc/batch_vfin.bin")]
wp_files = [("wsc.bin", f"{WD}/wp/wsc.bin"),
            ("acc/claims.bin", f"{WD}/wp/acc/claims.bin"),
            ("wacc/claims.bin", f"{WD}/wp/wacc/claims.bin"),
            (f"ipa_batch_{IN_pad}", f"{WD}/wp/acc/ipa_batch_{IN_pad}.bin"),
            (f"ipa_batch_{OUT_pad}", f"{WD}/wp/acc/ipa_batch_{OUT_pad}.bin"),
            ("batch_sumcheck", f"{WD}/wp/acc/batch_sumcheck.bin"),
            ("batch_vfin", f"{WD}/wp/acc/batch_vfin.bin"),
            ("wbatch_sumcheck", f"{WD}/wp/wacc/wbatch_sumcheck.bin"),
            ("wbatch_vfin", f"{WD}/wp/wacc/wbatch_vfin.bin"),
            (f"wipa_batch_{OUT_pad}", f"{WD}/wp/wacc/wipa_batch_{OUT_pad}.bin")]
pt = sum(fsize(p) for _, p in plain_files)
wt = sum(fsize(p) for _, p in wp_files)
for n, p in plain_files: print(f"  plain {n:24s} {fsize(p):10d}")
for n, p in wp_files:    print(f"  wpriv {n:24s} {fsize(p):10d}")
print(f"  proof-side total: plain={pt}  wpriv={wt}  delta={wt-pt:+d}")

# ---------- D4 leak regression ----------
print("\n==== D4 leak regression ====")
def read_cblinds(path):
    out = {}
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    while off < len(data):
        n = struct.unpack_from("<I", data, off)[0]; off += 4
        cid = data[off:off+n].decode(); off += n
        v = data[off:off+32]; off += 32
        t = data[off:off+32]; off += 32
        out[cid] = (v, t)
    return out

claim_W = read_cblinds(f"{WD}/wp/wacc/cblinds.bin")["bench.matmul:W"][0]

# every verifier-consumed artifact in the wpriv pipeline
artifacts = []
for base in (f"{WD}/wp",):
    for fn in ("dims.bin", "com_X.bin", "com_Y.bin", "wsc.bin", "com_W.bin"):
        artifacts.append(f"{base}/{fn}")
for accd in (f"{WD}/wp/acc", f"{WD}/wp/vacc", f"{WD}/wp/wacc", f"{WD}/wp/wvacc"):
    for fn in os.listdir(accd):
        if fn.startswith("wit_") or fn in ("witrefs.txt", "cblinds.bin", "blindrefs.txt"):
            continue  # prover-private, never shipped
        artifacts.append(f"{accd}/{fn}")

hits = [p for p in artifacts if claim_W in open(p, "rb").read()]
print(f"  scanned {len(artifacts)} artifacts for claim_W: "
      f"{'CLEAN (0 hits)' if not hits else 'LEAK: ' + str(hits)}")

# positive control: the PLAIN path must leak claim_W in sumcheck.bin + claims.bin
with open(f"{WD}/plain/sumcheck.bin", "rb") as f:
    plain_claim_W = f.read()[-32:]
ctrl = [p for p in (f"{WD}/plain/sumcheck.bin", f"{WD}/plain/acc/claims.bin")
        if plain_claim_W in open(p, "rb").read()]
print(f"  positive control (plain path): claim_W found in {len(ctrl)}/2 expected artifacts")

# prover-private files DO contain it (sanity that the secret exists prover-side)
priv = [p for p in (f"{WD}/wp/wacc/cblinds.bin",) if claim_W in open(p, "rb").read()]
print(f"  prover-private blind stash contains claim_W: {bool(priv)}")

ok = (not hits) and len(ctrl) == 2 and bool(priv)
print(f"\nD4 BENCH REGRESSION: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
