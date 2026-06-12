#!/usr/bin/env python3
"""Stage B pair walk (TRANSPORT_REBUILD_DESIGN §6 Stage B, gate T3).

Two-driver fc -> rescale chain through the orchestrator conventions:
  run_seed = sha256(public.json bytes); per-obligation transcript seed =
  f"{run_seed}:{obligation_id}" (verify_walk.py line 210); registration gens
  (ONE gen<G>.bin per domain size) + registered weight commitments; byte-
  equality chain edges (common.py: "gate_fc com_Y == gate_rescale com_X").

Pairs (real shapes from the faithful-arch-v1 walk):
  layer0.mlp.gate_proj.matmul -> .rescaling   B=1024 768x3072  gen4096  sf 2^20
  lm_head.matmul              -> .rescaling   B=1024 768x32000 gen32768 sf 2^16

Both drivers run in claim mode (--claims): all 12 claims land in ONE prover
accumulator (<run>/acc), discharged by ONE zkob_batchopen; the verify side
re-runs each driver (fresh verifier accumulator <run>/vacc*), checks the chain
edges, asserts the one-gen-file-per-domain-size registration invariant and the
relative-comref policy (ZKOB_REQUIRE_RELATIVE_COMREF=1), then batch-verifies.
Verdict = ACCEPT only if every driver is ACCEPT-conditional AND every edge
holds AND opening_batch ACCEPTs (the F12 gating done by this harness, as in
the selftests; full orchestrator wiring is Stage C).

All driver/batch invocations use cwd=<run_dir> with RELATIVE paths, so every
comref in claims.bin is run-dir-relative (the Stage-A flag-3 canonicalization).

usage: pair_walk.py setup|prove|baseline|verify|battery|measure <run_dir>
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time

ZKLLM = "/root/zkllm"
SRC = "/root/zkorch/stage3v2-fa"          # faithful-arch-v1 official run
SEQ = 1024
GENS = {1024: "registration/gen1024.bin",
        4096: "registration/gen4096.bin",
        32768: "registration/gen32768.bin"}
Q = "registration/q.bin"

PAIRS = [
    {   # layer0.mlp.gate_proj
        "mm": "layer0.mlp.gate_proj.matmul", "rs": "layer0.mlp.gate_proj.rescaling",
        "IN": 768, "OUT": 3072, "log_sf": 20,
        "gen_in": GENS[1024], "gen_out": GENS[4096],
        "com_W": "registration/weights/layer0.mlp.gate_proj-com.bin",
        "W_int": "registration/weights/layer0.mlp.gate_proj-int.bin",
        "X_src": SRC + "/data/layer0.post_attn_norm.rmsnorm.out.i32.bin",
        "X": "data/gate_X.i32.bin", "Y": "data/gate_Y.i64.bin",
        "out": "data/gate_out.i32.bin",
    },
    {   # lm_head (the gen32768 stress instances)
        "mm": "lm_head.matmul", "rs": "lm_head.rescaling",
        "IN": 768, "OUT": 32000, "log_sf": 16,
        "gen_in": GENS[1024], "gen_out": GENS[32768],
        "com_W": "registration/weights/lm_head-com.bin",
        "W_int": "registration/weights/lm_head-int.bin",
        "X_src": SRC + "/data/final_norm.rmsnorm.out.i32.bin",
        "X": "data/lm_X.i32.bin", "Y": "data/lm_Y.i64.bin",
        "out": "data/lm_out.i32.bin",
    },
]


def sh(run, cmd, env=None, log=None, check=True):
    e = dict(os.environ)
    if env:
        e.update(env)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=run, env=e, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if log is not None:
        log.append((cmd, dt, p.returncode, p.stdout))
    if check and p.returncode not in (0, 1):
        sys.stderr.write(p.stdout)
        raise RuntimeError(f"driver error (exit {p.returncode}): {' '.join(cmd)}")
    return p.returncode, p.stdout, dt


def sha256f(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_seed_of(run):
    with open(os.path.join(run, "public.json"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---- claims.bin parser (format: zkob_claims.cuh claim_blob) ----------------
def parse_claims(path):
    b = open(path, "rb").read()
    assert b[:4] == b"ZKCL", "bad claims magic"
    ver, n = struct.unpack_from("<II", b, 4)
    assert ver == 1
    off = 12
    out = []
    for _ in range(n):
        (l,) = struct.unpack_from("<I", b, off); off += 4
        cid = b[off:off + l].decode(); off += l
        (l,) = struct.unpack_from("<I", b, off); off += 4
        comref = b[off:off + l].decode(); off += l
        dom, rows, np_ = struct.unpack_from("<III", b, off); off += 12
        off += np_ * 32
        tag = b[off]; off += 1
        off += 32 if tag == 0 else 144
        out.append({"id": cid, "comref": comref, "domain": dom, "n_rows": rows})
    assert off == len(b), "trailing bytes"
    return out


def setup(run):
    os.makedirs(run, exist_ok=True)
    for d in ("registration/weights", "data", "proofs", "acc"):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    # registration: symlink the official faithful-arch-v1 files (read-only use)
    for rel in list(GENS.values()) + [Q] + [p["com_W"] for p in PAIRS] + [p["W_int"] for p in PAIRS]:
        dst = os.path.join(run, rel)
        if not os.path.exists(dst):
            os.symlink(os.path.join(SRC, rel), dst)
    for p in PAIRS:
        dst = os.path.join(run, p["X"])
        if not os.path.exists(dst):
            shutil.copyfile(p["X_src"], dst)
    for p in PAIRS:
        for sub in (p["mm"], p["rs"]):
            os.makedirs(os.path.join(run, "proofs", sub), exist_ok=True)
    # the pair statement: registration hashes + shapes (run_seed = sha256 of this)
    pub = {
        "statement": "stageB-pair fc->rescale (gate_proj + lm_head) over faithful-arch-v1 registration",
        "submission": "faithful-arch-v1",
        "registration": {rel: sha256f(os.path.join(run, rel))
                         for rel in list(GENS.values()) + [Q]
                         + [p["com_W"] for p in PAIRS]},
        "inputs": {p["X"]: sha256f(os.path.join(run, p["X"])) for p in PAIRS},
        "pairs": [{"mm": p["mm"], "rs": p["rs"], "B": SEQ, "IN": p["IN"],
                   "OUT": p["OUT"], "log_sf": p["log_sf"]} for p in PAIRS],
    }
    with open(os.path.join(run, "public.json"), "w") as f:
        json.dump(pub, f, indent=1, sort_keys=True)
    print(f"setup done; run_seed = {run_seed_of(run)}")


def prove(run, prof=False):
    seed0 = run_seed_of(run)
    env = {"ZKOB_PROF": "1"} if prof else {}
    log = []
    for p in PAIRS:
        mm, rs = p["mm"], p["rs"]
        rc, out, dt = sh(run, [os.path.join(ZKLLM, "zkob_fc"), "prove",
                               f"proofs/{mm}", f"{seed0}:{mm}",
                               p["X"], p["W_int"], str(SEQ), str(p["IN"]), str(p["OUT"]),
                               p["gen_in"], p["gen_out"], Q, p["Y"],
                               "--claims", "acc", mm, p["com_W"]], env=env, log=log)
        assert rc == 0, out
        print(f"  prove {mm}: {dt:.1f} s")
        rc, out, dt = sh(run, [os.path.join(ZKLLM, "zkob_rescale"), "prove",
                               f"proofs/{rs}", f"{seed0}:{rs}",
                               p["Y"], str(SEQ), str(p["OUT"]), str(p["log_sf"]),
                               p["gen_out"], Q, p["out"],
                               "--claims", "acc", rs], env=env, log=log)
        assert rc == 0, out
        print(f"  prove {rs}: {dt:.1f} s")
    genspec = [f"{g}={rel}" for g, rel in sorted(GENS.items())]
    rc, out, dt = sh(run, [os.path.join(ZKLLM, "zkob_batchopen"), "prove",
                           "acc", seed0, Q] + genspec,
                     env={**env, "ZKOB_PROF": "1"}, log=log)
    assert rc == 0, out
    print(f"  batch prove: {dt:.1f} s")
    for line in out.splitlines():
        if line.startswith("PROF"):
            print("   ", line)
    return log


def baseline(run):
    """OLD-tail (inline-IPA) prove+verify of the same four obligations into
    proofs_old/ — the Stage-A 'before' timings on THIS box and data."""
    seed0 = run_seed_of(run)
    os.makedirs(os.path.join(run, "proofs_old"), exist_ok=True)
    times = {}
    for p in PAIRS:
        mm, rs = p["mm"], p["rs"]
        for sub in (mm, rs):
            os.makedirs(os.path.join(run, "proofs_old", sub), exist_ok=True)
        sh(run, [os.path.join(ZKLLM, "zkob_fc"), "prove",
                 f"proofs_old/{mm}", f"{seed0}:{mm}",
                 p["X"], p["W_int"], str(SEQ), str(p["IN"]), str(p["OUT"]),
                 p["gen_in"], p["gen_out"], Q])
        sh(run, [os.path.join(ZKLLM, "zkob_rescale"), "prove",
                 f"proofs_old/{rs}", f"{seed0}:{rs}",
                 p["Y"], str(SEQ), str(p["OUT"]), str(p["log_sf"]),
                 p["gen_out"], Q])
        rc, out, dt = sh(run, [os.path.join(ZKLLM, "zkob_fc"), "verify",
                               f"proofs_old/{mm}", f"{seed0}:{mm}",
                               str(SEQ), str(p["IN"]), str(p["OUT"]), p["com_W"],
                               p["gen_in"], p["gen_out"], Q], env={"ZKOB_PROF": "1"})
        assert rc == 0, out
        times[mm] = (dt, [l for l in out.splitlines() if l.startswith("PROF")])
        print(f"  OLD verify {mm}: {dt:.2f} s")
        rc, out, dt = sh(run, [os.path.join(ZKLLM, "zkob_rescale"), "verify",
                               f"proofs_old/{rs}", f"{seed0}:{rs}",
                               str(SEQ), str(p["OUT"]), str(p["log_sf"]),
                               p["gen_out"], Q], env={"ZKOB_PROF": "1"})
        assert rc == 0, out
        times[rs] = (dt, [l for l in out.splitlines() if l.startswith("PROF")])
        print(f"  OLD verify {rs}: {dt:.2f} s")
    with open(os.path.join(run, "baseline_times.json"), "w") as f:
        json.dump({k: {"wall_s": v[0], "prof": v[1]} for k, v in times.items()}, f, indent=1)
    return times


def verify(run, vtag="vacc", env_extra=None, quiet=False):
    """Full pair verify: drivers (claim mode, fresh verifier accumulator) ->
    chain edges -> registration/comref invariants -> opening_batch."""
    seed0 = run_seed_of(run)
    vacc = os.path.join(run, vtag)
    if os.path.exists(vacc):
        shutil.rmtree(vacc)
    os.makedirs(vacc)
    env = {"ZKOB_PROF": "1", "ZKOB_REQUIRE_RELATIVE_COMREF": "1"}
    if env_extra:
        env.update(env_extra)
    t = {}
    ok = True
    reasons = []
    for p in PAIRS:
        mm, rs = p["mm"], p["rs"]
        rc, out, dt = sh(run, [os.path.join(ZKLLM, "zkob_fc"), "verify",
                               f"proofs/{mm}", f"{seed0}:{mm}",
                               str(SEQ), str(p["IN"]), str(p["OUT"]), p["com_W"],
                               p["gen_in"], p["gen_out"], Q,
                               "--claims", vtag, mm], env=env)
        t[mm] = dt
        if rc != 0 or "ACCEPT-conditional" not in out:
            ok = False; reasons.append(f"driver {mm}: " + out.strip().splitlines()[-1])
        rc, out, dt = sh(run, [os.path.join(ZKLLM, "zkob_rescale"), "verify",
                               f"proofs/{rs}", f"{seed0}:{rs}",
                               str(SEQ), str(p["OUT"]), str(p["log_sf"]),
                               p["gen_out"], Q,
                               "--claims", vtag, rs], env=env)
        t[rs] = dt
        if rc != 0 or "ACCEPT-conditional" not in out:
            ok = False; reasons.append(f"driver {rs}: " + out.strip().splitlines()[-1])
        # chain edge (common.py: "gate_fc com_Y == gate_rescale com_X")
        ey = os.path.join(run, "proofs", mm, "com_Y.bin")
        ex = os.path.join(run, "proofs", rs, "com_X.bin")
        if open(ey, "rb").read() != open(ex, "rb").read():
            ok = False; reasons.append(f"chain edge {mm}.com_Y != {rs}.com_X")
    if not ok:
        print("PAIR VERIFY: REJECT (driver/edge stage)")
        for r in reasons:
            print("   ", r)
        return False, t, reasons
    # registration invariant: every claim domain maps 1:1 onto the unique
    # registration gen file of that size; comrefs are run-dir-relative
    claims = parse_claims(os.path.join(vacc, "claims.bin"))
    assert len(claims) == 12, f"expected 12 claims, got {len(claims)}"
    for c in claims:
        assert c["domain"] in GENS, f"claim {c['id']}: domain {c['domain']} has no registration gen file"
        assert not c["comref"].startswith("/"), f"absolute comref: {c['comref']}"
    assert len({c["domain"] for c in claims}) == 3
    genspec = [f"{g}={rel}" for g, rel in sorted(GENS.items())]
    rc, out, dt = sh(run, [os.path.join(ZKLLM, "zkob_batchopen"), "verify",
                           "acc", vtag, seed0, Q] + genspec, env=env)
    t["opening_batch"] = dt
    if not quiet:
        for line in out.splitlines():
            if line.startswith("PROF") or "ACCEPT" in line or "REJECT" in line:
                print("   ", line)
    if rc != 0:
        locus = [l for l in out.splitlines() if l.startswith("REJECT")]
        print("PAIR VERIFY: REJECT (opening_batch)")
        return False, t, locus
    print(f"PAIR VERIFY: ACCEPT (4 drivers conditional + 2 edges + opening_batch; "
          f"batch verify {dt:.2f} s)")
    return True, t, []


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    mode, run = sys.argv[1], sys.argv[2]
    if mode == "setup":
        setup(run)
    elif mode == "prove":
        prove(run, prof=True)
    elif mode == "baseline":
        baseline(run)
    elif mode == "verify":
        ok, t, _ = verify(run)
        print(json.dumps(t, indent=1))
        sys.exit(0 if ok else 1)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
