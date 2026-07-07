"""Stage-C2 wiring smoke test: a 2-driver (gate fc -> rescale) claim-mode
chain through the orchestrator's OWN conventions (common.run_driver,
rel_argv, genspec_args, cwd=run_dir, relative comrefs, batch seed shape),
one zkob_batchopen prove + verify, ZKOB_REQUIRE_RELATIVE_COMREF on verify.
Validates the driver CLIs and the accumulator plumbing prove_walk/verify_walk
depend on, before the long full run."""
import os
import shutil
import sys

sys.path.insert(0, "/workspace/projects/zk-hillclimb/orchestrator")
import common as C

SRC = "/root/zkorch/stage3v2-fa"
RUN = "/tmp/c2smoke"
seed = "c2smoke-runseed"

if os.path.exists(RUN):
    shutil.rmtree(RUN)
os.makedirs(os.path.join(RUN, "data"))
os.symlink(os.path.join(SRC, "registration"), os.path.join(RUN, "registration"))
for d in ("proofs/gate_mm", "proofs/gate_rs", "proofs/opening_batch/b0", "vacc/b0"):
    os.makedirs(os.path.join(RUN, d))
shutil.copy(os.path.join(SRC, "data/layer0.post_attn_norm.rmsnorm.out.i32.bin"),
            os.path.join(RUN, "data/X.i32.bin"))

P = C.reg_paths(RUN)
acc = os.path.join(RUN, "proofs/opening_batch/b0")
vacc = os.path.join(RUN, "vacc/b0")
W = C.wpath(RUN, "layer0.mlp.gate_proj", "int")
comW = C.wpath(RUN, "layer0.mlp.gate_proj", "com")
B, IN, OUT = str(C.SEQ), str(C.EMBED), str(C.INTER)

def rr(cmd, label, **kw):
    ok, dt, out = C.run_driver(C.rel_argv(cmd, RUN), label, cwd=RUN, **kw)
    print(f"    -> {'ACCEPT' if ok else 'REJECT'} ({dt:.1f}s)")
    return ok, out

# prove side (exact prove_walk shapes: trailing --claims <acc> <obid> [extra])
rr([C.drv("zkob_fc"), "prove", os.path.join(RUN, "proofs/gate_mm"), f"{seed}:gate_mm",
    os.path.join(RUN, "data/X.i32.bin"), W, B, IN, OUT,
    P["gen1024"], P["gen4096"], P["q"], os.path.join(RUN, "data/Y.i64.bin"),
    "--claims", os.path.relpath(acc, RUN), "gate_mm", comW], "prove gate fc")
rr([C.drv("zkob_rescale"), "prove", os.path.join(RUN, "proofs/gate_rs"), f"{seed}:gate_rs",
    os.path.join(RUN, "data/Y.i64.bin"), B, OUT, "20", P["gen4096"], P["q"],
    os.path.join(RUN, "data/Xr.i32.bin"),
    "--claims", os.path.relpath(acc, RUN), "gate_rs"], "prove gate rescale")
rr([C.drv("zkob_batchopen"), "prove", os.path.relpath(acc, RUN),
    C.batch_seed(seed, 0), "registration/q.bin"] + C.genspec_args(),
   "prove opening_batch")

claims = C.parse_claims(os.path.join(acc, "claims.bin"))
print(f"  claims.bin: {len(claims)} claims")
for c in claims:
    print(f"    {c['id']:<12} comref={c['comref']} dom={c['domain']} rows={c['n_rows']} tag={c['tag']}")
assert all(not c["comref"].startswith("/") for c in claims), "absolute comref leaked"

# verify side (exact verify_walk shapes)
os.environ["ZKOB_REQUIRE_RELATIVE_COMREF"] = "1"
ok1, _ = rr([C.drv("zkob_fc"), "verify", os.path.join(RUN, "proofs/gate_mm"),
             f"{seed}:gate_mm", B, IN, OUT, comW, P["gen1024"], P["gen4096"], P["q"],
             "--claims", os.path.relpath(vacc, RUN), "gate_mm"],
            "verify gate fc", expect_reject_ok=True)
ok2, out2 = rr([C.drv("zkob_rescale"), "verify", os.path.join(RUN, "proofs/gate_rs"),
                f"{seed}:gate_rs", B, OUT, "20", P["gen4096"], P["q"],
                "--claims", os.path.relpath(vacc, RUN), "gate_rs"],
               "verify gate rescale", expect_reject_ok=True)
assert ok1 and ok2 and "ACCEPT-conditional" in out2, "driver verify failed"
ok3, out3 = rr([C.drv("zkob_batchopen"), "verify", os.path.relpath(acc, RUN),
                os.path.relpath(vacc, RUN), C.batch_seed(seed, 0),
                "registration/q.bin"] + C.genspec_args(),
               "verify opening_batch", expect_reject_ok=True)
assert ok3, "batch verify failed"

# one tamper: claims.bin eval byte -> claims_match (the named-locus pin)
p = os.path.join(acc, "claims.bin")
b = bytearray(open(p, "rb").read()); b[-1] ^= 1; open(p, "wb").write(bytes(b))
okt, outt = rr([C.drv("zkob_batchopen"), "verify", os.path.relpath(acc, RUN),
                os.path.relpath(vacc, RUN), C.batch_seed(seed, 0),
                "registration/q.bin"] + C.genspec_args(),
               "verify opening_batch (tampered claims)", expect_reject_ok=True)
b[-1] ^= 1; open(p, "wb").write(bytes(b))
assert not okt and "claims_match" in outt, f"tamper locus wrong: {outt[-300:]}"
print("SMOKE: ALL PASS (honest accept + claims_match tamper locus)")
