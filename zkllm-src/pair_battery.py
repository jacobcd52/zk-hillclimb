#!/usr/bin/env python3
"""Stage B BO forgery battery at pair scale (TRANSPORT_REVIEW §5.2 BO-1..BO-12
with the F8 locus split and the F3/F10 structural cases, plus the two Stage-B
additions: cross-driver claim-drop and chain-edge tamper).

Runs against an already-proven pair run (pair_walk.py setup+prove). Every case
must be rejected by EXACTLY the named check; honest/restored verifies must
ACCEPT. Cases that only touch batch artifacts / the claim lists reuse the
verifier accumulator from the initial honest full verify and re-run
zkob_batchopen verify alone (the drivers' claims are unchanged by
construction); cases touching driver artifacts re-run the full harness verify.

usage: pair_battery.py <run_dir>
"""
import os
import shutil
import struct
import subprocess
import sys

import pair_walk as PW

ZKLLM = PW.ZKLLM
GENS = PW.GENS
Q = PW.Q


# ---- full-fidelity claims.bin codec ----------------------------------------
def cl_parse(path):
    b = open(path, "rb").read()
    assert b[:4] == b"ZKCL"
    ver, n = struct.unpack_from("<II", b, 4)
    assert ver == 1
    off = 12
    out = []
    for _ in range(n):
        c = {}
        (l,) = struct.unpack_from("<I", b, off); off += 4
        c["id"] = b[off:off + l]; off += l
        (l,) = struct.unpack_from("<I", b, off); off += 4
        c["comref"] = b[off:off + l]; off += l
        c["domain"], c["n_rows"], np_ = struct.unpack_from("<III", b, off); off += 12
        c["point"] = [b[off + 32 * i:off + 32 * (i + 1)] for i in range(np_)]; off += np_ * 32
        c["tag"] = b[off]; off += 1
        evl = 32 if c["tag"] == 0 else 144
        c["eval"] = b[off:off + evl]; off += evl
        out.append(c)
    assert off == len(b)
    return out


def cl_write(path, cs):
    out = [b"ZKCL", struct.pack("<II", 1, len(cs))]
    for c in cs:
        out.append(struct.pack("<I", len(c["id"]))); out.append(c["id"])
        out.append(struct.pack("<I", len(c["comref"]))); out.append(c["comref"])
        out.append(struct.pack("<III", c["domain"], c["n_rows"], len(c["point"])))
        out.extend(c["point"])
        out.append(bytes([c["tag"]]))
        out.append(c["eval"])
    open(path, "wb").write(b"".join(out))


def tamper(path, off, delta=1):
    with open(path, "r+b") as f:
        f.seek(off, 0 if off >= 0 else 2)
        c = f.read(1)[0]
        f.seek(-1, 1)
        f.write(bytes([(c + delta) & 0xFF]))


total = 0
fails = []


def record(name, ok, want, got):
    global total
    total += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} -> expected {want}, got {got}")
    if not ok:
        fails.append(name)


def main():
    run = sys.argv[1]
    seed0 = PW.run_seed_of(run)
    acc = os.path.join(run, "acc")
    vacc = os.path.join(run, "vacc")
    genspec = [f"{g}={rel}" for g, rel in sorted(GENS.items())]
    env_base = {"ZKOB_REQUIRE_RELATIVE_COMREF": "1"}

    def batch_verify_locus(extra_env=None):
        """zkob_batchopen verify against the existing honest vacc; returns
        'accept' or the REJECT locus."""
        e = dict(os.environ); e.update(env_base)
        if extra_env:
            e.update(extra_env)
        p = subprocess.run([os.path.join(ZKLLM, "zkob_batchopen"), "verify",
                            "acc", "vacc", seed0, Q] + genspec,
                           cwd=run, env=e, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        if p.returncode == 0:
            return "accept"
        for line in p.stdout.splitlines():
            if line.startswith("REJECT[opening_batch."):
                return line.split("REJECT[opening_batch.", 1)[1].split("]", 1)[0]
        return f"error({p.returncode})"

    def expect_batch(name, want, extra_env=None):
        got = batch_verify_locus(extra_env)
        record(name, got == want, want, got)

    def batch_prove(evil=None, seed=None):
        e = dict(os.environ)
        if evil is not None:
            e["ZKOB_EVIL"] = str(evil)
        p = subprocess.run([os.path.join(ZKLLM, "zkob_batchopen"), "prove",
                            "acc", seed or seed0, Q] + genspec,
                           cwd=run, env=e, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        if p.returncode != 0:
            sys.stderr.write(p.stdout)
            raise RuntimeError("batch prove failed")

    # ---- honest full verify (drivers + edges + batch); fills vacc ----------
    print("=== honest (full pair verify) ===")
    ok, _, _ = PW.verify(run, vtag="vacc", quiet=True)
    record("honest full pair verify", ok, "accept", "accept" if ok else "reject")
    # at-scale convention cross-checks ON for one verify (fold + slow IPA path)
    print("=== at-scale convention cross-checks (fold xcheck + slow-IPA path) ===")
    got = batch_verify_locus({"ZKOB_FOLD_CROSSCHECK": "1"})
    record("batched-fold == fold_chain element-exact at pair scale (and ACCEPT)",
           got == "accept", "accept", got)
    got = batch_verify_locus({"ZKOB_SLOW_FOLD": "1", "ZKOB_SLOW_IPA": "1"})
    record("slow-path (Stage-A fold_chain + header ipa_verify) same verdict",
           got == "accept", "accept", got)

    # backups of everything the battery mutates
    honest_claims = cl_parse(acc + "/claims.bin")
    honest_vclaims = cl_parse(vacc + "/claims.bin")
    bak = os.path.join(run, "battery_bak")
    os.makedirs(bak, exist_ok=True)
    batch_files = ["batch_sumcheck.bin", "batch_vfin.bin"] + \
        [f"ipa_batch_{g}.bin" for g in sorted(GENS)]
    for f in batch_files + ["claims.bin", "drvstates.bin"]:
        shutil.copyfile(os.path.join(acc, f), os.path.join(bak, f))
    shutil.copyfile(vacc + "/drvstates.bin", bak + "/v_drvstates.bin")

    def restore_lists():
        cl_write(acc + "/claims.bin", honest_claims)
        cl_write(vacc + "/claims.bin", honest_vclaims)
        shutil.copyfile(bak + "/drvstates.bin", acc + "/drvstates.bin")
        shutil.copyfile(bak + "/v_drvstates.bin", vacc + "/drvstates.bin")

    def restore_batch():
        for f in batch_files:
            shutil.copyfile(os.path.join(bak, f), os.path.join(acc, f))

    # sanity: byte-identical round trip of the codec
    cl_write(acc + "/claims.bin", honest_claims)
    assert open(acc + "/claims.bin", "rb").read() == \
        open(bak + "/claims.bin", "rb").read(), "claims codec not byte-faithful"

    # ---- BO-1a / BO-1b (the F8 split) --------------------------------------
    print("=== BO-1a / BO-1b (F8 locus split) ===")
    falsified = cl_parse(bak + "/claims.bin")
    falsified[1]["eval"] = bytes([falsified[1]["eval"][0] ^ 1]) + falsified[1]["eval"][1:]
    cl_write(acc + "/claims.bin", falsified)
    cl_write(vacc + "/claims.bin", falsified)
    batch_prove(evil=1)
    expect_batch("BO-1a false eval (gate fc X), honest-procedure batch prover", "round0")
    batch_prove(evil=2)
    # last tensor in canonical claim order = lm_head.rescaling:m (gen32768)
    expect_batch("BO-1b fully adaptive prover (forged v' passes G3)", "ipa32768")
    restore_lists()
    restore_batch()
    expect_batch("restored after BO-1", "accept")

    # ---- claims_match family ------------------------------------------------
    print("=== BO-2/BO-3/BO-10 + cross-driver drop (claims_match) ===")
    drop = [c for c in honest_claims if not c["id"].startswith(b"lm_head.rescaling")]
    assert len(drop) == 9
    cl_write(acc + "/claims.bin", drop)
    expect_batch("cross-driver claim-drop (rescale's 3 claims omitted, fc's present)",
                 "claims_match")
    one = honest_claims[:5] + honest_claims[6:]
    cl_write(acc + "/claims.bin", one)
    expect_batch("BO-2 single claim omitted (gate rescale m)", "claims_match")
    swp = [dict(c) for c in honest_claims]
    swp[3]["comref"], swp[4]["comref"] = swp[4]["comref"], swp[3]["comref"]  # gate A <-> rem
    cl_write(acc + "/claims.bin", swp)
    expect_batch("BO-3 comrefs of two same-shape claims swapped (list only)", "claims_match")
    dup = honest_claims + [honest_claims[-1]]
    cl_write(acc + "/claims.bin", dup)
    expect_batch("BO-10 duplicate claim appended", "claims_match")
    ro = list(honest_claims)
    ro[0], ro[1] = ro[1], ro[0]
    cl_write(acc + "/claims.bin", ro)
    expect_batch("reordered claims in prover list", "claims_match")
    restore_lists()
    expect_batch("restored after claims_match family", "accept")

    # ---- BO-4 (rho from doctored list, honest list shipped) ----------------
    print("=== BO-4 ===")
    batch_prove(evil=3)
    expect_batch("BO-4 batch transcript built over doctored list", "round0")
    restore_batch()

    # ---- BO-5 / BO-6 / BO-8 byte tampers ------------------------------------
    print("=== BO-5/BO-6/BO-8 (artifact byte tampers) ===")
    bs = acc + "/batch_sumcheck.bin"
    tamper(bs, 16 + 32); expect_batch("BO-5 round-0 p(1) tamper", "round0"); tamper(bs, 16 + 32, -1)
    tamper(bs, 16 + 3 * 32); expect_batch("BO-5 round-1 p(0) tamper", "round1"); tamper(bs, 16 + 3 * 32, -1)
    tamper(bs, 4); expect_batch("BO-8 batch_sumcheck n_claims field", "xcheck"); tamper(bs, 4, -1)
    bv = acc + "/batch_vfin.bin"
    tamper(bv, -32); expect_batch("BO-6 vfin tamper (last v')", "terminal"); tamper(bv, -32, -1)
    for g in sorted(GENS):
        ip = acc + f"/ipa_batch_{g}.bin"
        tamper(ip, -32)
        expect_batch(f"BO-8 ipa_batch_{g} a_final tamper", f"ipa{g}")
        tamper(ip, -32, -1)
    tamper(acc + f"/ipa_batch_4096.bin", 8 + 16)
    expect_batch("BO-8 ipa_batch_4096 round-0 L tamper", "ipa4096")
    tamper(acc + f"/ipa_batch_4096.bin", 8 + 16, -1)
    tamper(acc + "/claims.bin", 12 + 4)   # first claim's id byte
    expect_batch("BO-8 claims.bin id byte tamper (file never parsed, F5)", "claims_match")
    tamper(acc + "/claims.bin", 12 + 4, -1)
    tamper(vacc + "/drvstates.bin", -1)
    expect_batch("BO-8 verifier-side drvstate divergence", "round0")
    tamper(vacc + "/drvstates.bin", -1, -1)
    expect_batch("restored after byte tampers", "accept")

    # ---- BO-7 (vars lie, both lists) ----------------------------------------
    print("=== BO-7 ===")
    lie = [dict(c) for c in honest_claims]
    lie[2] = dict(lie[2]); lie[2]["point"] = lie[2]["point"] + [b"\x01" + b"\x00" * 31]
    cl_write(acc + "/claims.bin", lie)
    cl_write(vacc + "/claims.bin", lie)
    expect_batch("BO-7 vars lie (junk-padded point, BOTH lists)", "shape")
    restore_lists()

    # ---- F3/F10 structural shape (covert-channel pin) ------------------------
    print("=== F3/F10 (trailing rows / truncation, before any fold) ===")
    comA = os.path.join(run, "proofs/layer0.mlp.gate_proj.rescaling/com_A.bin")
    shutil.copyfile(comA, bak + "/com_A.bin")
    with open(comA, "ab") as f:
        f.write(open(comA, "rb").read(144))   # one extra trailing row point
    expect_batch("F10 com_A with extra trailing row (the F3 covert channel)", "shape")
    shutil.copyfile(bak + "/com_A.bin", comA)
    b = open(comA, "rb").read()
    open(comA, "wb").write(b[:-144])
    expect_batch("F10 truncated com_A", "shape")
    shutil.copyfile(bak + "/com_A.bin", comA)
    expect_batch("restored after F10", "accept")

    # ---- BO-9b (batch artifacts from a different statement) ------------------
    print("=== BO-9b (foreign-seed batch artifacts; full replay is toy-covered) ===")
    batch_prove(seed=seed0 + ":otherstatement")
    expect_batch("BO-9b batch artifacts proven under a different run_seed", "round0")
    restore_batch()

    # ---- BO-11 (substituted com file post-prove, honest hash) ----------------
    print("=== BO-11 ===")
    comrem = os.path.join(run, "proofs/layer0.mlp.gate_proj.rescaling/com_rem.bin")
    shutil.copyfile(comA, bak + "/com_A2.bin")
    shutil.copyfile(comrem, comA)   # same-shape substitute, self-consistent bytes
    expect_batch("BO-11 substituted com_A (verifier hashes substituted bytes)", "round0")
    shutil.copyfile(bak + "/com_A2.bin", comA)
    expect_batch("restored after BO-11", "accept")

    # ---- BO-12 (group structure) ---------------------------------------------
    print("=== BO-12 ===")
    ip4096 = acc + "/ipa_batch_4096.bin"
    os.rename(ip4096, ip4096 + ".bak")
    expect_batch("BO-12 missing domain-4096 group IPA", "group_missing")
    os.rename(ip4096 + ".bak", ip4096)

    # ---- EvalVar / empty / comref policy --------------------------------------
    print("=== EvalVar / empty / relative-comref policy ===")
    cv = [dict(c) for c in honest_claims]
    cv[0] = dict(cv[0]); cv[0]["tag"] = 1; cv[0]["eval"] = b"\x11" * 144
    cl_write(acc + "/claims.bin", cv)
    cl_write(vacc + "/claims.bin", cv)
    expect_batch("Committed EvalVar claim (path closed until Stage D)", "evalvar")
    cl_write(acc + "/claims.bin", [])
    cl_write(vacc + "/claims.bin", [])
    expect_batch("n_claims == 0", "empty")
    ab = [dict(c) for c in honest_claims]
    ab[0] = dict(ab[0])
    ab[0]["comref"] = os.path.join(run, ab[0]["comref"].decode()).encode()
    cl_write(acc + "/claims.bin", ab)
    cl_write(vacc + "/claims.bin", ab)
    expect_batch("absolute comref under relative-comref policy", "shape")
    restore_lists()
    expect_batch("restored after policy cases", "accept")

    # ---- chain-edge tamper (harness edge check is the catcher) ----------------
    print("=== chain-edge tamper ===")
    rs = "layer0.mlp.gate_proj.rescaling"
    obdir = os.path.join(run, "proofs", rs)
    shutil.move(obdir, obdir + ".bak")
    os.makedirs(obdir)
    # re-prove gate rescale HONESTLY over a doctored input (one int64 bumped):
    # internally consistent driver artifacts, but com_X no longer byte-equals
    # the fc run's com_Y — only the chain edge can catch it.
    ybytes = bytearray(open(os.path.join(run, PW.PAIRS[0]["Y"]), "rb").read())
    (v,) = struct.unpack_from("<q", ybytes, 8 * 100)
    struct.pack_into("<q", ybytes, 8 * 100, v + (1 << PW.PAIRS[0]["log_sf"]))
    open(os.path.join(run, "data/gate_Y_doctored.i64.bin"), "wb").write(ybytes)
    eacc = os.path.join(run, "acc_edge")
    os.makedirs(eacc, exist_ok=True)
    p = PW.PAIRS[0]
    rc, out, _ = PW.sh(run, [os.path.join(ZKLLM, "zkob_rescale"), "prove",
                             f"proofs/{rs}", f"{seed0}:{rs}",
                             "data/gate_Y_doctored.i64.bin", str(PW.SEQ), str(p["OUT"]),
                             str(p["log_sf"]), p["gen_out"], Q,
                             "--claims", "acc_edge", rs])
    assert rc == 0, out
    ok, _, reasons = PW.verify(run, vtag="vacc_edge", quiet=True)
    edge_named = (not ok) and any("chain edge" in r for r in reasons)
    record("chain-edge tamper (rescale re-proven over doctored Y; edge check fires)",
           edge_named, "chain-edge reject", "chain-edge reject" if edge_named
           else ("accept(!!)" if ok else str(reasons)))
    shutil.rmtree(obdir)
    shutil.move(obdir + ".bak", obdir)
    shutil.rmtree(eacc, ignore_errors=True)
    shutil.rmtree(os.path.join(run, "vacc_edge"), ignore_errors=True)
    os.remove(os.path.join(run, "data/gate_Y_doctored.i64.bin"))

    # ---- final restored full verify -------------------------------------------
    print("=== final restored (full pair verify) ===")
    ok, _, _ = PW.verify(run, vtag="vacc_final", quiet=True)
    record("final restored full pair verify", ok, "accept", "accept" if ok else "reject")

    print(f"PAIR-BO BATTERY: {total - len(fails)}/{total} "
          + ("ALL PASS" if not fails else "FAIL: " + "; ".join(fails)))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
