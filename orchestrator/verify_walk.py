"""SEPARATE verifier process for a stage-3 run (full forward pass + head).

Reads ONLY: public.json, registration/ (hash-checked against public.json),
proofs/, and the frozen harness manifest. NEVER reads data/ (witness files),
never invokes a prove mode, re-derives run_seed = sha256(public.json) itself.

Checks, in order:
 1. registration hashes (gens incl. gen64 + gen32768, registered weight
    commitments incl. final_norm.g + lm_head, input + its commitment, the
    registered served tokens tstar.i32.bin — edge L2 of STAGE3 §3.4 — and ALL
    public tables: swiglu, rope cos/sin, softmax exp) vs public.json —
    mismatch => REJECT and STOP (fail-closed: a tampered t* never reaches a
    driver);
 1b. structure: every registration/ path the walk consumes is hash-pinned,
    and proofs/ holds only contained regular files — violation => REJECT and STOP;
 2. every covered obligation's zkob_* verify with registered public inputs
    (incl. the attention chain: rope/headslice/headmerge + per-head fc,
    rescale, softmax — com_W for the per-head matmuls = the headslice's slice
    commitment files, passed as path args and absorbed by the driver; and the
    stage-3 head: final_norm trio, lm_head fc vs the REGISTERED lm_head com,
    lm_head rescale, zkob_rowmax vpad with the verifier's own registered t*);
 3. every chain edge: byte-equality (incl. ALL of ROPE_ATTENTION_DESIGN §7.4
    A1..A15 — A15 closes edge S1's former open attention boundary — and
    STAGE3 §3.4 F0..F6 + L1), skip edges via zkob_skip point checks, and com_W
    path bindings (A7/A12: structural check that the fc verify argv references
    exactly the slice commitment);
 4. statement obligations. statement.logit_binding enters `checked` iff the
    rowmax verify ACCEPTs AND edge L1 holds AND the registration hash check
    covered tstar.i32.bin (§3.3 — the last is fail-closed in step 1).
A manifest id enters `checked` only if ALL its composed sub-runs and ALL its
edges pass (ROPE_ATTENTION_DESIGN §9.5).
Emits transcript.json (harness format + explicit skipped/details sections).

Run: /root/int-model-env/bin/python verify_walk.py <run_dir> [--out transcript.json]
"""
import argparse
import json
import os
import stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # MINOR-1: drivers run with cwd=/root/zkllm; absolutize so the hash check,
    # byte edges and every driver resolve the SAME files regardless of caller cwd.
    run_dir = os.path.abspath(args.run_dir)
    out_path = args.out or os.path.join(run_dir, "transcript.json")
    # MINOR-2: read public.json ONCE; constants/hash dict and run_seed come from
    # the same bytes (no second open that a swap could decouple).
    pub_bytes = open(os.path.join(run_dir, "public.json"), "rb").read()
    public = json.loads(pub_bytes)
    cst = public["constants"]
    run_seed = C.run_seed_of_bytes(pub_bytes)
    P = C.reg_paths(run_dir)
    t_start = time.time()
    print(f"== verify_walk {run_dir} ==\n  run_seed (re-derived) = {run_seed}")

    details = {}
    timing = {}
    skipped = C.skipped_ids()

    def fail(mid, reason):
        d = details.setdefault(mid, {"ok": True, "reasons": []})
        d["ok"] = False
        d["reasons"].append(reason)
        print(f"  REJECT {mid}: {reason}")

    def note(mid, reason):
        details.setdefault(mid, {"ok": True, "reasons": []})["reasons"].append(reason)

    def stop_reject(why, n_hashes):
        """Fail-closed STOP before any driver runs (untrusted run structure)."""
        for d in details.values():
            d["reason"] = "; ".join(d.pop("reasons"))
        transcript = {
            "verdict": "REJECT",
            "checked": [],
            "skipped": skipped,
            "details": details,
            "registration": {"ok": False, "n_hashes": n_hashes},
            "note": f"{why}; obligation verifies not run",
        }
        json.dump(transcript, open(out_path, "w"), indent=2)
        print(f"VERDICT: REJECT ({why})")
        sys.exit(1)

    # ---- 1. registration hash checks ------------------------------------
    reg_ok = True
    checks = [(f"gens.{k}", P[k], public["gens"][k])
              for k in sorted(public["gens"])]                 # gen64/gen1024/gen4096/q
    checks += [(f"weights.{wid}", C.wpath(run_dir, wid, "com"), h)
               for wid, h in public["registered_weight_commitments"].items()]
    checks += [("input.file", P["input"], public["input"]["sha256"]),
               ("input.commitment", P["com_input"], public["input"]["commitment_sha256"]),
               # STAGE3 §3.4 edge L2: the registered served tokens. Fail-closed
               # here means a tampered t* never reaches the rowmax driver.
               ("served_tokens.file", P["tstar"], public["served_tokens"]["sha256"])]
    checks += [(f"tables.{fname}", os.path.join(P["reg"], fname), h)
               for fname, h in sorted(public["tables"].items())]  # swiglu + rope cos/sin + exp
    for label, path, want in checks:
        got = C.sha256_file(path) if os.path.exists(path) else "<missing>"
        if got != want:
            reg_ok = False
            fail("statement.registered_weight_hash", f"hash mismatch: {label}")
    if reg_ok:
        note("statement.registered_weight_hash",
             f"{len(checks)} registered files re-hashed and matched public.json")
        print(f"  registration: {len(checks)} hashes OK")
    else:
        stop_reject("registration check failed (untrusted registration)", len(checks))

    # ---- 1b. structural checks (before any driver runs) ------------------
    spec, edges = C.walk_spec(run_dir)

    # MINOR-5: every registration/ path the walk consumes (driver argv or chain
    # edge) must have been hash-pinned by the step-1 checks above. Guards
    # against a future walk addition silently consuming an unpinned file.
    reg_root = os.path.realpath(P["reg"])
    pinned = {os.path.realpath(p) for _, p, _ in checks}

    def under_reg(p):
        rp = os.path.realpath(p)
        return rp if rp == reg_root or rp.startswith(reg_root + os.sep) else None

    used_reg = set()
    for subs in spec.values():
        for s in subs.values():
            for c in (s["verify"] or []):
                rp = under_reg(c) if isinstance(c, str) and os.sep in c else None
                if rp:
                    used_reg.add(rp)
    for e in edges:
        # file-path positions per kind: byte (a, b); skip (a, b, z); path (file,)
        for p in (e[3:5] if e[0] == "byte" else e[3:6] if e[0] == "skip" else e[3:4]):
            rp = under_reg(p)
            if rp:
                used_reg.add(rp)
    unpinned = sorted(used_reg - pinned)
    if unpinned:
        for p in unpinned:
            fail("statement.registered_weight_hash",
                 f"registration path consumed by the walk but NOT hash-pinned: {p}")
        stop_reject("walk consumes unpinned registration path(s)", len(checks))
    note("statement.registered_weight_hash",
         f"all {len(used_reg)} registration paths consumed by the walk are hash-pinned")

    # MINOR-6: reject non-regular files (FIFOs, devices, sockets) and symlink
    # escapes anywhere under proofs/ before the verifier or a driver opens them.
    proofs_root = os.path.join(run_dir, "proofs")
    real_proofs = os.path.realpath(proofs_root)
    fs_bad = []
    for dirpath, dirnames, filenames in os.walk(proofs_root, followlinks=False):
        for name, is_dir in [(n, True) for n in dirnames] + [(n, False) for n in filenames]:
            p = os.path.join(dirpath, name)
            rp = os.path.realpath(p)
            if not (rp == real_proofs or rp.startswith(real_proofs + os.sep)):
                fs_bad.append(f"{p}: resolves outside proofs/ ({rp})")
                continue
            try:
                st = os.stat(p)  # follows symlinks: check the file actually opened
            except OSError as exc:
                fs_bad.append(f"{p}: stat failed ({exc})")
                continue
            if is_dir and not stat.S_ISDIR(st.st_mode):
                fs_bad.append(f"{p}: not a directory")
            elif not is_dir and not stat.S_ISREG(st.st_mode):
                fs_bad.append(f"{p}: not a regular file")
    if fs_bad:
        for b in fs_bad:
            fail("statement.registered_weight_hash", f"proofs/ filesystem hygiene: {b}")
        stop_reject("non-regular or escaping file under proofs/", len(checks))

    # ---- 2. per-obligation driver verifies -------------------------------
    for mid, subs in spec.items():
        for sub, s in subs.items():
            if s["verify"] is None:
                continue   # skip-connection ids: checked purely via edges below
            cmd = list(s["verify"])
            cmd[3] = f"{run_seed}:{s['seed_id']}"
            cmd = [str(cst["C_eps"]) if c is None else c for c in cmd]
            try:
                ok, dt, out = C.run_driver(cmd, f"verify {mid} [{sub}]", expect_reject_ok=True)
            except RuntimeError as e:
                ok, dt, out = False, 0.0, str(e)
            timing[f"{mid}[{sub}]"] = round(dt, 3)
            if ok:
                note(mid, f"{sub}: driver verify ACCEPT")
            else:
                tail = [ln for ln in out.splitlines() if "REJECT" in ln][-1:] or [out[-200:]]
                fail(mid, f"{sub}: driver verify REJECT ({tail[0].strip()})")

    # ---- 3. chain edges ---------------------------------------------------
    edge_results = []
    for e in edges:
        kind, owner, label = e[0], e[1], e[2]
        if kind == "byte":
            a, b = e[3], e[4]
            ok = (os.path.exists(a) and os.path.exists(b)
                  and open(a, "rb").read() == open(b, "rb").read())
            if not ok:
                fail(owner, f"chain byte-equality FAILED: {label}")
        elif kind == "path":
            # com_W path binding (§7.4 A7/A12): the named sub's verify argv must
            # reference exactly this slice commitment file (and it must exist) —
            # the driver absorbs the file, so a divergent operand already
            # rejects at the transcript level; this check pins the wiring.
            fpath, mid, sub = e[3], e[4], e[5]
            argv = spec.get(mid, {}).get(sub, {}).get("verify") or []
            ok = fpath in argv and os.path.exists(fpath)
            if not ok:
                fail(owner, f"com_W path binding FAILED: {label}")
        else:  # Pedersen point check: com_Z == com_A + com_B (Jacobian reps differ)
            a, b, z = e[3], e[4], e[5]
            if not (os.path.exists(a) and os.path.exists(b) and os.path.exists(z)):
                ok = False
            else:
                try:
                    ok, dt, _ = C.run_driver([C.drv("zkob_skip"), "verify", a, b, z],
                                             f"edge {label}", expect_reject_ok=True)
                    timing[f"edge:{owner}"] = round(dt, 3)
                except RuntimeError:
                    ok = False
            if not ok:
                fail(owner, f"skip point-equality FAILED: {label}")
        if ok:
            note(owner, f"edge OK: {label}")
        edge_results.append({"edge": label, "ok": ok})

    # commitment_opening ids are discharged by the same fc proof as the matmul
    # (IPA(W) against the REGISTERED commitment, absorbed into the transcript).
    # Discharged AFTER the edge phase (MINOR-4) so a matmul whose driver passed
    # but whose chain edge failed also drags its opening id out of `checked`.
    for mid in list(C.covered_ids()):
        if mid.endswith(".commitment_opening"):
            mm = mid.replace(".commitment_opening", ".matmul")
            ok = details.get(mm, {}).get("ok", False)
            d = details.setdefault(mid, {"ok": True, "reasons": []})
            d["ok"] = ok
            d["reasons"].append(f"discharged by {mm}: IPA(W) vs registered commitment "
                                f"({'ACCEPT' if ok else 'REJECT'})")

    # ---- 4. statement obligations ----------------------------------------
    # statement.logit_binding (§3.3): ok iff its rowmax verify ACCEPTed AND
    # edge L1 held (both already folded into its details above) AND the
    # registration hash check covered tstar.i32.bin (fail-closed in step 1 —
    # reaching this point implies it passed). Record the semantics explicitly.
    lb = details.setdefault("statement.logit_binding", {"ok": True, "reasons": []})
    lb["reasons"].append(
        "t*[i] = argmax_{v<32000} logits[i,v] bound at all 1024 positions "
        "(zkob_rowmax vpad T-BIND vs the verifier's own registered, hash-pinned "
        "tstar.i32.bin; logits grid chained by edge L1; t* pinned into run_seed "
        "via public.json — STAGE3 §3.3)")

    drivers_ok = all(details.get(m, {}).get("ok", False) for m in C.covered_ids()
                     if m.startswith(("layer", "final_norm", "lm_head")))
    d = details.setdefault("statement.prompt_binding", {"ok": True, "reasons": []})
    d["ok"] = drivers_ok
    d["reasons"].append(
        "run_seed = sha256(public.json) re-derived by this verifier (binds input digest, "
        "registered hashes, constants — prompt_token_ids analog for the random-input "
        "pipeline); every FS transcript above verified under seeds derived from it"
        if drivers_ok else "dependent obligation verifies failed")

    # ---- transcript --------------------------------------------------------
    checked = [m for m in C.covered_ids() if details.get(m, {}).get("ok", False)]
    rejected = [m for m in C.covered_ids() if not details.get(m, {}).get("ok", False)]
    verdict = "ACCEPT" if not rejected else "REJECT"
    for m, d in details.items():
        d["reason"] = "; ".join(d.pop("reasons"))
    transcript = {
        "verdict": verdict,
        "checked": checked,
        "rejected": rejected,
        "skipped": skipped,
        "details": details,
        "chain_edges": edge_results,
        "registration": {"ok": True, "n_hashes": len(checks)},
        # harness-reader echo (§3.3): the registered served tokens this
        # transcript's logit binding was verified against.
        "served_tokens_sha256": public["served_tokens"]["sha256"],
        "timing": {**timing, "total_verify_wall_s": round(time.time() - t_start, 2)},
    }
    json.dump(transcript, open(out_path, "w"), indent=2)
    print(f"checked={len(checked)} rejected={len(rejected)} skipped={len(skipped)}")
    print(f"VERDICT: {verdict}  ({out_path})")
    sys.exit(0 if verdict == "ACCEPT" else 1)


if __name__ == "__main__":
    main()
