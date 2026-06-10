"""SEPARATE verifier process for a stage-1 run.

Reads ONLY: public.json, registration/ (hash-checked against public.json),
proofs/, and the frozen harness manifest. NEVER reads data/ (witness files),
never invokes a prove mode, re-derives run_seed = sha256(public.json) itself.

Checks, in order:
 1. registration hashes (gens, registered weight commitments, input + its
    commitment, swiglu table) vs public.json — mismatch => REJECT and STOP;
 2. every covered obligation's zkob_* verify with registered public inputs;
 3. every chain edge (byte-equality; skip edges via zkob_skip point checks);
 4. statement obligations.
Emits transcript.json (harness format + explicit skipped/details sections).

Run: /root/int-model-env/bin/python verify_walk.py <run_dir> [--out transcript.json]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = args.run_dir
    out_path = args.out or os.path.join(run_dir, "transcript.json")
    public = json.load(open(os.path.join(run_dir, "public.json")))
    cst = public["constants"]
    run_seed = C.run_seed_of(run_dir)
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

    # ---- 1. registration hash checks ------------------------------------
    reg_ok = True
    checks = [(f"gens.{k}", P[k], public["gens"][k]) for k in ("gen1024", "gen4096", "q")]
    checks += [(f"weights.{wid}", C.wpath(run_dir, wid, "com"), h)
               for wid, h in public["registered_weight_commitments"].items()]
    checks += [("input.file", P["input"], public["input"]["sha256"]),
               ("input.commitment", P["com_input"], public["input"]["commitment_sha256"]),
               ("tables.swiglu", P["table"], public["tables"]["swiglu-table.bin"])]
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
        transcript = {
            "verdict": "REJECT",
            "checked": [],
            "skipped": skipped,
            "details": details,
            "registration": {"ok": False, "n_hashes": len(checks)},
            "note": "registration check failed; obligation verifies not run (untrusted registration)",
        }
        json.dump(transcript, open(out_path, "w"), indent=2)
        print("VERDICT: REJECT (registration)")
        sys.exit(1)

    # ---- 2. per-obligation driver verifies -------------------------------
    spec, edges = C.walk_spec(run_dir)
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

    # commitment_opening ids are discharged by the same fc proof as the matmul
    # (IPA(W) against the REGISTERED commitment, absorbed into the transcript)
    for mid in list(C.covered_ids()):
        if mid.endswith(".commitment_opening"):
            mm = mid.replace(".commitment_opening", ".matmul")
            ok = details.get(mm, {}).get("ok", False)
            d = details.setdefault(mid, {"ok": True, "reasons": []})
            d["ok"] = ok
            d["reasons"].append(f"discharged by {mm}: IPA(W) vs registered commitment "
                                f"({'ACCEPT' if ok else 'REJECT'})")

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

    # ---- 4. statement obligations ----------------------------------------
    drivers_ok = all(details.get(m, {}).get("ok", False) for m in C.covered_ids()
                     if m.startswith("layer"))
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
        "timing": {**timing, "total_verify_wall_s": round(time.time() - t_start, 2)},
    }
    json.dump(transcript, open(out_path, "w"), indent=2)
    print(f"checked={len(checked)} rejected={len(rejected)} skipped={len(skipped)}")
    print(f"VERDICT: {verdict}  ({out_path})")
    sys.exit(0 if verdict == "ACCEPT" else 1)


if __name__ == "__main__":
    main()
