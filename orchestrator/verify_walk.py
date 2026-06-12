"""SEPARATE verifier process for a stage-3 / faithful-arch-v1 run (full
forward pass + head). The submission mode is part of the public STATEMENT
(public.json "submission"/"headmerge_perm" — inside the run_seed hash):
faithful-arch-v1 switches the attention chain to rowmax (causal) + softmax8
per head with edges RM1/RM2/SX8a/SX8b, headmerge concat, and o_proj fc +
rescale (edges O1/O2/O3, discharging the six o_proj.* covered-waived ids);
an unknown submission value or a headmerge_perm inconsistent with the
submission's pinned mode is a fail-closed REJECT before any driver runs.

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

Stage D (public.json "weight_privacy": "hiding", inside run_seed): the 20
registered-weight-opening verifies additionally run --wpriv (their W/g claim
is recomputed as a Committed record into the verifier's weight accumulator),
q.bin must be the 2-slot [Q, H] file, and ONE zkob_batchopen wverify weight
sub-batch (wclaims_match + committed-round sumcheck + blinded ZK IPAs) must
ACCEPT with every registered weight comref opened HIDDEN there (and never
Plain in a public sub-batch) — all gating the verdict like the public
sub-batches. Fail-closed before any driver on an unknown mode, a non-batched
transport, or a 1-slot q.bin.
Emits transcript.json (harness format + explicit skipped/details sections).

Run: /root/int-model-env/bin/python verify_walk.py <run_dir> [--out transcript.json]
"""
import argparse
import json
import os
import shutil
import stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-pool", action="store_true",
                    help="batched mode: one subprocess per driver invocation "
                         "(pre-C2 transport) instead of the single-process "
                         "serve-worker pool")
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

    # ---- 0. submission mode (part of the STATEMENT, inside the run_seed) --
    submission = public.get("submission", "baseline")
    if submission not in C.SUBMISSIONS:
        fail("statement.registered_weight_hash",
             f"unknown submission {submission!r} in public.json")
        stop_reject(f"unknown submission {submission!r}", 0)
    perm = public.get("headmerge_perm", "pi157")
    if perm != C.PERM_FOR[submission]:
        # PHASE0 §21 MINOR-7: the verifier passes headmerge_perm as argv; a
        # statement whose perm contradicts its submission mode is incoherent.
        fail("statement.registered_weight_hash",
             f"headmerge_perm={perm!r} inconsistent with submission {submission!r} "
             f"(pinned: {C.PERM_FOR[submission]!r})")
        stop_reject("public.json submission/headmerge_perm inconsistent", 0)
    # Stage C2: the proof-transport mode is part of the statement (inside the
    # run_seed hash). "batched" switches every driver verify to claim mode
    # (recompute claims into the verifier accumulator, NO inline IPAs) and
    # demands one zkob_batchopen discharge per sub-batch — fail-closed: an
    # unknown mode rejects before any driver runs.
    transport = public.get("transport", "inline")
    if transport not in C.TRANSPORTS:
        fail("statement.registered_weight_hash",
             f"unknown transport {transport!r} in public.json")
        stop_reject(f"unknown transport {transport!r}", 0)
    batched = transport == "batched"
    if batched:
        # Stage-B flag 7 + measurement hygiene: forgery/slow-path envs must
        # never be active in a production batched verify.
        for v in ("ZKOB_EVIL", "ZKOB_SLOW_FOLD", "ZKOB_SLOW_IPA"):
            assert v not in os.environ, f"{v} set in a production batched verify"
        # Stage-B comref canonicalization: reject absolute comrefs in the batch
        os.environ["ZKOB_REQUIRE_RELATIVE_COMREF"] = "1"
    # Stage D: weight privacy is part of the statement (inside the run_seed).
    # "hiding" switches every registered-weight-opening driver verify to
    # --wpriv (its W/g claim is recomputed as a Committed record into the
    # verifier's weight accumulator) and demands the zkob_batchopen wverify
    # weight sub-batch on top of the public discharges — fail-closed: an
    # unknown mode, or weight privacy without the batched transport, rejects
    # before any driver runs.
    wpriv = public.get("weight_privacy")
    if wpriv is not None and wpriv not in C.WEIGHT_PRIVACY_MODES:
        fail("statement.registered_weight_hash",
             f"unknown weight_privacy {wpriv!r} in public.json")
        stop_reject(f"unknown weight_privacy {wpriv!r}", 0)
    if wpriv and not batched:
        fail("statement.registered_weight_hash",
             "weight_privacy requires the batched transport")
        stop_reject("weight_privacy without batched transport", 0)
    print(f"  submission: {submission} (headmerge {perm}; transport {transport}"
          + (f"; weight privacy: {wpriv}" if wpriv else "") + ")")

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

    # Stage D: under weight privacy the blinding generator H IS q.bin slot 1 —
    # the (hash-pinned, just re-checked) q.bin must be the 2-slot [Q, H] file,
    # or every "hiding" commitment claim is vacuous. Fail-closed before drivers.
    if wpriv:
        want = C.WPRIV_Q_SLOTS * C.G1_POINT_BYTES
        qsz = os.path.getsize(P["q"])
        if qsz != want:
            fail("statement.registered_weight_hash",
                 f"weight_privacy={wpriv!r} but q.bin is {qsz} bytes, not the "
                 f"2-slot [Q, H] file ({want})")
            stop_reject("weight_privacy without a 2-slot q.bin (no H)", len(checks))
        note("statement.registered_weight_hash",
             "q.bin is the 2-slot [Q, H] file (H = slot 1, hash-pinned)")

    # ---- 1b. structural checks (before any driver runs) ------------------
    spec, edges = C.walk_spec(run_dir, submission)

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
    # Stage C2 single-process transport: in batched mode every driver verify,
    # the batch discharges and the skip edges run through ONE persistent
    # serve-mode worker per driver binary (CUDA init paid ~12x, not ~235x;
    # same zkw_run1 entry, same FS schedules, same plan order).
    pool = C.DriverPool(run_dir) if batched and not args.no_pool else None

    def drive(cmd, label, cwd=None):
        if pool is not None:
            return pool.run(cmd, label, expect_reject_ok=True)
        return C.run_driver(cmd, label, expect_reject_ok=True, cwd=cwd)

    def run_verify(mid, sub, s, suffix=(), cwd=None, want=None):
        cmd = list(s["verify"])
        cmd[3] = f"{run_seed}:{s['seed_id']}"
        cmd = [str(cst["C_eps"]) if c is None else c for c in cmd]
        cmd += list(suffix)
        if cwd:
            cmd = C.rel_argv(cmd, cwd)
        try:
            ok, dt, out = drive(cmd, f"verify {mid} [{sub}]", cwd=cwd)
            if ok and want and want not in out:
                ok = False
        except RuntimeError as e:
            ok, dt, out = False, 0.0, str(e)
        timing[f"{mid}[{sub}]"] = round(dt, 3)
        if ok:
            note(mid, f"{sub}: driver verify "
                      + ("ACCEPT-conditional (claims recomputed; gated on opening_batch)"
                         if want else "ACCEPT"))
        else:
            tail = [ln for ln in out.splitlines() if "REJECT" in ln][-1:] or [out[-200:]]
            fail(mid, f"{sub}: driver verify REJECT ({tail[0].strip()})")

    if batched:
        # claim mode: plan order IS the canonical claim order both sides must
        # reproduce (claims_match byte-compares per sub-batch); each driver
        # recomputes its claims from its own FS replay into the verifier
        # accumulator; NO inline IPAs run anywhere in this loop.
        plan, n_batches = C.claim_plan(run_dir, submission)
        # the plan must cover EXACTLY the spec's driver-verified sub-runs
        # (skip ids have verify=None and emit no claims) — a future walk
        # addition missing from the plan would silently skip its verify
        want_runs = {(m, s) for m, subs in spec.items()
                     for s, v in subs.items() if v["verify"] is not None}
        plan_runs = {(e["mid"], e["sub"]) for e in plan}
        assert plan_runs == want_runs, \
            f"claim plan != walk spec: only-in-spec {sorted(want_runs - plan_runs)[:4]} " \
            f"only-in-plan {sorted(plan_runs - want_runs)[:4]}"
        if os.path.exists(os.path.join(run_dir, "vacc")):
            shutil.rmtree(os.path.join(run_dir, "vacc"))
        for k in range(n_batches):
            os.makedirs(C.vacc_dir(run_dir, k))
        if wpriv:
            os.makedirs(C.wvacc_dir(run_dir))
        wvacc_rel = os.path.relpath(C.wvacc_dir(run_dir), run_dir)
        for e in plan:
            s = spec[e["mid"]][e["sub"]]
            assert s["seed_id"] == e["obid"], (e, s["seed_id"])
            suffix = ["--claims",
                      os.path.relpath(C.vacc_dir(run_dir, e["batch"]), run_dir),
                      e["obid"]]
            if wpriv and e.get("wid"):
                # Stage D: this run opens a REGISTERED weight commitment — the
                # verifier-side driver recomputes its W/g claim as a Committed
                # record into the weight accumulator (mirrors prove exactly;
                # claim order = plan order on both sides, wclaims_match).
                suffix += ["--wpriv", wvacc_rel]
            run_verify(e["mid"], e["sub"], s, suffix=suffix,
                       cwd=run_dir, want="ACCEPT-conditional")
    else:
        for mid, subs in spec.items():
            for sub, s in subs.items():
                if s["verify"] is None:
                    continue   # skip-connection ids: checked purely via edges below
                run_verify(mid, sub, s)

    # ---- 2b. opening_batch: ONE zkob_batchopen discharge per sub-batch ----
    # F12 conditional-verdict gating: in batched mode the overall verdict is
    # ACCEPT only if every driver passed its (non-opening) checks AND every
    # sub-batch's claims_match + batch-evaluation sumcheck + per-domain IPAs
    # ACCEPT AND every chain edge holds.
    batch_results = []
    batch_ok = True      # the public sub-batches (F12 gate)
    wbatch_ok = True     # the Stage-D weight sub-batch (same gate class)
    if batched:
        ob_d = details.setdefault("opening_batch", {"ok": True, "reasons": []})
        all_comrefs = set()
        total_claims = 0
        obid2mid = {e["obid"]: e["mid"] for e in plan}
        for k in range(n_batches):
            pacc, vacc = C.acc_dir(run_dir, k), C.vacc_dir(run_dir, k)
            entry = {"batch": k, "ok": False, "locus": None, "claims": 0}
            try:
                claims = C.parse_claims(os.path.join(vacc, "claims.bin"))
                entry["claims"] = len(claims)
                total_claims += len(claims)
                # structural pins (Stage B flags 2/3 + the F5 discharge pin):
                # relative comrefs; every claim domain maps onto the unique
                # registration gen file of its size; Plain evals only pre-D.
                for c in claims:
                    assert not c["comref"].startswith("/"), \
                        f"absolute comref {c['comref']} ({c['id']})"
                    assert c["domain"] in C.GEN_FOR, \
                        f"claim domain {c['domain']} has no registration gen file"
                    # the PUBLIC batches carry Plain evals only, in BOTH modes
                    # (Stage D routes Committed claims to the weight batch; a
                    # Committed claim here is the closed BO-13b path)
                    assert c["tag"] == 0, \
                        f"non-Plain EvalVar in a public sub-batch ({c['id']})"
                    all_comrefs.add(c["comref"])
                cmd = [C.drv("zkob_batchopen"), "verify",
                       os.path.relpath(pacc, run_dir), os.path.relpath(vacc, run_dir),
                       C.batch_seed(run_seed, k), "registration/q.bin"] + C.genspec_args()
                ok, dt, out = drive(cmd, f"verify opening_batch [b{k}]", cwd=run_dir)
                timing[f"opening_batch[b{k}]"] = round(dt, 3)
                loci = [ln for ln in out.splitlines() if ln.startswith("REJECT[opening_batch")]
                entry["ok"] = ok
                entry["locus"] = "accept" if ok else (loci[0] if loci else out[-200:].strip())
            except (AssertionError, RuntimeError, OSError) as exc:
                entry["ok"] = False
                entry["locus"] = str(exc)
            if entry["ok"]:
                ob_d["reasons"].append(f"b{k}: opening_batch ACCEPT ({entry['claims']} claims)")
            else:
                batch_ok = False
                ob_d["ok"] = False
                ob_d["reasons"].append(f"b{k}: opening_batch REJECT ({entry['locus']})")
                print(f"  REJECT opening_batch b{k}: {entry['locus']}")
                # Stage C2 localization: pinpoint the offending id(s) instead
                # of blaming every id sharing the sub-batch. The verdict is
                # already REJECT (F12); this only sets WHICH manifest ids the
                # transcript names, at the same loci the inline transport
                # named (a diverging claim record == a diverged driver
                # transcript or a forged prover record — both owned by the
                # claim's obligation id).
                try:
                    div_ids, loc_note = C.localize_batch_failure(
                        os.path.join(pacc, "claims.bin"),
                        os.path.join(vacc, "claims.bin"))
                except (AssertionError, OSError) as exc:
                    div_ids, loc_note = [], f"localization unavailable ({exc})"
                entry["diverging_claims"] = div_ids
                ob_d["reasons"].append(f"b{k} localization: {loc_note}")
                impl_mids = {}
                for cid in div_ids:
                    mid = obid2mid.get(cid.rsplit(":", 1)[0])
                    if mid is None:
                        ob_d["reasons"].append(
                            f"b{k}: diverging claim {cid!r} has no plan obligation "
                            f"(forged id — stays on opening_batch)")
                    else:
                        impl_mids.setdefault(mid, []).append(cid)
                entry["implicated"] = sorted(impl_mids)
                for mid, cids in sorted(impl_mids.items()):
                    fail(mid, f"opening_batch b{k}: {len(cids)} recomputed claim(s) "
                              f"diverge from the prover list ({', '.join(cids[:3])}"
                              f"{', ...' if len(cids) > 3 else ''})")
            batch_results.append(entry)
        # ---- 2c. Stage D: the ZK weight sub-batch ------------------------
        # ONE zkob_batchopen wverify over every hidden weight claim
        # (wclaims_match byte-compares the prover's weight claims.bin against
        # the verifier-recomputed list; committed-round batch sumcheck;
        # blinded per-domain ZK IPAs). The verdict gates on it exactly like
        # the public sub-batches (F12 clause).
        wbatch_entry = None
        wcomrefs = set()
        if wpriv:
            wb_d = details.setdefault("opening_batch_w", {"ok": True, "reasons": []})
            pwacc, wvacc = C.wacc_dir(run_dir), C.wvacc_dir(run_dir)
            wbatch_entry = {"ok": False, "locus": None, "claims": 0}
            try:
                wclaims = C.parse_claims(os.path.join(wvacc, "claims.bin"))
                wbatch_entry["claims"] = len(wclaims)
                for c in wclaims:
                    assert not c["comref"].startswith("/"), \
                        f"absolute comref {c['comref']} ({c['id']})"
                    assert c["domain"] in C.GEN_FOR, \
                        f"claim domain {c['domain']} has no registration gen file"
                    # the weight batch is uniformly COMMITTED (a Plain weight
                    # eval here is the wevalvar leak path — fail-closed early)
                    assert c["tag"] == 1, \
                        f"plain (non-Committed) claim in the weight batch ({c['id']})"
                    wcomrefs.add(c["comref"])
                cmd = [C.drv("zkob_batchopen"), "wverify",
                       os.path.relpath(pwacc, run_dir),
                       os.path.relpath(wvacc, run_dir),
                       run_seed, "registration/q.bin"] + C.genspec_args()
                ok, dt, out = drive(cmd, "verify opening_batch_w", cwd=run_dir)
                timing["opening_batch_w"] = round(dt, 3)
                loci = [ln for ln in out.splitlines()
                        if ln.startswith("REJECT[opening_batch_w")]
                wbatch_entry["ok"] = ok
                wbatch_entry["locus"] = ("accept" if ok else
                                         (loci[0] if loci else out[-200:].strip()))
            except (AssertionError, RuntimeError, OSError) as exc:
                wbatch_entry["ok"] = False
                wbatch_entry["locus"] = str(exc)
            if wbatch_entry["ok"]:
                wb_d["reasons"].append(
                    f"weight sub-batch ACCEPT ({wbatch_entry['claims']} hidden "
                    f"claims, all Committed; committed-round sumcheck + ZK IPAs)")
            else:
                wbatch_ok = False
                wb_d["ok"] = False
                wb_d["reasons"].append(
                    f"weight sub-batch REJECT ({wbatch_entry['locus']})")
                print(f"  REJECT opening_batch_w: {wbatch_entry['locus']}")
                # same localization discipline as the public sub-batches
                try:
                    div_ids, loc_note = C.localize_batch_failure(
                        os.path.join(pwacc, "claims.bin"),
                        os.path.join(wvacc, "claims.bin"))
                except (AssertionError, OSError) as exc:
                    div_ids, loc_note = [], f"localization unavailable ({exc})"
                wbatch_entry["diverging_claims"] = div_ids
                wb_d["reasons"].append(f"localization: {loc_note}")
                impl_mids = {}
                for cid in div_ids:
                    mid = obid2mid.get(cid.rsplit(":", 1)[0])
                    if mid is None:
                        wb_d["reasons"].append(
                            f"diverging claim {cid!r} has no plan obligation "
                            f"(forged id — stays on opening_batch_w)")
                    else:
                        impl_mids.setdefault(mid, []).append(cid)
                wbatch_entry["implicated"] = sorted(impl_mids)
                for mid, cids in sorted(impl_mids.items()):
                    fail(mid, f"opening_batch_w: {len(cids)} recomputed hidden "
                              f"claim(s) diverge from the prover list "
                              f"({', '.join(cids[:3])}"
                              f"{', ...' if len(cids) > 3 else ''})")

        # the F5 discharge pin: every registered weight commitment the walk
        # consumes must be opened under its registered comref — in the WEIGHT
        # batch when weight privacy is on (and then NEVER in a public batch:
        # a registered weight claimed Plain there would be the leak path), in
        # the public batches otherwise.
        reg_comrefs = C.registered_weight_comrefs(submission)
        if wpriv:
            wb_d = details["opening_batch_w"]
            missing_reg = sorted(reg_comrefs - wcomrefs)
            leaked_reg = sorted(reg_comrefs & all_comrefs)
            for m in missing_reg:
                wbatch_ok = False
                wb_d["ok"] = False
                wb_d["reasons"].append(
                    f"registered commitment NOT opened in the weight batch: {m}")
                print(f"  REJECT opening_batch_w: registered comref missing: {m}")
            for m in leaked_reg:
                wbatch_ok = False
                wb_d["ok"] = False
                wb_d["reasons"].append(
                    f"registered weight commitment opened PLAIN in a public "
                    f"sub-batch under weight privacy: {m}")
                print(f"  REJECT opening_batch_w: registered comref in a "
                      f"public batch: {m}")
            if not missing_reg and not leaked_reg:
                wb_d["reasons"].append(
                    f"all {len(reg_comrefs)} registered weight commitments "
                    f"opened HIDDEN in the weight batch under their registered "
                    f"comrefs (commitment_opening discharge explicit), none in "
                    f"a public sub-batch")
            wbatch_entry["ok"] = wbatch_ok   # fold the pins into the entry
        else:
            missing_reg = sorted(reg_comrefs - all_comrefs)
            if missing_reg:
                batch_ok = False
                ob_d["ok"] = False
                for m in missing_reg:
                    ob_d["reasons"].append(f"registered commitment NOT opened in any batch: {m}")
                    print(f"  REJECT opening_batch: registered comref missing: {m}")
            else:
                ob_d["reasons"].append(
                    f"all {len(reg_comrefs)} registered weight "
                    f"commitments opened under their registered comrefs (commitment_opening "
                    f"discharge explicit); {total_claims} claims total")

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
                    ok, dt, _ = drive([C.drv("zkob_skip"), "verify", a, b, z],
                                      f"edge {label}")
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
    # In batched mode the discharge is ACCEPT-conditional like every driver
    # verdict: it follows the matmul's detail (which now includes any
    # batch-localized blame on its W claim), NOT the blanket batch_ok — a
    # failed sub-batch elsewhere must not drag unrelated opening ids out of
    # `checked` (Stage C2 localization; the overall verdict still gates on
    # opening_batch via the F12 clause).
    for mid in list(C.covered_ids(submission)):
        if mid.endswith(".commitment_opening"):
            mm = mid.replace(".commitment_opening", ".matmul")
            ok = details.get(mm, {}).get("ok", False)
            d = details.setdefault(mid, {"ok": True, "reasons": []})
            d["ok"] = ok
            via = "opening_batch_w (hidden claim)" if wpriv else "opening_batch"
            d["reasons"].append(
                (f"discharged by {mm}'s W claim vs the registered commitment via "
                 f"{via} ({f'ACCEPT-conditional, gated on {via}' if ok else 'REJECT'})")
                if batched else
                (f"discharged by {mm}: IPA(W) vs registered commitment "
                 f"({'ACCEPT' if ok else 'REJECT'})"))

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

    drivers_ok = all(details.get(m, {}).get("ok", False) for m in C.covered_ids(submission)
                     if m.startswith(("layer", "final_norm", "lm_head")))
    d = details.setdefault("statement.prompt_binding", {"ok": True, "reasons": []})
    d["ok"] = drivers_ok
    d["reasons"].append(
        "run_seed = sha256(public.json) re-derived by this verifier (binds input digest, "
        "registered hashes, constants — prompt_token_ids analog for the random-input "
        "pipeline); every FS transcript above verified under seeds derived from it"
        if drivers_ok else "dependent obligation verifies failed")

    if pool is not None:
        timing["pool_spawn_total_s"] = round(pool.spawn_s, 2)
        pool.close()

    # ---- transcript --------------------------------------------------------
    checked = [m for m in C.covered_ids(submission) if details.get(m, {}).get("ok", False)]
    rejected = [m for m in C.covered_ids(submission) if not details.get(m, {}).get("ok", False)]
    # F12 gating: in batched mode every per-driver verdict above is
    # ACCEPT-conditional; the overall ACCEPT additionally requires every
    # opening_batch sub-batch (claims_match + sumcheck + per-domain IPAs).
    # `opening_batch` deliberately does NOT enter `checked`: it is not a
    # manifest obligation id and check_transcript rejects unknown ids
    # (design §8.11 resolved — the batch gates the verdict via details).
    verdict = ("ACCEPT" if not rejected and (not batched or batch_ok)
               and (not wpriv or wbatch_ok) else "REJECT")
    for m, d in details.items():
        d["reason"] = "; ".join(d.pop("reasons"))
    transcript = {
        "verdict": verdict,
        "submission": submission,
        "transport": transport,
        "weight_privacy": wpriv,
        "checked": checked,
        "rejected": rejected,
        "skipped": skipped,
        "details": details,
        "chain_edges": edge_results,
        "opening_batch": ({"ok": batch_ok, "n_batches": len(batch_results),
                           "claims": sum(b["claims"] for b in batch_results),
                           "batches": batch_results} if batched else None),
        "opening_batch_w": (wbatch_entry if wpriv else None),
        "registration": {"ok": True, "n_hashes": len(checks)},
        # harness-reader echo (§3.3): the registered served tokens this
        # transcript's logit binding was verified against.
        "served_tokens_sha256": public["served_tokens"]["sha256"],
        "timing": {**timing, "total_verify_wall_s": round(time.time() - t_start, 2)},
    }
    json.dump(transcript, open(out_path, "w"), indent=2)
    print(f"checked={len(checked)} rejected={len(rejected)} skipped={len(skipped)}"
          + (f" opening_batch={'ACCEPT' if batch_ok else 'REJECT'}"
             f" ({len(batch_results)} sub-batches)" if batched else "")
          + (f" opening_batch_w={'ACCEPT' if wbatch_entry and wbatch_entry['ok'] else 'REJECT'}"
             f" ({wbatch_entry['claims'] if wbatch_entry else 0} hidden claims)"
             if wpriv else ""))
    print(f"VERDICT: {verdict}  ({out_path})")
    sys.exit(0 if verdict == "ACCEPT" else 1)


if __name__ == "__main__":
    main()
