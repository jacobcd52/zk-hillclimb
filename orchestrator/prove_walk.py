"""Prover walk over the stage-3 covered subgraph: the FULL forward pass of
both layers (rmsnorm sites, complete attention chain, MLP path, skips) PLUS
the stage-3 head (STAGE3_FAITHFUL_DESIGN §3: final_norm trio with exact-R
authority, lm_head fc + rescale on the registered 768x32000 weight, and the
statement.logit_binding zkob_rowmax vpad instance binding the registered
served tokens t*) on real llama-68m weights + the registered run input.

Submission faithful-arch-v1 (STAGE3 §4, selected by public.json "submission"):
per head, rowmax (causal) chains from rescale10 and its driver-emitted mx
chain file feeds softmax8 (temperature 8, edges RM1/RM2/SX8a/SX8b); headmerge
runs concat; o_proj fc + rescale slots between headmerge and attn_skip (edges
O1/O2/O3). The §2.4(ii) selector-tie duty is measured for EVERY rowmax
instance (24 causal + 1 vpad) and reported in prove_manifest.json.

Witness authority (ORCHESTRATOR_DESIGN.md §3, extended by ROPE_ATTENTION_DESIGN
§1.5/§7.3):
 - chain data files come from the drivers (fc Y.i64, rescale Xr.i32, glu H.i64,
   rmsnorm W.i64/Y.i64, rope Y64.i64, headslice slice files, softmax P.i32,
   headmerge O2 = attn_out.i32);
 - the orchestrator computes only: skip sums, the rmsnorm advice R
   (INTEGER-EXACT bracket), and the int32->int64 widening shim between the two
   score rescales (lossless; data files are not trust-carrying).
 - attn_out comes from the INTEGER chain (headmerge O2 output) — the stage-1
   python-float attention segment is gone; com_attn_out is chained to the
   merge's com_O2 by edge A15 (edge S1's former open boundary is closed).

Run: /root/int-model-env/bin/python prove_walk.py <run_dir>
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RUNS = []   # prove-manifest entries
TIES = {}   # rowmax selector-tie duty per instance (STAGE3 §2.4(ii))

# Batched-transport state (TRANSPORT_REBUILD_DESIGN Stage C2). When
# public.json says "transport": "batched", every claim-emitting prove call
# gets a --claims block routed into its sub-batch accumulator (the common.py
# claim_plan, EXACT prove order), drivers run with cwd=run_dir + relative
# argv (run-relative comrefs), and zkob_batchopen prove runs as soon as each
# sub-batch's last driver completes — after which that batch's witness dumps
# (wit_*.fr, the ~32 B/element Fr expansions batch_prove streams from) are
# DELETED, capping witness disk at one sub-batch (~7 GB) instead of the full
# walk's ~42 GB.
BATCH = {"on": False, "plan": [], "cursor": 0, "cur": 0, "run_dir": None,
         "run_seed": None, "stats": []}


def finalize_batch(k):
    run_dir, run_seed = BATCH["run_dir"], BATCH["run_seed"]
    acc = C.acc_dir(run_dir, k)
    acc_rel = os.path.relpath(acc, run_dir)
    cmd = [C.drv("zkob_batchopen"), "prove", acc_rel, C.batch_seed(run_seed, k),
           "registration/q.bin"] + C.genspec_args()
    ok, dt, out = C.run_driver(cmd, f"prove opening_batch [b{k}]", cwd=run_dir)
    record("opening_batch", f"b{k}", f"b{k}", cmd, dt)
    claims = C.parse_claims(os.path.join(acc, "claims.bin"))
    elems = sum({c["comref"]: c["n_rows"] * c["domain"] for c in claims}.values())
    tensors = len({c["comref"] for c in claims})
    assert elems <= C.SUBBATCH_HARD_CAP_ELEMS, \
        f"sub-batch b{k}: {elems} elements exceeds the hard cap (estimate drift)"
    n_wit = 0
    wit_bytes = 0
    for f in os.listdir(acc):
        if f.startswith("wit_") and f.endswith(".fr"):
            p = os.path.join(acc, f)
            wit_bytes += os.path.getsize(p)
            os.remove(p)
            n_wit += 1
    wr = os.path.join(acc, "witrefs.txt")   # prover-only pointer file, spent
    if os.path.exists(wr):
        os.remove(wr)
    BATCH["stats"].append({"batch": k, "claims": len(claims), "tensors": tensors,
                           "elements": elems, "prove_s": round(dt, 2),
                           "witness_bytes_freed": wit_bytes, "wit_files": n_wit})
    print(f"  opening_batch b{k}: {len(claims)} claims, {tensors} tensors, "
          f"{elems/1e6:.1f}M elems, {dt:.1f}s; freed {wit_bytes/2**30:.2f} GiB witness")


def record_ties(instance, kmax):
    """STAGE3 §2.4(ii) / PHASE0 §19 tie-count duty: per rowmax instance, the
    rows with >1 allowed maximizers and sum over rows of log2(#maximizers) —
    the selector tie channel, observable in proof bytes (com_S + openings)
    only. kmax = per-row count of allowed positions attaining the row max."""
    TIES[instance] = {
        "rows": int(kmax.size),
        "rows_with_ties": int((kmax > 1).sum()),
        "sum_log2_maximizers_bits": round(float(np.log2(kmax.astype(np.float64)).sum()), 4),
    }


def record(mid, sub, seed_id, cmd, seconds):
    RUNS.append({"manifest_id": mid, "sub": sub, "seed_id": seed_id,
                 "cmd": [os.path.basename(cmd[0])] + cmd[1:], "seconds": round(seconds, 3)})


def prove(mid, sub, seed_id, cmd, run_seed):
    cmd = [c if c is not None else f"{run_seed}:{seed_id}" for c in cmd]
    cwd = None
    if BATCH["on"] and cmd[1] == "prove":
        # every "prove"-mode call is a claim-emitting run and must be the next
        # entry of the canonical plan (order drift would silently desynchronize
        # claims_match — assert loudly instead)
        e = BATCH["plan"][BATCH["cursor"]]
        assert (e["mid"], e["sub"], e["obid"]) == (mid, sub, seed_id), \
            f"claim plan drift: expected {e['mid']}[{e['sub']}] ({e['obid']}), " \
            f"got {mid}[{sub}] ({seed_id})"
        if e["batch"] != BATCH["cur"]:        # previous sub-batch is complete
            finalize_batch(BATCH["cur"])
            BATCH["cur"] = e["batch"]
        BATCH["cursor"] += 1
        run_dir = BATCH["run_dir"]
        cmd = cmd + ["--claims", os.path.relpath(C.acc_dir(run_dir, e["batch"]), run_dir),
                     seed_id] + ([e["extra"]] if e["extra"] else [])
        cmd = C.rel_argv(cmd, run_dir)
        cwd = run_dir
    ok, dt, _ = C.run_driver(cmd, f"prove {mid}" + (f" [{sub}]" if sub != "main" else ""),
                             cwd=cwd)
    record(mid, sub, seed_id, cmd, dt)
    return ok


def exact_R(M, Cdim):
    """Largest r with r^2 * M <= 2^64 * C, exact (python ints == __int128 path
    of zkob_rmsnorm.cu::exact_R, with isqrt instead of float sqrt + fix-ups)."""
    c64 = Cdim << 64
    r = math.isqrt(c64 // M)
    while (r + 1) * (r + 1) * M <= c64:
        r += 1
    while r > 0 and r * r * M > c64:
        r -= 1
    if r < 1 or r > 0x7FFFFFFF:
        raise RuntimeError(f"exact_R out of int32 range: {r}")
    return r


def compute_R(X_i32, c_eps):
    X = X_i32.astype(np.int64)
    M = (X * X).sum(axis=1)               # < 2^47 at this scale; int64-safe
    return np.array([exact_R(int(m) + c_eps, X.shape[1]) for m in M], dtype=np.int32)


def widen_i32_to_i64(src, dst):
    """int32 -> int64 widening shim (lossless). Needed when a rescale stage
    feeds another rescale stage (attention scores 2^13 -> 2^10 chain,
    SOFTMAX_DESIGN §7.3). Data files are not trust-carrying — commitments are."""
    np.fromfile(src, dtype=np.int32).astype(np.int64).tofile(dst)


def attention_chain(run_dir, l, attn_in_path, run_seed, cst, submission, perm,
                    log=print):
    """The integer attention chain, fully proven. Baseline (ROPE §7.3):

      q/k/v fc+rescale -> rope.q/k (+rescales) -> headslice -> per head:
      scores fc -> rescale13 -> [widen] -> rescale10 -> softmax -> values fc
      -> values rescale -> headmerge (pi157) -> attn_out (int32 @2^16)

    faithful-arch-v1 (STAGE3 §4.3/§4.1) replaces the softmax step per head with
    rowmax (causal, chained from rescale10; its mx-out CHAIN FILE — the pinned
    witness authority — feeds softmax8, edges RM1/RM2/SX8a/SX8b) and appends,
    after headmerge in concat mode, the o_proj fc + rescale (edges O1/O2/O3).

    Every chain file comes from the driver that proves it; the orchestrator
    only widens int32->int64 between the two score rescales (lossless shim)
    and measures the §2.4 selector-tie duty from the witness scores.
    Returns the attn_out activation (headmerge O2 baseline / o_proj rescale
    output faithful — com chained to attn_skip by A15 / O3).
    """
    P = C.reg_paths(run_dir)
    faithful = submission == "faithful-arch-v1"
    d = os.path.join(run_dir, "data")
    B, Cw, HD = str(C.SEQ), str(C.EMBED), str(C.HEAD_DIM)
    SM = f"layer{l}.attn.scores_matmul"
    SX = f"layer{l}.attn.softmax"
    VM = f"layer{l}.attn.values_matmul"

    # q/k/v projections: zkob_fc (registered weights) + rescale 2^16
    proj_out = {}
    for pj in ("q_proj", "k_proj", "v_proj"):
        mm, rs = f"layer{l}.attn.{pj}.matmul", f"layer{l}.attn.{pj}.rescaling"
        y = os.path.join(d, f"{mm}.Y.i64.bin")
        o = os.path.join(d, f"{rs}.out.i32.bin")
        os.makedirs(C.ob(run_dir, mm), exist_ok=True)
        os.makedirs(C.ob(run_dir, rs), exist_ok=True)
        prove(mm, "fc", mm,
              [C.drv("zkob_fc"), "prove", C.ob(run_dir, mm), None, attn_in_path,
               C.wpath(run_dir, f"layer{l}.attn.{pj}", "int"),
               B, Cw, Cw, P["gen1024"], P["gen1024"], P["q"], y], run_seed)
        prove(rs, "rescale", rs,
              [C.drv("zkob_rescale"), "prove", C.ob(run_dir, rs), None, y,
               B, Cw, str(C.QKV_RESCALE_LOG), P["gen1024"], P["q"], o], run_seed)
        proj_out[pj] = o

    # RoPE on Q and K (registered cos/sin tables) + rescale 2^16
    roped = {}
    for t, pj in (("q", "q_proj"), ("k", "k_proj")):
        y64 = os.path.join(d, f"layer{l}.attn.rope.{t}.Y64.i64.bin")
        o = os.path.join(d, f"layer{l}.attn.rope.{t}.out.i32.bin")
        os.makedirs(C.ob(run_dir, SM, f"rope.{t}"), exist_ok=True)
        os.makedirs(C.ob(run_dir, SM, f"rope.{t}.rescale"), exist_ok=True)
        prove(SM, f"rope.{t}", f"layer{l}.attn.rope.{t}",
              [C.drv("zkob_rope"), "prove", C.ob(run_dir, SM, f"rope.{t}"), None,
               proj_out[pj], B, Cw, HD, P["rope_cos"], P["rope_sin"],
               P["gen1024"], P["q"], y64], run_seed)
        prove(SM, f"rope.{t}.rescale", f"layer{l}.attn.rope.{t}.rescale",
              [C.drv("zkob_rescale"), "prove", C.ob(run_dir, SM, f"rope.{t}.rescale"), None,
               y64, B, Cw, str(C.ROPE_RESCALE_LOG), P["gen1024"], P["q"], o], run_seed)
        roped[t] = o

    # headslice: 12x {Qh, KhT, Vh} slice files + commitments pinned to Qr/Kr/V
    slice_dir = os.path.join(d, f"layer{l}.attn.slice")
    os.makedirs(slice_dir, exist_ok=True)
    os.makedirs(C.ob(run_dir, SM, "slice"), exist_ok=True)
    prove(SM, "slice", f"layer{l}.attn.slice",
          [C.drv("zkob_headslice"), "prove", C.ob(run_dir, SM, "slice"), None,
           roped["q"], roped["k"], proj_out["v_proj"], B, Cw, HD,
           P["gen1024"], P["gen64"], P["q"], slice_dir + os.sep], run_seed)

    # per head: scores fc -> rescale13 -> widen -> rescale10 -> softmax
    #           -> values fc -> values rescale
    values_dir = os.path.join(d, f"layer{l}.attn.values")
    os.makedirs(values_dir, exist_ok=True)
    for hh in C.HH:
        z = os.path.join(d, f"layer{l}.attn.scores.h{hh}.z.i64.bin")
        z13 = os.path.join(d, f"layer{l}.attn.scores.h{hh}.z13.i32.bin")
        z13w = os.path.join(d, f"layer{l}.attn.scores.h{hh}.z13.i64.bin")
        z_ = os.path.join(d, f"layer{l}.attn.scores.h{hh}.z_.i32.bin")
        pp = os.path.join(d, f"layer{l}.attn.softmax.h{hh}.P.i32.bin")
        out64 = os.path.join(d, f"layer{l}.attn.values.h{hh}.out64.i64.bin")
        sx_subs = ((f"rowmax.h{hh}", f"softmax8.h{hh}") if faithful
                   else (f"softmax.h{hh}",))
        for mid, sub in ((SM, f"fc.h{hh}"), (SX, f"rescale13.h{hh}"),
                         (SX, f"rescale10.h{hh}"),
                         *[(SX, s) for s in sx_subs],
                         (VM, f"fc.h{hh}"), (VM, f"rescale.h{hh}")):
            os.makedirs(C.ob(run_dir, mid, sub), exist_ok=True)
        prove(SM, f"fc.h{hh}", f"layer{l}.attn.scores.h{hh}",
              [C.drv("zkob_fc"), "prove", C.ob(run_dir, SM, f"fc.h{hh}"), None,
               os.path.join(slice_dir, f"Qh{hh}.i32.bin"),
               os.path.join(slice_dir, f"KhT{hh}.i32.bin"),
               B, HD, B, P["gen64"], P["gen1024"], P["q"], z], run_seed)
        prove(SX, f"rescale13.h{hh}", f"layer{l}.attn.scores_rescale13.h{hh}",
              [C.drv("zkob_rescale"), "prove", C.ob(run_dir, SX, f"rescale13.h{hh}"), None,
               z, B, B, str(C.SCORES_RESCALE13_LOG), P["gen1024"], P["q"], z13], run_seed)
        widen_i32_to_i64(z13, z13w)   # SOFTMAX_DESIGN §7.3 widening shim
        prove(SX, f"rescale10.h{hh}", f"layer{l}.attn.scores_rescale10.h{hh}",
              [C.drv("zkob_rescale"), "prove", C.ob(run_dir, SX, f"rescale10.h{hh}"), None,
               z13w, B, B, str(C.SCORES_RESCALE10_LOG), P["gen1024"], P["q"], z_], run_seed)
        zv = np.fromfile(z_, dtype=np.int32)
        if zv.min() < C.SOFTMAX_LOW_E or zv.max() >= C.SOFTMAX_LOW_E + C.SOFTMAX_LEN_E:
            # one bound, two consumers: the baseline exp-table domain and the
            # faithful rowmax/softmax8 envelope are both exactly [-2^19, 2^19)
            raise RuntimeError(f"layer{l} h{hh}: scores leave the +-2^19 score envelope "
                               f"[{zv.min()}, {zv.max()}] — completeness failure, report honestly")
        if faithful:
            # rowmax causal (chained from rescale10) -> softmax8 consuming the
            # driver-emitted mx CHAIN FILE (the pinned witness authority).
            mx = os.path.join(d, f"layer{l}.attn.scores.h{hh}.mx.i32.bin")
            prove(SX, f"rowmax.h{hh}", f"layer{l}.attn.rowmax.h{hh}",
                  [C.drv("zkob_rowmax"), "prove", C.ob(run_dir, SX, f"rowmax.h{hh}"), None,
                   z_, B, B, "causal", "0", str(C.SCORES_ROWMAX_LEN_R),
                   str(C.SCORES_ROWMAX_NPL), P["gen1024"], P["gen1024"], P["q"], mx],
                  run_seed)
            # §2.4(ii) tie duty, measured from the witness scores (allowed = j <= i)
            zg = zv.reshape(C.SEQ, C.SEQ).astype(np.int64)
            tril = np.tri(C.SEQ, dtype=bool)
            mxv = np.where(tril, zg, np.int64(-(1 << 62))).max(axis=1)
            record_ties(f"layer{l}.attn.rowmax.h{hh}",
                        ((zg == mxv[:, None]) & tril).sum(axis=1))
            prove(SX, f"softmax8.h{hh}", f"layer{l}.attn.softmax8.h{hh}",
                  [C.drv("zkob_softmax8"), "prove", C.ob(run_dir, SX, f"softmax8.h{hh}"), None,
                   z_, mx, B, B, str(C.SOFTMAX8_LOW), str(C.SOFTMAX8_LEN),
                   P["exp8_table"], str(C.SOFTMAX8_LEN_R), P["gen1024"], P["q"], pp],
                  run_seed)
        else:
            prove(SX, f"softmax.h{hh}", f"layer{l}.attn.softmax.h{hh}",
                  [C.drv("zkob_softmax"), "prove", C.ob(run_dir, SX, f"softmax.h{hh}"), None,
                   z_, B, B, str(C.SOFTMAX_LOW_E), str(C.SOFTMAX_LEN_E), P["exp_table"],
                   str(C.SOFTMAX_LEN_R), P["gen1024"], P["q"], pp], run_seed)
        prove(VM, f"fc.h{hh}", f"layer{l}.attn.values.h{hh}",
              [C.drv("zkob_fc"), "prove", C.ob(run_dir, VM, f"fc.h{hh}"), None,
               pp, os.path.join(slice_dir, f"Vh{hh}.i32.bin"),
               B, B, HD, P["gen1024"], P["gen64"], P["q"], out64], run_seed)
        prove(VM, f"rescale.h{hh}", f"layer{l}.attn.values_rescale.h{hh}",
              [C.drv("zkob_rescale"), "prove", C.ob(run_dir, VM, f"rescale.h{hh}"), None,
               out64, B, HD, str(C.VALUES_RESCALE_LOG), P["gen64"], P["q"],
               os.path.join(values_dir, f"out{hh}.i32.bin")], run_seed)

    # headmerge: pi157 (baseline, the line-157 permutation) or concat (faithful)
    attn_path = os.path.join(d, f"layer{l}.attn_out.i32.bin")
    merge_path = (os.path.join(d, f"layer{l}.attn_merge.i32.bin") if faithful
                  else attn_path)
    os.makedirs(C.ob(run_dir, VM, "merge"), exist_ok=True)
    prove(VM, "merge", f"layer{l}.attn.merge",
          [C.drv("zkob_headmerge"), "prove", C.ob(run_dir, VM, "merge"), None,
           os.path.join(values_dir, "out"), B, Cw, HD, perm,
           P["gen1024"], P["gen64"], P["q"], merge_path], run_seed)
    if faithful:
        # §4.1: o_proj fc (REGISTERED weight) + rescale 2^16 between headmerge
        # and attn_skip; attn_out := the o_proj rescale output (edges O1/O2/O3).
        o_mm = f"layer{l}.attn.o_proj.matmul"
        o_rs = f"layer{l}.attn.o_proj.rescaling"
        o64 = os.path.join(d, f"{o_mm}.Y.i64.bin")
        os.makedirs(C.ob(run_dir, o_mm), exist_ok=True)
        os.makedirs(C.ob(run_dir, o_rs), exist_ok=True)
        prove(o_mm, "fc", o_mm,
              [C.drv("zkob_fc"), "prove", C.ob(run_dir, o_mm), None, merge_path,
               C.wpath(run_dir, f"layer{l}.attn.o_proj", "int"),
               B, Cw, Cw, P["gen1024"], P["gen1024"], P["q"], o64], run_seed)
        prove(o_rs, "rescale", o_rs,
              [C.drv("zkob_rescale"), "prove", C.ob(run_dir, o_rs), None, o64,
               B, Cw, str(C.OPROJ_RESCALE_LOG), P["gen1024"], P["q"], attn_path],
              run_seed)
    return np.fromfile(attn_path, dtype=np.int32).reshape(C.SEQ, C.EMBED)


def rmsnorm_site(run_dir, l, site, X_i32, run_seed, c_eps, log=print):
    """rmsnorm + wrescale + yrescale; returns the rescaled output activation.
    l=None, site='final_norm' = the stage-3 final-norm site (STAGE3 §3.1:
    same trio, same C_eps, exact-R advice — the witness-authority switch)."""
    if l is None:
        mid, gwid = "final_norm.rmsnorm", "final_norm.g"
    else:
        mid, gwid = f"layer{l}.{site}.rmsnorm", f"layer{l}.{site}.g"
    P = C.reg_paths(run_dir)
    d = os.path.join(run_dir, "data")
    xp = os.path.join(d, f"{mid}.X.i32.bin")
    rp = os.path.join(d, f"{mid}.R.i32.bin")
    wp = os.path.join(d, f"{mid}.W.i64.bin")
    yp = os.path.join(d, f"{mid}.Y.i64.bin")
    yo = os.path.join(d, f"{mid}.out.i32.bin")
    X_i32.tofile(xp)
    t0 = time.time()
    compute_R(X_i32, c_eps).tofile(rp)
    record(mid, "advice-R", "-", ["python:exact_R"], time.time() - t0)
    for sub in ("rmsnorm", "wrescale", "yrescale"):
        os.makedirs(C.ob(run_dir, mid, sub), exist_ok=True)
    prove(mid, "rmsnorm", mid,
          [C.drv("zkob_rmsnorm"), "prove", C.ob(run_dir, mid, "rmsnorm"), None,
           xp, rp, C.wpath(run_dir, gwid, "int"),
           str(C.SEQ), str(C.EMBED), str(c_eps), P["gen1024"], P["gen1024"], P["q"], wp, yp],
          run_seed)
    prove(mid, "wrescale", mid + ".wrescale",
          [C.drv("zkob_rescale"), "prove", C.ob(run_dir, mid, "wrescale"), None,
           wp, str(C.SEQ), str(C.EMBED), str(C.LOG_SF), P["gen1024"], P["q"]], run_seed)
    prove(mid, "yrescale", mid + ".yrescale",
          [C.drv("zkob_rescale"), "prove", C.ob(run_dir, mid, "yrescale"), None,
           yp, str(C.SEQ), str(C.EMBED), str(C.LOG_SF), P["gen1024"], P["q"], yo], run_seed)
    return np.fromfile(yo, dtype=np.int32).reshape(C.SEQ, C.EMBED)


def skip_add(a_i32, b_i32):
    z = a_i32.astype(np.int64) + b_i32.astype(np.int64)
    assert np.abs(z).max() < (1 << 31), "skip add overflows int32"
    return z.astype(np.int32)


def main():
    run_dir = os.path.abspath(sys.argv[1])
    public = json.load(open(os.path.join(run_dir, "public.json")))
    cst = public["constants"]
    c_eps = cst["C_eps"]
    submission = public.get("submission", "baseline")
    assert submission in C.SUBMISSIONS, f"unknown submission {submission!r}"
    transport = public.get("transport", "inline")
    assert transport in C.TRANSPORTS, f"unknown transport {transport!r}"
    if transport == "batched":
        # Stage-B flag 7: the selftest-only forgery env must never reach a
        # production batch prove; the measurement envs must not skew T5 either.
        for v in ("ZKOB_EVIL", "ZKOB_SLOW_FOLD", "ZKOB_SLOW_IPA"):
            assert v not in os.environ, f"{v} set in a production batched prove"
        plan, n_batches = C.claim_plan(run_dir, submission)
        BATCH.update(on=True, plan=plan, cursor=0, cur=0, run_dir=run_dir,
                     run_seed=C.run_seed_of(run_dir))
        for k in range(n_batches):
            os.makedirs(C.acc_dir(run_dir, k), exist_ok=True)
        print(f"  transport: batched ({len(plan)} claim-emitting runs -> "
              f"{n_batches} opening sub-batches)")
    # PHASE0 §21 MINOR-7: headmerge_perm comes from public.json (argv to BOTH
    # prove and verify); it must agree with the submission's pinned mode.
    perm = public.get("headmerge_perm", "pi157")
    assert perm == C.PERM_FOR[submission], \
        f"public.json headmerge_perm={perm!r} inconsistent with submission {submission!r}"
    run_seed = C.run_seed_of(run_dir)
    P = C.reg_paths(run_dir)
    d = os.path.join(run_dir, "data")
    print(f"== prove_walk {run_dir} (submission: {submission}) ==\n  run_seed = {run_seed}")
    t_start = time.time()

    resid = np.fromfile(P["input"], dtype=np.int32).reshape(C.SEQ, C.EMBED)

    for l in range(C.N_LAYERS):
        print(f"== layer {l} ==")
        # input rmsnorm -> attention input
        rmsnorm_site(run_dir, l, "input_norm", resid, run_seed, c_eps)
        attn_in_path = os.path.join(d, f"layer{l}.input_norm.rmsnorm.out.i32.bin")

        # attention: the full integer chain; attn_out = headmerge O2 (baseline)
        # or the o_proj rescale output (faithful, §4.1)
        attn_out = attention_chain(run_dir, l, attn_in_path, run_seed, cst,
                                   submission, perm)
        a_skip = f"layer{l}.attn_skip.add"
        os.makedirs(C.ob(run_dir, a_skip), exist_ok=True)
        prove(a_skip, "commit-attn-out", a_skip,
              [C.drv("zkob_fc"), "commit", os.path.join(d, f"layer{l}.attn_out.i32.bin"),
               str(C.SEQ), str(C.EMBED), P["gen1024"],
               os.path.join(C.ob(run_dir, a_skip), "com_attn_out.bin")], run_seed)
        z1 = skip_add(resid, attn_out)

        # post-attention rmsnorm -> ffn input
        ffn_in = rmsnorm_site(run_dir, l, "post_attn_norm", z1, run_seed, c_eps)
        ffn_in_path = os.path.join(d, f"layer{l}.post_attn_norm.rmsnorm.out.i32.bin")

        # MLP: gate/up fc -> rescales -> glu -> hrescale -> down fc -> rescale
        paths = {}
        for pj, IN, OUT, rs_log in (("gate_proj", C.EMBED, C.INTER, cst["GATE_RESCALE_LOG"]),
                                    ("up_proj", C.EMBED, C.INTER, cst["UP_RESCALE_LOG"]),
                                    ("down_proj", C.INTER, C.EMBED, cst["DOWN_RESCALE_LOG"])):
            mm, rs = f"layer{l}.mlp.{pj}.matmul", f"layer{l}.mlp.{pj}.rescaling"
            x_path = ffn_in_path if pj != "down_proj" else paths["Hr"]
            y_path = os.path.join(d, f"{mm}.Y.i64.bin")
            o_path = os.path.join(d, f"{rs}.out.i32.bin")
            gen_in = P[C.GEN_FOR[C.pad2(IN)].split(".")[0]]
            gen_out = P[C.GEN_FOR[C.pad2(OUT)].split(".")[0]]
            os.makedirs(C.ob(run_dir, mm), exist_ok=True)
            os.makedirs(C.ob(run_dir, rs), exist_ok=True)
            prove(mm, "fc", mm,
                  [C.drv("zkob_fc"), "prove", C.ob(run_dir, mm), None, x_path,
                   C.wpath(run_dir, f"layer{l}.mlp.{pj}", "int"),
                   str(C.SEQ), str(IN), str(OUT), gen_in, gen_out, P["q"], y_path], run_seed)
            prove(rs, "rescale", rs,
                  [C.drv("zkob_rescale"), "prove", C.ob(run_dir, rs), None, y_path,
                   str(C.SEQ), str(OUT), str(rs_log), gen_out, P["q"], o_path], run_seed)
            paths[pj] = o_path

            if pj == "up_proj":   # both inputs ready -> swiglu (glu + hidden rescale)
                G = np.fromfile(paths["gate_proj"], dtype=np.int32)
                if G.min() < cst["SWIGLU_LOW"] or G.max() >= cst["SWIGLU_LOW"] + cst["SWIGLU_LEN"]:
                    raise RuntimeError(f"layer{l}: gate activations leave the silu table domain "
                                       f"[{G.min()}, {G.max()}] — completeness failure, report honestly")
                sw = f"layer{l}.mlp.swiglu"
                h_path = os.path.join(d, f"{sw}.H.i64.bin")
                hr_path = os.path.join(d, f"{sw}.Hr.i32.bin")
                os.makedirs(C.ob(run_dir, sw, "glu"), exist_ok=True)
                os.makedirs(C.ob(run_dir, sw, "hrescale"), exist_ok=True)
                prove(sw, "glu", sw,
                      [C.drv("zkob_glu"), "prove", C.ob(run_dir, sw, "glu"), None,
                       paths["gate_proj"], paths["up_proj"], str(C.SEQ), str(C.INTER),
                       str(cst["SWIGLU_LOW"]), str(cst["SWIGLU_LEN"]), P["table"],
                       P["gen4096"], P["q"], h_path], run_seed)
                prove(sw, "hrescale", sw + ".hrescale",
                      [C.drv("zkob_rescale"), "prove", C.ob(run_dir, sw, "hrescale"), None,
                       h_path, str(C.SEQ), str(C.INTER), str(cst["HIDDEN_RESCALE_LOG"]),
                       P["gen4096"], P["q"], hr_path], run_seed)
                paths["Hr"] = hr_path

        # mlp skip
        ffn_out = np.fromfile(paths["down_proj"], dtype=np.int32).reshape(C.SEQ, C.EMBED)
        z2 = skip_add(z1, ffn_out)
        m_skip = f"layer{l}.mlp_skip.add"
        os.makedirs(C.ob(run_dir, m_skip), exist_ok=True)
        if l + 1 == C.N_LAYERS:
            # terminal output commitment: FRESH gen1024 commitment of z2.
            # Stage 3's edge F0 byte-chains the final-norm com_X to this file,
            # so it must be a fresh commitment (stage 2 used a homomorphic
            # zkob_skip sum here — Jacobian bytes, not byte-comparable). The S2
            # point check (zkob_skip verify) validates the same file unchanged:
            # commitment linearity makes the fresh com of z1+ffn_out point-equal
            # to com_z1 + com_ffn_out.
            fo = os.path.join(d, "final_output.i32.bin")
            z2.tofile(fo)
            prove(m_skip, "commit-Z", m_skip,
                  [C.drv("zkob_fc"), "commit", fo, str(C.SEQ), str(C.EMBED),
                   P["gen1024"], os.path.join(C.ob(run_dir, m_skip), "com_Z.bin")],
                  run_seed)
        resid = z2

    # ---- stage-3 head (STAGE3 §3.4 walk order) ---------------------------
    print("== head: final_norm -> lm_head -> statement.logit_binding ==")
    rmsnorm_site(run_dir, None, "final_norm", resid, run_seed, c_eps)
    normed_path = os.path.join(d, "final_norm.rmsnorm.out.i32.bin")

    MM, RS, LB = "lm_head.matmul", "lm_head.rescaling", "statement.logit_binding"
    logits64 = os.path.join(d, "lm_head.logits64.i64.bin")
    logits_p = os.path.join(d, "lm_head.logits.i32.bin")
    for mid, sub in ((MM, None), (RS, None), (LB, "rowmax")):
        os.makedirs(C.ob(run_dir, mid, sub), exist_ok=True)
    prove(MM, "fc", MM,
          [C.drv("zkob_fc"), "prove", C.ob(run_dir, MM), None, normed_path,
           C.wpath(run_dir, "lm_head", "int"), str(C.SEQ), str(C.EMBED),
           str(C.VOCAB), P["gen1024"], P["gen32768"], P["q"], logits64], run_seed)
    prove(RS, "rescale", RS,
          [C.drv("zkob_rescale"), "prove", C.ob(run_dir, RS), None, logits64,
           str(C.SEQ), str(C.VOCAB), str(C.LM_RESCALE_LOG), P["gen32768"], P["q"],
           logits_p], run_seed)

    logits = np.fromfile(logits_p, dtype=np.int32).reshape(C.SEQ, C.VOCAB)
    if np.abs(logits).max() >= (1 << 25):
        raise RuntimeError(f"logits leave the rowmax vpad envelope |z| < 2^25 "
                           f"(max |logit| = {np.abs(logits).max()}) — completeness "
                           "failure, report honestly")
    # §3.3 completeness guard: the driver-emitted logits must reproduce the
    # registered t* (a mismatch means chain and head pass diverged — a BUG,
    # never a soundness event).
    tstar = np.fromfile(P["tstar"], dtype=np.int32)
    tstar_drv = np.argmax(logits, axis=1).astype(np.int32)
    if not np.array_equal(tstar, tstar_drv):
        bad = np.flatnonzero(tstar != tstar_drv)
        raise RuntimeError(f"argmax(driver logits) != registered t* at rows "
                           f"{bad[:8].tolist()} — head pass / chain divergence (bug)")
    # tie-count duty (STAGE3 §2.4(ii) / PHASE0 §19): sum over rows of
    # log2(#row maximizers) — the selector tie channel, proof-bytes-only.
    # (logits holds exactly the V real columns, so every position is allowed.)
    mxv = logits.max(axis=1)
    record_ties("statement.logit_binding", (logits == mxv[:, None]).sum(axis=1))
    total_tie_bits = round(sum(t["sum_log2_maximizers_bits"] for t in TIES.values()), 4)
    total_tie_rows = sum(t["rows_with_ties"] for t in TIES.values())
    print(f"  rowmax selector ties ({len(TIES)} instances): {total_tie_rows} tied rows, "
          f"sum log2(#maximizers) = {total_tie_bits:.3f} bits")

    prove(LB, "rowmax", LB,
          [C.drv("zkob_rowmax"), "prove", C.ob(run_dir, LB, "rowmax"), None,
           logits_p, str(C.SEQ), str(C.VOCAB_PAD), "vpad", str(C.VOCAB),
           str(C.LOGIT_LEN_R), str(C.LOGIT_NPL), P["gen32768"], P["gen1024"],
           P["q"], "-", P["tstar"]], run_seed)

    if BATCH["on"]:
        assert BATCH["cursor"] == len(BATCH["plan"]), \
            f"claim plan incomplete: {BATCH['cursor']}/{len(BATCH['plan'])} runs"
        finalize_batch(BATCH["cur"])          # the last open sub-batch

    # prove manifest + totals
    proof_bytes = 0
    for root, _dirs, files in os.walk(os.path.join(run_dir, "proofs")):
        proof_bytes += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    manifest = {
        "run_seed": run_seed,
        "submission": submission,
        "transport": transport,
        "opening_batch": ({"n_batches": len(BATCH["stats"]),
                           "batches": BATCH["stats"]} if BATCH["on"] else None),
        "covered_ids": C.covered_ids(submission),
        "skipped_ids": C.skipped_ids(),
        "rowmax_selector_ties": {
            "note": "covert selector tie channel duty (STAGE3 §2.4(ii)): per rowmax "
                    "instance, sum over rows of log2(#allowed maximizers); observable "
                    "in proof bytes (com_S + openings) only, never in served tensors",
            "instances": TIES,
            "rows_with_ties": total_tie_rows,
            "total_bits": total_tie_bits,
        },
        "runs": RUNS,
        "totals": {
            "prove_wall_s": round(time.time() - t_start, 2),
            "driver_prove_s": round(sum(r["seconds"] for r in RUNS), 2),
            "proof_bytes": proof_bytes,
        },
    }
    with open(os.path.join(run_dir, "prove_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"PROVE WALK DONE: {manifest['totals']}")


if __name__ == "__main__":
    main()
