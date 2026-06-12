"""Stage D D4 leakage regression at WALK scale (STAGE_D_REPORT §5, walk-scale
gate): scan EVERY verifier-visible artifact of a weight-private run for the
plaintext hidden weight-MLE evaluations.

What the secrets are: each --wpriv driver run emits exactly one Committed
weight claim (fc `<obid>:W` = W̃(u_output,u_input); rmsnorm `<obid>:g` =
g̃(u_c3)) whose scalar value v ships ONLY inside C_v = v·Q + t·H. The honest
prover's (v, t) pairs live in the PROVER-PRIVATE stash data/wpriv/cblinds.bin
(relocated out of proofs/ by prove_walk right after wprove) — this scanner
reads the secrets to scan FOR from there, which is exactly why that file must
never ship.

What is scanned (the verifier-visible surface):
  - public.json, prove_manifest.json, transcript*.json (everything a verifier
    or harness reader consumes at the run root);
  - every hash-pinned registration file (gens incl. the 2-slot q, ALL
    registered weight commitments, input + its commitment, tstar, tables) —
    the same list verify_walk re-hashes; the weight *-int.bin files and
    data/ are PROVER-PRIVATE by the run layout (the verifier never reads
    them) and are deliberately NOT part of the shipped surface;
  - everything under proofs/ (driver artifacts, public sub-batches, the
    weight sub-batch);
  - everything under vacc/ (the verifier-side recomputed accumulators —
    they too must never materialize a hidden eval).

Patterns: each secret v as its 32-byte little-endian Fr image (the exact
representation every proof artifact uses — same convention as the in-driver
wp_leak_scan). Positive control: every secret IS found in the prover-private
cblinds.bin stash (validates pattern extraction + matcher end-to-end).

Exit 0 + "D4 WALK SCAN: CLEAN" iff zero hits and the control passes.

Run: /root/int-model-env/bin/python wpriv_leak_scan.py <run_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def visible_files(run_dir, public):
    """The verifier-visible surface (see module docstring)."""
    P = C.reg_paths(run_dir)
    files = []
    for f in sorted(os.listdir(run_dir)):
        if f.endswith(".json"):
            files.append(os.path.join(run_dir, f))
    files += [P[k] for k in sorted(public["gens"])]
    files += [C.wpath(run_dir, wid, "com")
              for wid in sorted(public["registered_weight_commitments"])]
    files += [P["input"], P["com_input"], P["tstar"]]
    files += [os.path.join(P["reg"], f) for f in sorted(public["tables"])]
    for root in (os.path.join(run_dir, "proofs"), os.path.join(run_dir, "vacc")):
        for dirpath, _dirs, fnames in os.walk(root):
            files += [os.path.join(dirpath, f) for f in sorted(fnames)]
    return [f for f in files if os.path.isfile(f)]


def main():
    run_dir = os.path.abspath(sys.argv[1])
    public = json.load(open(os.path.join(run_dir, "public.json")))
    if public.get("weight_privacy") not in C.WEIGHT_PRIVACY_MODES:
        print("not a weight-private run (public.json has no weight_privacy)")
        sys.exit(2)
    stash = os.path.join(run_dir, "data", "wpriv", "cblinds.bin")
    secrets = C.parse_cblinds(stash)         # {claim_id: (v_bytes, t_bytes)}
    assert secrets, "empty cblinds stash"
    print(f"== D4 walk-scale leak scan: {run_dir} ==")
    print(f"  secrets: {len(secrets)} hidden weight-MLE evals "
          f"(from the prover-private stash {os.path.relpath(stash, run_dir)})")

    files = visible_files(run_dir, public)
    total_bytes = sum(os.path.getsize(f) for f in files)
    print(f"  surface: {len(files)} verifier-visible files, "
          f"{total_bytes / 2**20:.1f} MiB")

    hits = []
    for path in files:
        with open(path, "rb") as fh:
            blob = fh.read()
        for cid, (v, _t) in secrets.items():
            if blob.find(v) != -1:
                hits.append((cid, os.path.relpath(path, run_dir)))
    for cid, p in hits:
        print(f"  LEAK: {cid} eval found in {p}")

    # positive control: the matcher finds every secret where it MUST be —
    # the prover-private stash itself (never shipped)
    stash_blob = open(stash, "rb").read()
    control_ok = all(stash_blob.find(v) != -1 for v, _t in secrets.values())
    print(f"  positive control: prover-private cblinds.bin contains all "
          f"{len(secrets)} secrets: {control_ok}")

    if not hits and control_ok:
        print(f"D4 WALK SCAN: CLEAN ({len(secrets)} hidden evals x "
              f"{len(files)} artifacts, 0 hits)")
        sys.exit(0)
    print(f"D4 WALK SCAN: FAIL ({len(hits)} hits, control_ok={control_ok})")
    sys.exit(1)


if __name__ == "__main__":
    main()
