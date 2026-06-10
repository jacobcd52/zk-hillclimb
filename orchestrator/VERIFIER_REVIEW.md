# VERIFIER_REVIEW — independent soundness audit of verify_walk.py (+ verifier-relevant common.py)

Reviewer: independent (read-only) audit, 2026-06-10.
Scope: the Python verifier layer — `verify_walk.py`, the verifier-consumed parts of
`common.py` (`walk_spec`, `covered_ids`, `skipped_ids`, `run_driver`, `reg_paths`,
`run_seed_of`), `make_stage1_manifest.py`, `selftest.sh`, and their interaction with
`harness/check_transcript.py`. Driver binaries (`zkob_*`) are previously audited and
in scope here only as contracts (exit codes, FS absorption, dims cross-check), which
I spot-checked in source.

Adversary model audited against: prover controls all of `proofs/` and `data/`;
`registration/` + `public.json` are honest (hash-pinned); verifier host honest.

Method: line-by-line review of the code against ORCHESTRATOR_DESIGN.md §4–§6 and
PHASE0_NOTES §3/§14/§15; spot-checks of driver source for the contracts the Python
layer relies on; and live experiments against a copy of the passing selftest run
(`/root/zkorch/selftest-20260610-221407`) under `/tmp/vaudit-scratch/` — including
adversarial cases the selftest does not exercise (see §7 below).

## VERDICT: **SOUND** (no CRITICAL, no MAJOR; 6 MINOR findings, all hardening/hygiene — none lets incorrect prover data be wrongly accepted under the stated adversary model)

---

## 1. Path/argument injection — CLEAN

Every subprocess invocation and the provenance of every argument:

| invocation (all via `common.run_driver`, list-argv, no shell) | args and provenance |
|---|---|
| `zkob_rmsnorm verify <obdir> <seed> 1024 768 <C_eps> <com_g> <gen1024> <gen1024> <q>` (×4) | obdir = `run_dir/proofs/<id>/<sub>` from hardcoded ids (common.py:139-159, 178-201); seed = sha256(public.json) + hardcoded suffix; dims = common.py constants; C_eps from honest public.json (verify_walk.py:92); com_g/gens/q = `registration/` paths from `reg_paths`/`wpath` (constants + run_dir only) |
| `zkob_rescale verify <obdir> <seed> 1024 <C> <log_sf> <gen> <q>` (×8) | all constants from common.py (LOG_SF/GATE/UP/HIDDEN/DOWN_RESCALE_LOG); paths as above |
| `zkob_fc verify <obdir> <seed> 1024 <IN> <OUT> <com_W-registered> <gen_in> <gen_out> <q>` (×6) | IN/OUT/gen selection from common.py constants via `pad2`/`GEN_FOR`; com_W is the REGISTERED path (common.py:215) |
| `zkob_glu verify <obdir> <seed> 1024 3072 <LOW> <LEN> <table-registered> <gen4096> <q>` (×2) | LOW/LEN from common.py constants; table = registered path |
| `zkob_skip verify <com_A> <com_B> <com_Z>` (×4) | paths built in `walk_spec` from run_dir + hardcoded ids only |

No argument is derived from anything **read out of** `proofs/` or `data/` — the
prover influences only the *contents* at fixed paths. `verify_walk.py` never reads
`data/` at all (verified empirically: a run with `data/` deleted re-verifies ACCEPT,
`/tmp/vaudit-scratch/t_nodata.json`, 30/30 checked). The C_eps hole-filling at
verify_walk.py:91-92 is order-safe: `cmd[3]` (seed) is assigned before the
None-substitution, so only the rmsnorm argv[6] hole receives C_eps. argv lengths
match each driver's `argc` gate (11/9/11/5), confirmed in driver `main()`s.

Driver-side: prover-controlled `dims.bin` is cross-checked against the CLI dims and
rejected on mismatch (zkob_fc.cu:226-230, zkob_rescale.cu:160-164,
zkob_rmsnorm.cu:458-463) — confirmed in source.

## 2. Registration hash checking — CLEAN

`verify_walk.py:55-82`, run BEFORE any driver: gens (gen1024/gen4096/q), all 16
registered weight commitments (iterated from honest public.json), `input.i32.bin`,
`com_input.bin`, `swiglu-table.bin` — 22 files, each fully re-hashed
(`sha256_file` streams the whole file) and compared; missing file ⇒ `"<missing>"` ⇒
mismatch. On any mismatch: verdict REJECT, `checked=[]`, exit 1, drivers never run
(fail-closed STOP, verify_walk.py:72-82).

Cross-check of "every file a driver verify reads": gens/q ✓, com_g ×4 ✓ (the 4 norm
gains are in `weight_specs()` → in public.json's dict → pinned), mlp com_W ×6 ✓,
swiglu table ✓. `com_input` is consumed only via byte-edge R0 — pinned ✓. The weight
`-int.bin` files are read by **no** verify invocation. q/k/v commitments are pinned
though unused (forward-compat). **run_seed is re-derived** as sha256 of the
public.json bytes on disk (common.py:56-58); no stored seed (prove_manifest.json,
meta-style files) is ever read.

Structural note (no current gap): the pinned-set is driven by public.json's dict,
and the used-set by `walk_spec` — both derive from `weight_specs()` today, but
nothing asserts the used registered paths ⊆ pinned paths. See MINOR-5.

## 3. Chain-edge completeness and correctness — CLEAN

The code (common.py:241-307) emits exactly the §4 map: per layer R1/W1/W2/Y1 ×2
sites (8), S1, M1–M9, S2 = 19; ×2 layers + R0 = **39 edges**, matching the 39 in the
passing transcript (all ok). W2 correctly uses the on-disk name `com_Wr.bin`
(PHASE0 §14 MINOR-7 honored, common.py:249-250). Byte edges compare **full file
contents** (`open(a).read() == open(b).read()`, verify_walk.py:121-122) with
existence checks on both sides. Skip edges call `zkob_skip verify A B Z` in the
(A,B,Z) order the driver expects (zkob_skip.cu:86), and the driver checks
per-row Jacobian point equality `com_Z[j] == com_A[j] + com_B[j]` with row-count
guards, exit 0 only on ACCEPT.

Anchor-graph (transitive chaining) analysis — every covered obligation's input
commitment reaches registration or a declared-open boundary:
- l0 `input_norm/com_X` → R0 → registered `com_input` (statement anchor);
- l0 `post_attn_norm/com_X` → S1 → input_norm com_X + `com_attn_out` (**declared
  OPEN** boundary; the only unconstrained inputs in the graph are the two
  com_attn_out files, exactly as §4/§6 declare);
- gate/up fc com_X → M1/M2 → yrescale com_Xr, whose com_X →Y1→ rmsnorm com_Y
  (internally bound to com_X by the rmsnorm hadamard); glu G/U/H → M5/M6/M7;
  down fc → M8/M9; l1 input via S2(l0); terminal com_Z via S2(l1). Complete.
- Weight bindings need no edges: fc/rmsnorm verify absorb the registered com path
  from the CLI; R1 is redundant defense-in-depth as documented.

"Could a prover pass with com files that differ where no edge looks?" — every com
file in every obdir is absorbed into that obligation's FS transcript by the driver
verify (confirmed in source: zkob_fc.cu:232-247, zkob_rescale.cu:166+, glu and
rmsnorm absorb schedules), so intra-obligation files are bound by the proof and
inter-obligation shared files are bound by the 39 edges. Files bound by NO driver
transcript are exactly `com_attn_out.bin` (open by design) and the terminal
`com_Z.bin` — both bound by skip point checks, and I verified empirically that
tampering either one REJECTs at the right id (§7).

Seed separation: every sub-run has a distinct `run_seed:<seed_id>` (mid, mid.wrescale,
mid.yrescale, mid.hrescale all distinct across layers/sites), so obdir replay across
positions diverges the FS transcript — no splice/replay at the Python layer.

Scope remark (not a finding): everything downstream of S1 is anchored only *modulo*
the open attention boundary; e.g. the post-attn rmsnorm's R-advice tightness (PHASE0
§14 MINOR-5) assumes an honest M, which a prover could influence by choosing
attn_out. This is inherent to — and subsumed by — the declared attention gap; the
transcript records attention SKIPPED, and §4 labels the boundary OPEN.

## 4. Verdict logic — CLEAN

- `checked` is computed as `[m for m in covered_ids() if details[m].ok]`
  (verify_walk.py:154); `covered_ids()` and `skipped_ids()` are disjoint (verified:
  intersection empty), so a SKIPPED id can never appear in `checked` — harness hack
  #1 excluded structurally, and `covered ∪ skipped ∪ frozen-manifest-waived` exactly
  partitions the manifest ids (verified against `manifest_llama68m.json`).
- An id with **no** details entry defaults to not-ok ⇒ `rejected` ⇒ REJECT.
  `fail()` sets ok=False permanently; `note()` uses `setdefault` and can never reset
  a False back to True (verify_walk.py:45-52).
- Driver nonzero exit: rc=1 ⇒ immediate reject (no retry, `expect_reject_ok=True`
  on every verify call); rc∉{0,1} ⇒ one retry then `RuntimeError`, caught at
  verify_walk.py:95-96 ⇒ fail. Missing obdir/files: drivers throw/abort ⇒ nonzero
  ⇒ same path (verified empirically: deleted obdir ⇒ REJECT, `t_missing.json`).
  Skip-edge missing files ⇒ ok=False without invoking the driver
  (verify_walk.py:127-128). Registration mismatch ⇒ REJECT before drivers.
  Exit status: 0 iff verdict ACCEPT (verify_walk.py:172, and exit 1 on the
  registration-stop path:82). Fail-closed everywhere I could construct.
- `statement.prompt_binding` is conditioned on ALL layer ids' final ok (computed
  after the edge phase, so edge failures propagate into it) — observed in every
  tamper transcript.

## 5. Subprocess result interpretation — CLEAN

ACCEPT is determined **only** by exit code (common.py:74-87) — stdout is never
parsed for acceptance (the "REJECT" grep at verify_walk.py:101 is for the reason
string only). Driver contract confirmed in source: `return verify(...) ? 0 : 1`
in all four verify drivers and zkob_skip; a driver that printed ACCEPT then
crashed would exit nonzero ⇒ not accepted. Retries occur only for rc∉{0,1}
(crash/OOM), never for REJECT, and a retry cannot mask a REJECT (rc=1 returns
immediately); deterministic re-verification of an invalid proof cannot
flip to accept on retry. PHASE0 §15 MINOR-4 ("treat ANY nonzero exit as reject,
don't parse for REJECT") is honored. No timeout — see MINOR-3.

## 6. TOCTOU / state leakage — CLEAN (with hygiene notes)

The verifier executes no prover content: no pickle, no eval, no exec, no dynamic
import; the only JSON parsed is honest `public.json`; `prove_manifest.json` and
everything under `data/` are never opened (empirically confirmed — `data/` deleted,
still ACCEPT). Nothing prove_walk cached is re-read: the walk spec is rebuilt from
constants, the seed from public.json bytes. Registration files are hash-checked
before the drivers re-open them; the re-open window is only exploitable by a live
writer on the verifier host during verification, which the model excludes — but see
MINOR-2/MINOR-6 for cheap hardening (single-read of public.json; reject non-regular
files in proofs/).

## 7. Selftest honesty — CLEAN (coverage gaps noted, and closed by this audit's experiments)

All three phases test the right layer: (b) and (c) tamper artifacts and re-run
**only** `verify_walk.py` (the prover is not re-invoked), so the REJECTs fire in the
audited verifier — confirmed by reading the existing run's tamper transcripts:
(b) caught **twice independently** (glu FS divergence at lookup round 0 AND chain
edge M7) with verdict REJECT and the right id; (c) caught at the registration check
with `checked=[]` and drivers never run. Phase (c)'s tamper script rewrites
public.json (re-serialized), but restore is from the byte-exact `.bak`, and the
final re-verify-ACCEPT guards against a dirty run dir. The full-manifest
check_transcript is asserted to FAIL (gap reported, not hidden) and the stage-1
manifest is a derived copy — the frozen harness file is untouched
(make_stage1_manifest.py only flips `waived` on the skipped ids).

Selftest gaps (now exercised by this audit, all fail-closed — recommend adding):
- tamper of a **skip-edge-only** file: `com_attn_out.bin` ⇒ REJECT at
  `layer0.attn_skip.add` (`/tmp/vaudit-scratch/t_attnout.json`);
- tamper of the terminal `com_Z.bin` ⇒ REJECT at `layer1.mlp_skip.add`
  (`t_comz.json`);
- deleted obdir ⇒ REJECT at the right id (`t_missing.json`);
- `data/` deleted entirely ⇒ still ACCEPT (verifier independence, `t_nodata.json`).

---

## Findings

### CRITICAL — none.

### MAJOR — none.

### MINOR

**MINOR-1 — relative `run_dir` splits path resolution between the hash check and the drivers.**
`verify_walk.py:32` uses `args.run_dir` as-is, while `run_driver` launches drivers
with `cwd=/root/zkllm` (common.py:74). With a relative run_dir, the Python-side
hash checks/byte-edges resolve against the caller's cwd but every driver resolves
the same strings against `/root/zkllm` — i.e. the file that was hash-pinned is not
the file the driver reads. What gets wrongly accepted: nothing under the stated
model — exploiting it requires planting a full fake run under `/root/zkllm/<rel>`,
which is not prover-writable; the innocent case fails closed (observed: REJECT with
confusing "missing file" driver errors, plus one segfaulted zkob_skip). Fix
(one line): `run_dir = os.path.abspath(args.run_dir)` at verify_walk.py:32.

**MINOR-2 — `public.json` is read twice.**
`json.load` at verify_walk.py:34 and `sha256_file` at :36 (via `run_seed_of`) open
the file separately; a swap between the two reads would decouple the constants/hash
dict from the seed. Honest-host model makes this moot; hardening: read the bytes
once, hash them, `json.loads` the same bytes.

**MINOR-3 — no subprocess timeout.**
`subprocess.run` (common.py:74) has no `timeout`; a wedged driver (or a FIFO planted
at a com path — see MINOR-6) hangs verification forever while holding the GPU lock.
Liveness only, never a wrong ACCEPT. Fix: `timeout=` per invocation + treat
TimeoutExpired as fail.

**MINOR-4 — `commitment_opening` ids are discharged before the chain-edge phase.**
verify_walk.py:106-113 snapshots the matmul's ok *before* edges run; if a matmul
later fails only an edge (e.g. an internally-valid fc proof over a non-chained X —
edge M1 fails, driver passes), the verdict correctly flips to REJECT via the matmul
id, but the opening id remains in `checked` of that REJECT transcript. Defensible —
the IPA(W)-vs-registered-commitment check the opening id names did genuinely pass —
and unreachable in any ACCEPT transcript, so no wrong acceptance. For transcript
precision, move the discharge loop after the edge phase.

**MINOR-5 — no structural assertion that every registered path used by `walk_spec` is hash-pinned.**
Today both sets derive from `weight_specs()` and the audit confirms used ⊆ pinned;
but a future walk addition referencing a com absent from
`public["registered_weight_commitments"]` would be consumed unpinned with no error.
Fix: in verify_walk, collect every `registration/` path appearing in the spec/edges
and assert each was covered by a step-1 hash check.

**MINOR-6 — non-regular files under `proofs/` are not rejected.**
Symlinks are harmless (content is content, and contents are FS-absorbed), but a
FIFO at a com path blocks the byte-edge read (DoS), and under any future relaxation
of the static-dir assumption could serve different bytes to the byte-compare vs. the
driver. Hardening: `os.path.realpath` containment + `stat.S_ISREG` check on every
proofs/ path the verifier or a driver will open.

### Clean categories (what was checked)

- **Injection** (§1): full argv provenance table; nothing prover-readable feeds any
  argument; list-argv, no shell.
- **Registration pinning** (§2): 22/22 files re-hashed before use; run_seed
  re-derived; weight int files never opened by the verifier.
- **Edges** (§3): 39/39 edges match the design map, full-content byte compares,
  correct zkob_skip arg order/exit contract, on-disk `com_Wr.bin` name honored;
  anchor graph closed up to the two declared-open com_attn_out boundaries.
- **Verdict** (§4): fail-closed on nonzero exit, missing files, exceptions,
  absent details entries; `checked` ∩ skipped = ∅ structurally; exit status
  mirrors verdict on both paths.
- **Subprocess interpretation** (§5): exit-code-only contract, confirmed in all
  five driver mains; REJECT never retried.
- **TOCTOU/state** (§6): no prover content executed or deserialized beyond byte
  compares and driver-parsed proof files; `data/`-independence proven empirically.
- **Selftest** (§7): tamper rejects fire in the verifier layer; gaps identified and
  exercised by this audit (skip-edge tamper, terminal-com tamper, missing obdir,
  no-data run) — all fail-closed.

Experiment artifacts: `/tmp/vaudit-scratch/{t_nodata,t_attnout,t_comz,t_missing}.json`
(+ logs). The existing run dirs under `/root/zkorch/` were not modified.
