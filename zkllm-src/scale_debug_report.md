# Scale debug report: streamed-prefix + structured blinds, and the logn=30 commit NTT

## Bug 1 (correctness): streamed prefix corrupts the transcript when structured blinds are active

**Symptom.** `seq=1024 d=64 zk=1` proves in 324 s, `verify_ok=0`, `FATAL: verify
failed: Dp sumcheck`. The Dp zero-check is the one chain big enough to stream
(`sc5z_gpu STREAM hwl-scP N=2^27 x 12 cols`), and at that size the chain uses
STRUCTURED Libra blinds, whose per-round message fixup `p3sg::ScFix` is round-
indexed.

**Mechanism.** `p3sg::sc_prove_gpu` (p3_scgpu.cuh) gained an `rd0` round-offset
parameter for exactly this split — its doc comment even says "so the ScFix
callbacks keep seeing chain-global round numbers" — but the loop body never
applied it: it called `fx->fix(rd, s, 5)` and `fx->bound(rd, a)` with the LOCAL
round index `rd = 0,1,...`. After a streamed prefix of `rd0` rounds (which
correctly used global rounds `0..rd0-1`), the resident phase re-ran the fixup
from round 0:

- `fix` adds `rho * [2^(v-1-rd) * (pref + g_rd(t)) + 2^(v-2-rd) * suf[rd+1]]`
  to the round message — with `rd` too small by `rd0`, both the power-of-two
  weights and the selected univariate `g_rd` are wrong, so every resident round
  message diverges from what the verifier's structured-blind replay recomputes;
- `bound` accumulates `pref += g_rd(a)` — with the wrong `rd`, the running
  prefix (and therefore the terminal claim `ystar = g(r)` fed to
  `mblind_claims`) is also wrong.

Non-structured chains were unaffected (their blind columns are bound as extra
sumcheck columns; `fx == nullptr`), which is why the earlier validation of the
streamed prefix passed. The other two suspects were checked and are sound: the
hoisted `sb_fix(mb)` captures only `mb`, which streaming does not mutate (in
structured mode `nbl == 0`, so the `mb.Bv` frees in the stream loop never
execute), and `mblind_claims` gets the full challenge list `r`, which is all
the structured claim math needs.

**Fix.** p3_scgpu.cuh, `sc_prove_gpu`: pass the chain-global round index to
both callbacks — `fx->fix(rd0 + rd, s, NT)` / `fx->bound(rd0 + rd, a)`. No-op
for every existing caller with `rd0 == 0`.

**Repro (structured mode forced with `P3_SBLIND_MIN=10`, multi-round streaming
forced with `P3_SC5ZG_CAP=800000000`, config `64 64 2 32 128 1 1
tables_ld6.bin`).** Pre-fix binary: `verify_ok=0`, `FATAL: verify failed: Dp
sumcheck` (the exact seq=1024 failure). Post-fix: capped and uncapped runs both
`verify_ok=1` with identical `proof_mb=41.569`; non-structured capped/uncapped
regression pair identical at `proof_mb=42.658`.

## Bug 2 (capacity): d=256 seq=256 zk=1 aborts at `ntt:Wi (4294967296 bytes)`

**Mechanism — two stacked defects at logn=30.** The P=2^27 dff zero-check
commits 2^28-coefficient augmented columns; with blowup R=2 the codeword is
M0=2^30 points.

1. *Twiddle-table footprint.* `P3Ntt` eagerly builds FULL twiddle tables:
   `n/2` powers per direction = 4 GiB each at logn=30, held for the process
   lifetime by the `ntt_plan` cache (which by that point also holds the full
   tables of every smaller plan the run touched). The first logn=30 plan is
   constructed inside `rs_encode_gpu_dev` AFTER the two M0-element codeword
   buffers (8 GiB each) are already allocated from the async pool:
   8 + 8 + 4 (Wf, succeeded) + 4 (Wi) = 24 GiB = the whole card. The
   existing trim+retry in `p3_cuda_malloc_persist` was insufficient because
   the pool blocks it trims are not idle hoard — they are the live codeword
   buffers of the very commit being executed.
2. *Loop-counter overflow.* With the alloc fixed, `P3Ntt::run` hangs at
   logn=30: the stage loop `for (m <<= 1; m <= n; m <<= 2)` uses `uint32_t m`;
   after the final stage at `m = n = 2^30`, `m <<= 2` wraps to 0 and
   `0 <= n` launches stage kernels with `m = 0` forever. Latent until now
   because no logn>=30 NTT had ever gotten past the twiddle alloc.

**Fix.** p3_ntt.cuh:

1. Split twiddle tables for big plans (`logn >= 26`, override
   `P3_NTT_SPLIT_MIN`): store `[2^hb powers base^lo | powers base^(hi<<hb)]`
   with `hb = (logn-1)/2` and fetch `W[k]` as
   `gl_mul(W[k & (2^hb-1)], W[2^hb + (k >> hb)])` (`p3_ntt_tw`, threaded
   through the four stage kernels). `base^lo * base^(hi<<hb)` IS `base^k` —
   the same field element — so every butterfly, codeword, Merkle root, and
   transcript byte is unchanged; only the table representation differs.
   Table size at logn=30 drops from 4 GiB to 384 KiB per direction (and the
   plan cache's cumulative hoard shrinks the same way).
2. `run`/`run_batch` stage loop counter widened to `size_t` (values passed to
   the kernels still fit `uint32_t`).

**Verification of transcript-identity.** Standalone check dumping
forward + inverse + batch NTT outputs for logn in {3,4,7,12,13,20,21} with
full tables vs `P3_NTT_SPLIT_MIN=1` (split everywhere, odd and even hb):
byte-identical. Full composed run `64 64 2 32 128 1 1` with
`P3_NTT_SPLIT_MIN=1`: `verify_ok=1`, `proof_mb=42.658` — identical to the
default-table run. Capacity smoke test replicating the failure shape (2x 8 GiB
codeword buffers live, then first-use logn=30 plan construction + forward run):
completes.

## Gate outputs

(appended below)
