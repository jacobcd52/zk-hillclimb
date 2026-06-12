"""Shared definitions for the zk-hillclimb orchestrator.

Stage 1 covered the MLP subgraph; stage 2 added the full attention chain
(ROPE_ATTENTION_DESIGN.md §7.3/§7.4). Stage 3 (STAGE3_FAITHFUL_DESIGN.md §3,
Part B) closes the manifest: final_norm.rmsnorm (the validated trio at the
final-norm site, exact-R authority), lm_head.matmul + .rescaling +
.commitment_opening (registered 768x32000 weight, gen32768), and
statement.logit_binding (one zkob_rowmax vpad instance binding the registered
served tokens t* = argmax(logits) at all 1024 positions). 56/56 non-waived
checked + 3 covered-waived; the only waived-and-uncovered id left is
embedding.lookup.

Submission faithful-arch-v1 (STAGE3 §4, Part C) re-registers the statement as
the faithful llama-68m: per-head zkob_rowmax (causal) + zkob_softmax8 replace
zkob_softmax (temperature 8 with the exact rowmax max-shift), zkob_headmerge
runs concat mode (line-157 fix), and o_proj (registered weight, fc + rescale)
slots between headmerge and attn_skip — 56 non-waived + 9 covered-waived = 65
checked. The mode is selected per run by public.json's "submission" field; the
baseline chain stays bit-reproducible behind the default.

The walk specification (which driver runs cover which manifest id, with which
public constants, and which chain edges bind them) is defined ONCE here and
consumed by both prove_walk.py and verify_walk.py, so the two sides cannot
drift apart silently.

Run layout (see ORCHESTRATOR_DESIGN.md):
  <run>/public.json            statement; run_seed = sha256(file bytes)
  <run>/registration/          gens, weights(+commitments), input, table
  <run>/data/                  PROVER-ONLY witness/chain files
  <run>/proofs/<id>[/<sub>]    obligation dirs (orchestrator mkdirs)
"""
import fcntl
import hashlib
import json
import os
import struct
import subprocess
import threading
import time

ZKLLM = "/root/zkllm"
RUN_ROOT = "/root/zkorch"
GPU_LOCK = "/tmp/zkorch.gpu.lock"

# Proof-transport modes (TRANSPORT_REBUILD_DESIGN, Stage C2). "inline" is the
# original per-driver inline-IPA tail; "batched" runs every driver in claim
# mode and discharges all claims through zkob_batchopen sub-batches. The mode
# is part of the public STATEMENT (public.json "transport", inside run_seed):
# the verifier demands the discharge the statement names, fail-closed.
TRANSPORTS = ("inline", "batched")

# Weight-privacy modes (STAGE_D_REPORT / TRANSPORT_REBUILD_DESIGN §4).
# "hiding": every registered weight tensor is committed with hiding Pedersen
# rows (com[r] += s_r*H, blinds prover-private under data/), fc and rmsnorm
# emit their weight claims as Committed (C_v, no scalar eval) into a SEPARATE
# weight accumulator, and ONE zkob_batchopen wprove/wverify weight sub-batch
# (committed-round sumcheck + ZK blinded IPAs) discharges them. Part of the
# public STATEMENT (public.json "weight_privacy", inside run_seed); requires
# transport=batched (the wpriv driver modes exist only in claim mode); H = the
# q.bin slot-1 generator (2-slot genq file, hash-pinned like every gen).
WEIGHT_PRIVACY_MODES = ("hiding",)
G1_POINT_BYTES = 144             # Commitment::save Jacobian point size
WPRIV_Q_SLOTS = 2                # q.bin = [Q, H] under weight privacy

SEQ, EMBED, INTER = 1024, 768, 3072
LOG_SF = 16                      # residual-stream / weight scale 2^16
GATE_RESCALE_LOG = 20            # gate lands at 2^12 for the silu table (ffn.cu)
UP_RESCALE_LOG = 16
HIDDEN_RESCALE_LOG = 16
DOWN_RESCALE_LOG = 16
SWIGLU_LOW, SWIGLU_LEN = -(1 << 21), 1 << 22
N_LAYERS = 2

# Attention chain (ROPE_ATTENTION_DESIGN.md; SOFTMAX_DESIGN.md §1.1/§7.3)
HEAD_DIM = 64
N_HEADS = EMBED // HEAD_DIM                  # 12
HH = [f"{h:02d}" for h in range(N_HEADS)]    # two-digit head labels, 00..11
QKV_RESCALE_LOG = 16
ROPE_RESCALE_LOG = 16
SCORES_RESCALE13_LOG = 13                    # scores 2^32 -> 2^19
SCORES_RESCALE10_LOG = 10                    # then 2^19 -> 2^9 (widening shim between)
VALUES_RESCALE_LOG = 16
SOFTMAX_LOW_E, SOFTMAX_LEN_E = -(1 << 19), 1 << 20
SOFTMAX_LEN_R = 1 << 20

# Submission faithful-arch-v1 (STAGE3_FAITHFUL_DESIGN §4; PHASE0 §19-21):
# per-head exact rowmax max-shift + temperature-8 softmax8 + headmerge concat
# mode + o_proj fc/rescale. The baseline statement keeps pi157/temp-128/no-o_proj.
SUBMISSIONS = ("baseline", "faithful-arch-v1")
PERM_FOR = {"baseline": "pi157", "faithful-arch-v1": "concat"}
SCORES_ROWMAX_LEN_R = 1 << 20                # causal Df in [0, 2^20): one 20-bit limb
SCORES_ROWMAX_NPL = 1
SOFTMAX8_LOW, SOFTMAX8_LEN = -(1 << 20) + 2, 1 << 20   # Dm domain [-1048574, +1]; SENT=+1
SOFTMAX8_LEN_R = 1 << 14                     # 14-bit limb pairs (r1, r2 < 2^27)
OPROJ_RESCALE_LOG = 16

# Stage 3 head (STAGE3_FAITHFUL_DESIGN §3.2/§3.3)
VOCAB = 32000                                # real lm_head width (padded to 32768)
VOCAB_PAD = 32768                            # gen32768; NCOL of the rowmax vpad grid
LM_RESCALE_LOG = 16                          # logits 2^32 -> 2^16
LOGIT_LEN_R = 1 << 20                        # rowmax limb table (two 20-bit limbs)
LOGIT_NPL = 2

GEN_FOR = {64: "gen64.bin", 1024: "gen1024.bin", 4096: "gen4096.bin",
           32768: "gen32768.bin"}


def pad2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def drv(name):
    return os.path.join(ZKLLM, name)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_seed_of_bytes(pub_bytes):
    """run_seed = sha256 of the public.json bytes (the full public statement)."""
    return hashlib.sha256(pub_bytes).hexdigest()


def run_seed_of(run_dir):
    """run_seed = sha256 of the public.json bytes (the full public statement)."""
    return sha256_file(os.path.join(run_dir, "public.json"))


# Per-invocation driver timeout (VERIFIER_REVIEW MINOR-3). The longest single
# driver run is ~20 s exclusive-GPU; the budget below leaves >40x headroom for
# contention from other lock-serialized GPU jobs. A wedged driver (or a FIFO
# planted at a path it opens) must not hold the GPU lock forever.
DRIVER_TIMEOUT_S = 900


def run_driver(cmd, label, expect_reject_ok=False, retries=1, log=print,
               timeout=DRIVER_TIMEOUT_S, cwd=None):
    """Run one driver invocation, serialized on the shared GPU via a lock file.

    Returns (accepted: bool, seconds: float, output: str).
    Exit 0 = ACCEPT/success; exit 1 = REJECT (meaningful for verify modes, not
    retried); anything else = crash (CUDA contention etc.) -> retried once.
    Timeout = fail (RuntimeError, no retry): liveness hardening, fail-closed.
    cwd: working directory (default ZKLLM). The batched transport runs drivers
    with cwd = run dir and RELATIVE argv paths so every comref in claims.bin
    is run-dir-relative (Stage-B comref canonicalization).
    """
    attempt = 0
    while True:
        attempt += 1
        t0 = time.time()
        with open(GPU_LOCK, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                r = subprocess.run(cmd, cwd=cwd or ZKLLM, capture_output=True, text=True,
                                   timeout=timeout)
            except subprocess.TimeoutExpired as e:
                # TimeoutExpired captures raw bytes even under text=True
                parts = [p.decode(errors="replace") if isinstance(p, bytes) else p
                         for p in (e.stdout, e.stderr) if p]
                out = "".join(parts)
                raise RuntimeError(
                    f"driver TIMEOUT after {timeout}s: {' '.join(cmd)}\n{out[-2000:]}")
        dt = time.time() - t0
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0:
            log(f"  [{dt:7.2f}s] {label}")
            return True, dt, out
        if r.returncode == 1 and expect_reject_ok:
            log(f"  [{dt:7.2f}s] {label} -> REJECT")
            return False, dt, out
        if attempt <= retries:
            log(f"  RETRY ({r.returncode}) {label}: {out[-300:]}")
            time.sleep(5.0)
            continue
        raise RuntimeError(f"driver failed rc={r.returncode}: {' '.join(cmd)}\n{out[-2000:]}")


class DriverPool:
    """Stage C2 single-process verifier transport (TRANSPORT_REBUILD_DESIGN
    §1.3/§2.7/§6 Stage C): one persistent `serve`-mode process per driver
    binary; every request runs through the same zkw_run1 entry as the
    one-shot CLI (byte-identical FS schedules, checks and verdicts), in the
    exact order the caller issues them (so vacc claims.bin append order stays
    canonical). CUDA init is paid once per DRIVER (~12) instead of once per
    OBLIGATION (~235). Holds the GPU lock for its lifetime (the per-call
    transport held it per invocation; the walk is serial either way).

    Fail-closed: a request that crashes the worker, returns an unexpected rc,
    or times out raises RuntimeError (callers already treat that as REJECT).
    NO automatic retry — re-running a claim-mode verify would double-append
    into the verifier accumulator. A dead worker is respawned only for the
    NEXT request."""

    def __init__(self, cwd, log=print):
        self.cwd = cwd
        self.log = log
        self.procs = {}
        self.spawn_s = 0.0
        self._lk = open(GPU_LOCK, "w")
        fcntl.flock(self._lk, fcntl.LOCK_EX)

    def _get(self, binpath):
        p = self.procs.get(binpath)
        if p is not None and p.poll() is None:
            return p
        if p is not None:
            self.log(f"  [pool] respawning dead {os.path.basename(binpath)} "
                     f"worker (rc={p.returncode})")
        t0 = time.time()
        p = subprocess.Popen([binpath, "serve"], cwd=self.cwd,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        line = p.stdout.readline()
        if "ZKW-READY" not in line:
            raise RuntimeError(f"serve worker failed to start: {binpath} ({line!r})")
        dt = time.time() - t0
        self.spawn_s += dt
        self.log(f"  [pool] {os.path.basename(binpath)} serve up ({dt:.2f}s)")
        self.procs[binpath] = p
        return p

    def run(self, cmd, label, expect_reject_ok=False, timeout=DRIVER_TIMEOUT_S):
        """Same contract as run_driver (accepted, seconds, output)."""
        binpath, args = cmd[0], [str(c) for c in cmd[1:]]
        for a in args:
            assert not any(ch.isspace() for ch in a), f"whitespace in argv: {a!r}"
        t0 = time.time()
        p = self._get(binpath)
        try:
            p.stdin.write(" ".join(args) + "\n")
            p.stdin.flush()
        except (BrokenPipeError, OSError):
            raise RuntimeError(f"serve worker died before request: {label}")
        # blocking readline + kill-timer (NOT select: readline buffers ahead,
        # so the sentinel can sit in the TextIO buffer with the fd quiet)
        out_lines = []
        rc = None
        timed_out = []
        watchdog = threading.Timer(timeout, lambda: (timed_out.append(1), p.kill()))
        watchdog.start()
        try:
            while True:
                line = p.stdout.readline()
                if line == "":
                    if timed_out:
                        raise RuntimeError(
                            f"serve request TIMEOUT after {timeout}s: {label}\n"
                            + "".join(out_lines)[-2000:])
                    raise RuntimeError(
                        f"serve worker died mid-request (rc={p.poll()}): {label}\n"
                        + "".join(out_lines)[-2000:])
                if line.startswith("ZKW-RC "):
                    rc = int(line.split()[1])
                    break
                out_lines.append(line)
        finally:
            watchdog.cancel()
        dt = time.time() - t0
        out = "".join(out_lines)
        if rc == 0:
            self.log(f"  [{dt:7.2f}s] {label}")
            return True, dt, out
        if rc == 1 and expect_reject_ok:
            self.log(f"  [{dt:7.2f}s] {label} -> REJECT")
            return False, dt, out
        raise RuntimeError(f"serve request failed rc={rc}: "
                           f"{' '.join([binpath] + args)}\n{out[-2000:]}")

    def close(self):
        for p in self.procs.values():
            try:
                p.stdin.close()
            except OSError:
                pass
        for p in self.procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs.clear()
        fcntl.flock(self._lk, fcntl.LOCK_UN)
        self._lk.close()


# ---------------------------------------------------------------------------
# Registered weights: (wid, pipeline dump stem, IN, OUT, gen size for commit)
# 2-D weights are exported as (IN, OUT) = w.float().T per pipeline semantics;
# rmsnorm gains as (1, EMBED). q/k/v registered now for attention-stage compat.
# ---------------------------------------------------------------------------
def weight_specs(submission="baseline"):
    specs = []
    for l in range(N_LAYERS):
        specs += [
            (f"layer{l}.mlp.gate_proj", f"layer-{l}-mlp.gate_proj.weight", EMBED, INTER, 4096),
            (f"layer{l}.mlp.up_proj",   f"layer-{l}-mlp.up_proj.weight",   EMBED, INTER, 4096),
            (f"layer{l}.mlp.down_proj", f"layer-{l}-mlp.down_proj.weight", INTER, EMBED, 1024),
            (f"layer{l}.attn.q_proj",   f"layer-{l}-self_attn.q_proj.weight", EMBED, EMBED, 1024),
            (f"layer{l}.attn.k_proj",   f"layer-{l}-self_attn.k_proj.weight", EMBED, EMBED, 1024),
            (f"layer{l}.attn.v_proj",   f"layer-{l}-self_attn.v_proj.weight", EMBED, EMBED, 1024),
            (f"layer{l}.input_norm.g",  f"layer-{l}-input_layernorm.weight", 1, EMBED, 1024),
            (f"layer{l}.post_attn_norm.g", f"layer-{l}-post_attention_layernorm.weight", 1, EMBED, 1024),
        ]
        if submission == "faithful-arch-v1":
            # §4.1: the pipeline's commit loop iterates ALL layer.named_parameters(),
            # so layer-{l}-self_attn.o_proj.weight-int.bin EXISTS in the pipeline
            # dump (it is just never used) — the full byte-compare provenance
            # guard applies to o_proj exactly as to q/k/v.
            specs.append((f"layer{l}.attn.o_proj", f"layer-{l}-self_attn.o_proj.weight",
                          EMBED, EMBED, 1024))
    return specs


def head_weight_specs():
    """Stage-3 registered weights: (wid, IN, OUT, gen size). Exported by
    register.export_head (the pipeline never dumps model.norm / lm_head — its
    commit loop only iterates model.model.layers[*] — so the provenance guard
    for these two is re-export comparison only, a documented deviation,
    STAGE3_FAITHFUL_DESIGN §3.1/§3.2)."""
    return [("final_norm.g", 1, EMBED, 1024),
            ("lm_head", EMBED, VOCAB, 32768)]


def reg_paths(run_dir):
    reg = os.path.join(run_dir, "registration")
    return {
        "reg": reg,
        "gen64": os.path.join(reg, "gen64.bin"),
        "gen1024": os.path.join(reg, "gen1024.bin"),
        "gen4096": os.path.join(reg, "gen4096.bin"),
        "gen32768": os.path.join(reg, "gen32768.bin"),
        "tstar": os.path.join(reg, "tstar.i32.bin"),
        "q": os.path.join(reg, "q.bin"),
        "weights": os.path.join(reg, "weights"),
        "input": os.path.join(reg, "input.i32.bin"),
        "com_input": os.path.join(reg, "com_input.bin"),
        "table": os.path.join(reg, "swiglu-table.bin"),
        "rope_cos": os.path.join(reg, "rope-cos-table.bin"),
        "rope_sin": os.path.join(reg, "rope-sin-table.bin"),
        "exp_table": os.path.join(reg, "softmax-exp-table.bin"),
        "exp8_table": os.path.join(reg, "softmax8-exp-table.bin"),
    }


def wpath(run_dir, wid, kind):  # kind in {"int", "com"}
    return os.path.join(run_dir, "registration", "weights", f"{wid}-{kind}.bin")


# ---------------------------------------------------------------------------
# Covered subgraph walk: per manifest id, the sub-runs (driver verify specs)
# and the chain byte-equality edges. Built from run_dir paths only — the
# verifier consumes exactly this and nothing prover-chosen.
# ---------------------------------------------------------------------------
def ob(run_dir, mid, sub=None):
    p = os.path.join(run_dir, "proofs", mid)
    return os.path.join(p, sub) if sub else p


def covered_ids(submission="baseline"):
    ids = []
    for l in range(N_LAYERS):
        ids += [f"layer{l}.input_norm.rmsnorm"]
        for pj in ("q_proj", "k_proj", "v_proj"):
            ids += [
                f"layer{l}.attn.{pj}.commitment_opening",
                f"layer{l}.attn.{pj}.matmul",
                f"layer{l}.attn.{pj}.rescaling",
            ]
        ids += [
            f"layer{l}.attn.scores_matmul",
            f"layer{l}.attn.softmax",
            f"layer{l}.attn.values_matmul",
        ]
        if submission == "faithful-arch-v1":
            # §4.1: the six o_proj.* ids are waived in the frozen manifest but
            # genuinely covered by the submission (covered-waived NOTE path).
            ids += [
                f"layer{l}.attn.o_proj.commitment_opening",
                f"layer{l}.attn.o_proj.matmul",
                f"layer{l}.attn.o_proj.rescaling",
            ]
        ids += [
            f"layer{l}.attn_skip.add",
            f"layer{l}.post_attn_norm.rmsnorm",
            f"layer{l}.mlp.gate_proj.commitment_opening",
            f"layer{l}.mlp.gate_proj.matmul",
            f"layer{l}.mlp.gate_proj.rescaling",
            f"layer{l}.mlp.up_proj.commitment_opening",
            f"layer{l}.mlp.up_proj.matmul",
            f"layer{l}.mlp.up_proj.rescaling",
            f"layer{l}.mlp.swiglu",
            f"layer{l}.mlp.down_proj.commitment_opening",
            f"layer{l}.mlp.down_proj.matmul",
            f"layer{l}.mlp.down_proj.rescaling",
            f"layer{l}.mlp_skip.add",
        ]
    # stage 3 (§3.4 walk order): final norm -> lm_head -> logit binding.
    # final_norm.rmsnorm / lm_head.matmul / lm_head.rescaling are WAIVED in the
    # frozen manifest but genuinely covered here (check_transcript's
    # covered-waived NOTE path, §3.5); the opening + logit_binding ids are the
    # last two non-waived ids.
    ids += [
        "final_norm.rmsnorm",
        "lm_head.commitment_opening",
        "lm_head.matmul",
        "lm_head.rescaling",
        "statement.logit_binding",
    ]
    ids += ["statement.registered_weight_hash", "statement.prompt_binding"]
    return ids


def skipped_ids():
    """Stage 3: nothing is stage-skipped. All 56 non-waived manifest ids are
    covered (plus 3 covered-waived ids); the only waived-and-uncovered id,
    embedding.lookup, is waived in the FROZEN manifest itself and needs no
    entry here (no integer path exists for the random-input pipeline; prompt
    binding remains the input digest via run_seed — STAGE3 §3.5)."""
    return {}


def rmsnorm_site(run_dir, l, site):
    """site in {'input_norm', 'post_attn_norm'} (per-layer) or 'final_norm'
    with l=None (stage 3, §3.1) -> (manifest_id, sub specs)."""
    if l is None:
        mid, gwid = "final_norm.rmsnorm", "final_norm.g"
    else:
        mid, gwid = f"layer{l}.{site}.rmsnorm", f"layer{l}.{site}.g"
    P = reg_paths(run_dir)
    gC = str(EMBED)
    com_g = wpath(run_dir, gwid, "com")
    subs = {
        "rmsnorm": {
            "obdir": ob(run_dir, mid, "rmsnorm"), "seed_id": mid,
            "verify": [drv("zkob_rmsnorm"), "verify", ob(run_dir, mid, "rmsnorm"), None,
                       str(SEQ), gC, None, com_g, P["gen1024"], P["gen1024"], P["q"]],
        },
        "wrescale": {
            "obdir": ob(run_dir, mid, "wrescale"), "seed_id": mid + ".wrescale",
            "verify": [drv("zkob_rescale"), "verify", ob(run_dir, mid, "wrescale"), None,
                       str(SEQ), gC, str(LOG_SF), P["gen1024"], P["q"]],
        },
        "yrescale": {
            "obdir": ob(run_dir, mid, "yrescale"), "seed_id": mid + ".yrescale",
            "verify": [drv("zkob_rescale"), "verify", ob(run_dir, mid, "yrescale"), None,
                       str(SEQ), gC, str(LOG_SF), P["gen1024"], P["q"]],
        },
    }
    return mid, subs


def fc_block(run_dir, l, proj, IN, OUT, rs_log):
    """gate/up/down_proj -> matmul id (fc) + rescaling id (rescale)."""
    P = reg_paths(run_dir)
    mid_mm = f"layer{l}.mlp.{proj}.matmul"
    mid_rs = f"layer{l}.mlp.{proj}.rescaling"
    com_W = wpath(run_dir, f"layer{l}.mlp.{proj}", "com")
    gen_in, gen_out = P[GEN_FOR[pad2(IN)].split(".")[0]], P[GEN_FOR[pad2(OUT)].split(".")[0]]
    return {
        mid_mm: {"fc": {
            "obdir": ob(run_dir, mid_mm), "seed_id": mid_mm,
            "verify": [drv("zkob_fc"), "verify", ob(run_dir, mid_mm), None,
                       str(SEQ), str(IN), str(OUT), com_W, gen_in, gen_out, P["q"]],
        }},
        mid_rs: {"rescale": {
            "obdir": ob(run_dir, mid_rs), "seed_id": mid_rs,
            "verify": [drv("zkob_rescale"), "verify", ob(run_dir, mid_rs), None,
                       str(SEQ), str(OUT), str(rs_log), gen_out, P["q"]],
        }},
    }


def attn_proj_block(run_dir, l, pj):
    """attn q/k/v_proj -> matmul id (fc vs REGISTERED com_W) + rescaling id."""
    P = reg_paths(run_dir)
    mid_mm = f"layer{l}.attn.{pj}.matmul"
    mid_rs = f"layer{l}.attn.{pj}.rescaling"
    com_W = wpath(run_dir, f"layer{l}.attn.{pj}", "com")
    return {
        mid_mm: {"fc": {
            "obdir": ob(run_dir, mid_mm), "seed_id": mid_mm,
            "verify": [drv("zkob_fc"), "verify", ob(run_dir, mid_mm), None,
                       str(SEQ), str(EMBED), str(EMBED), com_W,
                       P["gen1024"], P["gen1024"], P["q"]],
        }},
        mid_rs: {"rescale": {
            "obdir": ob(run_dir, mid_rs), "seed_id": mid_rs,
            "verify": [drv("zkob_rescale"), "verify", ob(run_dir, mid_rs), None,
                       str(SEQ), str(EMBED), str(QKV_RESCALE_LOG), P["gen1024"], P["q"]],
        }},
    }


def attention_spec(run_dir, l, submission="baseline"):
    """The integer attention chain for layer l, composed under the frozen
    manifest ids per ROPE_ATTENTION_DESIGN §4.0 (baseline) extended by
    STAGE3_FAITHFUL_DESIGN §4.3 (faithful-arch-v1):

      scores_matmul  = rope.q/k (+rescales) + headslice + 12 scores fc
      softmax        = 12 x (rescale13 + rescale10 + softmax)            [baseline]
                     = 12 x (rescale13 + rescale10 + rowmax + softmax8)  [faithful]
      values_matmul  = 12 x (values fc + values rescale) + headmerge
                       (perm = pi157 baseline / concat faithful, PHASE0 §21)
      o_proj.matmul/.rescaling (faithful only, §4.1: fc vs REGISTERED com +
                       rescale 2^16 between headmerge and attn_skip)

    Returns (spec_update, edges) with EVERY §7.4 edge A1..A15 (baseline) /
    A1..A14 + RM1/RM2/SX8a/SX8b per head + O1/O2/O3 (faithful — A10/A11 are
    superseded by the rowmax/softmax8 edges, A15 by the o_proj path). Edge kinds:
      ("byte", owner, label, a, b)            byte-equality of commitment files
      ("path", owner, label, file, mid, sub)  com_W path binding: the named
        sub's verify argv must reference `file` (structural; the driver absorbs
        the file so a divergent operand rejects at the transcript level)
    """
    P = reg_paths(run_dir)
    faithful = submission == "faithful-arch-v1"
    perm = PERM_FOR[submission]
    SM = f"layer{l}.attn.scores_matmul"
    SX = f"layer{l}.attn.softmax"
    VM = f"layer{l}.attn.values_matmul"
    B, C_, HD = str(SEQ), str(EMBED), str(HEAD_DIM)
    spec = {SM: {}, SX: {}, VM: {}}
    edges = []

    # -- scores_matmul subs: rope.q/k (+rescale), slice, fc.h{hh} -----------
    for t in ("q", "k"):
        spec[SM][f"rope.{t}"] = {
            "obdir": ob(run_dir, SM, f"rope.{t}"), "seed_id": f"layer{l}.attn.rope.{t}",
            "verify": [drv("zkob_rope"), "verify", ob(run_dir, SM, f"rope.{t}"), None,
                       B, C_, HD, P["rope_cos"], P["rope_sin"], P["gen1024"], P["q"]],
        }
        spec[SM][f"rope.{t}.rescale"] = {
            "obdir": ob(run_dir, SM, f"rope.{t}.rescale"),
            "seed_id": f"layer{l}.attn.rope.{t}.rescale",
            "verify": [drv("zkob_rescale"), "verify", ob(run_dir, SM, f"rope.{t}.rescale"), None,
                       B, C_, str(ROPE_RESCALE_LOG), P["gen1024"], P["q"]],
        }
    spec[SM]["slice"] = {
        "obdir": ob(run_dir, SM, "slice"), "seed_id": f"layer{l}.attn.slice",
        "verify": [drv("zkob_headslice"), "verify", ob(run_dir, SM, "slice"), None,
                   B, C_, HD, P["gen1024"], P["gen64"], P["q"]],
    }
    for hh in HH:
        spec[SM][f"fc.h{hh}"] = {
            "obdir": ob(run_dir, SM, f"fc.h{hh}"), "seed_id": f"layer{l}.attn.scores.h{hh}",
            "verify": [drv("zkob_fc"), "verify", ob(run_dir, SM, f"fc.h{hh}"), None,
                       B, HD, B, os.path.join(ob(run_dir, SM, "slice"), f"com_KhT{hh}.bin"),
                       P["gen64"], P["gen1024"], P["q"]],
        }

    # -- softmax subs: rescale13/rescale10 + softmax (baseline, SOFTMAX §4.0)
    #    or + rowmax/softmax8 (faithful, STAGE3 §4.3 — manifest composition:
    #    layer{l}.attn.softmax = 12 x (rescale13 + rescale10 + rowmax + softmax8))
    for hh in HH:
        spec[SX][f"rescale13.h{hh}"] = {
            "obdir": ob(run_dir, SX, f"rescale13.h{hh}"),
            "seed_id": f"layer{l}.attn.scores_rescale13.h{hh}",
            "verify": [drv("zkob_rescale"), "verify", ob(run_dir, SX, f"rescale13.h{hh}"), None,
                       B, B, str(SCORES_RESCALE13_LOG), P["gen1024"], P["q"]],
        }
        spec[SX][f"rescale10.h{hh}"] = {
            "obdir": ob(run_dir, SX, f"rescale10.h{hh}"),
            "seed_id": f"layer{l}.attn.scores_rescale10.h{hh}",
            "verify": [drv("zkob_rescale"), "verify", ob(run_dir, SX, f"rescale10.h{hh}"), None,
                       B, B, str(SCORES_RESCALE10_LOG), P["gen1024"], P["q"]],
        }
        if faithful:
            spec[SX][f"rowmax.h{hh}"] = {
                "obdir": ob(run_dir, SX, f"rowmax.h{hh}"),
                "seed_id": f"layer{l}.attn.rowmax.h{hh}",
                "verify": [drv("zkob_rowmax"), "verify", ob(run_dir, SX, f"rowmax.h{hh}"), None,
                           B, B, "causal", "0", str(SCORES_ROWMAX_LEN_R),
                           str(SCORES_ROWMAX_NPL), P["gen1024"], P["gen1024"], P["q"]],
            }
            spec[SX][f"softmax8.h{hh}"] = {
                "obdir": ob(run_dir, SX, f"softmax8.h{hh}"),
                "seed_id": f"layer{l}.attn.softmax8.h{hh}",
                "verify": [drv("zkob_softmax8"), "verify", ob(run_dir, SX, f"softmax8.h{hh}"), None,
                           B, B, str(SOFTMAX8_LOW), str(SOFTMAX8_LEN), P["exp8_table"],
                           str(SOFTMAX8_LEN_R), P["gen1024"], P["q"]],
            }
        else:
            spec[SX][f"softmax.h{hh}"] = {
                "obdir": ob(run_dir, SX, f"softmax.h{hh}"),
                "seed_id": f"layer{l}.attn.softmax.h{hh}",
                "verify": [drv("zkob_softmax"), "verify", ob(run_dir, SX, f"softmax.h{hh}"), None,
                           B, B, str(SOFTMAX_LOW_E), str(SOFTMAX_LEN_E), P["exp_table"],
                           str(SOFTMAX_LEN_R), P["gen1024"], P["q"]],
            }

    # -- values_matmul subs: fc/rescale per head + merge --------------------
    for hh in HH:
        spec[VM][f"fc.h{hh}"] = {
            "obdir": ob(run_dir, VM, f"fc.h{hh}"), "seed_id": f"layer{l}.attn.values.h{hh}",
            "verify": [drv("zkob_fc"), "verify", ob(run_dir, VM, f"fc.h{hh}"), None,
                       B, B, HD, os.path.join(ob(run_dir, SM, "slice"), f"com_Vh{hh}.bin"),
                       P["gen1024"], P["gen64"], P["q"]],
        }
        spec[VM][f"rescale.h{hh}"] = {
            "obdir": ob(run_dir, VM, f"rescale.h{hh}"),
            "seed_id": f"layer{l}.attn.values_rescale.h{hh}",
            "verify": [drv("zkob_rescale"), "verify", ob(run_dir, VM, f"rescale.h{hh}"), None,
                       B, HD, str(VALUES_RESCALE_LOG), P["gen64"], P["q"]],
        }
    spec[VM]["merge"] = {
        "obdir": ob(run_dir, VM, "merge"), "seed_id": f"layer{l}.attn.merge",
        "verify": [drv("zkob_headmerge"), "verify", ob(run_dir, VM, "merge"), None,
                   B, C_, HD, perm, P["gen1024"], P["gen64"], P["q"]],
    }
    if faithful:
        # §4.1: o_proj fc (vs the REGISTERED commitment) + rescale 2^16,
        # slotted between headmerge and attn_skip.
        o_mm = f"layer{l}.attn.o_proj.matmul"
        o_rs = f"layer{l}.attn.o_proj.rescaling"
        spec[o_mm] = {"fc": {
            "obdir": ob(run_dir, o_mm), "seed_id": o_mm,
            "verify": [drv("zkob_fc"), "verify", ob(run_dir, o_mm), None,
                       B, C_, C_, wpath(run_dir, f"layer{l}.attn.o_proj", "com"),
                       P["gen1024"], P["gen1024"], P["q"]],
        }}
        spec[o_rs] = {"rescale": {
            "obdir": ob(run_dir, o_rs), "seed_id": o_rs,
            "verify": [drv("zkob_rescale"), "verify", ob(run_dir, o_rs), None,
                       B, C_, str(OPROJ_RESCALE_LOG), P["gen1024"], P["q"]],
        }}

    # -- §7.4 edges ----------------------------------------------------------
    nid = f"layer{l}.input_norm.rmsnorm"
    attn_in_com = os.path.join(ob(run_dir, nid), "yrescale/com_Xr.bin")
    smd, sxd, vmd = ob(run_dir, SM), ob(run_dir, SX), ob(run_dir, VM)
    for pj, t in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
        mm, rs = f"layer{l}.attn.{pj}.matmul", f"layer{l}.attn.{pj}.rescaling"
        edges += [
            ("byte", mm, f"A1{t}: layer{l} {pj} fc com_X == input_norm yrescale com_Xr",
             os.path.join(ob(run_dir, mm), "com_X.bin"), attn_in_com),
            ("byte", rs, f"A2{t}: layer{l} {pj} fc com_Y == {pj} rescale com_X",
             os.path.join(ob(run_dir, mm), "com_Y.bin"), os.path.join(ob(run_dir, rs), "com_X.bin")),
        ]
        if t in ("q", "k"):
            edges += [
                ("byte", SM, f"A3{t}: layer{l} rope.{t} com_T == {pj} rescale com_Xr",
                 os.path.join(smd, f"rope.{t}/com_T.bin"), os.path.join(ob(run_dir, rs), "com_Xr.bin")),
                ("byte", SM, f"A4{t}: layer{l} rope.{t} com_Y64 == rope.{t}.rescale com_X",
                 os.path.join(smd, f"rope.{t}/com_Y64.bin"), os.path.join(smd, f"rope.{t}.rescale/com_X.bin")),
                ("byte", SM, f"A5{t}: layer{l} slice com_{t.upper()} == rope.{t}.rescale com_Xr",
                 os.path.join(smd, f"slice/com_{t.upper()}.bin"), os.path.join(smd, f"rope.{t}.rescale/com_Xr.bin")),
            ]
        else:
            edges.append(("byte", SM, f"A3v: layer{l} slice com_V == v_proj rescale com_Xr",
                          os.path.join(smd, "slice/com_V.bin"), os.path.join(ob(run_dir, rs), "com_Xr.bin")))
    for hh in HH:
        edges += [
            ("byte", SM, f"A6.{hh}: layer{l} slice com_Qh{hh} == scores fc.h{hh} com_X",
             os.path.join(smd, f"slice/com_Qh{hh}.bin"), os.path.join(smd, f"fc.h{hh}/com_X.bin")),
            ("path", SM, f"A7.{hh}: layer{l} slice com_KhT{hh} IS scores fc.h{hh} com_W argv",
             os.path.join(smd, f"slice/com_KhT{hh}.bin"), SM, f"fc.h{hh}"),
            ("byte", SM, f"A8.{hh}: layer{l} scores fc.h{hh} com_Y == rescale13.h{hh} com_X",
             os.path.join(smd, f"fc.h{hh}/com_Y.bin"), os.path.join(sxd, f"rescale13.h{hh}/com_X.bin")),
            ("byte", SX, f"A9.{hh}: layer{l} rescale13.h{hh} com_Xr == rescale10.h{hh} com_X",
             os.path.join(sxd, f"rescale13.h{hh}/com_Xr.bin"), os.path.join(sxd, f"rescale10.h{hh}/com_X.bin")),
        ]
        if faithful:
            # STAGE3 §2.7/§4.3 + PHASE0 §20 MINOR-1 (load-bearing): rowmax and
            # softmax8 bind the SAME score grid (RM1/SX8a), mx chains rowmax ->
            # softmax8 (RM2) — softmax8 alone neither range-binds z_ nor proves
            # allowed z-mx <= 0; the chained rowmax instance is the defense.
            edges += [
                ("byte", SX, f"RM1.{hh}: layer{l} rescale10.h{hh} com_Xr == rowmax.h{hh} com_z",
                 os.path.join(sxd, f"rescale10.h{hh}/com_Xr.bin"), os.path.join(sxd, f"rowmax.h{hh}/com_z.bin")),
                ("byte", SX, f"RM2.{hh}: layer{l} rowmax.h{hh} com_mx == softmax8.h{hh} com_mx",
                 os.path.join(sxd, f"rowmax.h{hh}/com_mx.bin"), os.path.join(sxd, f"softmax8.h{hh}/com_mx.bin")),
                ("byte", SX, f"SX8a.{hh}: layer{l} rescale10.h{hh} com_Xr == softmax8.h{hh} com_z",
                 os.path.join(sxd, f"rescale10.h{hh}/com_Xr.bin"), os.path.join(sxd, f"softmax8.h{hh}/com_z.bin")),
                ("byte", SX, f"SX8b.{hh}: layer{l} softmax8.h{hh} com_P == values fc.h{hh} com_X",
                 os.path.join(sxd, f"softmax8.h{hh}/com_P.bin"), os.path.join(vmd, f"fc.h{hh}/com_X.bin")),
            ]
        else:
            edges += [
                ("byte", SX, f"A10.{hh}: layer{l} rescale10.h{hh} com_Xr == softmax.h{hh} com_z",
                 os.path.join(sxd, f"rescale10.h{hh}/com_Xr.bin"), os.path.join(sxd, f"softmax.h{hh}/com_z.bin")),
                ("byte", SX, f"A11.{hh}: layer{l} softmax.h{hh} com_P == values fc.h{hh} com_X",
                 os.path.join(sxd, f"softmax.h{hh}/com_P.bin"), os.path.join(vmd, f"fc.h{hh}/com_X.bin")),
            ]
        edges += [
            ("path", VM, f"A12.{hh}: layer{l} slice com_Vh{hh} IS values fc.h{hh} com_W argv",
             os.path.join(smd, f"slice/com_Vh{hh}.bin"), VM, f"fc.h{hh}"),
            ("byte", VM, f"A13.{hh}: layer{l} values fc.h{hh} com_Y == rescale.h{hh} com_X",
             os.path.join(vmd, f"fc.h{hh}/com_Y.bin"), os.path.join(vmd, f"rescale.h{hh}/com_X.bin")),
            ("byte", VM, f"A14.{hh}: layer{l} values rescale.h{hh} com_Xr == merge com_O{hh}",
             os.path.join(vmd, f"rescale.h{hh}/com_Xr.bin"), os.path.join(vmd, f"merge/com_O{hh}.bin")),
        ]
    if faithful:
        # §4.1 edges O1/O2/O3 replace A15: the S1-closure now runs THROUGH o_proj.
        o_mm_d, o_rs_d = ob(run_dir, o_mm), ob(run_dir, o_rs)
        edges += [
            ("byte", o_mm, f"O1: layer{l} merge com_O2 == o_proj fc com_X (plain head-concat M)",
             os.path.join(vmd, "merge/com_O2.bin"), os.path.join(o_mm_d, "com_X.bin")),
            ("byte", o_rs, f"O2: layer{l} o_proj fc com_Y == o_proj rescale com_X",
             os.path.join(o_mm_d, "com_Y.bin"), os.path.join(o_rs_d, "com_X.bin")),
            ("byte", o_rs, f"O3: layer{l} o_proj rescale com_Xr == attn_skip com_attn_out "
                           "(the S1-closure edge, now through o_proj)",
             os.path.join(o_rs_d, "com_Xr.bin"),
             os.path.join(ob(run_dir, f"layer{l}.attn_skip.add"), "com_attn_out.bin")),
        ]
    else:
        edges.append(("byte", VM, f"A15: layer{l} merge com_O2 == attn_skip com_attn_out "
                                  "(closes edge S1's former OPEN BOUNDARY)",
                      os.path.join(vmd, "merge/com_O2.bin"),
                      os.path.join(ob(run_dir, f"layer{l}.attn_skip.add"), "com_attn_out.bin")))
    return spec, edges


def head_spec(run_dir):
    """Stage-3 head (§3.4): final_norm trio -> lm_head fc + rescale ->
    statement.logit_binding (zkob_rowmax vpad + registered t*).

    Returns (spec_update, edges). Edge F0 chains the final-norm input to the
    terminal residual commitment (layer1.mlp_skip com_Z — a FRESH gen1024
    commitment of the same int32 tensor, so byte-equality applies; the skip
    point check S2 validates the same file). L1 chains the committed logits
    grid into the rowmax instance; the registered tstar.i32.bin is passed to
    the rowmax verify as argv (hash-pinned via public.json — edge L2 is the
    registration hash check itself, enforced fail-closed before any driver).
    """
    P = reg_paths(run_dir)
    FN, MM, RS, LB = ("final_norm.rmsnorm", "lm_head.matmul",
                      "lm_head.rescaling", "statement.logit_binding")
    _, fn_subs = rmsnorm_site(run_dir, None, "final_norm")
    spec = {
        FN: fn_subs,
        MM: {"fc": {
            "obdir": ob(run_dir, MM), "seed_id": MM,
            "verify": [drv("zkob_fc"), "verify", ob(run_dir, MM), None,
                       str(SEQ), str(EMBED), str(VOCAB),
                       wpath(run_dir, "lm_head", "com"),
                       P["gen1024"], P["gen32768"], P["q"]],
        }},
        RS: {"rescale": {
            "obdir": ob(run_dir, RS), "seed_id": RS,
            "verify": [drv("zkob_rescale"), "verify", ob(run_dir, RS), None,
                       str(SEQ), str(VOCAB), str(LM_RESCALE_LOG),
                       P["gen32768"], P["q"]],
        }},
        LB: {"rowmax": {
            "obdir": ob(run_dir, LB, "rowmax"), "seed_id": LB,
            "verify": [drv("zkob_rowmax"), "verify", ob(run_dir, LB, "rowmax"), None,
                       str(SEQ), str(VOCAB_PAD), "vpad", str(VOCAB),
                       str(LOGIT_LEN_R), str(LOGIT_NPL),
                       P["gen32768"], P["gen1024"], P["q"], P["tstar"]],
        }},
    }
    fnd = ob(run_dir, FN)
    edges = [
        ("byte", FN, "F0: final_norm rmsnorm com_X == layer1.mlp_skip com_Z (terminal residual)",
         os.path.join(fnd, "rmsnorm/com_X.bin"),
         os.path.join(ob(run_dir, f"layer{N_LAYERS-1}.mlp_skip.add"), "com_Z.bin")),
        ("byte", FN, "F1: final_norm com_g == registered final_norm.g-com",
         os.path.join(fnd, "rmsnorm/com_g.bin"), wpath(run_dir, "final_norm.g", "com")),
        ("byte", FN, "F2: final_norm com_W == wrescale com_X",
         os.path.join(fnd, "rmsnorm/com_W.bin"), os.path.join(fnd, "wrescale/com_X.bin")),
        ("byte", FN, "F3: final_norm com_Wr == wrescale com_Xr",
         os.path.join(fnd, "rmsnorm/com_Wr.bin"), os.path.join(fnd, "wrescale/com_Xr.bin")),
        ("byte", FN, "F4: final_norm com_Y == yrescale com_X",
         os.path.join(fnd, "rmsnorm/com_Y.bin"), os.path.join(fnd, "yrescale/com_X.bin")),
        ("byte", MM, "F5: final_norm yrescale com_Xr == lm_head fc com_X",
         os.path.join(fnd, "yrescale/com_Xr.bin"), os.path.join(ob(run_dir, MM), "com_X.bin")),
        ("byte", RS, "F6: lm_head fc com_Y == lm_head rescale com_X",
         os.path.join(ob(run_dir, MM), "com_Y.bin"), os.path.join(ob(run_dir, RS), "com_X.bin")),
        ("byte", LB, "L1: lm_head rescale com_Xr == logit_binding rowmax com_z (committed logits grid)",
         os.path.join(ob(run_dir, RS), "com_Xr.bin"),
         os.path.join(ob(run_dir, LB, "rowmax"), "com_z.bin")),
    ]
    return spec, edges


def walk_spec(run_dir, submission="baseline"):
    """Full covered-subgraph spec: {manifest_id: {sub: spec}}, plus edges.

    Each sub spec's `verify` is a ready argv with two None holes: argv[3] = seed
    (verifier fills with run_seed:seed_id) and, for rmsnorm only, argv[6] = C_eps
    (filled from public.json constants).
    Edge kinds: ("byte", a, b) file byte-equality; ("skip", A, B, Z) Pedersen
    point check via zkob_skip verify.
    """
    P = reg_paths(run_dir)
    spec, edges = {}, []
    for l in range(N_LAYERS):
        nid, nsubs = rmsnorm_site(run_dir, l, "input_norm")
        pid, psubs = rmsnorm_site(run_dir, l, "post_attn_norm")
        spec[nid] = nsubs
        # attention chain: q/k/v projections, then scores/softmax/values
        for pj in ("q_proj", "k_proj", "v_proj"):
            spec.update(attn_proj_block(run_dir, l, pj))
        attn_spec, attn_edges = attention_spec(run_dir, l, submission)
        spec.update(attn_spec)
        edges += attn_edges
        spec[pid] = psubs
        for sid, site in ((nid, "input_norm"), (pid, "post_attn_norm")):
            o = ob(run_dir, sid)
            edges += [
                ("byte", sid, f"{sid}: com_g == registered", os.path.join(o, "rmsnorm/com_g.bin"),
                 wpath(run_dir, f"layer{l}.{site}.g", "com")),
                ("byte", sid, f"{sid}: com_W == wrescale com_X", os.path.join(o, "rmsnorm/com_W.bin"),
                 os.path.join(o, "wrescale/com_X.bin")),
                # pinned PHASE0 §14 MINOR-7: the driver saves com_W_ as com_Wr.bin
                ("byte", sid, f"{sid}: com_Wr == wrescale com_Xr", os.path.join(o, "rmsnorm/com_Wr.bin"),
                 os.path.join(o, "wrescale/com_Xr.bin")),
                ("byte", sid, f"{sid}: com_Y == yrescale com_X", os.path.join(o, "rmsnorm/com_Y.bin"),
                 os.path.join(o, "yrescale/com_X.bin")),
            ]
        # layer-0 statement boundary: residual stream starts at the registered input
        if l == 0:
            edges.append(("byte", nid, f"{nid}: com_X == registered com_input",
                          os.path.join(ob(run_dir, nid), "rmsnorm/com_X.bin"), P["com_input"]))

        # attn skip: Z1 = resid + attn_out (com_attn_out chained to merge com_O2
        # via A15 in baseline; through o_proj via O3 in faithful-arch-v1)
        a_skip = f"layer{l}.attn_skip.add"
        closure = "O3" if submission == "faithful-arch-v1" else "A15"
        spec[a_skip] = {"skip": {"obdir": ob(run_dir, a_skip), "seed_id": a_skip, "verify": None}}
        edges.append(("skip", a_skip, f"{a_skip}: com_X(input_norm) + com_attn_out == com_X(post_attn_norm) [boundary closed by {closure}]",
                      os.path.join(ob(run_dir, nid), "rmsnorm/com_X.bin"),
                      os.path.join(ob(run_dir, a_skip), "com_attn_out.bin"),
                      os.path.join(ob(run_dir, pid), "rmsnorm/com_X.bin")))

        # MLP path
        spec.update(fc_block(run_dir, l, "gate_proj", EMBED, INTER, GATE_RESCALE_LOG))
        spec.update(fc_block(run_dir, l, "up_proj", EMBED, INTER, UP_RESCALE_LOG))
        spec.update(fc_block(run_dir, l, "down_proj", INTER, EMBED, DOWN_RESCALE_LOG))
        sw = f"layer{l}.mlp.swiglu"
        spec[sw] = {
            "glu": {"obdir": ob(run_dir, sw, "glu"), "seed_id": sw,
                    "verify": [drv("zkob_glu"), "verify", ob(run_dir, sw, "glu"), None,
                               str(SEQ), str(INTER), str(SWIGLU_LOW), str(SWIGLU_LEN),
                               P["table"], P["gen4096"], P["q"]]},
            "hrescale": {"obdir": ob(run_dir, sw, "hrescale"), "seed_id": sw + ".hrescale",
                         "verify": [drv("zkob_rescale"), "verify", ob(run_dir, sw, "hrescale"), None,
                                    str(SEQ), str(INTER), str(HIDDEN_RESCALE_LOG), P["gen4096"], P["q"]]},
        }
        ffn_in_com = os.path.join(ob(run_dir, pid), "yrescale/com_Xr.bin")
        gmm, grs = ob(run_dir, f"layer{l}.mlp.gate_proj.matmul"), ob(run_dir, f"layer{l}.mlp.gate_proj.rescaling")
        umm, urs = ob(run_dir, f"layer{l}.mlp.up_proj.matmul"), ob(run_dir, f"layer{l}.mlp.up_proj.rescaling")
        dmm, drs = ob(run_dir, f"layer{l}.mlp.down_proj.matmul"), ob(run_dir, f"layer{l}.mlp.down_proj.rescaling")
        swd = ob(run_dir, sw)
        edges += [
            ("byte", f"layer{l}.mlp.gate_proj.matmul", f"layer{l}: gate_fc com_X == post_attn_norm yrescale com_Xr", os.path.join(gmm, "com_X.bin"), ffn_in_com),
            ("byte", f"layer{l}.mlp.up_proj.matmul", f"layer{l}: up_fc com_X == post_attn_norm yrescale com_Xr", os.path.join(umm, "com_X.bin"), ffn_in_com),
            ("byte", f"layer{l}.mlp.gate_proj.rescaling", f"layer{l}: gate_fc com_Y == gate_rescale com_X", os.path.join(gmm, "com_Y.bin"), os.path.join(grs, "com_X.bin")),
            ("byte", f"layer{l}.mlp.up_proj.rescaling", f"layer{l}: up_fc com_Y == up_rescale com_X", os.path.join(umm, "com_Y.bin"), os.path.join(urs, "com_X.bin")),
            ("byte", sw, f"layer{l}: glu com_G == gate_rescale com_Xr", os.path.join(swd, "glu/com_G.bin"), os.path.join(grs, "com_Xr.bin")),
            ("byte", sw, f"layer{l}: glu com_U == up_rescale com_Xr", os.path.join(swd, "glu/com_U.bin"), os.path.join(urs, "com_Xr.bin")),
            ("byte", sw, f"layer{l}: glu com_H == hrescale com_X", os.path.join(swd, "glu/com_H.bin"), os.path.join(swd, "hrescale/com_X.bin")),
            ("byte", f"layer{l}.mlp.down_proj.matmul", f"layer{l}: down_fc com_X == hrescale com_Xr", os.path.join(dmm, "com_X.bin"), os.path.join(swd, "hrescale/com_Xr.bin")),
            ("byte", f"layer{l}.mlp.down_proj.rescaling", f"layer{l}: down_fc com_Y == down_rescale com_X", os.path.join(dmm, "com_Y.bin"), os.path.join(drs, "com_X.bin")),
        ]

        # mlp skip: Z2 = Z1 + ffn_out
        m_skip = f"layer{l}.mlp_skip.add"
        spec[m_skip] = {"skip": {"obdir": ob(run_dir, m_skip), "seed_id": m_skip, "verify": None}}
        if l + 1 < N_LAYERS:
            z_com = os.path.join(ob(run_dir, f"layer{l+1}.input_norm.rmsnorm"), "rmsnorm/com_X.bin")
        else:
            z_com = os.path.join(ob(run_dir, m_skip), "com_Z.bin")  # terminal output commitment
        edges.append(("skip", m_skip, f"{m_skip}: com_X(post_attn_norm) + com_Xr(down_rescale) == next com_X",
                      os.path.join(ob(run_dir, pid), "rmsnorm/com_X.bin"),
                      os.path.join(drs, "com_Xr.bin"), z_com))

    # stage-3 head: final norm -> lm_head -> logit binding (§3.4)
    h_spec, h_edges = head_spec(run_dir)
    spec.update(h_spec)
    edges += h_edges

    return spec, edges


# ---------------------------------------------------------------------------
# Batched transport (TRANSPORT_REBUILD_DESIGN §6 Stage C2): the claim plan.
#
# Every claim-emitting driver run, in EXACT prove_walk order (the canonical
# claim order both sides must reproduce byte-for-byte for claims_match), with
# a deterministic sub-batch assignment. Sub-batches exist because batch_prove
# round-0 residency is ~2x sum_j 2^vars_j of the batch's distinct tensors
# (Fr = 32 B/element, post-streaming-fix) and the full walk's ~1.3 G elements
# (~42 GB) exceed the 24 GB card; claims partition freely across batches
# (design §8.3: Lemma 3 applies per batch). The partition is a deterministic
# function of the walk spec computed identically by prove_walk and
# verify_walk; a prover using ANY other partition fails claims_match on the
# first divergent sub-batch (fail-closed).
#
# The per-run element numbers are conservative static estimates of
# sum_{distinct tensors} 2^vars (from each driver's claim table); prove_walk
# asserts the EXACT per-batch totals (parsed from claims.bin) stay under
# SUBBATCH_HARD_CAP after each batch closes, so a drifting estimate fails
# loudly, never silently OOMs.
# ---------------------------------------------------------------------------
SUBBATCH_BUDGET_ELEMS = 220_000_000   # ~14 GB resident (2 x 32 B) + overhead
SUBBATCH_HARD_CAP_ELEMS = 260_000_000


def _est_fc(IN, OUT):
    INp, OUTp = pad2(IN), pad2(OUT)
    return INp * OUTp + SEQ * INp + SEQ * OUTp           # W, X, Y


def _est_rescale(C, sf_log):
    return 2 * SEQ * pad2(C) + (1 << sf_log)             # A, rem, m-table


def _est_rowmax(NCOL, len_r, npl):
    NCOLp = pad2(NCOL)
    return (2 * npl + 2) * SEQ * NCOLp + len_r + 2 * SEQ  # A_L, L, S, z (+m_L, mx)


_EST_RMSNORM = 16 << 20      # 17 claims; AL/L limb planes + 5 full grids
_EST_SOFTMAX8 = 17 << 20     # 19 claims; A_L/L 4B-row planes dominate
_EST_SOFTMAX = 17 << 20
_EST_GLU = 7 * SEQ * 4096 + SWIGLU_LEN
_EST_HEADSLICE = 3 * SEQ * 1024 + 36 * SEQ * 64
_EST_HEADMERGE = 2 << 20
_EST_ROPE = 2 * SEQ * 1024   # T (2 claims, one tensor) + Y64


def claim_plan(run_dir, submission="baseline"):
    """Ordered claim-emitting runs: [{mid, sub, obid, est, batch, extra, wid}].
    obid == the sub's seed_id (the --claims obligation id, both sides);
    extra == the registered/slice commitment path appended to fc/rmsnorm
    prove --claims blocks (fc: <registered-com_W>; rmsnorm: <registered-com_g>);
    wid == the registered weight tensor id when extra IS a registered weight
    commitment (None for slice commitments / non-weight runs) — under
    weight_privacy exactly these runs get --wpriv (their W/g claim routes to
    the weight accumulator; per-head scores/values fc open ACTIVATION slices
    and stay public by design). Returns (plan, n_batches)."""
    faithful = submission == "faithful-arch-v1"
    P = reg_paths(run_dir)
    entries = []

    def add(mid, sub, obid, est, extra=None, wid=None):
        entries.append({"mid": mid, "sub": sub, "obid": obid, "est": int(est),
                        "extra": extra, "wid": wid})

    def norm_site(mid, gwid):
        add(mid, "rmsnorm", mid, _EST_RMSNORM, wpath(run_dir, gwid, "com"),
            wid=gwid)
        add(mid, "wrescale", mid + ".wrescale", _est_rescale(EMBED, LOG_SF))
        add(mid, "yrescale", mid + ".yrescale", _est_rescale(EMBED, LOG_SF))

    for l in range(N_LAYERS):
        SM = f"layer{l}.attn.scores_matmul"
        SX = f"layer{l}.attn.softmax"
        VM = f"layer{l}.attn.values_matmul"
        norm_site(f"layer{l}.input_norm.rmsnorm", f"layer{l}.input_norm.g")
        for pj in ("q_proj", "k_proj", "v_proj"):
            mm, rs = f"layer{l}.attn.{pj}.matmul", f"layer{l}.attn.{pj}.rescaling"
            add(mm, "fc", mm, _est_fc(EMBED, EMBED),
                wpath(run_dir, f"layer{l}.attn.{pj}", "com"),
                wid=f"layer{l}.attn.{pj}")
            add(rs, "rescale", rs, _est_rescale(EMBED, QKV_RESCALE_LOG))
        for t in ("q", "k"):
            add(SM, f"rope.{t}", f"layer{l}.attn.rope.{t}", _EST_ROPE)
            add(SM, f"rope.{t}.rescale", f"layer{l}.attn.rope.{t}.rescale",
                _est_rescale(EMBED, ROPE_RESCALE_LOG))
        add(SM, "slice", f"layer{l}.attn.slice", _EST_HEADSLICE)
        for hh in HH:
            add(SM, f"fc.h{hh}", f"layer{l}.attn.scores.h{hh}",
                _est_fc(HEAD_DIM, SEQ),
                os.path.join(ob(run_dir, SM, "slice"), f"com_KhT{hh}.bin"))
            add(SX, f"rescale13.h{hh}", f"layer{l}.attn.scores_rescale13.h{hh}",
                _est_rescale(SEQ, SCORES_RESCALE13_LOG))
            add(SX, f"rescale10.h{hh}", f"layer{l}.attn.scores_rescale10.h{hh}",
                _est_rescale(SEQ, SCORES_RESCALE10_LOG))
            if faithful:
                add(SX, f"rowmax.h{hh}", f"layer{l}.attn.rowmax.h{hh}",
                    _est_rowmax(SEQ, SCORES_ROWMAX_LEN_R, SCORES_ROWMAX_NPL))
                add(SX, f"softmax8.h{hh}", f"layer{l}.attn.softmax8.h{hh}",
                    _EST_SOFTMAX8)
            else:
                add(SX, f"softmax.h{hh}", f"layer{l}.attn.softmax.h{hh}",
                    _EST_SOFTMAX)
            add(VM, f"fc.h{hh}", f"layer{l}.attn.values.h{hh}",
                _est_fc(SEQ, HEAD_DIM),
                os.path.join(ob(run_dir, SM, "slice"), f"com_Vh{hh}.bin"))
            add(VM, f"rescale.h{hh}", f"layer{l}.attn.values_rescale.h{hh}",
                _est_rescale(HEAD_DIM, VALUES_RESCALE_LOG))
        add(VM, "merge", f"layer{l}.attn.merge", _EST_HEADMERGE)
        if faithful:
            o_mm = f"layer{l}.attn.o_proj.matmul"
            o_rs = f"layer{l}.attn.o_proj.rescaling"
            add(o_mm, "fc", o_mm, _est_fc(EMBED, EMBED),
                wpath(run_dir, f"layer{l}.attn.o_proj", "com"),
                wid=f"layer{l}.attn.o_proj")
            add(o_rs, "rescale", o_rs, _est_rescale(EMBED, OPROJ_RESCALE_LOG))
        norm_site(f"layer{l}.post_attn_norm.rmsnorm", f"layer{l}.post_attn_norm.g")
        for pj, IN, OUT, rs_log in (("gate_proj", EMBED, INTER, GATE_RESCALE_LOG),
                                    ("up_proj", EMBED, INTER, UP_RESCALE_LOG),
                                    ("down_proj", INTER, EMBED, DOWN_RESCALE_LOG)):
            mm, rs = f"layer{l}.mlp.{pj}.matmul", f"layer{l}.mlp.{pj}.rescaling"
            add(mm, "fc", mm, _est_fc(IN, OUT),
                wpath(run_dir, f"layer{l}.mlp.{pj}", "com"),
                wid=f"layer{l}.mlp.{pj}")
            add(rs, "rescale", rs, _est_rescale(OUT, rs_log))
            if pj == "up_proj":   # prove_walk order: glu + hrescale after up
                sw = f"layer{l}.mlp.swiglu"
                add(sw, "glu", sw, _EST_GLU)
                add(sw, "hrescale", sw + ".hrescale",
                    _est_rescale(INTER, HIDDEN_RESCALE_LOG))
    norm_site("final_norm.rmsnorm", "final_norm.g")
    add("lm_head.matmul", "fc", "lm_head.matmul", _est_fc(EMBED, VOCAB),
        wpath(run_dir, "lm_head", "com"), wid="lm_head")
    add("lm_head.rescaling", "rescale", "lm_head.rescaling",
        _est_rescale(VOCAB, LM_RESCALE_LOG))
    add("statement.logit_binding", "rowmax", "statement.logit_binding",
        _est_rowmax(VOCAB_PAD, LOGIT_LEN_R, LOGIT_NPL))

    # greedy deterministic sub-batch assignment over prove order
    batch, cur = 0, 0
    for e in entries:
        if cur > 0 and cur + e["est"] > SUBBATCH_BUDGET_ELEMS:
            batch += 1
            cur = 0
        e["batch"] = batch
        cur += e["est"]
    return entries, batch + 1


def acc_dir(run_dir, k):
    """Prover accumulator + batch artifacts for sub-batch k (under proofs/ —
    the batch artifacts ARE proof artifacts the verifier consumes)."""
    return os.path.join(run_dir, "proofs", "opening_batch", f"b{k}")


def vacc_dir(run_dir, k):
    """Verifier-recomputed accumulator for sub-batch k (verifier-internal,
    F6: NEVER under proofs/, never a prover artifact)."""
    return os.path.join(run_dir, "vacc", f"b{k}")


def wacc_dir(run_dir):
    """Prover WEIGHT accumulator + weight sub-batch artifacts (Stage D; ONE
    batch — the 20 registered tensors sum well under the residency budget).
    Under proofs/ because claims.bin/drvstates.bin/wbatch_*/wipa_* ARE
    verifier-consumed; the prover-private files the wpriv drivers stage here
    (cblinds.bin, blindrefs.txt, wit_*.fr, witrefs.txt) are deleted/relocated
    to data/wpriv/ by prove_walk right after wprove, so the shipped proofs/
    tree never carries a hidden eval or blind."""
    return os.path.join(run_dir, "proofs", "opening_batch_w")


def wvacc_dir(run_dir):
    """Verifier-recomputed WEIGHT accumulator (verifier-internal, like vacc)."""
    return os.path.join(run_dir, "vacc", "w")


def wblind_path(run_dir, wid):
    """Registration row blinds for a hiding-registered weight tensor.
    PROVER-PRIVATE: lives under data/ (which the verifier NEVER reads, by the
    same rule as every witness file); consumed by prove --wpriv only."""
    return os.path.join(run_dir, "data", "wpriv", f"{wid}.blinds.bin")


def parse_cblinds(path):
    """cblinds.bin parser (zkob_wpriv.cuh wp_cblind_emit records: u32 id-len,
    id, v (32 B Fr LE), t (32 B Fr LE)). Returns {claim_id: (v_bytes, t_bytes)}.
    v IS the hidden weight-MLE evaluation — this file is PROVER-PRIVATE (the
    D4 scan reads the secrets to scan FOR from here)."""
    with open(path, "rb") as f:
        b = f.read()
    off, out = 0, {}
    while off < len(b):
        (l,) = struct.unpack_from("<I", b, off); off += 4
        cid = b[off:off + l].decode(); off += l
        v, t = b[off:off + 32], b[off + 32:off + 64]; off += 64
        out[cid] = (v, t)
    assert off == len(b), "trailing bytes in cblinds.bin"
    return out


def batch_seed(run_seed, k):
    """Per-sub-batch seed; zkob_batchopen appends ':opening_batch' itself."""
    return f"{run_seed}:b{k}"


def genspec_args(rel=True):
    """zkob_batchopen generator spec: ONE registration gen file per domain
    size (the registration invariant the per-domain RLC is a commitment
    under), run-dir-relative."""
    return [f"{g}=registration/{fname}" for g, fname in sorted(GEN_FOR.items())]


def rel_argv(cmd, run_dir):
    """Relativize every argv path under run_dir (driver binary stays
    absolute). Batched-transport invocations use cwd=run_dir so claims.bin
    comrefs come out run-dir-relative on both sides."""
    out = []
    for c in cmd:
        if isinstance(c, str) and c.startswith(run_dir + os.sep):
            out.append(os.path.relpath(c, run_dir))
        else:
            out.append(c)
    return out


def parse_claims(path):
    """claims.bin parser (format: zkob_claims.cuh claim_blob); returns dicts
    with id/comref/domain/n_rows/n_point/tag."""
    with open(path, "rb") as f:
        b = f.read()
    assert b[:4] == b"ZKCL", "bad claims magic"
    ver, n = struct.unpack_from("<II", b, 4)
    assert ver == 1, f"bad claims version {ver}"
    off = 12
    out = []
    for _ in range(n):
        start = off
        (l,) = struct.unpack_from("<I", b, off); off += 4
        cid = b[off:off + l].decode(); off += l
        (l,) = struct.unpack_from("<I", b, off); off += 4
        comref = b[off:off + l].decode(); off += l
        dom, rows, np_ = struct.unpack_from("<III", b, off); off += 12
        off += np_ * 32
        tag = b[off]; off += 1
        off += 32 if tag == 0 else 144
        out.append({"id": cid, "comref": comref, "domain": dom,
                    "n_rows": rows, "n_point": np_, "tag": tag,
                    "raw": b[start:off]})
    assert off == len(b), "trailing bytes in claims.bin"
    return out


def localize_batch_failure(pacc_claims_path, vacc_claims_path):
    """Stage C2 localization: a failed sub-batch must pinpoint the offending
    id(s), never blame every id sharing the batch. Multiset-diff the prover's
    claims.bin against the VERIFIER-recomputed list (the ground truth) on the
    canonical per-claim record bytes; the diverging records' claim ids name
    exactly the tampered/forged obligations (a driver whose transcript
    diverged emits different points; a driver that rejected driver-side never
    emitted, so its claims appear prover-side only; a forged prover record
    appears prover-side only).

    Returns (diverging_claim_ids, note). Empty ids + note means the claim
    lists are byte-identical: the failure lives in the batch proof artifacts
    themselves (sumcheck/vfin/ipa bytes) or is an SZ-caught false-claim batch
    — information-theoretically not attributable to a single id, so the named
    locus stays the batch check (the C1 §3 relocated-locus discipline)."""
    vclaims = parse_claims(vacc_claims_path)   # verifier-internal: must parse
    try:
        pclaims = parse_claims(pacc_claims_path)
    except (AssertionError, OSError, UnicodeDecodeError, struct.error):
        return [], ("prover claims.bin unparseable — batch proof artifact "
                    "(no per-id attribution)")
    from collections import Counter
    pc = Counter(c["raw"] for c in pclaims)
    vc = Counter(c["raw"] for c in vclaims)
    raw2id = {c["raw"]: c["id"] for c in pclaims + vclaims}
    ids = sorted({raw2id[r] for r in (pc - vc) | (vc - pc)})
    if not ids:
        return [], ("prover and verifier claim lists byte-identical — failure "
                    "is in the batch proof artifacts (or an SZ-caught false "
                    "claim); no single id implicable")
    return ids, f"{len(ids)} claim record(s) diverge from verifier recomputation"


def registered_weight_comrefs(submission="baseline"):
    """Every registered weight commitment the walk must open IN THE BATCH
    (the F5 discharge pin: the orchestrator asserts these comrefs appear in
    the verifier-recomputed claim lists, making the *.commitment_opening
    discharge explicit). Run-dir-relative comref strings."""
    wids = [wid for wid, *_ in weight_specs(submission)]
    wids += [wid for wid, *_ in head_weight_specs()]
    return {f"registration/weights/{wid}-com.bin" for wid in wids}
