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
import subprocess
import time

ZKLLM = "/root/zkllm"
RUN_ROOT = "/root/zkorch"
GPU_LOCK = "/tmp/zkorch.gpu.lock"

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
               timeout=DRIVER_TIMEOUT_S):
    """Run one driver invocation, serialized on the shared GPU via a lock file.

    Returns (accepted: bool, seconds: float, output: str).
    Exit 0 = ACCEPT/success; exit 1 = REJECT (meaningful for verify modes, not
    retried); anything else = crash (CUDA contention etc.) -> retried once.
    Timeout = fail (RuntimeError, no retry): liveness hardening, fail-closed.
    """
    attempt = 0
    while True:
        attempt += 1
        t0 = time.time()
        with open(GPU_LOCK, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                r = subprocess.run(cmd, cwd=ZKLLM, capture_output=True, text=True,
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


# ---------------------------------------------------------------------------
# Registered weights: (wid, pipeline dump stem, IN, OUT, gen size for commit)
# 2-D weights are exported as (IN, OUT) = w.float().T per pipeline semantics;
# rmsnorm gains as (1, EMBED). q/k/v registered now for attention-stage compat.
# ---------------------------------------------------------------------------
def weight_specs():
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


def covered_ids():
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


def attention_spec(run_dir, l):
    """The §7.3 integer attention chain for layer l, composed under the frozen
    manifest ids per ROPE_ATTENTION_DESIGN §4.0:

      scores_matmul  = rope.q/k (+rescales) + headslice + 12 scores fc
      softmax        = 12 x (rescale13 + rescale10 + softmax)
      values_matmul  = 12 x (values fc + values rescale) + headmerge

    Returns (spec_update, edges) with EVERY §7.4 edge A1..A15. Edge kinds:
      ("byte", owner, label, a, b)            byte-equality of commitment files
      ("path", owner, label, file, mid, sub)  com_W path binding: the named
        sub's verify argv must reference `file` (structural; the driver absorbs
        the file so a divergent operand rejects at the transcript level)
    """
    P = reg_paths(run_dir)
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

    # -- softmax subs: rescale13/rescale10/softmax per head (SOFTMAX §4.0) --
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
                   B, C_, HD, "pi157", P["gen1024"], P["gen64"], P["q"]],
    }

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
            ("byte", SX, f"A10.{hh}: layer{l} rescale10.h{hh} com_Xr == softmax.h{hh} com_z",
             os.path.join(sxd, f"rescale10.h{hh}/com_Xr.bin"), os.path.join(sxd, f"softmax.h{hh}/com_z.bin")),
            ("byte", SX, f"A11.{hh}: layer{l} softmax.h{hh} com_P == values fc.h{hh} com_X",
             os.path.join(sxd, f"softmax.h{hh}/com_P.bin"), os.path.join(vmd, f"fc.h{hh}/com_X.bin")),
            ("path", VM, f"A12.{hh}: layer{l} slice com_Vh{hh} IS values fc.h{hh} com_W argv",
             os.path.join(smd, f"slice/com_Vh{hh}.bin"), VM, f"fc.h{hh}"),
            ("byte", VM, f"A13.{hh}: layer{l} values fc.h{hh} com_Y == rescale.h{hh} com_X",
             os.path.join(vmd, f"fc.h{hh}/com_Y.bin"), os.path.join(vmd, f"rescale.h{hh}/com_X.bin")),
            ("byte", VM, f"A14.{hh}: layer{l} values rescale.h{hh} com_Xr == merge com_O{hh}",
             os.path.join(vmd, f"rescale.h{hh}/com_Xr.bin"), os.path.join(vmd, f"merge/com_O{hh}.bin")),
        ]
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


def walk_spec(run_dir):
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
        attn_spec, attn_edges = attention_spec(run_dir, l)
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

        # attn skip: Z1 = resid + attn_out (com_attn_out chained to merge com_O2 via A15)
        a_skip = f"layer{l}.attn_skip.add"
        spec[a_skip] = {"skip": {"obdir": ob(run_dir, a_skip), "seed_id": a_skip, "verify": None}}
        edges.append(("skip", a_skip, f"{a_skip}: com_X(input_norm) + com_attn_out == com_X(post_attn_norm) [boundary closed by A15]",
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
