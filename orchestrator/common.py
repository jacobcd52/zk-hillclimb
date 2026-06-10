"""Shared definitions for the zk-hillclimb orchestrator (stage 1: MLP subgraph).

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

GEN_FOR = {1024: "gen1024.bin", 4096: "gen4096.bin"}


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


def reg_paths(run_dir):
    reg = os.path.join(run_dir, "registration")
    return {
        "reg": reg,
        "gen1024": os.path.join(reg, "gen1024.bin"),
        "gen4096": os.path.join(reg, "gen4096.bin"),
        "q": os.path.join(reg, "q.bin"),
        "weights": os.path.join(reg, "weights"),
        "input": os.path.join(reg, "input.i32.bin"),
        "com_input": os.path.join(reg, "com_input.bin"),
        "table": os.path.join(reg, "swiglu-table.bin"),
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
        ids += [
            f"layer{l}.input_norm.rmsnorm",
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
    ids += ["statement.registered_weight_hash", "statement.prompt_binding"]
    return ids


def skipped_ids():
    sk = {}
    for l in range(N_LAYERS):
        for pj in ("q_proj", "k_proj", "v_proj"):
            for kind in ("commitment_opening", "matmul", "rescaling"):
                sk[f"layer{l}.attn.{pj}.{kind}"] = "stage1: attention chain pending (zkob_softmax in flight)"
        sk[f"layer{l}.attn.scores_matmul"] = "stage1: attention chain pending"
        sk[f"layer{l}.attn.softmax"] = "stage1: zkob_softmax being built by a parallel agent"
        sk[f"layer{l}.attn.values_matmul"] = "stage1: attention chain pending"
    sk["lm_head.commitment_opening"] = ("stage1: lm_head not proven; a standalone opening with no "
                                        "matmul claim to discharge would be vacuous — deferred with lm_head.matmul")
    sk["statement.logit_binding"] = ("stage1: no proven logits exist (lm_head skipped); reserved slot — will also "
                                     "carry served-token == argmax binding (THREAT_MODEL_NOTES §1)")
    return sk


def rmsnorm_site(run_dir, l, site):
    """site in {'input_norm', 'post_attn_norm'} -> (manifest_id, sub specs)."""
    mid = f"layer{l}.{site}.rmsnorm"
    P = reg_paths(run_dir)
    gC = str(EMBED)
    com_g = wpath(run_dir, f"layer{l}.{site}.g", "com")
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

        # attn skip: Z1 = resid + attn_out (com_attn_out is the OPEN attention boundary)
        a_skip = f"layer{l}.attn_skip.add"
        spec[a_skip] = {"skip": {"obdir": ob(run_dir, a_skip), "seed_id": a_skip, "verify": None}}
        edges.append(("skip", a_skip, f"{a_skip}: com_X(input_norm) + com_attn_out == com_X(post_attn_norm) [attn boundary OPEN]",
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

    return spec, edges
