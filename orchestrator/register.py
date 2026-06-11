"""One-time registration for a ZK-verified llama-68m run (stage 2).

Produces <run>/registration/ (gens incl. gen64, integer weights + registered
commitments, public input + its commitment, swiglu table, rope cos/sin tables,
softmax exp table) and <run>/public.json whose sha256 IS the run seed.
See ORCHESTRATOR_DESIGN.md §2 + ROPE_ATTENTION_DESIGN.md §2.1/§4.5.

Run with the pipeline env: /root/int-model-env/bin/python register.py [--run-id X]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MODEL_CARD = "JackFram/llama-68m"
CACHE_DIR = os.path.join(C.ZKLLM, "model-storage")
PIPELINE_DUMP = os.path.join(C.ZKLLM, "zkllm-workdir", "llama-68m")


def export_weights(run_dir, log=print):
    """Export integer weights with the pipeline's exact semantics and
    cross-check against the pipeline's own dumps where they exist."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_CARD, cache_dir=CACHE_DIR)
    eps = model.config.rms_norm_eps
    sf = 1 << C.LOG_SF
    name_of = {  # wid stem -> pipeline parameter name within the layer
        "mlp.gate_proj": "mlp.gate_proj.weight", "mlp.up_proj": "mlp.up_proj.weight",
        "mlp.down_proj": "mlp.down_proj.weight",
        "attn.q_proj": "self_attn.q_proj.weight", "attn.k_proj": "self_attn.k_proj.weight",
        "attn.v_proj": "self_attn.v_proj.weight",
        "input_norm.g": "input_layernorm.weight", "post_attn_norm.g": "post_attention_layernorm.weight",
    }
    os.makedirs(os.path.join(run_dir, "registration", "weights"), exist_ok=True)
    for wid, dump_stem, IN, OUT, _gen in C.weight_specs():
        l = int(wid.split(".")[0][len("layer"):])
        pname = name_of[wid.split(".", 1)[1]]
        w = dict(model.model.layers[l].named_parameters())[pname]
        # m68-pipeline.py lines 92-97: 2-D -> w.float().T, 1-D -> w.float(); round(*2^16)
        w_orig = w.float().T if len(w.shape) == 2 else w.float()
        w_int = torch.round(w_orig * sf).to(torch.int32).detach().cpu().numpy().astype(np.int32)
        assert w_int.size == IN * OUT, (wid, w_int.shape)
        path = C.wpath(run_dir, wid, "int")
        w_int.tofile(path)
        dump = os.path.join(PIPELINE_DUMP, f"{dump_stem}-int.bin")
        if os.path.exists(dump):
            if open(dump, "rb").read() != open(path, "rb").read():
                raise RuntimeError(f"provenance check FAILED: {path} != pipeline dump {dump}")
            log(f"  exported {wid} ({IN}x{OUT}) [matches pipeline dump]")
        else:
            log(f"  exported {wid} ({IN}x{OUT}) [no pipeline dump to compare]")
    return eps


def gen_input(run_dir, run_id, log=print):
    """Pipeline convention (m68-pipeline.py line 112): round(randn(seq,embed)*2^16).
    The pipeline uses unseeded torch.randn; we pin a numpy seed for reproducibility."""
    seed = int.from_bytes(hashlib.sha256(f"zkorch-input:{run_id}".encode()).digest()[:4], "little")
    rng = np.random.RandomState(seed)
    x = np.rint(rng.standard_normal((C.SEQ, C.EMBED)) * (1 << C.LOG_SF)).astype(np.int32)
    p = C.reg_paths(run_dir)["input"]
    x.tofile(p)
    log(f"  input: randn seed {seed}, {x.shape}, int32 @2^{C.LOG_SF}")
    return p


def gen_swiglu_table(run_dir, log=print):
    """m68-pipeline.py lines 104-105 verbatim (GPU float32, like the pipeline)."""
    import torch
    Xs = torch.arange(-(1 << 9), 1 << 9, step=1 / (1 << 12), device=0)
    vals = torch.round(Xs * torch.sigmoid(Xs) * (1 << 16)).to(torch.int32)
    p = C.reg_paths(run_dir)["table"]
    vals.cpu().numpy().astype(np.int32).tofile(p)
    assert vals.numel() == C.SWIGLU_LEN
    assert int(vals[-C.SWIGLU_LOW].item()) == 0, "table must map 0 -> 0 (zkob_glu layout)"
    log(f"  swiglu table: len {vals.numel()}, low {C.SWIGLU_LOW}")
    return p


def gen_attention_tables(run_dir, log=print):
    """Run the PINNED generator scripts from /root/zkllm (the sole authority
    for table bytes: gen_rope_tables.py per ROPE_ATTENTION_DESIGN §2.1,
    gen_softmax_exp_table.py per SOFTMAX_DESIGN §7.4 / PHASE0 §15). The
    scripts write to cwd, so they run with cwd = registration/ — /root/zkllm
    is never written. The sha256 registration below, not regeneration, is the
    source of truth thereafter."""
    import subprocess
    P = C.reg_paths(run_dir)
    for script, outs in (("gen_rope_tables.py", ("rope-cos-table.bin", "rope-sin-table.bin")),
                         ("gen_softmax_exp_table.py", ("softmax-exp-table.bin",))):
        spath = os.path.join(C.ZKLLM, script)
        r = subprocess.run([sys.executable, spath], cwd=P["reg"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"{script} failed: {r.stdout}{r.stderr}")
        for o in outs:
            p = os.path.join(P["reg"], o)
            if not os.path.exists(p):
                raise RuntimeError(f"{script} did not produce {o}")
            log(f"  {o}: {os.path.getsize(p)} bytes (from {spath})")
    # sanity anchors pinned by the designs: cos(0) = 2^16; exp(0) = 2^16
    cos0 = np.fromfile(P["rope_cos"], dtype=np.int32, count=1)[0]
    assert cos0 == (1 << 16), f"rope cos table anchor wrong: cos[0,0]={cos0}"
    exp = np.fromfile(P["exp_table"], dtype=np.int32)
    assert exp.size == C.SOFTMAX_LEN_E and exp[-C.SOFTMAX_LOW_E] == (1 << 16), \
        "softmax exp table anchor wrong (exp(0) != 2^16)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=time.strftime("run-%Y%m%d-%H%M%S"))
    ap.add_argument("--root", default=C.RUN_ROOT)
    args = ap.parse_args()
    run_dir = os.path.join(args.root, args.run_id)
    P = C.reg_paths(run_dir)
    os.makedirs(P["weights"], exist_ok=True)
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "proofs"), exist_ok=True)
    print(f"== register: {run_dir} ==")

    print("-- gens (ppgen) --")
    for n, key in ((64, "gen64"), (1024, "gen1024"), (4096, "gen4096"), (1, "q")):
        C.run_driver([C.drv("ppgen"), str(n), P[key]], f"ppgen {n}")

    print("-- weights (pipeline semantics) --")
    eps = export_weights(run_dir)
    c_eps = round(eps * C.EMBED * (1 << 32))
    print(f"  rms_norm_eps={eps} -> C_eps={c_eps}")

    print("-- registered commitments (zkob_fc commit) --")
    for wid, _stem, IN, OUT, gen in C.weight_specs():
        C.run_driver([C.drv("zkob_fc"), "commit", C.wpath(run_dir, wid, "int"),
                      str(IN), str(OUT), P[C.GEN_FOR[gen].split(".")[0]],
                      C.wpath(run_dir, wid, "com")], f"commit {wid}")

    print("-- input + commitment --")
    gen_input(run_dir, args.run_id)
    C.run_driver([C.drv("zkob_fc"), "commit", P["input"], str(C.SEQ), str(C.EMBED),
                  P["gen1024"], P["com_input"]], "commit input")

    print("-- swiglu table --")
    gen_swiglu_table(run_dir)

    print("-- attention tables (rope cos/sin + softmax exp) --")
    gen_attention_tables(run_dir)

    print("-- public.json --")
    public = {
        "model": MODEL_CARD, "seq_len": C.SEQ, "run_id": args.run_id,
        "prompt_token_ids": None,
        "note_input": "pipeline starts from a random-normal activation (embedding waived); "
                      "the input file digest below is the prompt-binding analog",
        "constants": {
            "LOG_SF": C.LOG_SF, "GATE_RESCALE_LOG": C.GATE_RESCALE_LOG,
            "UP_RESCALE_LOG": C.UP_RESCALE_LOG, "HIDDEN_RESCALE_LOG": C.HIDDEN_RESCALE_LOG,
            "DOWN_RESCALE_LOG": C.DOWN_RESCALE_LOG,
            "SWIGLU_LOW": C.SWIGLU_LOW, "SWIGLU_LEN": C.SWIGLU_LEN,
            "rms_norm_eps": eps, "C_eps": c_eps,
            "EMBED": C.EMBED, "INTER": C.INTER, "N_LAYERS": C.N_LAYERS,
            # attention chain (ROPE_ATTENTION_DESIGN §1.5; SOFTMAX_DESIGN §1.1)
            "HEAD_DIM": C.HEAD_DIM, "N_HEADS": C.N_HEADS,
            "QKV_RESCALE_LOG": C.QKV_RESCALE_LOG, "ROPE_RESCALE_LOG": C.ROPE_RESCALE_LOG,
            "SCORES_RESCALE13_LOG": C.SCORES_RESCALE13_LOG,
            "SCORES_RESCALE10_LOG": C.SCORES_RESCALE10_LOG,
            "VALUES_RESCALE_LOG": C.VALUES_RESCALE_LOG,
            "SOFTMAX_LOW_E": C.SOFTMAX_LOW_E, "SOFTMAX_LEN_E": C.SOFTMAX_LEN_E,
            "SOFTMAX_LEN_R": C.SOFTMAX_LEN_R,
        },
        "gens": {k: C.sha256_file(P[k]) for k in ("gen64", "gen1024", "gen4096", "q")},
        "registered_weight_commitments": {
            wid: C.sha256_file(C.wpath(run_dir, wid, "com")) for wid, *_ in C.weight_specs()
        },
        "input": {"file": "registration/input.i32.bin", "sha256": C.sha256_file(P["input"]),
                  "commitment_sha256": C.sha256_file(P["com_input"])},
        "tables": {
            "swiglu-table.bin": C.sha256_file(P["table"]),
            "rope-cos-table.bin": C.sha256_file(P["rope_cos"]),
            "rope-sin-table.bin": C.sha256_file(P["rope_sin"]),
            "softmax-exp-table.bin": C.sha256_file(P["exp_table"]),
        },
        "covered_subgraph": "stage2-full-forward (MLP + rmsnorm + skips + complete attention "
                            "chain incl. rope/headslice/softmax/headmerge; lm_head pending)",
        "future_slots": {
            "statement.final_argmax": "reserved: served token == argmax(logits) within DiFR tolerance "
                                      "(THREAT_MODEL_NOTES §1); lands with lm_head",
            "lm_head.commitment_opening": "reserved: needs gen32768 registration; lands with lm_head.matmul",
        },
    }
    pj = os.path.join(run_dir, "public.json")
    with open(pj, "w") as f:
        json.dump(public, f, indent=2, sort_keys=True)
    print(f"  run_seed = {C.run_seed_of(run_dir)}")
    print(f"REGISTERED {run_dir}")


if __name__ == "__main__":
    main()
