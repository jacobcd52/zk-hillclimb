"""One-time registration for a ZK-verified llama-68m run (stage 3).

Produces <run>/registration/ (gens incl. gen64 + gen32768, integer weights +
registered commitments incl. final_norm.g and lm_head, public input + its
commitment, swiglu table, rope cos/sin tables, softmax exp table) and
<run>/public.json whose sha256 IS the run seed. Stage 3 (STAGE3_FAITHFUL_DESIGN
§3.2-§3.4) adds Phase B: the deterministic integer-chain head pass from the
registered input through the registered weights to integer logits, whose
argmax t* is written to registration/tstar.i32.bin and sha256-pinned INSIDE
public.json BEFORE sealing — so run_seed = sha256(public.json) binds the
served tokens into every FS transcript.
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
    return eps, model


def export_head(run_dir, model, log=print):
    """Stage-3 head weights (STAGE3 §3.1/§3.2): final_norm.g and lm_head with
    the same pipeline export semantics. The pipeline never dumps model.norm or
    lm_head (its commit loop only iterates model.model.layers[*]), so the
    provenance guard here is re-export comparison only — documented deviation
    from the byte-compare-vs-pipeline-dump rule."""
    import torch
    sf = 1 << C.LOG_SF
    # PINNED assert (§3.1): do NOT silently inherit the per-layer eps value.
    fin_eps = float(model.model.norm.variance_epsilon)
    if fin_eps != 1e-6:
        raise RuntimeError(f"model.model.norm.variance_epsilon = {fin_eps} != 1e-6 "
                           "(STAGE3 §3.1 pins C_eps=3298535 for the final-norm site)")
    g = torch.round(model.model.norm.weight.float() * sf).to(torch.int32)
    g.cpu().numpy().astype(np.int32).tofile(C.wpath(run_dir, "final_norm.g", "int"))
    log(f"  exported final_norm.g (1x{C.EMBED}) [no pipeline dump exists: re-export guard only]")
    # llama-68m may tie lm_head to the embedding; HF materializes
    # model.lm_head.weight either way — semantics identical, flag recorded (§6.8).
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    w = torch.round(model.lm_head.weight.float().T * sf).to(torch.int32).cpu().numpy()
    assert w.shape == (C.EMBED, C.VOCAB), f"lm_head shape {w.shape} != ({C.EMBED},{C.VOCAB})"
    w.astype(np.int32).tofile(C.wpath(run_dir, "lm_head", "int"))
    log(f"  exported lm_head ({C.EMBED}x{C.VOCAB}) [tie_word_embeddings={tied}; "
        "no pipeline dump exists: re-export guard only]")
    return tied


def head_pass(run_dir, c_eps, log=print):
    """Phase B (STAGE3 §3.4): deterministic integer-chain head pass — the
    validated numpy forward (measure/int_chain.py semantics) from the
    registered input through the registered weights to integer logits, with
    the final-norm advice R SWITCHED to the integer-exact bracket (§3.1; the
    same witness-authority rule as the four per-layer sites). t*[i] =
    np.argmax(logits[i, :V]) — lowest-index tie-break, matching zkob_rowmax's
    canonical witness. Writes registration/tstar.i32.bin."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "measure"))
    from int_chain import IntChain, imatmul, rescale
    from prove_walk import compute_R
    P = C.reg_paths(run_dir)
    chain = IntChain(P["reg"])
    chain.g_final = chain.w["final_norm.g"].astype(np.int64)
    chain.w_lm = chain.w["lm_head"].reshape(C.EMBED, C.VOCAB)
    x0 = np.fromfile(P["input"], dtype=np.int32).reshape(C.SEQ, C.EMBED)
    t0 = time.time()
    resid = chain.forward(x0)
    R = compute_R(resid, c_eps).astype(np.int64)        # exact-R switch (§3.1)
    normed = chain._rmsnorm_body(resid, R, gain=chain.g_final)
    logits = rescale(imatmul(normed, chain.w_lm), C.LOG_SF)   # int @2^16
    assert logits.shape == (C.SEQ, C.VOCAB)
    tstar = np.argmax(logits, axis=1).astype(np.int32)  # lowest-index tie-break
    tstar.tofile(P["tstar"])
    log(f"  head pass: {time.time()-t0:.1f}s; logits in [{logits.min()}, {logits.max()}] "
        f"@2^{C.LOG_SF}; t* -> {P['tstar']}")
    return P["tstar"]


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
    for n, key in ((64, "gen64"), (1024, "gen1024"), (4096, "gen4096"),
                   (32768, "gen32768"), (1, "q")):
        C.run_driver([C.drv("ppgen"), str(n), P[key]], f"ppgen {n}")

    print("-- weights (pipeline semantics) --")
    eps, model = export_weights(run_dir)
    c_eps = round(eps * C.EMBED * (1 << 32))
    print(f"  rms_norm_eps={eps} -> C_eps={c_eps}")
    lm_head_tied = export_head(run_dir, model)
    del model

    print("-- registered commitments (zkob_fc commit) --")
    for wid, _stem, IN, OUT, gen in C.weight_specs():
        C.run_driver([C.drv("zkob_fc"), "commit", C.wpath(run_dir, wid, "int"),
                      str(IN), str(OUT), P[C.GEN_FOR[gen].split(".")[0]],
                      C.wpath(run_dir, wid, "com")], f"commit {wid}")
    for wid, IN, OUT, gen in C.head_weight_specs():
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

    print("-- phase B: head pass -> registered served tokens t* (STAGE3 §3.4) --")
    head_pass(run_dir, c_eps)

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
            # stage-3 head (STAGE3 §3.2/§3.3)
            "VOCAB": C.VOCAB, "VOCAB_PAD": C.VOCAB_PAD,
            "LM_RESCALE_LOG": C.LM_RESCALE_LOG,
            "LOGIT_LEN_R": C.LOGIT_LEN_R, "LOGIT_NPL": C.LOGIT_NPL,
        },
        "lm_head_tied": lm_head_tied,
        "gens": {k: C.sha256_file(P[k])
                 for k in ("gen64", "gen1024", "gen4096", "gen32768", "q")},
        "registered_weight_commitments": {
            wid: C.sha256_file(C.wpath(run_dir, wid, "com"))
            for wid in ([w for w, *_ in C.weight_specs()]
                        + [w for w, *_ in C.head_weight_specs()])
        },
        "served_tokens": {
            "file": "registration/tstar.i32.bin", "sha256": C.sha256_file(P["tstar"]),
            "note": "t*[i] = argmax_v logits[i, v<32000] of the registered random input's "
                    "logits (lowest-index tie-break), bound by statement.logit_binding "
                    "(zkob_rowmax vpad + T-BIND) at all 1024 positions; for random-input "
                    "runs these are the served tokens of the registered input (STAGE3 §3.3)",
        },
        "input": {"file": "registration/input.i32.bin", "sha256": C.sha256_file(P["input"]),
                  "commitment_sha256": C.sha256_file(P["com_input"])},
        "tables": {
            "swiglu-table.bin": C.sha256_file(P["table"]),
            "rope-cos-table.bin": C.sha256_file(P["rope_cos"]),
            "rope-sin-table.bin": C.sha256_file(P["rope_sin"]),
            "softmax-exp-table.bin": C.sha256_file(P["exp_table"]),
        },
        "covered_subgraph": "stage3-full-statement (MLP + rmsnorm + skips + complete attention "
                            "chain + final_norm + lm_head + served-token argmax binding; "
                            "the only waived-and-uncovered manifest id is embedding.lookup)",
    }
    pj = os.path.join(run_dir, "public.json")
    with open(pj, "w") as f:
        json.dump(public, f, indent=2, sort_keys=True)
    print(f"  run_seed = {C.run_seed_of(run_dir)}")
    print(f"REGISTERED {run_dir}")


if __name__ == "__main__":
    main()
