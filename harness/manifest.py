"""Generate the proof-obligation manifest for a model architecture.

The manifest is the anti-coverage-shrinkage mechanism (hack vector #2 in
HARNESS.md): it lists every component a complete proof of one forward pass
must cover, derived from the architecture itself — NOT from what any given
pipeline happens to implement. A submission's verifier must emit a transcript
of obligation ids it checked; harness/check_transcript.py compares that
against this manifest.

Known gaps in the current zkLLM pipeline are recorded as explicit WAIVERS
(coordinator-owned, each with a justification). A waived obligation still
exists — the LEDGER reports coverage as checked/required including waivers,
so shrinkage is visible, never silent.

Usage: python manifest.py [--model JackFram/llama-68m] [--seq 1024]
Writes manifest_<short>.json next to this file.
"""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoConfig

# Coordinator-owned waivers: obligations the CURRENT pipeline is known not to
# cover. Removing a waiver tightens the gate; adding one requires coordinator
# sign-off and a justification string.
WAIVERS = {
    "embedding.lookup": "zkLLM pipeline starts at layer 0 input; embedding table lookup not yet proven",
    "lm_head.matmul": "zkLLM pipeline proves through final hidden states; lm_head not yet proven",
    "lm_head.rescaling": "follows lm_head.matmul",
    "final_norm.rmsnorm": "not in current m68 pipeline scope",
}
for _l in range(2):  # llama-68m has 2 layers; o_proj omission is per-layer
    WAIVERS[f"layer{_l}.attn.o_proj.matmul"] = "zkLLM upstream omits o_proj (documented caveat)"
    WAIVERS[f"layer{_l}.attn.o_proj.rescaling"] = "zkLLM upstream omits o_proj"
    WAIVERS[f"layer{_l}.attn.o_proj.commitment_opening"] = "zkLLM upstream omits o_proj"


def linear_obligations(prefix, m, k, n, with_input_quant):
    """Every FP8->int linear contributes: weight commitment opening, the
    matmul sumcheck, the output rescaling lookup, and (for dynamic-quant
    schemes) input quantization well-formedness."""
    obs = [
        {"id": f"{prefix}.commitment_opening", "kind": "commitment_opening",
         "binds": "registered public weight commitment", "shape": [n, k]},
        {"id": f"{prefix}.matmul", "kind": "sumcheck_matmul", "shape": [m, k, n]},
        {"id": f"{prefix}.rescaling", "kind": "rescaling_lookup", "rows": m * n},
    ]
    if with_input_quant:
        obs += [
            {"id": f"{prefix}.input_codebook", "kind": "codebook_lookup",
             "rows": m * k, "note": "input values on FP8 codebook grid"},
            {"id": f"{prefix}.input_scale", "kind": "range_lookup",
             "rows": m, "note": "per-token absmax/scale well-formedness"},
        ]
    return obs


def build_manifest(model_id, seq, input_quant=False):
    cfg = AutoConfig.from_pretrained(model_id)
    h, ffn, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    nh = cfg.num_attention_heads
    obs = [{"id": "embedding.lookup", "kind": "table_lookup", "rows": seq}]
    for l in range(L):
        p = f"layer{l}"
        obs += [{"id": f"{p}.input_norm.rmsnorm", "kind": "rmsnorm", "rows": seq * h}]
        for proj, n in [("q_proj", h), ("k_proj", h), ("v_proj", h)]:
            obs += linear_obligations(f"{p}.attn.{proj}", seq, h, n, input_quant)
        obs += [
            {"id": f"{p}.attn.scores_matmul", "kind": "sumcheck_matmul",
             "shape": [seq, h // nh, seq], "per_head": nh},
            {"id": f"{p}.attn.softmax", "kind": "zkattn_softmax", "rows": seq * seq,
             "note": "incl. causal-mask + padding constraints (forgery D6)"},
            {"id": f"{p}.attn.values_matmul", "kind": "sumcheck_matmul",
             "shape": [seq, seq, h // nh], "per_head": nh},
        ]
        obs += linear_obligations(f"{p}.attn.o_proj", seq, h, h, input_quant)
        obs += [{"id": f"{p}.attn_skip.add", "kind": "skip_connection", "rows": seq * h},
                {"id": f"{p}.post_attn_norm.rmsnorm", "kind": "rmsnorm", "rows": seq * h}]
        for proj, (k, n) in [("gate_proj", (h, ffn)), ("up_proj", (h, ffn)),
                             ("down_proj", (ffn, h))]:
            obs += linear_obligations(f"{p}.mlp.{proj}", seq, k, n, input_quant)
        obs += [{"id": f"{p}.mlp.swiglu", "kind": "nonlinearity_lookup", "rows": seq * ffn},
                {"id": f"{p}.mlp_skip.add", "kind": "skip_connection", "rows": seq * h}]
    obs += [{"id": "final_norm.rmsnorm", "kind": "rmsnorm", "rows": seq * h}]
    obs += linear_obligations("lm_head", seq, h, cfg.vocab_size, input_quant)[:2] + [
        {"id": "lm_head.rescaling", "kind": "rescaling_lookup", "rows": seq * cfg.vocab_size}]
    obs += [{"id": "statement.logit_binding", "kind": "statement",
             "note": "proof binds FULL per-position logits/committed outputs, "
                     "not argmax only (forgeries A2/F3)"},
            {"id": "statement.prompt_binding", "kind": "statement",
             "note": "Fiat-Shamir transcript includes prompt token ids (A3)"},
            {"id": "statement.registered_weight_hash", "kind": "statement",
             "note": "commitment checked against REGISTERED hash, not "
                     "proof-supplied (B4)"}]
    for o in obs:
        o["waived"] = WAIVERS.get(o["id"])
    return {
        "model": model_id, "seq_len": seq, "input_quant_obligations": input_quant,
        "n_obligations": len(obs),
        "n_waived": sum(1 for o in obs if o["waived"]),
        "obligations": obs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="JackFram/llama-68m")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--input-quant", action="store_true",
                    help="include dynamic input-quantization obligations (codebook splice)")
    a = ap.parse_args()
    man = build_manifest(a.model, a.seq, a.input_quant)
    short = a.model.split("/")[-1].replace("-", "")
    out = Path(__file__).parent / f"manifest_{short}.json"
    body = json.dumps(man, indent=2)
    out.write_text(body)
    print(f"{man['n_obligations']} obligations ({man['n_waived']} waived) -> {out}")
    print("sha256:", hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
