"""Core library for the token-level covert-capacity measurement.

Models the ONE prover freedom the zk verifier tolerates: the per-row inverse-RMS
integer advice R may be any value satisfying

    (R-1)^2 * M  <=  2^64 * C  <=  (R+1)^2 * M

i.e. R within +-1 of the exact real-valued scaled inverse-RMS R* = rms_inv * 2^16
(integer-bracket interval [R*-1, R*+1]). We reproduce the zkLLM integer rmsnorm
EXACTLY (two zkLLM-convention rescales, weight at 2^16) so that perturbing R by its
accepted +-1 is modelled with the real rounding behaviour, then run the rest of the
forward pass and read out greedy (argmax) tokens + logits.

Faithfulness notes (honest scope):
  - rmsnorm is the channel under test and is modelled in EXACT int64 arithmetic.
  - The 7 linear projections + lm_head are run in float (cuBLAS).  The channel is a
    DIFFERENCE between two R choices through the *same* pipeline, so the float-vs-
    fixedpoint discrepancy (~1e-4 rel, per SCORES_RANGE.md) is common-mode and
    cancels in every delta we report.  A robustness check re-runs baseline+coarse on
    the zkLLM fixed-point integer student to confirm the conclusion is unchanged.
  - Greedy decoding is the pinned threat model (THREAT_MODEL_NOTES.md sec 1): served
    token == argmax(logits).  All token deltas are argmax flips on the raw logits.
"""
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CARD = "JackFram/llama-68m"
CACHE_DIR = "/workspace/projects/zk-hillclimb/zkllm-src/model-storage"
SEQ = 1024
DEV = "cuda:0"
SF = 1 << 16            # 2^16 fixed-point scale
EPS = 1e-6             # JackFram/llama-68m config rms_norm_eps

# The three real-text contexts from SCORES_RANGE.md (identical prompts/tiling).
PROMPTS = {
    "real_lorem": ("The quick brown fox jumps over the lazy dog. " * 200),
    "real_wiki": ("Attention is a mechanism in neural networks that allows a model to "
                  "weigh the importance of different parts of the input when producing "
                  "an output. The transformer architecture relies entirely on attention. " * 60),
    "real_code": ("def softmax(x):\n    m = max(x)\n    e = [exp(v - m) for v in x]\n"
                  "    s = sum(e)\n    return [v / s for v in e]\n" * 80),
}


def _rescale(v, S):
    """zkLLM Rescaling convention: remainder in [-S/2, S/2), y = (v - rem)/S. int64."""
    hsf = S >> 1
    rem = (v + hsf) % S
    rem = torch.where(rem < 0, rem + S, rem) - hsf
    return (v - rem) // S


class IntRMSNorm(torch.nn.Module):
    """Drop-in replacement for LlamaRMSNorm reproducing the zkLLM integer rmsnorm
    obligation, with a per-row integer advice R that can be overridden (the channel).

    Holds, after a baseline forward pass, R_star / R_round / accepted-set masks for
    every row, so attacks can pick R in {R_round-1, R_round, R_round+1} ∩ accepted.
    """

    def __init__(self, weight, eps, site_id):
        super().__init__()
        self.weight = weight                  # original norm weight (float, [embed])
        self.eps = eps
        self.site_id = site_id
        self.g = torch.round(weight.double() * SF).to(torch.int64)   # weight at 2^16
        self.R_override = None                # None -> use R_round; else int64 [seq]
        self.record = False
        # filled on a baseline pass:
        self.R_star = None     # exact real R* per row (float64 [seq])
        self.R_round = None    # honest integer advice per row (int64 [seq])
        self.acc_lo = None     # smallest accepted integer R per row
        self.acc_hi = None     # largest  accepted integer R per row

    def forward(self, x):
        orig_dtype = x.dtype
        sq = x.dim() == 3
        xf = x.double()
        if sq:
            xf = xf.squeeze(0)                 # [seq, embed]
        x_int = torch.round(xf * SF).to(torch.int64)
        X = x_int.double() / SF
        ms = X.pow(2).mean(dim=-1) + self.eps  # [seq]
        rms_inv = 1.0 / torch.sqrt(ms)
        R_star = rms_inv * SF                   # exact real scaled inverse-rms
        R_round = torch.round(R_star).to(torch.int64)

        if self.record or self.R_star is None:
            self.R_star = R_star.detach()
            self.R_round = R_round.detach()
            # accepted set = integers in [R*-1, R*+1]
            self.acc_lo = torch.ceil(R_star - 1.0).to(torch.int64).detach()
            self.acc_hi = torch.floor(R_star + 1.0).to(torch.int64).detach()

        R = self.R_round if self.R_override is None else self.R_override
        # W_ = rescale(R ⊗ g, 2^16) ; Y = rescale(W_ ⊙ x_int, 2^16)
        W_ = _rescale(R.view(-1, 1) * self.g.view(1, -1), SF)          # [seq, embed]
        Y = _rescale(W_ * x_int, SF)
        out = (Y.double() / SF).to(orig_dtype)
        if sq:
            out = out.unsqueeze(0)
        return out


SITE_ORDER = ["L0.input", "L0.post_attn", "L1.input", "L1.post_attn", "final"]


def build_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_CARD, cache_dir=CACHE_DIR,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="eager").to(DEV).eval()
    norms = {}
    for li, layer in enumerate(model.model.layers):
        for which, attr in [("input", "input_layernorm"),
                            ("post_attn", "post_attention_layernorm")]:
            mod = getattr(layer, attr)
            sid = f"L{li}.{which}"
            new = IntRMSNorm(mod.weight.detach().to(DEV), getattr(mod, "variance_epsilon", EPS), sid).to(DEV)
            setattr(layer, attr, new)
            norms[sid] = new
    fn = model.model.norm
    new = IntRMSNorm(fn.weight.detach().to(DEV), getattr(fn, "variance_epsilon", EPS), "final").to(DEV)
    model.model.norm = new
    norms["final"] = new
    return model, norms


def load_inputs(tok):
    out = {}
    for name, text in PROMPTS.items():
        ids = tok(text, return_tensors="pt").input_ids
        n = ids.shape[1]
        if n < SEQ:
            reps = (SEQ + n - 1) // n
            ids = ids.repeat(1, reps)[:, :SEQ]
        else:
            ids = ids[:, :SEQ]
        out[name] = ids.to(DEV)
    return out


@torch.no_grad()
def forward_logits(model, ids):
    return model(input_ids=ids, use_cache=False).logits.float().squeeze(0)  # [seq, V]


def clear_overrides(norms):
    for m in norms.values():
        m.R_override = None


def set_record(norms, on):
    for m in norms.values():
        m.record = on
