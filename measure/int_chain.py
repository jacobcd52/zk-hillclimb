"""The stage-2 integer witness chain as a pure-python forward pass (numpy,
exact integer semantics; every convention byte-validated against the
stage2-official1 driver-emitted data by probe_semantics.py / validate_chain.py).

Witness authority per segment (ORCHESTRATOR_DESIGN §3, ROPE_ATTENTION_DESIGN
§1.5/§2.2, SOFTMAX_DESIGN §2 — the pinned witness-authority switch):
  - rmsnorm sites (4): advice R = INTEGER-EXACT bracket (prove_walk.compute_R)
  - q/k/v fc + all rescales: driver semantics (int matmul, round-half-up)
  - RoPE: integer spec, registered int cos/sin tables
  - softmax: integer spec (exp table; P = round_half_up(2^16*MK*E/S), no advice)
  - headslice/headmerge: gathers + the line-157 pi permutation (bug-compatible)
  - skips: int32 adds
Segments with NO proof-chain semantics yet (pipeline-authority extensions,
flagged in every report that uses this module):
  - embedding: round(embed_tokens(ids).float32 * 2^16)  [manifest: waived]
  - final norm: pipeline float advice R = round(2^16/sqrt(mean(X_real^2)+eps))
    (m68-pipeline.py lines 124-127 semantics), then the same driver structure
  - lm_head: pipeline linear path round(W.T * 2^16) + int matmul + rescale 2^16
NOTE the chain contains NO o_proj (zkLLM upstream omits it; manifest waives it)
and the line-157 permutation — frozen pipeline behaviour, bound by the proofs.

Integer matmuls run as float64 BLAS with an asserted |.| < 2^52 exactness bound.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/projects/zk-hillclimb/orchestrator")
from prove_walk import compute_R, skip_add  # pinned witness functions

SEQ, EMBED, INTER, HD, NH = 1024, 768, 3072, 64, 12
LOG_SF = 16
SF = 1 << LOG_SF
C_EPS = 3298535
RMS_EPS = 1e-6
SWIGLU_LOW, SWIGLU_LEN = -(1 << 21), 1 << 22
SOFTMAX_LOW_E, SOFTMAX_LEN_E = -(1 << 19), 1 << 20
GATE_RESCALE_LOG = 20
N_LAYERS = 2


def rescale(v, log_sf):
    """zkob_rescale: rem in [-sf/2, sf/2), y = floor((v + sf/2)/sf). int64 in/out."""
    return (v + (1 << (log_sf - 1))) >> log_sf


def imatmul(X, W):
    """Exact int64 matmul via float64 BLAS (exact while |Y| < 2^52, asserted)."""
    Y = X.astype(np.float64) @ W.astype(np.float64)
    m = np.abs(Y).max()
    assert m < 2**52, f"imatmul exactness bound violated: max|Y| = {m:.3g}"
    return Y.astype(np.int64)


class IntChain:
    def __init__(self, reg_dir):
        wdir = os.path.join(reg_dir, "weights")
        self.w = {f[:-8]: np.fromfile(os.path.join(wdir, f), dtype=np.int32)
                  for f in os.listdir(wdir) if f.endswith("-int.bin")}
        self.swiglu_tab = np.fromfile(os.path.join(reg_dir, "swiglu-table.bin"), dtype=np.int32)
        self.exp_tab = np.fromfile(os.path.join(reg_dir, "softmax-exp-table.bin"), dtype=np.int32)
        cos = np.fromfile(os.path.join(reg_dir, "rope-cos-table.bin"), dtype=np.int32).reshape(SEQ, HD)
        sin = np.fromfile(os.path.join(reg_dir, "rope-sin-table.bin"), dtype=np.int32).reshape(SEQ, HD)
        e = np.arange(EMBED)
        sigma = np.where((e & 32) != 0, 1, -1).astype(np.int64)
        self.W1 = cos[:, e % HD].astype(np.int64)               # (SEQ, EMBED)
        self.W2 = sigma[None, :] * sin[:, e % HD].astype(np.int64)
        self.flip = e ^ 32
        self.MK = np.tril(np.ones((SEQ, SEQ), dtype=np.int64))  # causal mask
        # final norm / lm_head (pipeline-authority, set via set_head())
        self.g_final = None
        self.w_lm = None
        self.stats = {}

    def verify_weights_against_model(self, model):
        """register.py provenance guard: registered ints == round(w.float().T * 2^16)."""
        import torch
        for l in range(N_LAYERS):
            layer = model.model.layers[l]
            pairs = [(f"layer{l}.mlp.gate_proj", layer.mlp.gate_proj.weight, (EMBED, INTER)),
                     (f"layer{l}.mlp.up_proj", layer.mlp.up_proj.weight, (EMBED, INTER)),
                     (f"layer{l}.mlp.down_proj", layer.mlp.down_proj.weight, (INTER, EMBED)),
                     (f"layer{l}.attn.q_proj", layer.self_attn.q_proj.weight, (EMBED, EMBED)),
                     (f"layer{l}.attn.k_proj", layer.self_attn.k_proj.weight, (EMBED, EMBED)),
                     (f"layer{l}.attn.v_proj", layer.self_attn.v_proj.weight, (EMBED, EMBED)),
                     (f"layer{l}.input_norm.g", layer.input_layernorm.weight, (EMBED,)),
                     (f"layer{l}.post_attn_norm.g", layer.post_attention_layernorm.weight, (EMBED,))]
            for wid, wt, shape in pairs:
                w = wt.float().T if wt.dim() == 2 else wt.float()
                exp = torch.round(w * SF).to(torch.int32).cpu().numpy().reshape(-1)
                if not np.array_equal(exp, self.w[wid]):
                    raise RuntimeError(f"registered weight mismatch: {wid}")

    def set_head(self, model):
        """Export final-norm gain + lm_head with pipeline semantics (round(w*2^16))."""
        import torch
        self.g_final = torch.round(model.model.norm.weight.float() * SF).to(torch.int64).cpu().numpy()
        self.w_lm = torch.round(model.lm_head.weight.float().T * SF).to(torch.int32).cpu().numpy()
        self.final_eps = float(getattr(model.model.norm, "variance_epsilon", RMS_EPS))

    # --- chain segments -----------------------------------------------------
    def rmsnorm_exact(self, X):
        """Proof-chain rmsnorm: integer-exact advice R (witness-authority rule)."""
        R = compute_R(X, C_EPS).astype(np.int64)
        return self._rmsnorm_body(X, R, gain=None)

    def _rmsnorm_body(self, X, R, gain):
        g = gain if gain is not None else self._g
        W = R[:, None] * g[None, :]                  # i64 @2^32
        Wr = rescale(W, LOG_SF)                      # @2^16
        Y = Wr * X.astype(np.int64)                  # @2^32
        out = rescale(Y, LOG_SF)                     # @2^16
        assert np.abs(out).max() < 2**31
        return out.astype(np.int32)

    def attention(self, l, attn_in):
        proj = {}
        for pj in ("q_proj", "k_proj", "v_proj"):
            Wp = self.w[f"layer{l}.attn.{pj}"].reshape(EMBED, EMBED)
            proj[pj] = rescale(imatmul(attn_in, Wp), LOG_SF).astype(np.int32)
        roped = {}
        for t, pj in (("q", "q_proj"), ("k", "k_proj")):
            T = proj[pj].astype(np.int64)
            Y64 = T * self.W1 + T[:, self.flip] * self.W2
            assert np.abs(Y64).max() < 2**47, "rope |Y64| completeness guard"
            roped[t] = rescale(Y64, LOG_SF).astype(np.int32)
        out_heads = []
        zmin, zmax = [], []
        for h in range(NH):
            sl = slice(64 * h, 64 * h + 64)
            Qh = roped["q"][:, sl]
            KhT = roped["k"][:, sl].T
            Vh = proj["v_proj"][:, sl]
            z = imatmul(Qh, KhT)                                   # @2^32
            z_ = rescale(rescale(z, 13), 10)                       # @2^9
            zmin.append(int(z_.min())); zmax.append(int(z_.max()))
            if z_.min() < SOFTMAX_LOW_E or z_.max() >= SOFTMAX_LOW_E + SOFTMAX_LEN_E:
                raise RuntimeError(f"layer{l} h{h}: scores leave the exp-table domain "
                                   f"[{z_.min()}, {z_.max()}] — completeness failure")
            E = self.exp_tab[z_ - SOFTMAX_LOW_E].astype(np.int64)
            S = (E * self.MK).sum(axis=1)
            P = (((self.MK * E) << 17) + S[:, None]) // (2 * S[:, None])   # @2^16
            out64 = imatmul(P, Vh)                                 # @2^32
            out_heads.append(rescale(out64, LOG_SF).astype(np.int32))
        self.stats.setdefault("z_range", []).append((min(zmin), max(zmax)))
        M = np.concatenate(out_heads, axis=1)                      # (SEQ, EMBED)
        return M.T.reshape(SEQ, EMBED)                             # line-157 pi permutation

    def mlp(self, l, ffn_in):
        Wg = self.w[f"layer{l}.mlp.gate_proj"].reshape(EMBED, INTER)
        Wu = self.w[f"layer{l}.mlp.up_proj"].reshape(EMBED, INTER)
        Wd = self.w[f"layer{l}.mlp.down_proj"].reshape(INTER, EMBED)
        G = rescale(imatmul(ffn_in, Wg), GATE_RESCALE_LOG)         # @2^12 (silu table index)
        self.stats.setdefault("gate_range", []).append((int(G.min()), int(G.max())))
        if G.min() < SWIGLU_LOW or G.max() >= SWIGLU_LOW + SWIGLU_LEN:
            raise RuntimeError(f"layer{l}: gate activations leave the silu table domain "
                               f"[{G.min()}, {G.max()}] — completeness failure")
        U = rescale(imatmul(ffn_in, Wu), LOG_SF)                   # @2^16
        H = self.swiglu_tab[(G - SWIGLU_LOW).astype(np.int64)].astype(np.int64) * U  # @2^32
        Hr = rescale(H, LOG_SF).astype(np.int32)                   # @2^16
        return rescale(imatmul(Hr, Wd), LOG_SF).astype(np.int32)   # @2^16

    def forward(self, x0_i32):
        """x0 int32 @2^16 -> final residual-stream output int32 @2^16 (proof-chain part)."""
        resid = x0_i32
        for l in range(N_LAYERS):
            self._g = self.w[f"layer{l}.input_norm.g"].astype(np.int64)
            attn_in = self.rmsnorm_exact(resid)
            attn_out = self.attention(l, attn_in)
            z1 = skip_add(resid, attn_out)
            self._g = self.w[f"layer{l}.post_attn_norm.g"].astype(np.int64)
            ffn_in = self.rmsnorm_exact(z1)
            ffn_out = self.mlp(l, ffn_in)
            resid = skip_add(z1, ffn_out)
        return resid

    def logits(self, final_resid):
        """Pipeline-authority completion: final norm (float advice) + lm_head."""
        X = final_resid
        # m68-pipeline.py lines 124-127: X_real in float64, R = save_int(1/sqrt(...), 2^16)
        Xr = X.astype(np.float64) / SF
        rms_inv = 1.0 / np.sqrt((Xr * Xr).mean(axis=1) + self.final_eps)
        R = np.rint(rms_inv * SF).astype(np.int64)        # torch.round == rint (half-even)
        normed = self._rmsnorm_body(X, R, gain=self.g_final)
        logits_i = rescale(imatmul(normed, self.w_lm), LOG_SF)     # int @2^16
        return logits_i.astype(np.float64) / SF                    # float logits
