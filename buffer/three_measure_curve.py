"""Three capacity MEASURES' treaty-threshold curves vs #tokens (FAITHFUL scheme).

Holds the scheme fixed (faithful corrected: reference = M_int post-Gumbel, served =
FP8 argmax; ~131 072 benign tokens / 128 prompt-blocks) and varies the *measure* of
per-token covert capacity the treaty would meter.  For each measure we plot

    y(N) = benign_rate  +  FPR_buffer(N)  +  counting_slack(N)        [bits / token]

the effective per-token leakage capacity the treaty must tolerate to hold a
false-tear-up rate <= 1e-10 after auditing N tokens.  Lower y = tighter = better.

The THREE measures:
  1. FIVE-TERM bound  R5  (the margin/afforded one; benign mean ~0.306).  Sum-like
     functional of s_t; buffer reuses the validated session-sum block-bootstrap +
     sub-exp tail-fit machinery from analyze_buffer.py.  slack_5 = 2*log2(N)/N.
  2. FULL RANK-ENTROPY  R_rank = H(empirical served-rank distribution).  Entropy
     functional; buffer = empirical-entropy fluctuation, block-bootstrapped over whole
     sessions (captures within-prompt autocorrelation) + sub-exp fit, extrapolated by
     the sigma_H/sqrt(N_eff) law.  slack_full = (V-1)*log2(N)/N   [V=32000 -> huge].
  3. TOP-K RANK-ENTROPY  R_topK = H(p) + (1-p)*S + p*log2(V-K), p = served-tokens
     OUTSIDE top-K, S = entropy of the in-top-K rank histogram.  slack_topK =
     (K-1)*log2(N)/N.  K=4 (headline) and K=16 (overlaid).

Buffer is OBSERVED (solid) for N <= the 131 072-token data limit (block-bootstrap +
sub-exp fit) and EXTRAPOLATED (dashed) beyond, by each measure's fitted ~c/sqrt(N) law.

Run:  /root/int-model-env/bin/python three_measure_curve.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_buffer as ab   # load_dumps, per_token, session_sums, block_bootstrap,
                              # fit_exptail_buffer, z_upper

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.dirname(HERE)
SEQ_LEN = 1024
EPS = 1e-10
B5, K5 = 0.3466331658291457, 4          # five-term calibrated optimum (K=4 floor)
KS_TOPK = (4, 16)                        # top-K settings (K=4 headline, K=16 overlay)

# audit-size grid (powers + half-decades of 10), 1e4 .. 1e12
NGRID = np.unique(np.round(10.0 ** np.arange(4.0, 12.01, 0.25)).astype(np.int64))
DATA_LIMIT = 128 * SEQ_LEN               # 131072 tokens = 128 prompt-blocks


# ----------------------------------------------------------------------------- #
#  measure helpers (vectorised over bootstrap reps)                              #
# ----------------------------------------------------------------------------- #
def _Hbern(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = (x > 0.0) & (x < 1.0)
    out[m] = -x[m] * np.log2(x[m]) - (1 - x[m]) * np.log2(1 - x[m])
    return out


def entropy_rows(counts):
    """Shannon entropy (bits) of each row's normalised histogram. counts (reps,A)."""
    tot = counts.sum(axis=1, keepdims=True)
    P = counts / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0.0, -P * np.log2(P), 0.0)
    return terms.sum(axis=1)


def topk_rows(counts, support, K, V):
    """R_topK per row.  counts (reps,A), support (A,) the rank values of the columns."""
    tot = counts.sum(axis=1).astype(np.float64)
    in_mask = support < K
    in_c = counts[:, in_mask]
    out_c = counts[:, ~in_mask].sum(axis=1).astype(np.float64)
    p_out = out_c / tot
    in_tot = in_c.sum(axis=1, keepdims=True)
    Pin = in_c / np.maximum(in_tot, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(Pin > 0.0, -Pin * np.log2(Pin), 0.0).sum(axis=1)
    return _Hbern(p_out) + (1.0 - p_out) * S + p_out * np.log2(V - K)


def session_count_matrix(ranks, block, support):
    """(n_sessions, A) integer per-session rank-count matrix over `support`."""
    blocks = np.unique(block)
    pos = {v: i for i, v in enumerate(support)}
    C = np.zeros((len(blocks), len(support)), dtype=np.float64)
    for bi, b in enumerate(blocks):
        rb = ranks[block == b]
        v, c = np.unique(rb, return_counts=True)
        for vv, cc in zip(v, c):
            C[bi, pos[int(vv)]] += cc
    return C


def boot_counts(C, m, reps, rng):
    """Block-bootstrap: resample m whole sessions w/ replacement, return pooled
    count matrix (reps, A).  Additive accumulation -> O(reps*A) memory."""
    n, A = C.shape
    counts = np.zeros((reps, A), dtype=np.float64)
    for _ in range(m):
        counts += C[rng.integers(0, n, size=reps)]
    return counts


# ----------------------------------------------------------------------------- #
#  buffer(N): observed (bootstrap + sub-exp) for N<=data, c/sqrt(N) extrapolation #
# ----------------------------------------------------------------------------- #
def fit_sqrt_coeff(per_tok_by_N, Nmin):
    """Median of per_tok*sqrt(N) over observable N>=Nmin -> coeff c (bits*sqrt tok)
    and relative spread (fit-quality flag)."""
    cs = [v * np.sqrt(N) for N, v in per_tok_by_N.items() if N >= Nmin and v is not None]
    if not cs:
        return None, None
    cs = np.array(cs)
    return float(np.median(cs)), float(cs.std() / cs.mean()) if cs.mean() else None


def buffer_curve_sumlike(Y, rng, reps=2_000_000, Nmin_fit=16384):
    """Per-token buffer for the SUM-like five-term measure: session-sum block
    bootstrap + sub-exp tail fit (validated in BUFFER_FPR.md), per-token = total/N.
    Returns observed dict {N: per_tok} for N<=DATA_LIMIT and fitted coeff c."""
    obs = {}
    for N in NGRID[NGRID <= DATA_LIMIT]:
        m = max(1, int(round(N / SEQ_LEN)))
        tot = ab.block_bootstrap(Y, m, reps, rng)
        sub = ab.fit_exptail_buffer(tot, EPS)
        obs[int(N)] = (sub["buffer"] / (m * SEQ_LEN)) if sub else None
    c, spread = fit_sqrt_coeff(obs, Nmin_fit)
    return obs, c, spread


def buffer_curve_functional(C, support, benign, measure_fn, rng,
                            reps=2_000_000, Nmin_fit=16384):
    """Per-token buffer for an ENTROPY/top-K functional: block-bootstrap whole
    sessions, evaluate the functional on the pooled histogram, sub-exp tail-fit the
    upper deviation (value-benign) to EPS.  Returns observed dict + fitted c."""
    obs = {}
    for N in NGRID[NGRID <= DATA_LIMIT]:
        m = max(1, int(round(N / SEQ_LEN)))
        counts = boot_counts(C, m, reps, rng)
        vals = measure_fn(counts)
        dev = vals - benign                     # upper-tail margin above the benign rate
        sub = ab.fit_exptail_buffer(dev, EPS)
        obs[int(N)] = (sub["buffer"] if sub else None)
    c, spread = fit_sqrt_coeff(obs, Nmin_fit)
    return obs, c, spread


def y_curve(benign, obs_buf, c_buf, slack_coeff):
    """Assemble y(N)=benign+buffer(N)+slack(N) on NGRID.  buffer is the observed
    sub-exp value where N<=DATA_LIMIT (and a value exists), else c_buf/sqrt(N).
    slack(N)=slack_coeff*log2(N)/N."""
    Ns, ys, buf, slk, observed = [], [], [], [], []
    for N in NGRID:
        if N <= DATA_LIMIT and obs_buf.get(int(N)) is not None:
            b = obs_buf[int(N)]
            obsflag = True
        else:
            b = c_buf / np.sqrt(N)
            obsflag = False
        s = slack_coeff * np.log2(N) / N
        Ns.append(int(N)); buf.append(b); slk.append(s)
        ys.append(benign + b + s); observed.append(obsflag)
    return {"N": Ns, "y": ys, "buffer": buf, "slack": slk, "observed": observed}


# ----------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(20260615)
    margins, ranks, Nb, bgrid, V, block = ab.load_dumps(use_extra=True)
    N_tot = margins.size
    n_blocks = len(np.unique(block))
    assert N_tot == DATA_LIMIT, (N_tot, DATA_LIMIT)
    print(f"faithful: {N_tot} tokens, {n_blocks} prompt-blocks, V={V}")

    support = np.unique(ranks)                      # observed rank support {0..9}
    A_obs = support.size
    C = session_count_matrix(ranks, block, support)
    P = C.sum(axis=0) / N_tot
    H_full = float(np.where(P > 0, -P * np.log2(P), 0.0).sum())
    sigmaH2 = float((P * np.log2(P) ** 2).sum() - H_full ** 2)
    sigmaH = float(np.sqrt(sigmaH2))

    out = {"eps": EPS, "V": int(V), "n_tokens": int(N_tot), "n_blocks": int(n_blocks),
           "data_limit_tokens": DATA_LIMIT, "A_obs": int(A_obs),
           "rank_support": support.tolist(), "rank_P": P.tolist(),
           "H_full": H_full, "sigma_H": sigmaH, "measures": {}}

    # ---- variance-inflation / N_eff (entropy fluctuation) cross-check ----------
    # i.i.d. entropy fluctuation std = sigma_H/sqrt(N).  Measure the block-bootstrap
    # std at the data limit and back out tau_var so that boot_std=sigma_H/sqrt(N/tau).
    counts128 = boot_counts(C, 128, 400_000, rng)
    boot_std_128 = float(entropy_rows(counts128).std(ddof=1))
    tau_entropy = (boot_std_128 * np.sqrt(N_tot) / sigmaH) ** 2
    z = ab.z_upper(EPS)
    bias_coeff = (A_obs - 1) / (2.0 * np.log(2.0))   # Miller-Madow: bias = -bias_coeff/N
    out["entropy_fluct"] = {
        "sigma_H": sigmaH, "boot_std_at_datalimit": boot_std_128,
        "tau_var_entropy": float(tau_entropy),
        "z_1e-10": float(z),
        "analytic_c_subexp_equiv_bits_sqrt_tok": float(z * sigmaH * np.sqrt(tau_entropy)),
        "MillerMadow_bias_coeff_(bias=-coeff/N)": float(bias_coeff)}
    print(f"H_full={H_full:.5f} sigma_H={sigmaH:.4f} A_obs={A_obs} "
          f"tau_entropy={tau_entropy:.2f} z={z:.3f} "
          f"analytic c={z*sigmaH*np.sqrt(tau_entropy):.2f} MM-bias coeff={bias_coeff:.2f}")

    # ============================ MEASURE 1: five-term ======================== #
    pt = ab.per_token(margins, ranks, Nb, bgrid, V, B5, K5)
    mu5 = pt["mu"]
    Y5 = ab.session_sums(pt["s"], block)
    obs5, c5, sp5 = buffer_curve_sumlike(Y5, rng)
    curve5 = y_curve(mu5, obs5, c5, slack_coeff=2.0)     # 2 params (p,q)
    out["measures"]["five_term"] = {
        "label": "five-term R5 (margin/afforded, K=4)", "benign_rate": mu5,
        "slack_coeff_params": 2, "slack_formula": "2*log2(N)/N",
        "buffer_law": "sum-like c/sqrt(N), c from session-sum bootstrap sub-exp",
        "c_buffer_bits_sqrt_tok": c5, "c_fit_rel_spread": sp5,
        "p": pt["p"], "q": pt["q"], "n_viol": pt["n_viol"], "n_tail": pt["n_tail"],
        "curve": curve5}
    print(f"\n[five-term] benign mu5={mu5:.5f}  c_buf={c5:.2f} (spread {sp5:.2f})")

    # ========================= MEASURE 2: full rank-entropy =================== #
    obs2, c2, sp2 = buffer_curve_functional(C, support, H_full, entropy_rows, rng)
    curve2 = y_curve(H_full, obs2, c2, slack_coeff=float(V - 1))   # (V-1) params
    out["measures"]["full_rank_entropy"] = {
        "label": "full rank-entropy H(served rank)", "benign_rate": H_full,
        "slack_coeff_params": int(V - 1), "slack_formula": "(V-1)*log2(N)/N",
        "buffer_law": "entropy fluctuation c/sqrt(N) ~ z*sigma_H*sqrt(tau)/sqrt(N)",
        "c_buffer_bits_sqrt_tok": c2, "c_fit_rel_spread": sp2,
        "sigma_H": sigmaH, "tau_var_entropy": float(tau_entropy),
        "curve": curve2}
    print(f"[full-rank] benign H={H_full:.5f}  c_buf={c2:.2f} (spread {sp2:.2f}) "
          f"  (analytic {z*sigmaH*np.sqrt(tau_entropy):.2f})")

    # ============================ MEASURE 3: top-K ============================ #
    out["measures"]["top_k"] = {}
    curves_topk = {}
    for K in KS_TOPK:
        mfn = lambda counts, K=K: topk_rows(counts, support, K, V)
        benignK = float(mfn(C.sum(axis=0, keepdims=True))[0])
        p_out = float((ranks >= K).mean())
        obsK, cK, spK = buffer_curve_functional(C, support, benignK, mfn, rng)
        curveK = y_curve(benignK, obsK, cK, slack_coeff=float(K - 1))
        out["measures"]["top_k"][f"K{K}"] = {
            "label": f"top-K rank-entropy (K={K})", "benign_rate": benignK,
            "p_out_of_topK": p_out, "slack_coeff_params": int(K - 1),
            "slack_formula": f"({K}-1)*log2(N)/N",
            "buffer_law": "c/sqrt(N), c from functional bootstrap sub-exp",
            "c_buffer_bits_sqrt_tok": cK, "c_fit_rel_spread": spK, "curve": curveK}
        curves_topk[K] = (benignK, curveK)
        print(f"[top-K K={K}] benign={benignK:.5f} p_out={p_out:.3e} "
              f"c_buf={cK:.2f} (spread {spK:.2f})")

    # ----------------------- table at 1e4,1e6,1e9,1e12 ------------------------- #
    table_Ns = [10**4, 10**6, 10**9, 10**12]

    def interp_y(curve, Nq):
        Ns = np.array(curve["N"], float); ys = np.array(curve["y"], float)
        return float(np.interp(np.log10(Nq), np.log10(Ns), ys))

    table = {}
    rows = {"five_term": curve5, "full_rank_entropy": curve2,
            "top_k_K4": curves_topk[4][1], "top_k_K16": curves_topk[16][1]}
    for name, cv in rows.items():
        table[name] = {str(N): interp_y(cv, N) for N in table_Ns}
    out["table_y_at_N"] = table

    # ----------------------- crossovers (V log n washout) ---------------------- #
    Nfine = np.unique(np.round(10.0 ** np.arange(4.0, 12.001, 0.002)).astype(np.int64))

    def yfun(curve):
        Ns = np.array(curve["N"], float); ys = np.array(curve["y"], float)
        return np.interp(np.log10(Nfine), np.log10(Ns), ys)

    yf = {"five_term": yfun(curve5), "full": yfun(curve2),
          "top4": yfun(curves_topk[4][1]), "top16": yfun(curves_topk[16][1])}

    def first_below(a, b):
        idx = np.where(a < b)[0]
        return int(Nfine[idx[0]]) if idx.size else None

    crossovers = {
        "full_drops_below_top4": first_below(yf["full"], yf["top4"]),
        "full_drops_below_top16": first_below(yf["full"], yf["top16"]),
        "full_drops_below_five_term": first_below(yf["full"], yf["five_term"]),
        "full_drops_below_five_term_note":
            "None => full rank-entropy never beats the five-term (benign 0.362>0.306)."}
    out["crossovers"] = crossovers
    print("\ncrossovers (N where full-rank y drops below):", crossovers)
    print("\ny(N) table:")
    for name, cv in rows.items():
        print(f"  {name:18s}: " +
              "  ".join(f"N={N:.0e}:{table[name][str(N)]:.4f}" for N in table_Ns))

    make_plot(out, curve5, curve2, curves_topk, crossovers)
    with open(os.path.join(HERE, "three_measure_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'three_measure_results.json')}")
    print(f"plot -> {os.path.join(OUTDIR, 'three_measure_curve.png')}")
    return out


def make_plot(out, curve5, curve2, curves_topk, crossovers):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    specs = [
        ("FIVE-TERM R5  (margin/afforded, K=4)", "#1f77b4", curve5,
         out["measures"]["five_term"]["benign_rate"]),
        ("FULL RANK-ENTROPY  H(served rank)", "#d62728", curve2,
         out["measures"]["full_rank_entropy"]["benign_rate"]),
        ("TOP-K RANK-ENTROPY  (K=4)", "#2ca02c", curves_topk[4][1], curves_topk[4][0]),
        ("TOP-K RANK-ENTROPY  (K=16)", "#9467bd", curves_topk[16][1], curves_topk[16][0]),
    ]
    for label, col, cv, benign in specs:
        Ns = np.array(cv["N"], float); ys = np.array(cv["y"], float)
        obs = np.array(cv["observed"], bool)
        # solid where buffer is observed (N<=data limit), dashed where extrapolated
        ax.plot(Ns[obs], ys[obs], "-", color=col, lw=2.4, label=label)
        # bridge last-observed to first-extrapolated, then dashed
        if obs.any() and (~obs).any():
            j = np.where(obs)[0][-1]
            ax.plot(Ns[j:], ys[j:], "--", color=col, lw=1.8)
        elif (~obs).all():
            ax.plot(Ns, ys, "--", color=col, lw=1.8, label=label)
        ax.axhline(benign, color=col, ls=":", lw=1.0, alpha=0.7)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e4, 1e12)
    ymin, ymax = ax.get_ylim()
    ax.axvline(out["data_limit_tokens"], color="k", ls="-.", lw=1.2, alpha=0.6)
    ax.text(out["data_limit_tokens"] * 1.15, ymax * 0.4,
            "data limit 131 072 tok\nsolid <= | dashed extrapolated",
            fontsize=8, va="top")
    cx = crossovers["full_drops_below_top4"]
    if cx:
        ax.axvline(cx, color="#d62728", ls=":", lw=1.0, alpha=0.5)
        ax.text(cx * 1.2, ymin * 2.2, f"full < top-4\nN~{cx:.1e}\n(V·log n washout)",
                fontsize=7.5, color="#d62728")
    ax.set_xlabel("N = tokens audited  (log)")
    ax.set_ylabel("y(N) = benign_rate + FPR_buffer(N) + counting_slack(N)   [bits / token]")
    ax.set_title("Treaty per-token leakage-capacity ceiling vs audit size — FAITHFUL scheme\n"
                 "three capacity measures (FPR <= 1e-10).  Lower = tighter treaty bound.")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUTDIR, "three_measure_curve.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
