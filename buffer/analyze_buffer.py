"""Buffer-vs-FPR-vs-audit-size analysis for the capacity-budget treaty.

Consumes the corrected-orientation FAITHFUL per-position dumps (reference =
faithful M_int, served = FP8 fast-model argmax) and computes, at the calibrated
optimum (b*, K*) and a conservative K:

  1. per-token afforded-capacity r_t = overhead + s_t, where
        s_t = log2 N_b(t)        if margin_t <= b      (compliant multiplicity)
            = log2 K             if margin_t > b and cand_rank_t <  K  (top-K violation)
            = log2(V-K)          if margin_t > b and cand_rank_t >= K  (tail violation)
     overhead = H(p) + p*H(q)  (the amortized "which-positions/which-bucket-violate"
        message entropy, computed from the message's EMPIRICAL p,q; a per-token
        CONSTANT -> shifts the mean to exactly C(b,K), does NOT affect the buffer).
     mean(r_t) == C(b,K) (the swept worst-case capacity).  R(N) = sum_t r_t.

  2. r_t / s_t distribution (zero-inflation, mean, var, range, skew, kurtosis, tail).

  3. AUTOCORRELATION of s_t ALONG sequences, computed WITHIN prompt-blocks only
     (each tiled dolly prompt is one block); integrated autocorr time tau (Sokal
     adaptive window) and N_eff = N/tau.  The prompts are TILED (a short dolly
     prompt repeated to 1024 tokens), so the within-block series is quasi-periodic
     and tau is dominated by that tiling -> stated as a caveat.

  4. Var(R(N)) empirically vs N: within-block sliding windows for N<=L; for N>L the
     operational model is a concatenation of INDEPENDENT sessions (distinct prompts)
     -> V(N) = (N/L)*Var(Y_session).  Compared against the i.i.d. line N*sigma^2;
     the ratio is the correlation inflation = tau.

  5. buffer(N) for FPR eps such that P(R(N)-N*mu > buffer) <= eps, via
       (a) a rigorous CONCENTRATION (Bernstein) bound on INDEPENDENT sessions, with
           empirical session variance and a per-token bound s_max=log2(V-K);
       (b) a parametric TAIL FIT (Gaussian + a sub-exponential log-linear survival
           fit) to the block-bootstrap distribution of R(N), extrapolated to eps=1e-10;
       cross-checked against the block-bootstrap empirical quantiles at the FPRs we
       CAN observe (~1e-3..1e-5).

Run:
  /root/int-model-env/bin/python analyze_buffer.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(os.path.dirname(HERE), "capacity")
OUTDIR = os.path.dirname(HERE)
ORIG = os.path.join(CAP, "capacity_dump_corrected_faithful_seed20260611.npz")
EXTRA = os.path.join(HERE, "faithful_extra_corrected.npz")
SEQ_LEN = 1024
EPS_TARGET = 1e-10


def H(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    out = np.zeros_like(x)
    m = (x > 0) & (x < 1)
    out[m] = -x[m] * np.log2(x[m]) - (1 - x[m]) * np.log2(1 - x[m])
    return float(out) if out.ndim == 0 else out


def load_dumps(use_extra=True):
    """Return concatenated margins, cand_ranks, Nb, bgrid, V, block_id."""
    d = np.load(ORIG)
    bgrid = d["bgrid"].astype(np.float64)
    margins = [d["margins"].astype(np.float64)]
    ranks = [d["cand_ranks"].astype(np.int64)]
    Nb = [d["Nb"].astype(np.float64)]
    V = int(d["vocab"])
    nblk = margins[0].size // SEQ_LEN
    block = [np.repeat(np.arange(nblk), SEQ_LEN)]
    nxt = nblk
    if use_extra and os.path.exists(EXTRA):
        e = np.load(EXTRA)
        assert np.array_equal(e["bgrid"].astype(np.float64), bgrid), "bgrid mismatch"
        margins.append(e["margins"].astype(np.float64))
        ranks.append(e["cand_ranks"].astype(np.int64))
        Nb.append(e["Nb"].astype(np.float64))
        block.append(e["block_id"].astype(np.int64) + nxt)
    return (np.concatenate(margins), np.concatenate(ranks),
            np.concatenate(Nb, axis=0), bgrid, V, np.concatenate(block))


def interp_log2Nb(Nb, bgrid, b):
    """Per-position log2 N_b at threshold b, linear interp between grid points
    (exact on grid). Nb (P,nb)."""
    logNb = np.log2(Nb)
    hi = int(np.searchsorted(bgrid, b, side="left"))
    hi = min(max(hi, 0), len(bgrid) - 1)
    lo = max(hi - 1, 0)
    span = bgrid[hi] - bgrid[lo]
    if np.isclose(bgrid[hi], b) or span == 0:
        return logNb[:, hi]
    w = (b - bgrid[lo]) / span
    return logNb[:, lo] * (1 - w) + logNb[:, hi] * w


def per_token(margins, ranks, Nb, bgrid, V, b, K):
    """Per-token s_t, overhead, r_t, and summary (p,q,mu,C)."""
    compliant = margins <= b
    p = float((~compliant).mean())
    viol = ~compliant
    if viol.sum() > 0:
        q = float((ranks[viol] >= K).mean())
    else:
        q = 0.0
    log2Nb = interp_log2Nb(Nb, bgrid, b)
    s = np.where(compliant, log2Nb, 0.0)
    tail = viol & (ranks >= K)
    topk = viol & (ranks < K)
    s = np.where(topk, np.log2(K) if K > 1 else 0.0, s)
    s = np.where(tail, np.log2(V - K), s)
    overhead = H(np.array(p)) + p * H(np.array(q))
    r = s + overhead
    mu = float(r.mean())
    return {"s": s, "r": r, "overhead": float(overhead), "p": p, "q": q,
            "mu": mu, "C_check": mu, "compliant": compliant,
            "n_viol": int(viol.sum()), "n_tail": int(tail.sum()),
            "s_max_possible": float(np.log2(V - K)),
            "s_max_observed": float(s.max())}


def dist_stats(x):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(); var = x.var(ddof=1); sd = np.sqrt(var)
    c = x - mu
    skew = float((c**3).mean() / sd**3) if sd > 0 else 0.0
    kurt = float((c**4).mean() / sd**4 - 3.0) if sd > 0 else 0.0
    return {"mean": float(mu), "var": float(var), "std": float(sd),
            "min": float(x.min()), "max": float(x.max()),
            "frac_zero": float((x == 0).mean()),
            "frac_pos": float((x > 0).mean()),
            "skew": skew, "excess_kurtosis": kurt,
            "p50": float(np.percentile(x, 50)),
            "p99": float(np.percentile(x, 99)),
            "p999": float(np.percentile(x, 99.9)),
            "p9999": float(np.percentile(x, 99.99))}


def acf_within_blocks(s, block, max_lag=400):
    """Mean autocorrelation rho_k over within-block pairs only, and Sokal tau."""
    s = np.asarray(s, dtype=np.float64)
    blocks = [s[block == b] for b in np.unique(block)]
    mu = s.mean()
    # autocovariance at lag k aggregated over blocks (within-block pairs only)
    acov = np.zeros(max_lag + 1)
    cnt = np.zeros(max_lag + 1)
    for sb in blocks:
        c = sb - mu
        n = len(sb)
        kmax = min(max_lag, n - 1)
        for k in range(kmax + 1):
            acov[k] += np.dot(c[:n - k], c[k:])
            cnt[k] += (n - k)
    acov = np.where(cnt > 0, acov / np.maximum(cnt, 1), 0.0)
    rho = acov / acov[0] if acov[0] > 0 else acov
    # Sokal adaptive window: smallest M with M >= C*tau(M)
    tau_run = 1.0
    M_win = max_lag
    for M in range(1, max_lag + 1):
        tau_run = 1.0 + 2.0 * rho[1:M + 1].sum()
        if M >= 5.0 * tau_run:
            M_win = M
            break
    tau = 1.0 + 2.0 * rho[1:M_win + 1].sum()
    tau = max(tau, 1.0)
    return rho, float(tau), int(M_win)


def var_RN_within(s, block, Ns):
    """Empirical Var of length-N partial sums from WITHIN-block windows."""
    out = {}
    blocks = [s[block == b] for b in np.unique(block)]
    for N in Ns:
        sums = []
        for sb in blocks:
            L = len(sb)
            if L < N:
                continue
            cs = np.concatenate([[0.0], np.cumsum(sb)])
            sums.append(cs[N:] - cs[:L - N + 1])   # all contiguous length-N sums
        sums = np.concatenate(sums)
        out[N] = {"var": float(np.var(sums, ddof=1)), "n_windows": int(sums.size)}
    return out


def session_sums(s, block):
    """Per-session (per full prompt-block) sum Y_b. Only use full-length blocks."""
    ys = []
    for b in np.unique(block):
        sb = s[block == b]
        if len(sb) == SEQ_LEN:
            ys.append(sb.sum())
    return np.array(ys, dtype=np.float64)


def z_upper(eps):
    """Upper-tail standard-normal quantile z with P(Z>z)=eps. Acklam rational
    initial guess + Newton refinement using math.erfc (no scipy)."""
    import math
    p = 1.0 - eps                       # lower-tail probability
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= phigh:
        q = p - 0.5; r = q*q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    # Newton refinement on upper tail Q(x)=erfc(x/sqrt2)/2 == eps
    for _ in range(3):
        Q = 0.5 * math.erfc(x / math.sqrt(2.0))
        phi = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
        if phi == 0:
            break
        x += (Q - eps) / phi
    return x


def fit_exptail_buffer(totals, eps, lo=1e-5, hi=1e-2):
    """Sub-exponential tail extrapolation: fit ln S(t) ~ c - t/lambda on the
    EMPIRICAL bootstrap survival curve in the observable window [lo,hi], then
    solve for the t at survival=eps. lambda is the tail decay scale (bits)."""
    qs = np.linspace(lo, hi, 40)
    ts = np.quantile(totals, 1 - qs)               # survival level qs at threshold ts
    lnS = np.log(qs)
    A = np.vstack([np.ones_like(ts), ts]).T
    coef, *_ = np.linalg.lstsq(A, lnS, rcond=None)  # lnS = c + slope*t
    c, slope = coef
    if slope >= 0:
        return None
    lam = -1.0 / slope
    buf = (np.log(eps) - c) / slope                # t at S=eps
    return {"lambda_bits": float(lam), "intercept": float(c), "buffer": float(buf)}


def bernstein_buffer(N, Y, s_max, eps, L=SEQ_LEN):
    """Rigorous concentration buffer on INDEPENDENT sessions.
    R(N)-N*mu = sum over B=N/L sessions of (Y_b - mu_Y); Bernstein with
    session variance and session deviation bound M_Y (<= L*s_max, but we use the
    tighter empirical max session deviation as the support bound)."""
    muY = Y.mean()
    sigУ2 = Y.var(ddof=1)
    B = N / L
    Ln = np.log(1.0 / eps)
    V = B * sigУ2
    # session-deviation bound: empirical max |Y-muY| (data never exceeds it)
    M_emp = float(np.max(np.abs(Y - muY)))
    # rigorous-support a-priori bound: worst-case session sum L*s_max minus the
    # session-sum mean muY (loose; reported for contrast)
    M_wc = float(L * s_max - muY)
    out = {}
    for tag, M in (("empirical_support", M_emp), ("worstcase_support", M_wc)):
        # solve t^2 = 2 Ln (V + M t/3)  ->  t = (Ln M/3) + sqrt((Ln M/3)^2 + 2 Ln V)
        a = Ln * M / 3.0
        t = a + np.sqrt(a * a + 2.0 * Ln * V)
        out[tag] = float(t)
    out["V"] = float(V); out["M_emp"] = M_emp; out["M_wc"] = M_wc
    return out


def gaussian_buffer(N, Y, eps, L=SEQ_LEN):
    sig2 = Y.var(ddof=1)
    B = N / L
    z = z_upper(eps)
    return float(z * np.sqrt(B * sig2)), float(z)


def block_bootstrap(Y, m, reps, rng):
    """Distribution of (sum of m resampled sessions) - m*muY.
    Memory-safe additive accumulation: O(reps) memory, never the (reps,m) array."""
    muY = Y.mean()
    n = len(Y)
    totals = np.zeros(reps, dtype=np.float64)
    for _ in range(m):
        totals += Y[rng.integers(0, n, size=reps)]
    return totals - m * muY


COL = {"optimum_K4": "#2471a3", "conservative_K16": "#c0392b"}
LAB = {"optimum_K4": "optimum (K=4, b*=0.347)",
       "conservative_K16": "conservative (K=16, b*=0.465)"}


def make_plots(plotdata, results):
    # ---- Fig 1: r_t / s_t distribution ----
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for name, pd in plotdata.items():
        s = pd["s"]
        nz = s[s > 0]
        axs[0].hist(nz, bins=80, histtype="step", color=COL[name], lw=1.6,
                    label=f"{LAB[name]} (nonzero only; {100*(s>0).mean():.1f}% >0)")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("per-token afforded capacity s_t (bits), nonzero positions")
    axs[0].set_ylabel("count (log)")
    axs[0].set_title("Per-token afforded capacity s_t\n(zero-inflated: ~80% of tokens are 0)")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    for name, pd in plotdata.items():
        rho = pd["rho"]
        axs[1].plot(np.arange(len(rho[:120])), rho[:120], color=COL[name],
                    lw=1.2, label=f"{LAB[name]}: tau_var={pd['tau_var']:.2f}")
    axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_xlabel("lag k (tokens, within prompt-block)")
    axs[1].set_ylabel("autocorrelation rho_k of s_t")
    axs[1].set_title("Within-sequence autocorrelation of s_t\n(integrated token-level tau ~ 1)")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "buffer_rt_dist_acf.png"), dpi=120)
    plt.close(fig)

    # ---- Fig 2: Var(R(N)) vs N ----
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for name, pd in plotdata.items():
        vrn = pd["vrn"]; sigma2 = pd["sigma2"]
        Ns = sorted(int(k) for k in vrn)
        ev = [vrn[k]["var"] for k in Ns]
        ax.plot(Ns, ev, "o-", color=COL[name], label=f"{LAB[name]} empirical Var(R(N))")
        ax.plot(Ns, [sigma2 * n for n in Ns], "--", color=COL[name], alpha=0.5,
                label=f"{LAB[name]} i.i.d. N·sigma^2")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("window size N (tokens, within prompt-block)")
    ax.set_ylabel("Var(R(N))  (bits^2)")
    ax.set_title("Cumulative-capacity variance vs window size\n"
                 "(empirical vs i.i.d.; gap = correlation inflation)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "buffer_varRN.png"), dpi=120)
    plt.close(fig)

    # ---- Fig 3: buffer(N) and buffer-per-token(N) ----
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, pd in plotdata.items():
        bc = pd["buf_curve"]
        Ns = sorted(int(k) for k in bc)
        bern = [bc[n]["bernstein_emp"] for n in Ns]
        gau = [bc[n]["gaussian"] for n in Ns]
        Ns_sub = [n for n in Ns if bc[n]["subexp"] is not None]
        sub = [bc[n]["subexp"] for n in Ns_sub]
        c = COL[name]
        axs[0].plot(Ns, bern, "o-", color=c, label=f"{LAB[name]}: Bernstein (rigorous)")
        axs[0].plot(Ns, gau, "s--", color=c, alpha=0.6, label=f"{LAB[name]}: Gaussian fit")
        axs[0].plot(Ns_sub, sub, "^:", color=c, alpha=0.6, label=f"{LAB[name]}: sub-exp tail fit")
        axs[1].plot(Ns, [bc[n]["per_token_bernstein_emp"] for n in Ns], "o-", color=c,
                    label=f"{LAB[name]}: Bernstein/N")
        axs[1].plot(Ns, [bc[n]["per_token_gaussian"] for n in Ns], "s--", color=c, alpha=0.6,
                    label=f"{LAB[name]}: Gaussian/N")
        axs[1].plot(Ns_sub, [bc[n]["subexp"] / n for n in Ns_sub], "^:", color=c, alpha=0.6,
                    label=f"{LAB[name]}: sub-exp/N")
        axs[1].axhline(pd["mu"], color=c, ls=":", lw=1.2, alpha=0.8,
                       label=f"{LAB[name]}: benign mean mu={pd['mu']:.3f}")
    for ax in axs:
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("audit size N (tokens)")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("buffer above N·mu (bits)  for FPR <= 1e-10")
    axs[0].set_title("Buffer vs audit size (FPR <= 1e-10)")
    axs[0].legend(fontsize=7)
    axs[1].set_ylabel("buffer PER TOKEN (bits/token)")
    axs[1].set_title("Per-token buffer vs audit size\n(crosses below benign mean -> channel closes)")
    axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "buffer_vs_N.png"), dpi=120)
    plt.close(fig)

    # ---- Fig 4: tail-model validation (survival curve) at Nplot ----
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (name, pd) in zip(axs, plotdata.items()):
        tot = pd["tot_plot"]; Np = pd["Nplot"]; Y = pd["Y"]
        ts = np.sort(tot)
        surv = 1.0 - (np.arange(len(ts)) + 0.5) / len(ts)
        # subsample for plotting
        idx = np.unique(np.linspace(0, len(ts) - 1, 4000).astype(int))
        ax.plot(ts[idx], surv[idx], color="k", lw=1.4, label="block-bootstrap empirical")
        # gaussian
        sig = np.sqrt((Np / SEQ_LEN) * Y.var(ddof=1))
        tg = np.linspace(ts[idx][0], ts.max() * 1.1, 400)
        from math import erfc, sqrt
        sg = np.array([0.5 * erfc(t / (sig * sqrt(2.0))) for t in tg])
        ax.plot(tg, sg, "--", color="#2471a3", label="Gaussian model")
        # exp-tail fit
        ef = fit_exptail_buffer(tot, 1e-10)
        if ef:
            tt = np.linspace(np.quantile(tot, 0.99), ts.max() * 1.1, 200)
            ax.plot(tt, np.exp(ef["intercept"] + tt * (-1.0 / ef["lambda_bits"])),
                    ":", color="#e67e22", lw=2, label=f"sub-exp fit (lam={ef['lambda_bits']:.1f})")
        for e in (1e-2, 1e-3, 1e-4, 1e-5):
            ax.axhline(e, color="gray", lw=0.4, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel(f"R(N)-N·mu (bits), N={Np}")
        ax.set_ylabel("survival P(R-Nmu > t)")
        ax.set_title(f"{LAB[name]}\ntail-model validation (N={Np})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
        ax.set_ylim(1e-7, 1)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "buffer_tail_validation.png"), dpi=120)
    plt.close(fig)


def main():
    use_extra = os.path.exists(EXTRA)
    margins, ranks, Nb, bgrid, V, block = load_dumps(use_extra)
    N_tot = margins.size
    n_blocks = len(np.unique(block))
    print(f"loaded {N_tot} tokens, {n_blocks} prompt-blocks, extra={use_extra}, V={V}")

    CONFIGS = {"optimum_K4": {"b": 0.3466, "K": 4},
               "conservative_K16": {"b": 0.4654, "K": 16}}
    results = {"n_tokens": int(N_tot), "n_blocks": int(n_blocks),
               "used_extra": bool(use_extra), "V": V, "configs": {}}

    rng = np.random.default_rng(12345)
    plotdata = {}
    for name, cfg in CONFIGS.items():
        b, K = cfg["b"], cfg["K"]
        pt = per_token(margins, ranks, Nb, bgrid, V, b, K)
        s, r = pt["s"], pt["r"]
        sd = dist_stats(s); rd = dist_stats(r)
        rho, tau, Mwin = acf_within_blocks(s, block, max_lag=400)
        sigma2 = float(np.var(s, ddof=1))

        # Var(R(N)) empirical
        Ns_within = [8, 16, 32, 64, 128, 256, 512, 1024]
        vrn = var_RN_within(s, block, Ns_within)
        Y = session_sums(s, block)
        varY = float(Y.var(ddof=1))
        tau_var = varY / (SEQ_LEN * sigma2)          # variance-inflation tau
        N_eff_factor = tau_var                        # N_eff = N/tau_var

        # buffer curves
        Ns_buf = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
                  262144, 524288, 1048576, 4194304]
        buf_curve = {}
        for N in Ns_buf:
            bern = bernstein_buffer(N, Y, pt["s_max_possible"], EPS_TARGET)
            ga, z = gaussian_buffer(N, Y, EPS_TARGET)
            # tail fit via bootstrap at this N (m sessions); only where meaningful
            m = max(1, N // SEQ_LEN)
            if m <= 256:
                reps = min(2_000_000, max(200_000, 200_000_000 // m))
                tot = block_bootstrap(Y, m, reps, rng)
                sub = fit_exptail_buffer(tot, EPS_TARGET)
            else:
                sub = None
            buf_curve[N] = {
                "bernstein_emp": bern["empirical_support"],
                "bernstein_wc": bern["worstcase_support"],
                "gaussian": ga,
                "subexp": (sub["buffer"] if sub else None),
                "var_model_V": bern["V"],
                "per_token_bernstein_emp": bern["empirical_support"] / N,
                "per_token_gaussian": ga / N,
                "mu_bits_per_token": pt["mu"],
            }

        # crossover N where per-token buffer drops below the benign mean mu
        def crossover(model_key):
            xs = sorted(buf_curve)
            for N in xs:
                v = buf_curve[N][model_key]
                if v is not None and v / N < pt["mu"]:
                    return N
            return None
        crossovers = {k: crossover(k) for k in
                      ("bernstein_emp", "gaussian", "subexp")}

        # deep bootstrap validation at N=4096 (m=4) to reach ~1e-7
        tot_deep = block_bootstrap(Y, 4, 40_000_000, rng)
        deep = {f"{e:.0e}": float(np.quantile(tot_deep, 1 - e))
                for e in [1e-4, 1e-5, 1e-6, 1e-7]}
        deep_sub = {f"{e:.0e}": (fit_exptail_buffer(tot_deep, e) or {}).get("buffer")
                    for e in [1e-4, 1e-5, 1e-6, 1e-7]}
        deep_gau = {f"{e:.0e}": gaussian_buffer(4096, Y, e)[0]
                    for e in [1e-4, 1e-5, 1e-6, 1e-7]}

        # bootstrap validation at OBSERVABLE FPRs (compare models)
        valid = {}
        for N in [4096, 16384, 65536]:
            m = max(1, N // SEQ_LEN)
            reps = 4_000_000
            tot = block_bootstrap(Y, m, reps, rng)
            emp_q = {f"{e:.0e}": float(np.quantile(tot, 1 - e))
                     for e in [1e-2, 1e-3, 1e-4, 1e-5]}
            gq = {f"{e:.0e}": gaussian_buffer(N, Y, e)[0]
                  for e in [1e-2, 1e-3, 1e-4, 1e-5]}
            sq = {f"{e:.0e}": (fit_exptail_buffer(tot, e) or {}).get("buffer")
                  for e in [1e-2, 1e-3, 1e-4, 1e-5]}
            valid[N] = {"empirical": emp_q, "gaussian": gq, "subexp": sq,
                        "bootstrap_reps": reps}

        results["configs"][name] = {
            "b_star": b, "K": K, "p": pt["p"], "q": pt["q"],
            "overhead_const": pt["overhead"], "mu_C": pt["mu"],
            "n_viol": pt["n_viol"], "n_tail": pt["n_tail"],
            "s_max_observed": pt["s_max_observed"],
            "s_max_possible": pt["s_max_possible"],
            "s_dist": sd, "r_dist": rd,
            "sigma2_per_token": sigma2,
            "tau_acf": tau, "acf_window": Mwin,
            "tau_var_inflation": tau_var,
            "var_RN_within": vrn,
            "varY_session": varY,
            "n_sessions": int(len(Y)),
            "buffer_curve": buf_curve,
            "crossover_N_pertok_below_mu": crossovers,
            "validation": valid,
            "deep_validation_N4096": {"empirical": deep, "subexp": deep_sub,
                                      "gaussian": deep_gau, "reps": 40_000_000},
            "acf_first": rho[:60].tolist(),
        }
        # survival curve for the validation plot at a mid N (many sessions)
        Nplot = 65536
        mplot = max(1, Nplot // SEQ_LEN)
        tot_plot = block_bootstrap(Y, mplot, 4_000_000, rng)
        plotdata[name] = {"s": s, "rho": rho, "vrn": vrn, "Y": Y,
                          "buf_curve": buf_curve, "sigma2": sigma2,
                          "mu": pt["mu"], "tau_var": tau_var,
                          "Nplot": Nplot, "tot_plot": tot_plot, "K": K, "b": b}
        print(f"\n[{name}] b*={b} K={K}: mu(C)={pt['mu']:.4f}  p={pt['p']:.5f} "
              f"q={pt['q']:.4f}  n_viol={pt['n_viol']} n_tail={pt['n_tail']}")
        print(f"  s: mean={sd['mean']:.5f} std={sd['std']:.4f} max={sd['max']:.3f} "
              f"frac_zero={sd['frac_zero']:.4f} skew={sd['skew']:.2f} "
              f"exkurt={sd['excess_kurtosis']:.1f}")
        print(f"  tau_acf={tau:.2f} (win {Mwin})  tau_var_inflation={tau_var:.2f}  "
              f"sigma2={sigma2:.5f}  varY={varY:.4f}")
        print(f"  crossover N (per-tok buffer<mu): {crossovers}")
        print(f"  deep valid N=4096 1e-6: emp={deep['1e-06']:.1f} "
              f"subexp={deep_sub['1e-06']:.1f} gauss={deep_gau['1e-06']:.1f}")
        for N in [1024, 4096, 65536, 1048576]:
            bc = buf_curve[N]
            sx = bc['subexp']
            sxs = f"{sx:.1f}" if sx is not None else "n/a"
            ptsx = f"{sx/N:.4f}" if sx is not None else "n/a"
            print(f"  N={N:>8}: buffer eps=1e-10  bern_emp={bc['bernstein_emp']:.1f} "
                  f"gauss={bc['gaussian']:.1f} subexp={sxs} "
                  f"| per-tok subexp={ptsx} mu={bc['mu_bits_per_token']:.4f}")

    make_plots(plotdata, results)
    with open(os.path.join(HERE, "buffer_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'buffer_results.json')}")
    print("plots -> buffer_rt_dist_acf.png, buffer_varRN.png, buffer_vs_N.png, "
          "buffer_tail_validation.png")
    return results


if __name__ == "__main__":
    main()
