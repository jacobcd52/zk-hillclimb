"""The treaty THRESHOLD curve: per-token threshold(N) = mu + buffer(N)/N vs audit
size N, at FPR <= 1e-10, for BOTH schemes (faithful and codebook), overlaid.

THRESHOLD(N) is the per-token average-capacity ceiling the treaty enforces after
auditing N tokens: a datacenter whose measured mean afforded capacity exceeds
THRESHOLD(N) is torn up.  buffer(N) is sized so an HONEST datacenter trips it with
probability <= 1e-10.  As N grows the buffer/N term falls ~1/sqrt(N) and
THRESHOLD(N) relaxes down toward the scheme's benign mean mu.

Reuses analyze_buffer.py's machinery verbatim (per_token, variance/autocorrelation,
Bernstein + sub-exponential tail-fit buffers, block bootstrap).  Adds:
  - both schemes loaded from their own (orig 8-prompt + extra 120-prompt) corrected
    dumps, each evaluated at its OWN calibrated optimum (K=4 floor) and conservative
    (K=16) (b*, K*) from topk_corrected_results_seed20260611.json;
  - the variance-scaling fit per-token-buffer(N) ~ c / sqrt(N) and the inversion
    "N needed to bring the threshold within 10% / 1% of mu at FPR 1e-10";
  - the overlaid THRESHOLD(N) plot (PNG).

Run:
  /root/int-model-env/bin/python analyze_threshold.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_buffer as ab   # per_token, dist_stats, acf, var, buffers, bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(os.path.dirname(HERE), "capacity")
OUTDIR = os.path.dirname(HERE)
SEQ_LEN = 1024
EPS = 1e-10

# Per-scheme corrected dumps: (original 8-prompt, extra 120-prompt) and the
# calibrated (b*, K*) at the K=4 floor (optimum) and K=16 (conservative), taken
# from capacity/topk_corrected_results_seed20260611.json.
SCHEMES = {
    "faithful": {
        "orig":  os.path.join(CAP, "capacity_dump_corrected_faithful_seed20260611.npz"),
        "extra": os.path.join(HERE, "faithful_extra_corrected.npz"),
        "optimum":      {"b": 0.3466331658291457,  "K": 4},
        "conservative": {"b": 0.46542713567839195, "K": 16},
        "color": "#1f77b4", "label": "FAITHFUL",
    },
    "codebook": {
        "orig":  os.path.join(CAP, "capacity_dump_corrected_codebook_seed20260611.npz"),
        "extra": os.path.join(HERE, "codebook_extra_corrected.npz"),
        "optimum":      {"b": 0.21954773869346733, "K": 4},
        "conservative": {"b": 0.2593467336683417,  "K": 16},
        "color": "#d62728", "label": "CODEBOOK",
    },
}

Ns_BUF = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
          262144, 524288, 1048576, 4194304, 16777216]


def load_scheme(orig_path, extra_path):
    """Concatenate the 8-prompt original + 120-prompt extra dumps for one scheme."""
    d = np.load(orig_path)
    bgrid = d["bgrid"].astype(np.float64)
    margins = [d["margins"].astype(np.float64)]
    ranks = [d["cand_ranks"].astype(np.int64)]
    Nb = [d["Nb"].astype(np.float64)]
    V = int(d["vocab"])
    nblk = margins[0].size // SEQ_LEN
    block = [np.repeat(np.arange(nblk), SEQ_LEN)]
    nxt = nblk
    e = np.load(extra_path)
    assert np.array_equal(e["bgrid"].astype(np.float64), bgrid), "bgrid mismatch"
    margins.append(e["margins"].astype(np.float64))
    ranks.append(e["cand_ranks"].astype(np.int64))
    Nb.append(e["Nb"].astype(np.float64))
    block.append(e["block_id"].astype(np.int64) + nxt)
    return (np.concatenate(margins), np.concatenate(ranks),
            np.concatenate(Nb, axis=0), bgrid, V, np.concatenate(block))


def buffer_at(N, Y, s_max, rng):
    """All three buffer models at audit size N (bits above N*mu)."""
    bern = ab.bernstein_buffer(N, Y, s_max, EPS)
    ga, _ = ab.gaussian_buffer(N, Y, EPS)
    m = max(1, N // SEQ_LEN)
    if m <= 256:
        reps = min(2_000_000, max(200_000, 200_000_000 // m))
        tot = ab.block_bootstrap(Y, m, reps, rng)
        sub = ab.fit_exptail_buffer(tot, EPS)
        sub = sub["buffer"] if sub else None
    else:
        sub = None
    return {"bernstein": bern["empirical_support"], "gaussian": ga, "subexp": sub}


def n_to_within(frac, mu, varY, M, model, c_sub=None):
    """Smallest N (tokens) such that the per-token buffer(N) <= frac*mu, for the
    given buffer model.  Operative variance model: Var(R(N)) = (N/L)*varY over
    independent sessions, so per-token buffer falls ~1/sqrt(N) and a unique
    crossing exists.

      gaussian : per-tok = z*sqrt(varY/L)/sqrt(N) -> closed form
                 N = (z^2 * varY / L) / (frac*mu)^2
      subexp   : same sqrt(N) law with fitted coefficient c_sub (bits*sqrt(tok)):
                 per-tok = c_sub/sqrt(N) -> N = (c_sub/(frac*mu))^2
      bernstein: buf(N) = a + sqrt(a^2 + 2 Ln (N/L) varY), a = Ln*M/3 (const in N);
                 per-tok = buf/N -> solved on a clean geometric grid.
    """
    target = frac * mu
    Ln = np.log(1.0 / EPS)
    z = ab.z_upper(EPS)
    if model == "gaussian":
        return float((z * z * varY / SEQ_LEN) / target**2)
    if model == "subexp":
        if c_sub is None:
            return None
        return float((c_sub / target) ** 2)
    if model == "bernstein":
        Ns = np.unique(np.round(2.0 ** np.arange(10.0, 60.0, 0.02)).astype(np.float64))
        a = Ln * M / 3.0
        Vv = (Ns / SEQ_LEN) * varY
        buf = a + np.sqrt(a * a + 2.0 * Ln * Vv)
        per = buf / Ns
        below = np.where(per <= target)[0]
        return float(Ns[below[0]]) if below.size else None
    raise ValueError(model)


def fit_sqrt_coeff(buf_curve, key, Nmin=16384):
    """Fit per-token buffer ~ c/sqrt(N) for buffer model `key`, using curve points
    with N>=Nmin (the asymptotic-sqrt regime).  Returns c (bits*sqrt(tokens)) as the
    median of per_tok*sqrt(N), plus a relative spread as a fit-quality flag."""
    cs = []
    for N, bc in buf_curve.items():
        if N < Nmin:
            continue
        v = bc.get(key)
        if v is None:
            continue
        per = v / N
        cs.append(per * np.sqrt(N))
    if not cs:
        return None, None
    cs = np.array(cs)
    return float(np.median(cs)), float(cs.std() / cs.mean()) if cs.mean() else None


def main():
    rng = np.random.default_rng(20260615)
    out = {"eps": EPS, "schemes": {}}
    plot = {}

    for sname, S in SCHEMES.items():
        margins, ranks, Nb, bgrid, V, block = load_scheme(S["orig"], S["extra"])
        N_tot = margins.size
        n_blocks = len(np.unique(block))
        print(f"\n##### {sname}: {N_tot} tokens, {n_blocks} prompt-blocks, V={V}")
        out["schemes"][sname] = {"n_tokens": int(N_tot), "n_blocks": int(n_blocks),
                                 "configs": {}}
        plot[sname] = {}

        for cfgname in ("optimum", "conservative"):
            b, K = S[cfgname]["b"], S[cfgname]["K"]
            pt = ab.per_token(margins, ranks, Nb, bgrid, V, b, K)
            s = pt["s"]
            mu = pt["mu"]
            sigma2 = float(np.var(s, ddof=1))
            Y = ab.session_sums(s, block)
            varY = float(Y.var(ddof=1))
            tau_var = varY / (SEQ_LEN * sigma2)
            M_emp = float(np.max(np.abs(Y - Y.mean())))

            buf_curve = {}
            for N in Ns_BUF:
                bc = buffer_at(N, Y, pt["s_max_possible"], rng)
                buf_curve[N] = {
                    **bc,
                    "threshold_bernstein": mu + bc["bernstein"] / N,
                    "threshold_gaussian": mu + bc["gaussian"] / N,
                    "threshold_subexp": (mu + bc["subexp"] / N) if bc["subexp"] else None,
                    "per_token_bernstein": bc["bernstein"] / N,
                    "per_token_gaussian": bc["gaussian"] / N,
                }

            # sqrt-law coefficients c (per-tok buffer ~ c/sqrt(N)).
            #  - Gaussian & Bernstein have CLOSED-FORM asymptotic coefficients
            #    (per-tok -> c/sqrt(N) with c = z*sqrt(varY/L) and
            #    c = sqrt(2 ln(1/eps) varY/L) resp.; Bernstein also carries a
            #    constant-in-N a-term a=ln(1/eps)*M/3 that dominates at MODERATE N
            #    and is kept exactly in the numeric N_to_within solve).
            #  - sub-exp has no closed form -> fit c from the curve (N>=16384);
            #    it has no a-term so the fit is clean. This grounds the validated
            #    1e-10 extrapolation.
            Ln = np.log(1.0 / EPS); z = ab.z_upper(EPS)
            c_gau = float(z * np.sqrt(varY / SEQ_LEN))
            c_bern = float(np.sqrt(2.0 * Ln * varY / SEQ_LEN))   # asymptotic (a-term excl.)
            c_sub, sp_sub = fit_sqrt_coeff(buf_curve, "subexp")

            # --- tail-model validation at OBSERVABLE FPRs (sub-exp vs empirical) ---
            # deep block-bootstrap at N=4096 (m=4 sessions) to ~1e-6; confirms the
            # sub-exp extrapolation tracks the empirical honest tail for THIS scheme.
            tot_deep = ab.block_bootstrap(Y, 4, 40_000_000, rng)
            validation = {}
            for e in (1e-4, 1e-5, 1e-6):
                emp = float(np.quantile(tot_deep, 1 - e))
                sub_f = ab.fit_exptail_buffer(tot_deep, e)
                gau = ab.gaussian_buffer(4096, Y, e)[0]
                validation[f"{e:.0e}"] = {
                    "empirical": emp,
                    "subexp": (sub_f["buffer"] if sub_f else None),
                    "gaussian": gau,
                    "subexp_rel_err": (sub_f["buffer"] / emp - 1.0) if sub_f else None,
                    "gaussian_rel_err": gau / emp - 1.0}

            within = {}
            for frac, tag in ((0.10, "within10pct"), (0.01, "within1pct")):
                within[tag] = {
                    "bernstein": n_to_within(frac, mu, varY, M_emp, "bernstein"),
                    "gaussian": n_to_within(frac, mu, varY, M_emp, "gaussian"),
                    "subexp": n_to_within(frac, mu, varY, M_emp, "subexp", c_sub),
                }

            cfg = {
                "b_star": b, "K": K, "mu": mu, "p": pt["p"], "q": pt["q"],
                "overhead": pt["overhead"], "sigma2": sigma2,
                "varY_session": varY, "tau_var": tau_var,
                "N_eff_factor": tau_var, "M_emp_session_dev": M_emp,
                "n_sessions": int(len(Y)),
                "n_viol": pt["n_viol"], "n_tail": pt["n_tail"],
                "s_max_observed": pt["s_max_observed"],
                "sqrt_law_coeff_bits_sqrt_tok": {
                    "bernstein_asymptotic": c_bern, "subexp": c_sub,
                    "gaussian": c_gau, "subexp_rel_spread": sp_sub,
                    "bernstein_a_term_const_bits": float(Ln * M_emp / 3.0)},
                "buffer_curve": {str(k): v for k, v in buf_curve.items()},
                "tail_validation_N4096": validation,
                "N_to_within": within,
            }
            out["schemes"][sname]["configs"][cfgname] = cfg
            plot[sname][cfgname] = {"mu": mu, "buf_curve": buf_curve,
                                    "tau_var": tau_var, "color": S["color"]}

            print(f"  [{cfgname} K={K} b*={b:.4f}] mu={mu:.4f} sigma2={sigma2:.4f} "
                  f"tau_var={tau_var:.2f} varY={varY:.1f} M_emp={M_emp:.1f} "
                  f"n_viol={pt['n_viol']} n_tail={pt['n_tail']}")
            for N in (16384, 65536, 262144, 1048576):
                bc = buf_curve[N]
                sx = bc["threshold_subexp"]
                sxs = f"{sx:.4f}" if sx else "n/a"
                print(f"    N={N:>8}: thr/tok  bern={bc['threshold_bernstein']:.4f} "
                      f"subexp={sxs}  (mu={mu:.4f})")
            def fmt(x):
                return "n/a" if x is None else f"{x:.3e}"
            print(f"    sqrt-law c (bits*sqrt(tok)): bern={c_bern:.1f} subexp="
                  f"{c_sub:.1f} gauss={c_gau:.1f}")
            for tag in ("within10pct", "within1pct"):
                w = within[tag]
                print(f"    N {tag:>11}: bern={fmt(w['bernstein'])} "
                      f"subexp={fmt(w['subexp'])} gauss={fmt(w['gaussian'])}")
            v6 = validation["1e-06"]
            print(f"    tail-valid N=4096 @1e-6: emp={v6['empirical']:.1f} "
                  f"subexp={v6['subexp']:.1f} ({100*v6['subexp_rel_err']:+.0f}%) "
                  f"gauss={v6['gaussian']:.1f} ({100*v6['gaussian_rel_err']:+.0f}%)")

    make_plot(plot)
    with open(os.path.join(HERE, "threshold_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'threshold_results.json')}")
    print("plot -> threshold_curve.png")
    return out


def make_plot(plot):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    for cfgname, ax in (("optimum", axs[0]), ("conservative", axs[1])):
        for sname, S in SCHEMES.items():
            pd = plot[sname][cfgname]
            mu = pd["mu"]; c = S["color"]; bc = pd["buf_curve"]
            Ns = sorted(bc)
            thr_b = [bc[n]["threshold_bernstein"] for n in Ns]
            Ns_s = [n for n in Ns if bc[n]["threshold_subexp"] is not None]
            thr_s = [bc[n]["threshold_subexp"] for n in Ns_s]
            ax.plot(Ns_s, thr_s, "^-", color=c, lw=2,
                    label=f"{S['label']}  THRESHOLD (tail-fit), mu={mu:.3f}")
            ax.plot(Ns, thr_b, "o--", color=c, lw=1.3, alpha=0.55,
                    label=f"{S['label']}  THRESHOLD (Bernstein, rigorous)")
            ax.axhline(mu, color=c, ls=":", lw=1.4, alpha=0.9)
            ax.text(Ns[-1], mu, f"  mu={mu:.3f}", color=c, va="center",
                    fontsize=8, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("audit size N (tokens averaged over)")
        ax.set_ylabel("THRESHOLD(N)  =  mu + buffer(N)/N   (bits / token)")
        K = SCHEMES['faithful'][cfgname]['K']
        ax.set_title(f"{cfgname.upper()} (K={K})\n"
                     "per-token capacity ceiling for honest-FPR <= 1e-10\n"
                     "(tear up the treaty if measured mean capacity > THRESHOLD)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(bottom=0)
    fig.suptitle("Treaty threshold vs audit size — relaxes toward the benign mean mu "
                 "as more tokens are audited", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTDIR, "threshold_curve.png"), dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    main()
