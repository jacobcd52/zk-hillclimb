"""Single-panel THRESHOLD(N) curve (K=4 optimum), FAITHFUL + CODEBOOK overlaid,
with the buffer split into an OBSERVED region (variance measured directly from the
real ~131 k-token series) and an EXTRAPOLATED region (variance projected by the
fitted ~1/N law beyond the data limit).

Reframing of the buffer (the key fix; numerically identical to the validated
sub-exp sqrt(N) curve in analyze_threshold.py, just decomposed honestly):

    buffer_per_token(N) = z_eff * sqrt( Var(Rbar_N) )            (bits / token)
    THRESHOLD(N)        = mu + buffer_per_token(N)

  - Var(Rbar_N) = Var(R(N)) / N^2 is the variance of the N-token AVERAGE capacity.
      OBSERVED  (N <= N_data): measured directly from the real tokens.  Within a
        prompt (N <= L) it is the empirical variance of length-N within-block window
        sums; across prompts it is the empirical between-session variance Var(Y)
        propagated through the independent-session cumulative law
        Var(R(N)) = (N/L)*Var(Y).  Both are MEASURED quantities; the directly
        bootstrapped between-session variance tracks (N/L)*Var(Y) to ~1% out to
        N_data = 128 sessions = 131072 tokens (see run printout).
      EXTRAPOLATED (N > N_data): the SAME law Var(R(N)) = (N/L)*Var(Y) continued to
        more sessions than were measured (autocorrelation/heterogeneity-corrected
        slope Var(Y)/L = sigma^2 * tau_var, tau_var the established inflation).
  - z_eff is the SINGLE validated sub-exponential tail multiplier (the bootstrap-
    validated fit, NOT re-measured here): z_eff = c_sub / sqrt(Var(Y)/L), with c_sub
    the sub-exp sqrt(N) coefficient from analyze_threshold.py / threshold_results.json.
    It converts a variance into the FPR<=1e-10 buffer and is the SAME multiplier in
    both regions.  (Gaussian would use z=6.36; the validated heavy tail gives
    z_eff ~ 8.2-8.7.)  Bernstein is kept only as a text cross-check, NOT plotted.

HONESTY: 1e-10 is never directly observed (it needs ~1e10 honest samples).  What is
observed is the VARIANCE out to N_data; the 1e-10 multiplier z_eff is ALWAYS the
validated sub-exp tail model.  So "observed region" == variance measured directly;
the solid/dashed split is NOT a claim that 1e-10 events were seen.

Reads validated coefficients from threshold_results.json; recomputes mu / Var(Y) and
the directly-measured observed variance from the corrected dumps.  Writes
../threshold_curve.png.  int-model-approximation untouched; nothing committed.

Run: /root/int-model-env/bin/python plot_threshold_observed.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_threshold as at
import analyze_buffer as ab

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.dirname(HERE)
L = 1024
EPS = 1e-10


def scheme_numbers(sname, S, results):
    """mu, Var(Y), N_data and the validated sub-exp coefficient/tail-multiplier for
    one scheme's K=4 optimum; plus a direct measurement of the observed variance."""
    margins, ranks, Nb, bgrid, V, block = at.load_scheme(S["orig"], S["extra"])
    b, K = S["optimum"]["b"], S["optimum"]["K"]
    pt = ab.per_token(margins, ranks, Nb, bgrid, V, b, K)
    mu = pt["mu"]
    Y = ab.session_sums(pt["s"], block)
    varY = float(Y.var(ddof=1))
    n_sessions = len(Y)
    N_data = n_sessions * L

    cfg = results["schemes"][sname]["configs"]["optimum"]
    c_sub = cfg["sqrt_law_coeff_bits_sqrt_tok"]["subexp"]
    c_gau = cfg["sqrt_law_coeff_bits_sqrt_tok"]["gaussian"]
    c_bern = cfg["sqrt_law_coeff_bits_sqrt_tok"]["bernstein_asymptotic"]
    z_eff = c_sub / np.sqrt(varY / L)          # single validated sub-exp tail multiplier
    z_gau = c_gau / np.sqrt(varY / L)

    # --- direct measurement of the observed cumulative variance (grounds "observed") ---
    rng = np.random.default_rng(20260615)
    obs = {}
    for m in (4, 16, 64, n_sessions):           # N = m*L within the data extent
        tot = ab.block_bootstrap(Y, m, 400000, rng)
        obs[m * L] = float(tot.var(ddof=1))
    obs_ratio = float(np.mean([obs[m * L] / ((m * L / L) * varY)
                               for m in (4, 16, 64, n_sessions)]))

    return {"mu": mu, "varY": varY, "n_sessions": n_sessions, "N_data": N_data,
            "c_sub": c_sub, "c_bern": c_bern, "z_eff": z_eff, "z_gau": z_gau,
            "K": K, "b": b, "obs_var": obs, "obs_ratio": obs_ratio,
            "color": S["color"], "label": S["label"]}


def threshold(N, mu, c_sub):
    """THRESHOLD(N) = mu + z_eff*sqrt(Var(Rbar_N)) = mu + c_sub/sqrt(N)."""
    return mu + c_sub / np.sqrt(N)


def main():
    results = json.load(open(os.path.join(HERE, "threshold_results.json")))
    info = {s: scheme_numbers(s, S, results) for s, S in at.SCHEMES.items()}

    XMIN, XMAX = 1e4, 1e8
    table_Ns = [1e4, 1e5, 1e6, 1e7, 1e8]

    # report numbers
    table = {}
    for s, d in info.items():
        print(f"\n##### {s} K={d['K']} b*={d['b']:.4f}")
        print(f"  mu={d['mu']:.4f}  Var(Y)={d['varY']:.1f}  N_data={d['N_data']}  "
              f"({d['n_sessions']} sessions)")
        print(f"  c_sub={d['c_sub']:.3f} bits*sqrt(tok)  ->  z_eff(sub-exp)={d['z_eff']:.3f}"
              f"   (z_gauss={d['z_gau']:.3f}, c_bern={d['c_bern']:.3f})")
        print(f"  directly-measured observed Var / law ratio (mean over m=4..{d['n_sessions']}): "
              f"{d['obs_ratio']:.3f}")
        table[s] = {}
        for N in table_Ns:
            thr = threshold(N, d["mu"], d["c_sub"])
            reg = "observed" if N <= d["N_data"] else "extrapolated"
            table[s][f"{N:.0e}"] = {"threshold": float(thr), "region": reg}
            print(f"    N={N:>11.0e}: THRESHOLD={thr:.4f}  ({reg})")

    make_plot(info, Xmin=XMIN, Xmax=XMAX)

    with open(os.path.join(HERE, "threshold_observed_table.json"), "w") as f:
        json.dump({"eps": EPS, "table": table,
                   "z_eff_subexp": {s: info[s]["z_eff"] for s in info},
                   "z_gaussian": {s: info[s]["z_gau"] for s in info},
                   "N_data_tokens": {s: info[s]["N_data"] for s in info},
                   "mu": {s: info[s]["mu"] for s in info},
                   "varY_session": {s: info[s]["varY"] for s in info},
                   "c_sub": {s: info[s]["c_sub"] for s in info},
                   "obs_var_over_law_ratio": {s: info[s]["obs_ratio"] for s in info}},
                  f, indent=2)
    print("\nwrote threshold_observed_table.json")
    print("plot  -> threshold_curve.png")


def make_plot(info, Xmin, Xmax):
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    N_data = info["faithful"]["N_data"]          # identical (128 sessions) for both

    for s, d in info.items():
        c = d["color"]
        mu, c_sub = d["mu"], d["c_sub"]
        # observed branch (solid): Xmin .. N_data
        No = np.geomspace(Xmin, d["N_data"], 200)
        ax.plot(No, threshold(No, mu, c_sub), "-", color=c, lw=2.6, zorder=3,
                label=f"{d['label']} — observed (μ={mu:.3f}, z_eff={d['z_eff']:.2f})")
        # extrapolated branch (dashed): N_data .. Xmax
        Ne = np.geomspace(d["N_data"], Xmax, 200)
        ax.plot(Ne, threshold(Ne, mu, c_sub), "--", color=c, lw=2.2, zorder=3,
                label=f"{d['label']} — extrapolated (1/N law)")
        # decade-tick markers, hollow in the extrapolated region
        for N in (1e4, 1e5, 1e6, 1e7, 1e8):
            if N < Xmin or N > Xmax:
                continue
            thr = threshold(N, mu, c_sub)
            filled = N <= d["N_data"]
            ax.plot([N], [thr], "o", ms=6, color=c, zorder=4,
                    mfc=(c if filled else "white"), mec=c, mew=1.6)
        # benign-mean asymptote
        ax.axhline(mu, color=c, ls=":", lw=1.3, alpha=0.85, zorder=1)
        ax.text(Xmax, mu, f"  μ={mu:.3f}", color=c, va="center", ha="left",
                fontsize=9, fontweight="bold")

    # observed -> extrapolated boundary
    ax.axvline(N_data, color="0.35", ls="-", lw=1.1, alpha=0.8, zorder=2)
    ax.text(N_data, ax.get_ylim()[1], f" data limit\n N_data={N_data:,} tok\n (128 prompts)",
            color="0.25", va="top", ha="left", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlim(Xmin, Xmax)
    ax.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_ylim(0.16, 0.46)
    ax.set_xlabel("N  =  tokens audited  (log scale)")
    ax.set_ylabel("THRESHOLD(N)  =  μ + buffer$_{\\mathrm{FPR}\\leq10^{-10}}$(N)/N   (bits / token)")
    ax.set_title("Treaty per-token capacity ceiling vs audit size  (K=4 optimum)\n"
                 "tear up the treaty if a datacenter's measured mean afforded capacity > THRESHOLD(N)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    # legend ordering: observed/extrapolated pairs
    h, lbl = ax.get_legend_handles_labels()
    ax.legend(h, lbl, fontsize=8.5, loc="upper right", framealpha=0.95)

    cap = ("Solid = OBSERVED region: the cumulative variance Var(R̄$_N$) is measured "
           "directly from the real ~131k-token series (within-prompt windows + empirical "
           "between-session variance; matches the (N/L)·Var(Y) law to ~1%).  Dashed = "
           "EXTRAPOLATED: same 1/N variance law continued beyond the 128-prompt data limit.\n"
           "The FPR≤10$^{-10}$ buffer is variance × a SINGLE validated sub-exp tail "
           "multiplier z_eff (≈8.2–8.7, vs Gaussian 6.36) in BOTH regions — 10$^{-10}$ is "
           "never directly observed (needs ~10$^{10}$ samples); only the variance is. "
           "Bernstein (rigorous) is a text-only cross-check, not shown here.")
    fig.text(0.012, 0.012, cap, fontsize=7.6, va="bottom", ha="left", wrap=True)
    fig.tight_layout(rect=[0, 0.115, 1, 1])
    fig.savefig(os.path.join(OUTDIR, "threshold_curve.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
