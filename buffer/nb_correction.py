"""Quantify the near-band over-count: log2(K) should be log2(K - N_b) for a
top-K-but-out-of-band violator (tail correction V-K -> V-max(K,N_b) is negligible).
Reports corrected mu at the dump optimum, the K-N_b histogram, and a re-optimised b."""
import numpy as np
import analyze_threshold as at
import analyze_buffer as ab

K = 4


def H(x):
    x = float(x)
    return 0.0 if x <= 0 or x >= 1 else -x*np.log2(x) - (1-x)*np.log2(1-x)


def main():
    for sname, S in at.SCHEMES.items():
        margins, ranks, Nb, bgrid, V, block = at.load_scheme(S["orig"], S["extra"])
        logNb_full = np.log2(Nb)                      # (P, nb) — once
        b0 = S["optimum"]["b"]

        def mu_at_b(b, corrected):
            # cheap interpolation of log2 N_b at threshold b
            hi = min(max(int(np.searchsorted(bgrid, b, "left")), 0), len(bgrid)-1)
            lo = max(hi-1, 0); span = bgrid[hi]-bgrid[lo]
            if np.isclose(bgrid[hi], b) or span == 0:
                log2Nb = logNb_full[:, hi]
            else:
                w = (b-bgrid[lo])/span
                log2Nb = logNb_full[:, lo]*(1-w) + logNb_full[:, hi]*w
            compliant = margins <= b
            viol = ~compliant
            p = float(viol.mean())
            q = float((ranks[viol] >= K).mean()) if viol.any() else 0.0
            Nb_int = np.clip(np.rint(2.0**log2Nb).astype(np.int64), 1, None)
            s = np.where(compliant, log2Nb, 0.0)
            topk = viol & (ranks < K); tail = viol & (ranks >= K)
            if corrected:
                near = np.maximum(K - np.minimum(Nb_int, ranks), 1)
                s = np.where(topk, np.log2(near), s)
                tch = np.maximum(V - np.maximum(K, Nb_int), 1).astype(np.float64)
                s = np.where(tail, np.log2(tch), s)
            else:
                s = np.where(topk, np.log2(K), s)
                s = np.where(tail, np.log2(V-K), s)
            return float(s.mean() + H(p) + p*H(q)), p, q, int(topk.sum()), int(tail.sum())

        m_old, p, q, n_top, n_tail = mu_at_b(b0, False)
        m_new = mu_at_b(b0, True)[0]
        print(f"\n===== {sname}  (K={K}, dump-optimum b={b0:.4f}) =====", flush=True)
        print(f"  p(violate)={p:.4f}  q(tail|viol)={q:.4f}  "
              f"near-band toks={n_top}  tail toks={n_tail}", flush=True)
        print(f"  mu uncorrected={m_old:.5f}  corrected={m_new:.5f}  "
              f"Delta={m_new-m_old:+.5f} bits/tok", flush=True)

        # K - N_b histogram at near-band violators
        hi = min(max(int(np.searchsorted(bgrid, b0, "left")), 0), len(bgrid)-1)
        log2Nb = logNb_full[:, hi]
        topk = (margins > b0) & (ranks < K)
        Nb_top = np.minimum(np.clip(np.rint(2.0**log2Nb[topk]).astype(np.int64), 1, None),
                            ranks[topk])
        vals, cnts = np.unique(np.maximum(K-Nb_top, 1), return_counts=True)
        print("  K-N_b histogram: " +
              ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts)), flush=True)

        # re-optimise b on coarse grid (grid points only)
        rows = [(float(b), mu_at_b(b, True)[0], mu_at_b(b, False)[0]) for b in bgrid]
        b_new, mu_new, _ = min(rows, key=lambda r: r[1])
        b_old, _, mu_old = min(rows, key=lambda r: r[2])
        print(f"  re-opt: uncorrected b*={b_old:.4f} mu*={mu_old:.5f} | "
              f"corrected b*={b_new:.4f} mu*={mu_new:.5f}", flush=True)


if __name__ == "__main__":
    main()
