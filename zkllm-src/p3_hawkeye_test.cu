// FULL-LAYER Hawkeye ZKP test suite (p3_hawkeye.cuh).
//
//   python3 hawkeye_ref.py --dumplayers hawkeye_layers.bin   (Triton-checked goldens)
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_hawkeye_test.cu -o p3_hawkeye_test
//   ./p3_hawkeye_test            (run from /root/zkllm)
//
// HONEST ACCEPT: every golden layer (random + directed edge layers: shift0,
// shift>=width, negative products, absent/masked lanes, NaN codes, K%32!=0,
// multi-group acc chains, zero/negative scales) proves AND verifies against
// the PUBLIC golden Y -- which is bitwise-identical to hawkeye_ref/Triton.
//
// MUST-REJECT: semantic forgeries built by replaying the layer honestly from
// ONE forged step (so the witness is consistent everywhere except the single
// sub-argument that must catch it), plus operand/scale/param/proof tampers.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
using namespace p3hwl;
using std::vector;

static int np_ = 0, nf_ = 0;
static void ck(const char* n, bool c, const char* why = nullptr) {
    printf("  [%s] %s%s%s%s\n", c ? "PASS" : "FAIL", n,
           why ? "  (why=" : "", why ? why : "", why ? ")" : "");
    if (c) np_++; else nf_++;
}
static const uint32_t R = 2, Q = 24;

static Tables* TT;

struct Run {
    LayerProof pf;
    Operands ops;
    bool proved = false;
};

// prove wt against ops, then verify against (roots, Ypub, dims) -- returns
// verifier verdict + reason.
static bool prove_verify(const LayerWit& wt, const Operands& ops, const vector<uint16_t>& Ypub,
                         uint32_t B, uint32_t K, uint32_t N, const char** why,
                         bool strict, LayerProof* pf_out = nullptr, Prof* prof = nullptr) {
    fs::Transcript tp("hwl");
    LayerProof pf = prove(tp, wt, *TT, ops, R, Q, true, strict, prof, &Ypub);
    if (pf_out) *pf_out = pf;
    fs::Transcript tv("hwl");
    return verify(tv, *TT, pf, ops.X.root, ops.W.root, ops.XS.root, ops.WS.root,
                  Ypub, B, K, N, Q, R, why);
}

int main() {
    printf("=== Hawkeye FULL-LAYER ZKP selftest (golden layers, Triton-checked) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    vector<Golden> Ls;
    if (!load_layers("hawkeye_layers.bin", Ls)) {
        printf("FATAL: run  python3 hawkeye_ref.py --dumplayers hawkeye_layers.bin  first\n");
        return 1;
    }
    Tables T = build_tables(); TT = &T;

    // ---------------- HONEST ACCEPT battery ----------------
    printf("-- honest accept: %zu golden layers --\n", Ls.size());
    double tot_prove = 0, tot_verify = 0; size_t tot_sz = 0;
    for (size_t i = 0; i < Ls.size(); i++) {
        Golden& L = Ls[i];
        LayerWit wt = gen_witness(L);
        bool ybit = wt.Y.size() == L.y.size() &&
                    memcmp(wt.Y.data(), L.y.data(), 2 * L.y.size()) == 0;
        Operands ops = commit_operands(wt, R);
        double t0 = now_ms();
        fs::Transcript tp("hwl");
        LayerProof pf = prove(tp, wt, T, ops, R, Q);
        double t1 = now_ms();
        fs::Transcript tv("hwl");
        bool ok = verify(tv, T, pf, ops.X.root, ops.W.root, ops.XS.root, ops.WS.root,
                         L.y, L.B, L.K, L.N, Q, R, &why);
        double t2 = now_ms();
        tot_prove += t1 - t0; tot_verify += t2 - t1; tot_sz += proof_size(pf);
        char name[128];
        snprintf(name, sizeof name, "layer %zu (B=%u K=%u N=%u): witness==golden bitwise "
                 "AND proof accepts", i, L.B, L.K, L.N);
        ck(name, ybit && ok, ok ? nullptr : why);
    }
    printf("  totals: prove %.1f s, verify %.1f s, avg proof %.1f MB\n",
           tot_prove / 1e3, tot_verify / 1e3, tot_sz / Ls.size() / 1048576.0);

    // base layers for the forgery battery
    Golden& L0 = Ls[0];                        // 4x64x8, 2 groups
    LayerWit w0 = gen_witness(L0);
    Operands ops0 = commit_operands(w0, R);

    printf("-- must-reject: output / rounding forgeries --\n");
    // (1) public output off by 1 ulp (proof honest, claim wrong)
    {
        vector<uint16_t> Ybad = L0.y; Ybad[0] ^= 1;
        bool ok = prove_verify(w0, ops0, Ybad, L0.B, L0.K, L0.N, &why, true);
        ck("output off by 1 ulp rejects", !ok, why);
    }
    // (2) wrong rounding direction in the per-product align (q+1, r-=pw), fully
    //     consistent downstream incl. its own Y claim -> only REM can catch
    {
        // scan for a location where the forged rounding actually corrupts the
        // bf16 output (strongest teeth); fall back to any valid location
        bool found = false, ydiff = false;
        int base_li = 0;
        LayerWit wt;
        for (int li : {0, 8, 9, 6, 10, 2}) {
            if (ydiff) break;
            Golden& LL = Ls[li];
            LayerWit wl = gen_witness(LL);
            int tries = 0;
            for (size_t p = 0; p < wl.d.P && tries < 400 && !ydiff; p++) {
                int64_t sh = (int64_t)wl.dp[P_SH][p], q = (int64_t)wl.dp[P_Q][p];
                int64_t r = (int64_t)wl.dp[P_R][p];
                if (!(wl.dp[P_PR][p] && sh >= 1 && sh <= 14 && r > 0 && q >= 1)) continue;
                size_t gi = p >> 5;
                Tamper cand{TM_ROUND_UP, (uint32_t)(gi & (wl.d.Opad - 1)),
                            (uint32_t)(gi >> wl.d.lo), (uint32_t)(p & 31)};
                LayerWit wc = gen_witness(LL, true, &cand);
                tries++;
                bool yd = memcmp(wc.Y.data(), LL.y.data(), 2 * LL.y.size()) != 0;
                if (!found || yd) { wt = std::move(wc); found = true; ydiff = yd; base_li = li; }
            }
        }
        ck("found row for round-up forgery", found);
        if (found) {
            printf("      (layer %d; forged rounding corrupts the bf16 output: %s)\n",
                   base_li, ydiff ? "YES" : "no");
            Golden& LL = Ls[base_li];
            LayerWit wh = gen_witness(LL);
            Operands opsL = commit_operands(wh, R);
            bool ok = prove_verify(wt, opsL, wt.Y, LL.B, LL.K, LL.N, &why, false);
            ck("round-up (q+1, r-=pw) align forgery rejects via REM", !ok, why);
        }
    }
    // (3) acc-realign round-up (aq+1, ar-=apw): needs a group with abase>0
    {
        Tamper tm{TM_AQ_UP, 0, 0, 0}; bool found = false;
        for (size_t gi = 0; gi < w0.d.G && !found; gi++) {
            uint32_t g = (uint32_t)(gi >> w0.d.lo);
            if (g >= 1 && g < w0.d.NG && (int64_t)w0.dg[G_ABASE][gi] > 0 &&
                (int64_t)w0.dg[G_AR][gi] == 0 && (int64_t)w0.dg[G_ASH][gi] == 0) continue;
            if (g >= 1 && g < w0.d.NG && (int64_t)w0.dg[G_ABASE][gi] > 0) {
                tm.o = gi & (w0.d.Opad - 1); tm.g = g; found = true;
            }
        }
        ck("found group for realign forgery", found);
        if (found) {
            LayerWit wt = gen_witness(L0, true, &tm);
            bool ok = prove_verify(wt, ops0, wt.Y, L0.B, L0.K, L0.N, &why, false);
            ck("tampered acc-realign witness (aq+1) rejects via REM", !ok, why);
        }
    }
    printf("-- must-reject: group max_exp / normalize / chain forgeries --\n");
    // (4) overstated max_exp (+1), consistent downstream -> attainment catches
    {
        Tamper tm{TM_MAXEXP, 0, 0, 0}; bool found = false;
        for (size_t gi = 0; gi < w0.d.G && !found; gi++)
            if ((uint32_t)(gi >> w0.d.lo) < w0.d.NG) {
                for (int kk = 0; kk < 32; kk++)
                    if (w0.dp[P_PR][(gi << 5) + kk]) {
                        tm.o = gi & (w0.d.Opad - 1); tm.g = (uint32_t)(gi >> w0.d.lo);
                        found = true; break;
                    }
            }
        if (found) {
            LayerWit wt = gen_witness(L0, true, &tm);
            bool ok = prove_verify(wt, ops0, wt.Y, L0.B, L0.K, L0.N, &why, false);
            ck("tampered (overstated) group max_exp rejects via attainment", !ok, why);
        } else ck("found group for max_exp forgery", false);
    }
    // (5) understated normalize bit-width (wd-1), consistent downstream
    {
        Tamper tm{TM_WD_DOWN, 0, 0, 0}; bool found = false;
        for (size_t gi = 0; gi < w0.d.G && !found; gi++)
            if ((uint32_t)(gi >> w0.d.lo) < w0.d.NG && (int64_t)w0.dg[G_WD][gi] >= 2) {
                tm.o = gi & (w0.d.Opad - 1); tm.g = (uint32_t)(gi >> w0.d.lo); found = true;
            }
        if (found) {
            LayerWit wt = gen_witness(L0, true, &tm);
            bool ydiff = memcmp(wt.Y.data(), L0.y.data(), 2 * L0.y.size()) != 0;
            printf("      (forged width changes Y: %s)\n", ydiff ? "yes" : "no");
            bool ok = prove_verify(wt, ops0, wt.Y, L0.B, L0.K, L0.N, &why, false);
            ck("tampered normalize width rejects via pow2 sandwich", !ok, why);
        } else ck("found group for width forgery", false);
    }
    // (6) teleported accumulator state entering a group (chain must catch)
    {
        Tamper tm{TM_STATE, 0, 0, 0}; bool found = false;
        for (size_t gi = 0; gi < w0.d.G && !found; gi++) {
            uint32_t g = (uint32_t)(gi >> w0.d.lo);
            if (g >= 1 && g <= w0.d.NG && (int64_t)w0.dg[G_ASIG][gi] > 0) {
                tm.o = gi & (w0.d.Opad - 1); tm.g = g; found = true;
            }
        }
        if (found) {
            LayerWit wt = gen_witness(L0, true, &tm);
            bool ok = prove_verify(wt, ops0, wt.Y, L0.B, L0.K, L0.N, &why, false);
            ck("teleported acc state rejects via the chain sumcheck", !ok, why);
        } else ck("found group for state forgery", false);
    }
    // (7) swapped group order: honest witness for the layer with K-groups 0,1
    //     swapped, proven against the ORIGINAL X/W commitments
    {
        int li = -1; Golden Ls2;
        for (int cand : {6, 0, 2}) {
            Golden L2 = Ls[cand];
            for (uint32_t r_ = 0; r_ < L2.B; r_++)
                for (uint32_t k = 0; k < 32 && 32 + k < L2.K; k++)
                    std::swap(L2.x[r_ * L2.K + k], L2.x[r_ * L2.K + 32 + k]);
            for (uint32_t r_ = 0; r_ < L2.N; r_++)
                for (uint32_t k = 0; k < 32 && 32 + k < L2.K; k++)
                    std::swap(L2.w[r_ * L2.K + k], L2.w[r_ * L2.K + 32 + k]);
            LayerWit wsw = gen_witness(L2, true);
            if (memcmp(wsw.Y.data(), Ls[cand].y.data(), 2 * wsw.Y.size()) != 0) {
                li = cand; Ls2 = L2; break;
            }
        }
        ck("found layer where group order changes Y", li >= 0);
        if (li >= 0) {
            LayerWit worig = gen_witness(Ls[li]);
            Operands opsO = commit_operands(worig, R);
            LayerWit wsw = gen_witness(Ls2, true);
            bool ok = prove_verify(wsw, opsO, wsw.Y, Ls2.B, Ls2.K, Ls2.N, &why, false);
            ck("swapped group order rejects via the X/W operand binding", !ok, why);
        }
    }
    printf("-- must-reject: lookup / operand / scale / param forgeries --\n");
    // (8) forged DM entry: mag+1 with r+1 (C1 and everything else consistent,
    //     aligned value unchanged) -> only the DM lookup catches
    {
        long pi = -1;
        for (size_t p = 0; p < w0.d.P; p++) {
            int64_t pw = (int64_t)w0.dp[P_PW][p], r = (int64_t)w0.dp[P_R][p];
            if (w0.dp[P_PR][p] && (int64_t)w0.dp[P_SH][p] >= 1 && r + 1 < pw) { pi = (long)p; break; }
        }
        ck("found row for DM forgery", pi >= 0);
        if (pi >= 0) {
            LayerWit wt = w0;
            wt.dp[P_MAG][pi] = gl_add(wt.dp[P_MAG][pi], 1ULL);
            wt.dp[P_R][pi] = gl_add(wt.dp[P_R][pi], 1ULL);
            wt.lidx[LU_RM][pi] += 1;
            bool ok = prove_verify(wt, ops0, wt.Y, L0.B, L0.K, L0.N, &why, false);
            ck("forged decode-multiply output rejects via DM lookup", !ok, why);
        }
    }
    // (9) attainment selector moved to a non-max product (sel*sh != 0)
    {
        long p1 = -1, p2 = -1;
        for (size_t gi = 0; gi < w0.d.G && p1 < 0; gi++) {
            long a = -1, b = -1;
            for (int kk = 0; kk < 32; kk++) {
                size_t p = (gi << 5) + kk;
                if (w0.dp[P_SEL][p]) a = (long)p;
                if (w0.dp[P_PR][p] && (int64_t)w0.dp[P_SH][p] >= 1) b = (long)p;
            }
            if (a >= 0 && b >= 0) { p1 = a; p2 = b; }
        }
        ck("found rows for selector forgery", p1 >= 0);
        if (p1 >= 0) {
            LayerWit wt = w0;
            wt.dp[P_SEL][p1] = 0; wt.dp[P_SEL][p2] = 1;
            bool ok = prove_verify(wt, ops0, wt.Y, L0.B, L0.K, L0.N, &why, false);
            ck("misplaced attainment selector rejects via sel*sh=0", !ok, why);
        }
    }
    // (10) wrong per-row scale: witness honest for xs'=2*xs, original commitment
    {
        Golden L2 = L0;
        float f; memcpy(&f, &L2.xs[0], 4); f *= 2.0f; memcpy(&L2.xs[0], &f, 4);
        LayerWit wt = gen_witness(L2);
        bool ok = prove_verify(wt, ops0, wt.Y, L2.B, L2.K, L2.N, &why, false);
        ck("wrong per-row scale rejects via the xs commitment binding", !ok, why);
    }
    // (11) corrupted committed operand: one X code flipped in the witness
    {
        Golden L2 = L0; L2.x[0] ^= 0x08;
        LayerWit wt = gen_witness(L2);
        bool ok = prove_verify(wt, ops0, wt.Y, L2.B, L2.K, L2.N, &why, false);
        ck("corrupted committed operand rejects via the X binding", !ok, why);
    }
    // (12) parameter forgeries: Q=0 and wrong dims must be caught by the
    //      verifier's OWN pinned params (never read from the proof)
    {
        fs::Transcript tp("hwl");
        LayerProof pf = prove(tp, w0, T, ops0, R, Q);
        fs::Transcript tv1("hwl");
        bool ok1 = verify(tv1, T, pf, ops0.X.root, ops0.W.root, ops0.XS.root, ops0.WS.root,
                          L0.y, L0.B, L0.K, L0.N, /*Q=*/0, R, &why);
        ck("Q=0 params forgery rejects", !ok1, why);
        fs::Transcript tv2("hwl");
        bool ok2 = verify(tv2, T, pf, ops0.X.root, ops0.W.root, ops0.XS.root, ops0.WS.root,
                          L0.y, L0.B, L0.K, L0.N + 1, Q, R, &why);
        ck("wrong public dims rejects", !ok2, why);
        fs::Transcript tv3("hwl");
        bool ok3 = verify(tv3, T, pf, ops0.W.root, ops0.X.root, ops0.XS.root, ops0.WS.root,
                          L0.y, L0.B, L0.K, L0.N, Q, R, &why);
        ck("swapped X/W commitment roots rejects", !ok3, why);
        // (13) proof-object tampers
        { auto p2 = pf; p2.mDp[0].s1 = gl_add(p2.mDp[0].s1, 1ULL);
          fs::Transcript tv("hwl");
          bool ok = verify(tv, T, p2, ops0.X.root, ops0.W.root, ops0.XS.root, ops0.WS.root,
                           L0.y, L0.B, L0.K, L0.N, Q, R, &why);
          ck("tampered constraint sumcheck message rejects", !ok, why); }
        { auto p2 = pf; p2.oY.y = gl_add(p2.oY.y, 1ULL);
          fs::Transcript tv("hwl");
          bool ok = verify(tv, T, p2, ops0.X.root, ops0.W.root, ops0.XS.root, ops0.WS.root,
                           L0.y, L0.B, L0.K, L0.N, Q, R, &why);
          ck("tampered opened Y-column value rejects", !ok, why); }
        { auto p2 = pf; p2.lu[LU_DM].S = gl_add(p2.lu[LU_DM].S, 1ULL);
          fs::Transcript tv("hwl");
          bool ok = verify(tv, T, p2, ops0.X.root, ops0.W.root, ops0.XS.root, ops0.WS.root,
                           L0.y, L0.B, L0.K, L0.N, Q, R, &why);
          ck("tampered DM lookup sum rejects", !ok, why); }
        { auto p2 = pf; p2.rdp[P_AL] = p2.rdp[P_Q];
          fs::Transcript tv("hwl");
          bool ok = verify(tv, T, p2, ops0.X.root, ops0.W.root, ops0.XS.root, ops0.WS.root,
                           L0.y, L0.B, L0.K, L0.N, Q, R, &why);
          ck("swapped witness column root rejects", !ok, why); }
        { auto p2 = pf; p2.lu[LU_DM].y_virt[0] = gl_add(p2.lu[LU_DM].y_virt[0], 1ULL);
          fs::Transcript tv("hwl");
          bool ok = verify(tv, T, p2, ops0.X.root, ops0.W.root, ops0.XS.root, ops0.WS.root,
                           L0.y, L0.B, L0.K, L0.N, Q, R, &why);
          ck("tampered virtual-column claim rejects", !ok, why); }
    }

    printf("\nHAWKEYE-LAYER: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
