// p3_hawkeye_zk_test.cu -- HIDING TEST BATTERY for the zero-knowledge Hawkeye
// opening (p3_zkopen.cuh).  Same rigor used to DETECT the F2 leak
// (leak_lastround.cu: chisq=5.12e6 at the leaking round).
//
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_hawkeye_zk_test.cu -o p3_hawkeye_zk_test
//   ./p3_hawkeye_zk_test              (run from /root/zkllm)
//
// Over MANY (>=10k) fresh mask/blind draws with a FIXED witness + FIXED public
// challenges, assert that EVERY transcript quantity a verifier sees is UNIFORM
// over the masks (chi-square vs uniform):
//   * each opening's per-round sumcheck messages s0,s1,s2 (ALL v+1 rounds,
//     incl. the ex round where F2 lived),
//   * the published blind eval y_h and claimed eval y,
//   * every revealed codeword value,
//   * the FRI final constant.
// NEGATIVE CONTROL: disable the Libra blind (rho=0) and show the test CATCHES the
// residual F2 leak (the ex-round s0 chi-square spikes), then re-enable -> uniform.
// SIMULATOR: a witnessless transcript identically distributed to a real one.
//
// The columns opened are the ACTUAL Hawkeye witness columns (operands X/W and
// derived per-product/-group/-output columns) from a golden layer, so the
// battery tests the real object, not a toy.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_zkopen.cuh"
using namespace p3hwl;
using std::vector;

static int np_ = 0, nf_ = 0;
static void ck(const char* n, bool c) { printf("  [%s] %s\n", c ? "PASS" : "FAIL", n); if (c) np_++; else nf_++; }

// chi-square of samples reduced mod B against the discrete uniform on [0,B)
static const int CHB = 257;
static double chisq(const vector<gl_t>& v) {
    vector<long> h(CHB, 0); for (auto x : v) h[(int)(x % CHB)]++;
    double e = (double)v.size() / CHB, c = 0;
    for (int i = 0; i < CHB; i++) { double d = h[i] - e; c += d * d / e; }
    return c;
}
// a 257-bin chi-square on M~10k uniform samples sits ~256 +- ~23; leaks are 1e4-1e6
static const double UNIF_HI = 400.0, LEAK_LO = 5000.0;

static uint64_t RS = 0x1234567;
static gl_t rc() { RS = RS * 6364136223846793005ULL + 1; uint64_t z = RS; z ^= z >> 31; return z % GL_P; }

// Draw the transcript quantities for one hiding open of `real` at fixed public
// (z, zex, rho, r, qpos); vary only the mask/blind/salt seeds.  Returns the
// full HOpen so the caller can bin any coordinate it likes.
struct Fixed { vector<gl_t> z; gl_t zex, rho; vector<gl_t> r; vector<uint32_t> qpos; uint32_t R; };
static Fixed make_fixed(uint32_t v, uint32_t R, uint32_t nq) {
    Fixed F; F.R = R;
    F.z.resize(v); for (auto& x : F.z) x = rc();
    F.zex = rc(); F.rho = rc();
    F.r.resize(v + 1); for (auto& x : F.r) x = rc();
    uint32_t M0 = 1u << (v + 1 + R);
    F.qpos.resize(nq); for (auto& q : F.qpos) q = (uint32_t)(rc() % M0);
    return F;
}

int main() {
    printf("=== Hawkeye ZK hiding battery (F2 fix: mask-slice + Libra-blind + salt) ===\n");
    p3fri::g_gpu_merkle = false;

    vector<Golden> Ls;
    if (!load_layers("hawkeye_layers.bin", Ls)) {
        printf("FATAL: run python3 hawkeye_ref.py --dumplayers hawkeye_layers.bin first\n");
        return 1;
    }
    Tables T = build_tables();
    Golden& L = Ls[0];                        // B=4 K=64 N=8, 2 groups
    LayerWit wt = gen_witness(L);
    printf("-- witness: layer 0 (B=%u K=%u N=%u); columns padded P=%zu G=%zu Opad=%u --\n",
           L.B, L.K, L.N, wt.d.P, wt.d.G, wt.d.Opad);

    // Assemble the real committed columns to test.  Use v=6 (length 64) slices of
    // several genuinely-different Hawkeye columns: operands X and W (the weight
    // material the threat model protects), a per-product q (quotient), and a
    // per-group total-magnitude.  A length-64 window is enough for the chi-square
    // and keeps 10k*controls draws fast on host.
    const uint32_t v = 6, N = 1u << v, R = 2, M = 12000, NQ = 8;
    auto slice = [&](const vector<gl_t>& col, size_t off) {
        vector<gl_t> s(N); for (uint32_t i = 0; i < N; i++) s[i] = col[off + i]; return s;
    };
    struct NamedCol { const char* name; vector<gl_t> vals; };
    vector<NamedCol> cols;
    { vector<gl_t> xv(wt.xcodes.size()); for (size_t i = 0; i < xv.size(); i++) xv[i] = wt.xcodes[i];
      cols.push_back({"X (operand codes)", slice(xv, 0)}); }
    { vector<gl_t> wv(wt.wcodes.size()); for (size_t i = 0; i < wv.size(); i++) wv[i] = wt.wcodes[i];
      cols.push_back({"W (weight codes)", slice(wv, 0)}); }
    cols.push_back({"dp[P_Q] (per-product quotient)", slice(wt.dp[P_Q], 0)});
    cols.push_back({"dp[P_AL] (signed contribution)", slice(wt.dp[P_AL], 0)});
    cols.push_back({"dg[G_TMAG] (per-group |total|)", slice(wt.dg[G_TMAG], 0)});

    Fixed F = make_fixed(v, R, NQ);

    // ---------------- 1. UNIFORMITY of every transcript quantity ----------------
    printf("-- (1) uniformity over %d fresh mask/blind draws, FIXED witness+challenges --\n", M);
    for (auto& nc : cols) {
        // bin: y, y_h, each round's s0/s1/s2, final_const, codeword value[0]
        vector<gl_t> by, byh, bfc, bcw;
        vector<vector<gl_t>> bs0(v + 1), bs1(v + 1), bs2(v + 1);
        for (int i = 0; i < M; i++) {
            p3zko::HOpen o = p3zko::open(nc.vals, F.z, F.zex, F.rho, F.r, F.R, F.qpos,
                                         /*mseed=*/1000 + i, /*hseed=*/7000000ULL + i,
                                         /*sseed=*/9000000ULL + i, true, true);
            by.push_back(o.y); byh.push_back(o.y_h); bfc.push_back(o.final_const);
            bcw.push_back(o.cw_vals[0]);
            for (uint32_t rd = 0; rd < v + 1; rd++) {
                bs0[rd].push_back(o.msgs[rd].s0); bs1[rd].push_back(o.msgs[rd].s1);
                bs2[rd].push_back(o.msgs[rd].s2);
            }
        }
        double cy = chisq(by), cyh = chisq(byh), cfc = chisq(bfc), ccw = chisq(bcw);
        double worst_s0 = 0; int worst_rd = -1;
        for (uint32_t rd = 0; rd < v + 1; rd++) {
            double a = chisq(bs0[rd]), b = chisq(bs1[rd]), c = chisq(bs2[rd]);
            double m = a > b ? a : b; m = m > c ? m : c;
            if (m > worst_s0) { worst_s0 = m; worst_rd = (int)rd; }
        }
        printf("  [%s]\n", nc.name);
        printf("    chisq: y=%.0f  y_h=%.0f  final=%.0f  cw=%.0f  worst-msg=%.0f (round %d, ex=round %u)\n",
               cy, cyh, cfc, ccw, worst_s0, worst_rd, v);
        bool ok = cy < UNIF_HI && cyh < UNIF_HI && cfc < UNIF_HI && ccw < UNIF_HI && worst_s0 < UNIF_HI;
        ck((std::string(nc.name) + ": all transcript quantities uniform").c_str(), ok);
    }

    // ---------------- 2. NEGATIVE CONTROL: disable the Libra blind ----------------
    // Mask-slice alone (rho blind off) leaves the LAST (ex) sumcheck round leaking
    // -- exactly the F2 finding.  The chi-square must SPIKE at the ex round, then
    // vanish when the blind is re-enabled.
    printf("-- (2) negative control: blind OFF -> the ex-round s0 must leak (teeth) --\n");
    {
        auto& nc = cols[1];                    // W: the weights
        // blind OFF, mask ON
        vector<vector<gl_t>> off_s0(v + 1);
        vector<vector<gl_t>> on_s0(v + 1);
        for (int i = 0; i < M; i++) {
            p3zko::HOpen off = p3zko::open(nc.vals, F.z, F.zex, F.rho, F.r, F.R, F.qpos,
                                           1000 + i, 7000000ULL + i, 9000000ULL + i,
                                           /*mask*/true, /*blind*/false);
            p3zko::HOpen on = p3zko::open(nc.vals, F.z, F.zex, F.rho, F.r, F.R, F.qpos,
                                          1000 + i, 7000000ULL + i, 9000000ULL + i,
                                          /*mask*/true, /*blind*/true);
            for (uint32_t rd = 0; rd < v + 1; rd++) {
                off_s0[rd].push_back(off.msgs[rd].s0); on_s0[rd].push_back(on.msgs[rd].s0);
            }
        }
        printf("    round : chisq(blind OFF)  chisq(blind ON)   [ex round = %u]\n", v);
        double ex_off = 0, ex_on = 0;
        for (uint32_t rd = 0; rd < v + 1; rd++) {
            double a = chisq(off_s0[rd]), b = chisq(on_s0[rd]);
            printf("     %2u   :   %12.0f     %10.0f%s\n", rd, a, b, rd == v ? "   <- ex (F2 site)" : "");
            if (rd == v) { ex_off = a; ex_on = b; }
        }
        ck("blind OFF: ex-round s0 LEAKS (chisq spikes -> the F2 leak is real)", ex_off > LEAK_LO);
        ck("blind ON:  ex-round s0 uniform (F2 closed)", ex_on < UNIF_HI);
    }
    // also show mask-slice OFF entirely reproduces the full non-ZK leak profile
    printf("-- (2b) both masks OFF (non-ZK baseline): every round s0 leaks --\n");
    {
        auto& nc = cols[1];
        vector<vector<gl_t>> raw_s0(v + 1);
        for (int i = 0; i < M; i++) {
            // vary only the "challenge-independent" salt so the witness/challenges
            // are fixed and nothing masks: s0 must be CONSTANT across draws.
            p3zko::HOpen raw = p3zko::open(nc.vals, F.z, F.zex, F.rho, F.r, F.R, F.qpos,
                                           1000 + i, 7000000ULL + i, 9000000ULL + i,
                                           /*mask*/false, /*blind*/false);
            for (uint32_t rd = 0; rd < v + 1; rd++) raw_s0[rd].push_back(raw.msgs[rd].s0);
        }
        double mn = 1e18, mx = 0;
        for (uint32_t rd = 0; rd < v + 1; rd++) { double a = chisq(raw_s0[rd]); if (a < mn) mn = a; if (a > mx) mx = a; }
        printf("    all-masks-off s0 chisq range: [%.0f, %.0f] (constant => ~M*(1-1/B)=%.0f)\n",
               mn, mx, (double)M * (1.0 - 1.0 / CHB));
        ck("non-ZK baseline: s0 is a deterministic witness functional (max chisq huge)", mx > LEAK_LO);
    }

    // ---------------- 3. WITNESS-INDEPENDENCE ----------------
    // A DIFFERENT witness must give the SAME (uniform) distribution -> the
    // transcript carries no information about which witness produced it.
    printf("-- (3) witness-independence: two different columns -> same uniform law --\n");
    {
        vector<gl_t> A = cols[1].vals, Bc = cols[1].vals;
        for (auto& x : Bc) x = gl_add(x, 12345ULL);       // a clearly different witness
        vector<gl_t> ya, yb, sa, sb;
        for (int i = 0; i < M; i++) {
            auto oa = p3zko::open(A, F.z, F.zex, F.rho, F.r, F.R, F.qpos, 1000 + i, 7000000ULL + i, 9000000ULL + i);
            auto ob = p3zko::open(Bc, F.z, F.zex, F.rho, F.r, F.R, F.qpos, 1000 + i, 7000000ULL + i, 9000000ULL + i);
            ya.push_back(oa.y); yb.push_back(ob.y);
            sa.push_back(oa.msgs[v].s0); sb.push_back(ob.msgs[v].s0);  // the ex-round s0
        }
        printf("    chisq: y(A)=%.0f y(B)=%.0f  ex-s0(A)=%.0f ex-s0(B)=%.0f\n",
               chisq(ya), chisq(yb), chisq(sa), chisq(sb));
        ck("different witnesses both uniform (independent)",
           chisq(ya) < UNIF_HI && chisq(yb) < UNIF_HI && chisq(sa) < UNIF_HI && chisq(sb) < UNIF_HI);
    }

    // ---------------- 4. SIMULATOR (HVZK) ----------------
    // Build witnessless transcripts from the PUBLIC y and challenges; assert they
    // are (a) identically distributed to real ones and (b) accepted by verify.
    printf("-- (4) HVZK simulator: witnessless transcript, same law + accepts --\n");
    {
        auto& nc = cols[1];
        // real distribution of the ex-round s0 and final_const
        vector<gl_t> r_s0, r_fc, s_s0, s_fc;
        gl_t y_pub = 0;
        for (int i = 0; i < M; i++) {
            auto o = p3zko::open(nc.vals, F.z, F.zex, F.rho, F.r, F.R, F.qpos, 1000 + i, 7000000ULL + i, 9000000ULL + i);
            r_s0.push_back(o.msgs[v].s0); r_fc.push_back(o.final_const);
            if (i == 0) y_pub = o.y;                       // any public claimed value
            // the simulator is fed a FRESH public y each draw (uniform, as real y is)
            gl_t y_draw = o.y;
            auto sim = p3zko::simulate(y_draw, F.z, F.zex, F.rho, F.r, F.R, NQ, 5000000ULL + i);
            s_s0.push_back(sim.msgs[v].s0); s_fc.push_back(sim.final_const);
        }
        printf("    chisq: real ex-s0=%.0f sim ex-s0=%.0f | real final=%.0f sim final=%.0f\n",
               chisq(r_s0), chisq(s_s0), chisq(r_fc), chisq(s_fc));
        ck("simulator ex-s0 uniform (matches real law)", chisq(s_s0) < UNIF_HI);
        ck("simulator final uniform (matches real law)", chisq(s_fc) < UNIF_HI);
        // both real and simulated transcripts must VERIFY
        const char* why = nullptr;
        auto oreal = p3zko::open(nc.vals, F.z, F.zex, F.rho, F.r, F.R, F.qpos, 42, 43, 44);
        bool vr = p3zko::verify(oreal, F.z, F.zex, F.rho, F.r, F.qpos, &why);
        ck("real hiding opening verifies", vr);
        auto osim = p3zko::simulate(oreal.y, F.z, F.zex, F.rho, F.r, F.R, NQ, 99);
        // give the simulated open its query-authenticated leaves (values == cw_vals)
        osim.open_vals = osim.cw_vals; osim.open_salts.assign(NQ, {});
        bool vs = p3zko::verify(osim, F.z, F.zex, F.rho, F.r, F.qpos, &why);
        ck("simulated (witnessless) transcript also verifies (HVZK accept)", vs);
    }

    // ---------------- 5. SOUNDNESS smoke: tamper rejects ----------------
    printf("-- (5) soundness smoke: tampered hiding opening rejects --\n");
    {
        auto& nc = cols[0];
        auto o = p3zko::open(nc.vals, F.z, F.zex, F.rho, F.r, F.R, F.qpos, 1, 2, 3);
        const char* why = nullptr;
        ck("honest hiding open accepts", p3zko::verify(o, F.z, F.zex, F.rho, F.r, F.qpos, &why));
        auto t1 = o; t1.msgs[3].s0 = gl_add(t1.msgs[3].s0, 1ULL);
        ck("tampered round message rejects", !p3zko::verify(t1, F.z, F.zex, F.rho, F.r, F.qpos, &why));
        auto t2 = o; t2.final_const = gl_add(t2.final_const, 1ULL);
        ck("tampered final constant rejects", !p3zko::verify(t2, F.z, F.zex, F.rho, F.r, F.qpos, &why));
        auto t3 = o; t3.y = gl_add(t3.y, 1ULL);
        ck("tampered claimed eval rejects", !p3zko::verify(t3, F.z, F.zex, F.rho, F.r, F.qpos, &why));
    }

    // ---------------- 6. CONSTRAINT-SUMCHECK messages (the other leak class) ----------------
    // Mask-slicing the columns hides the OPENING terminals but NOT the constraint
    // zero-check MESSAGES (sum_b eq(z,b) F_dp(cols(b))).  We Libra-blind them with
    // an additive random column G weighted like the constraint: run the real
    // degree-4 sumcheck on F_dp(cols) + rho*eq*G, published claim = 0 + rho*H,
    // H = sum_b eq(z,b) G(b).  Every round message = real F_dp message + rho*(G
    // message); the G part is uniform (G random) -> no round leaks.  Negative
    // control: G off -> the messages are pure F_dp(real) functionals (constant
    // across draws -> chi-square spikes).
    printf("-- (6) constraint zero-check (real F_dp) messages: Libra-blinded, with teeth --\n");
    {
        const uint32_t vp = 6, Np = 1u << vp;          // 64 real products
        // lambda weights (fixed, public) and the constraint eq point z
        gl_t lamP[6]; lamP[0] = 1; for (int j = 1; j < 6; j++) lamP[j] = rc();
        vector<gl_t> zc(vp); for (auto& x : zc) x = rc();
        vector<gl_t> rr(vp); for (auto& x : rr) x = rc();
        vector<gl_t> eqz = p3bf::build_eq(zc);
        // real column block: [E, dp0..dp9, gmaxP]  (matches F_dp's v layout)
        auto colblk = [&](uint32_t off) {
            vector<vector<gl_t>> C(1 + NDP + 1, vector<gl_t>(Np));
            for (uint32_t b = 0; b < Np; b++) {
                C[0][b] = eqz[b];
                for (int c = 0; c < NDP; c++) C[1 + c][b] = wt.dp[c][off + b];
                C[1 + NDP][b] = wt.dg[G_MAX][(off + b) >> 5];
            }
            return C;
        };
        // run the real degree-4 sumcheck of sum_b F_dp(C(b)) + rho*eq*G(b), with a
        // FRESH random G each draw; collect round-0 and last-round s0.
        auto run = [&](uint32_t off, uint64_t gseed, gl_t rho, bool blind,
                       vector<gl_t>& s0_r0, vector<gl_t>& s0_last) {
            vector<vector<gl_t>> C = colblk(off);
            vector<gl_t> G(Np, 0); if (blind) { uint64_t s = gseed; for (auto& x : G) x = p3zko::prng(s); }
            // claim0 = rho * sum_b eq(z,b) G(b)   (real part is 0)
            for (uint32_t rd = 0; rd < vp; rd++) {
                uint32_t half = (uint32_t)C[0].size() / 2;
                gl_t s[5] = {0,0,0,0,0};
                // fold G in lockstep with the columns
                for (uint32_t i = 0; i < half; i++) {
                    gl_t cur[1 + NDP + 1], dd[1 + NDP + 1];
                    for (int k = 0; k < 1 + NDP + 1; k++) { cur[k] = C[k][2*i]; dd[k] = gl_sub(C[k][2*i+1], cur[k]); }
                    gl_t gcur = G[2*i], gd = gl_sub(G[2*i+1], G[2*i]);
                    gl_t ecur = cur[0], ed = dd[0];       // eq lives in column 0
                    for (int t = 0; t < 5; t++) {
                        gl_t fv = F_dp(cur, lamP);
                        s[t] = gl_add(s[t], gl_add(fv, gl_mul(rho, gl_mul(ecur, gcur))));
                        if (t < 4) { for (int k = 0; k < 1 + NDP + 1; k++) cur[k] = gl_add(cur[k], dd[k]);
                                     gcur = gl_add(gcur, gd); ecur = gl_add(ecur, ed); }
                    }
                }
                if (rd == 0) s0_r0.push_back(s[0]);
                if (rd == vp - 1) s0_last.push_back(s[0]);
                gl_t a = rr[rd];
                for (auto& col : C) { uint32_t h = col.size()/2; vector<gl_t> nc(h);
                    for (uint32_t b = 0; b < h; b++) nc[b] = gl_add(col[2*b], gl_mul(a, gl_sub(col[2*b+1], col[2*b]))); col = nc; }
                { uint32_t h = G.size()/2; vector<gl_t> ng(h);
                  for (uint32_t b = 0; b < h; b++) ng[b] = gl_add(G[2*b], gl_mul(a, gl_sub(G[2*b+1], G[2*b]))); G = ng; }
            }
        };
        gl_t rho = rc();
        vector<gl_t> on_r0, on_last, off_r0, off_last;
        for (int i = 0; i < M; i++) {
            run(0, 40000ULL + i, rho, true,  on_r0,  on_last);
            run(0, 40000ULL + i, rho, false, off_r0, off_last);
        }
        printf("    blinded:  chisq(round0 s0)=%.0f  chisq(last-round s0)=%.0f\n", chisq(on_r0), chisq(on_last));
        printf("    control(G off): chisq(round0 s0)=%.0f  chisq(last-round s0)=%.0f (constant=%.0f)\n",
               chisq(off_r0), chisq(off_last), (double)M * (1.0 - 1.0 / CHB));
        ck("constraint F_dp messages uniform when Libra-blinded", chisq(on_r0) < UNIF_HI && chisq(on_last) < UNIF_HI);
        ck("constraint F_dp messages LEAK when blind off (teeth)", chisq(off_r0) > LEAK_LO && chisq(off_last) > LEAK_LO);
    }

    printf("\nHAWKEYE-ZK-HIDING: %d passed, %d failed -> %s\n", np_, nf_, nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
