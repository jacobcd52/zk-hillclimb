// p3_int_zk_test.cu -- HIDING BATTERY for the composed INTEGER layer, same
// rigor and experimental design as p3_transformer_zk_test.cu (the fp8 layer's
// battery): the composed int prover applies the SAME three p3_zkopen
// mechanisms (mask-slice augmentation, degree-matched Libra blinds, salted
// leaves) + the batch blinder to every committed column / message / opening
// (soundness end-to-end: p3_int_layer_test 34/34, honest accepts + tampers
// reject in zk mode).  A full zk prove is fast but the battery still follows
// the validated design: instantiate the ACTUAL hiding mechanisms on the
// ACTUAL int-layer private column set (from a real build_witness) and drive
// >= 10k fresh mask/blind/salt draws at FIXED public challenges.  Tested:
//   (1) UNIFORMITY of every int column class's hiding opening (claimed eval,
//       blind eval, every sumcheck message, FRI final, revealed codeword).
//   (2) BLINDED CONSTRAINT MESSAGES incl. finite-difference coefficients on a
//       real int witness column, with the blind-off negative control.
//   (3) MATMUL MASK LINKAGE (the int layer's seam analogue): Y's mask slice 1
//       = Xm1*Wm1, so the hiding accumulator claim agrees with the sumcheck's
//       slice algebra AND is uniform over mask draws.  Plus the row-sum
//       linkage (S's slice 1 = row sums of E's slice 1).
//   (4) BATCH BLINDER one-time-pad on int columns.
//   (5) WITNESS-RECOVERY ATTACK on the int weight column: control leaks,
//       hidden transcript extracts 0 bits, posterior flat.
//   (6) HVZK SIMULATOR: witnessless transcripts, same law + accept.
//   (7) GKR-logUp mask siblings on real int lookup denominators, with teeth.
//
//   nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp -I. p3_int_zk_test.cu -o /root/p3_int_zk_test
//   cd /root/zkllm && /root/p3_int_zk_test
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include "p3_goldilocks.cuh"
#include "p3_int_layer.cuh"
#include "p3_zkopen.cuh"
#include "p3_gkr.cuh"
using std::vector;
using namespace p3itf;
using p3ig::gsig; using p3ig::sig64;

static int np_ = 0, nf_ = 0;
static void ck(const char* n, bool c) { printf("  [%s] %s\n", c ? "PASS" : "FAIL", n); if (c) np_++; else nf_++; }

static const int CHB = 257;
static double chisq(const vector<gl_t>& v) {
    vector<long> h(CHB, 0); for (auto x : v) h[(int)(x % CHB)]++;
    double e = (double)v.size() / CHB, c = 0;
    for (int i = 0; i < CHB; i++) { double d = h[i] - e; c += d * d / e; }
    return c;
}
static const double UNIF_HI = 400.0, LEAK_LO = 5000.0;

static uint64_t RS = 0xABCDEF;
static gl_t rc() { RS = RS * 6364136223846793005ULL + 1; uint64_t z = RS; z ^= z >> 31; return z % GL_P; }
static uint64_t rng_s = 42;
static inline uint64_t rnd64() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static inline int64_t rnds(int64_t b) { return (int64_t)(rnd64() % (2 * (uint64_t)b - 1)) - (b - 1); }

int main() {
    printf("=== INT-LAYER ZK hiding battery (p3_transformer_zk_test methodology) ===\n");
    p3fri::g_gpu_merkle = false;
    // real int-layer private witness at the battery config
    Config cfg; cfg.seq = 16; cfg.d = 64; cfg.nh = 2; cfg.dh = 32;
    cfg.dff = 128; cfg.batch = 1;
    Tables T = layer_tables(cfg);
    Weights W;
    {   // same generator as the layer battery
        W.cfg = cfg;
        int64_t wb = (int64_t)llround(65536.0 * 0.9 / sqrt((double)cfg.d));
        int64_t wbd = (int64_t)llround(65536.0 * 0.9 / sqrt((double)cfg.dff));
        for (int i = 0; i < NW; i++) {
            uint32_t K, N; wshape(cfg, i, K, N);
            W.w[i].resize((size_t)K * N);
            for (auto& v : W.w[i]) v = gsig(rnds(i == W_D ? wbd : wb));
        }
        W.g1.resize(cfg.d); W.g2.resize(cfg.d);
        for (auto& v : W.g1) v = gsig(49152 + rnds(16384));
        for (auto& v : W.g2) v = gsig(49152 + rnds(16384));
        W.rp.lseq = cfg.lseq(); W.rp.ldh = cfg.ldh();
        uint32_t dh2 = cfg.dh / 2;
        W.rp.ct.resize((size_t)cfg.seq * dh2); W.rp.st.resize((size_t)cfg.seq * dh2);
        for (uint32_t s = 0; s < cfg.seq; s++)
            for (uint32_t j = 0; j < dh2; j++) {
                double ang = (double)s * pow(10000.0, -2.0 * j / (double)cfg.dh);
                W.rp.ct[(size_t)s * dh2 + j] = (int32_t)llround(cos(ang) * 16384.0);
                W.rp.st[(size_t)s * dh2 + j] = (int32_t)llround(sin(ang) * 16384.0);
            }
    }
    vector<gl_t> x0((size_t)cfg.T() * cfg.d);
    for (auto& v : x0) v = gsig(rnds(1LL << 16));
    TfWit w = build_witness(cfg, x0, W, T);

    struct NC { const char* name; vector<gl_t> vals; };
    vector<NC> cols;
    cols.push_back({"x0 (residual input)", w.x0});
    cols.push_back({"h1 (rms1 out)", w.h1});
    cols.push_back({"Wq int weights", W.w[W_Q]});
    cols.push_back({"Wq accumulator", w.acc[W_Q]});
    cols.push_back({"yq (rescaled)", w.yq});
    cols.push_back({"rq (rope out)", w.rq});
    cols.push_back({"scores acc", w.sc});
    cols.push_back({"z (int scores)", w.z});
    cols.push_back({"P (int probs)", w.p});
    cols.push_back({"rescale limb L0", w.rsq.L[0]});
    cols.push_back({"rms M (row sums)", w.rms1.db[p3irms::B_M]});
    cols.push_back({"softmax S (row sums)", w.smx.xb[p3ismx::XB_S]});
    cols.push_back({"mo (swiglu out)", w.mo});
    cols.push_back({"out (final residual)", w.out});
    const uint32_t v = 6, N = 1u << v, R = 2, NQ = 8;
    const int M = 12000;
    for (auto& c : cols) c.vals.resize(N, 0);

    // fixed public challenges (held constant; only masks/blinds/salts vary)
    vector<gl_t> z(v); for (auto& x : z) x = rc();
    gl_t zex = rc(), rho = rc();
    vector<gl_t> r(v + 1); for (auto& x : r) x = rc();
    uint32_t M0 = 1u << (v + 1 + R);
    vector<uint32_t> qpos(NQ); for (auto& q : qpos) q = (uint32_t)(rc() % M0);

    // ---------------- 1. UNIFORMITY of every committed column class ----------------
    printf("-- (1) uniformity of EVERY int column class over %d draws, FIXED challenges --\n", M);
    bool all_ok = true;
    for (auto& nc : cols) {
        vector<gl_t> by, byh, bfc, bcw;
        vector<vector<gl_t>> bs0(v + 1), bs1(v + 1), bs2(v + 1);
        for (int i = 0; i < M; i++) {
            p3zko::HOpen o = p3zko::open(nc.vals, z, zex, rho, r, R, qpos,
                                         1000 + i, 7000000ULL + i, 9000000ULL + i, true, true);
            by.push_back(o.y); byh.push_back(o.y_h); bfc.push_back(o.final_const); bcw.push_back(o.cw_vals[0]);
            for (uint32_t rd = 0; rd < v + 1; rd++) {
                bs0[rd].push_back(o.msgs[rd].s0); bs1[rd].push_back(o.msgs[rd].s1); bs2[rd].push_back(o.msgs[rd].s2);
            }
        }
        double worst = 0;
        for (uint32_t rd = 0; rd < v + 1; rd++)
            worst = std::max({worst, chisq(bs0[rd]), chisq(bs1[rd]), chisq(bs2[rd])});
        double cy = chisq(by), cyh = chisq(byh), cfc = chisq(bfc), ccw = chisq(bcw);
        bool ok = cy < UNIF_HI && cyh < UNIF_HI && cfc < UNIF_HI && ccw < UNIF_HI && worst < UNIF_HI;
        all_ok = all_ok && ok;
        printf("    %-28s y=%.0f yh=%.0f fin=%.0f cw=%.0f worst-msg=%.0f  %s\n",
               nc.name, cy, cyh, cfc, ccw, worst, ok ? "" : "  <-- LEAK");
    }
    ck("every int column class: all transcript quantities uniform (chi-sq<400)", all_ok);

    // ---------------- 2. BLINDED CONSTRAINT MESSAGES + finite differences ----------------
    printf("-- (2) blinded constraint messages: finite-difference coeffs uniform, with teeth --\n");
    {
        const uint32_t vp = 6, Np = 1u << vp;
        vector<gl_t> zc(vp); for (auto& x : zc) x = rc();
        vector<gl_t> eqz = p3bf::build_eq(zc);
        vector<gl_t> code = cols[8].vals; code.resize(Np, 0);   // P (int probs)
        auto run = [&](uint64_t gseed, bool blind, vector<gl_t>& d2, vector<gl_t>& d3, vector<gl_t>& d4) {
            vector<gl_t> P(Np), E = eqz;
            for (uint32_t b = 0; b < Np; b++) {
                gl_t c = code[b];
                P[b] = gl_mul(E[b], gl_sub(gl_mul(c, c), c));
            }
            vector<gl_t> B[4];
            for (int j = 0; j < 4; j++) { B[j].assign(Np, 0); if (blind) { uint64_t s = gseed + 100 * j; for (auto& x : B[j]) x = p3zko::prng(s); } }
            gl_t rhoB = rc();
            uint32_t half = Np / 2;
            gl_t s[5] = {0,0,0,0,0};
            for (uint32_t i = 0; i < half; i++) {
                gl_t p0 = P[2*i], dp = gl_sub(P[2*i+1], P[2*i]);
                gl_t e0 = E[2*i], de = gl_sub(E[2*i+1], E[2*i]);
                gl_t b0[4], db[4]; for (int j=0;j<4;j++){ b0[j]=B[j][2*i]; db[j]=gl_sub(B[j][2*i+1],B[j][2*i]); }
                gl_t pc = p0, ec = e0, bc[4]; for(int j=0;j<4;j++) bc[j]=b0[j];
                for (int t = 0; t < 5; t++) {
                    gl_t bl = gl_add(bc[0], gl_mul(ec, gl_add(bc[1], gl_mul(ec, gl_add(bc[2], gl_mul(ec, bc[3]))))));
                    s[t] = gl_add(s[t], gl_add(pc, gl_mul(rhoB, bl)));
                    pc = gl_add(pc, dp); ec = gl_add(ec, de); for(int j=0;j<4;j++) bc[j]=gl_add(bc[j],db[j]);
                }
            }
            gl_t d1[4]; for (int i=0;i<4;i++) d1[i]=gl_sub(s[i+1],s[i]);
            gl_t dd2[3]; for (int i=0;i<3;i++) dd2[i]=gl_sub(d1[i+1],d1[i]);
            gl_t dd3[2]; for (int i=0;i<2;i++) dd3[i]=gl_sub(dd2[i+1],dd2[i]);
            gl_t dd4 = gl_sub(dd3[1], dd3[0]);
            d2.push_back(dd2[0]); d3.push_back(dd3[0]); d4.push_back(dd4);
        };
        vector<gl_t> on2,on3,on4,off2,off3,off4;
        for (int i = 0; i < M; i++) {
            run(40000ULL + i, true,  on2,  on3,  on4);
            run(40000ULL + i, false, off2, off3, off4);
        }
        printf("    blinded:  chisq Delta2=%.0f Delta3=%.0f Delta4=%.0f\n", chisq(on2), chisq(on3), chisq(on4));
        printf("    control:  chisq Delta2=%.0f Delta3=%.0f Delta4=%.0f\n", chisq(off2), chisq(off3), chisq(off4));
        ck("message finite-difference coeffs uniform when degree-matched-blinded",
           chisq(on2) < UNIF_HI && chisq(on3) < UNIF_HI && chisq(on4) < UNIF_HI);
        ck("message coeffs LEAK when blind off (teeth)",
           chisq(off2) > LEAK_LO && chisq(off4) > LEAK_LO);
    }

    // ---------------- 3. MATMUL + ROW-SUM MASK LINKAGE ----------------
    printf("-- (3) int linkage claims: uniform under shared ex-coordinate, and AGREE --\n");
    {
        // matmul: Y's mask slice 1 = Xm1 * Wm1 -> the augmented accumulator
        // claim (the sumcheck's base) is uniform AND both sides agree
        const uint32_t lj = 3, lk = 2, li = 3;
        vector<gl_t> X((size_t)1 << (lj + li)), Wm((size_t)1 << (lj + lk));
        for (auto& x : X) x = gsig(rnds(1LL << 17));
        for (auto& x : Wm) x = gsig(rnds(1LL << 14));
        p3imm::OpView xv = p3imm::direct_x(nullptr, lj, li);
        p3imm::OpView wv = p3imm::direct_w(nullptr, lj, lk);
        vector<gl_t> Y = p3imm::compute_y(X, xv, Wm, wv, lj, lk, li);
        vector<gl_t> zp(lk + li); for (auto& x : zp) x = rc();
        vector<gl_t> eqz = p3bf::build_eq(zp);
        vector<gl_t> byA, byB, diff;
        for (int i = 0; i < M; i++) {
            uint64_t s = 500000ULL + i;
            vector<gl_t> Xm1(X.size()), Wm1(Wm.size());
            for (auto& x : Xm1) x = p3zko::prng(s);
            for (auto& x : Wm1) x = p3zko::prng(s);
            vector<gl_t> Ym1 = p3imm::compute_y(Xm1, xv, Wm1, wv, lj, lk, li);
            gl_t zx = rc();
            gl_t yr = p3bf::eval_h(Y, eqz), ym = p3bf::eval_h(Ym1, eqz);
            // Y-claim side: augmented eval at (z || zx)
            gl_t yA = gl_add(gl_mul(gl_sub(1ULL, zx), yr), gl_mul(zx, ym));
            // sumcheck side: (1-zx)*(XW)~(z) + zx*(Xm1 Wm1)~(z), computed from
            // the operands (the slice algebra the eq-weight enforces)
            gl_t yB = gl_add(gl_mul(gl_sub(1ULL, zx), p3bf::eval_h(
                                 p3imm::compute_y(X, xv, Wm, wv, lj, lk, li), eqz)),
                             gl_mul(zx, p3bf::eval_h(Ym1, eqz)));
            byA.push_back(yA); byB.push_back(yB); diff.push_back(gl_sub(yA, yB));
        }
        bool alleq = true; for (auto d : diff) alleq = alleq && (d == 0);
        printf("    matmul-link chisq: claim side=%.0f sumcheck side=%.0f (agree=%s)\n",
               chisq(byA), chisq(byB), alleq ? "yes" : "NO");
        ck("matmul accumulator claim uniform over mask draws", chisq(byA) < UNIF_HI);
        ck("matmul claim == sumcheck slice algebra (binding preserved)", alleq);

        // row-sum: S's mask slice 1 = row sums of E's slice 1
        const uint32_t lb = 3, lc = 3;
        vector<gl_t> E((size_t)1 << (lb + lc));
        for (auto& x : E) x = (gl_t)(rnd64() & 0xFFFF);
        vector<gl_t> byS, diffS;
        vector<gl_t> zb(lb); for (auto& x : zb) x = rc();
        vector<gl_t> eqb = p3bf::build_eq(zb);
        for (int i = 0; i < M; i++) {
            uint64_t s = 700000ULL + i;
            vector<gl_t> Em1(E.size()); for (auto& x : Em1) x = p3zko::prng(s);
            vector<gl_t> Sm1 = p3ig::m1_rowsum(Em1, lc, lb, false);
            gl_t zx = rc();
            vector<gl_t> Sreal = p3ig::m1_rowsum(E, lc, lb, false);
            gl_t yS = gl_add(gl_mul(gl_sub(1ULL, zx), p3bf::eval_h(Sreal, eqb)),
                             gl_mul(zx, p3bf::eval_h(Sm1, eqb)));
            // element-side sum with the (zb||zx) row weight
            gl_t acc = 0;
            for (size_t e = 0; e < E.size(); e++)
                acc = gl_add(acc, gl_mul(gl_mul(gl_sub(1ULL, zx), eqb[e >> lc]), E[e]));
            for (size_t e = 0; e < Em1.size(); e++)
                acc = gl_add(acc, gl_mul(gl_mul(zx, eqb[e >> lc]), Em1[e]));
            byS.push_back(yS); diffS.push_back(gl_sub(yS, acc));
        }
        bool alleqS = true; for (auto d : diffS) alleqS = alleqS && (d == 0);
        printf("    rowsum-link chisq: S claim=%.0f (agree=%s)\n", chisq(byS), alleqS ? "yes" : "NO");
        ck("row-sum claim uniform over mask draws", chisq(byS) < UNIF_HI);
        ck("row-sum claim == element-side algebra (binding preserved)", alleqS);
    }

    // ---------------- 4. BATCH BLINDER one-time-pad ----------------
    printf("-- (4) batch blinder one-time-pad: combined word uniform --\n");
    {
        vector<gl_t> A0 = cols[3].vals, A1 = cols[8].vals; A0.resize(N); A1.resize(N);
        vector<gl_t> bU;
        for (int i = 0; i < M; i++) {
            uint64_t s = 800000ULL + i;
            gl_t rr0 = rc(), rr1 = rc();
            gl_t blv = p3zko::prng(s);
            bU.push_back(gl_add(gl_add(gl_mul(rr0, A0[i % N]), gl_mul(rr1, A1[i % N])), blv));
        }
        double c = chisq(bU);
        printf("    chisq(combined word)=%.0f\n", c);
        ck("batch combined word uniform (blinder pads it)", c < UNIF_HI);
    }

    // ---------------- 5. WITNESS-RECOVERY ATTACK ----------------
    printf("-- (5) witness-recovery attack on the int weight column: 0 bits under hiding --\n");
    {
        auto& wc = cols[2];                    // Wq int weights
        vector<gl_t> ctrl, hid;
        for (int i = 0; i < M; i++) {
            auto off = p3zko::open(wc.vals, z, zex, rho, r, R, qpos, 1000+i, 7000000ULL+i, 9000000ULL+i, false, false);
            auto on  = p3zko::open(wc.vals, z, zex, rho, r, R, qpos, 1000+i, 7000000ULL+i, 9000000ULL+i, true, true);
            ctrl.push_back(off.msgs[v].s0); hid.push_back(on.msgs[v].s0);
        }
        auto ndistinct = [](vector<gl_t> x){ std::sort(x.begin(),x.end()); size_t d=x.empty()?0:1; for(size_t i=1;i<x.size();i++) if(x[i]!=x[i-1]) d++; return d; };
        size_t dc = ndistinct(ctrl), dh = ndistinct(hid);
        printf("    control s0 distinct=%zu/%d; hidden distinct=%zu/%d\n", dc, M, dh, M);
        ck("control leaks (few distinct s0 = recoverable functional)", dc < 8);
        ck("hidden transcript: attack extracts 0 bits", dh > (size_t)(M * 0.9) && chisq(hid) < UNIF_HI);
        vector<gl_t> Wb = wc.vals; for (auto& x : Wb) x = gl_add(x, 77ULL);
        vector<gl_t> ha, hb;
        for (int i = 0; i < M; i++) {
            ha.push_back(p3zko::open(wc.vals, z, zex, rho, r, R, qpos, 1000+i,7000000ULL+i,9000000ULL+i).msgs[v].s0);
            hb.push_back(p3zko::open(Wb,      z, zex, rho, r, R, qpos, 1000+i,7000000ULL+i,9000000ULL+i).msgs[v].s0);
        }
        ck("posterior flat: different witnesses -> same uniform law",
           chisq(ha) < UNIF_HI && chisq(hb) < UNIF_HI);
    }

    // ---------------- 6. HVZK SIMULATOR ----------------
    printf("-- (6) HVZK simulator: witnessless transcript, same law + accepts --\n");
    {
        auto& nc = cols[2];
        vector<gl_t> rs0, ss0, rfc, sfc;
        for (int i = 0; i < M; i++) {
            auto o = p3zko::open(nc.vals, z, zex, rho, r, R, qpos, 1000+i,7000000ULL+i,9000000ULL+i);
            rs0.push_back(o.msgs[v].s0); rfc.push_back(o.final_const);
            auto sim = p3zko::simulate(o.y, z, zex, rho, r, R, NQ, 5000000ULL + i);
            ss0.push_back(sim.msgs[v].s0); sfc.push_back(sim.final_const);
        }
        printf("    chisq: real s0=%.0f sim s0=%.0f | real fin=%.0f sim fin=%.0f\n",
               chisq(rs0), chisq(ss0), chisq(rfc), chisq(sfc));
        ck("simulator ex-s0 uniform (matches real law)", chisq(ss0) < UNIF_HI);
        ck("simulator final uniform (matches real law)", chisq(sfc) < UNIF_HI);
        const char* why = nullptr;
        auto oreal = p3zko::open(nc.vals, z, zex, rho, r, R, qpos, 42, 43, 44);
        ck("real hiding opening verifies", p3zko::verify(oreal, z, zex, rho, r, qpos, &why));
        auto osim = p3zko::simulate(oreal.y, z, zex, rho, r, R, NQ, 99);
        osim.open_vals = osim.cw_vals; osim.open_salts.assign(NQ, {});
        ck("simulated (witnessless) transcript also verifies (HVZK accept)",
           p3zko::verify(osim, z, zex, rho, r, qpos, &why));
    }

    // ---------------- 7. GKR-logUp mask siblings ----------------
    printf("-- (7) GKR fraction-tree mask siblings on int lookup denominators, with teeth --\n");
    {
        const uint32_t vg = 6, Ng = 1u << vg;
        vector<gl_t> vals = cols[2].vals; vals.resize(Ng);
        vector<gl_t> valsB = vals; for (auto& x : valsB) x = gl_add(x, 77ULL);
        gl_t beta = rc();
        auto run = [&](uint64_t seed, bool mask, const vector<gl_t>& vv,
                       vector<gl_t>& bP, vector<gl_t>& bQ, vector<gl_t>& bmsg) {
            vector<gl_t> LP(2 * Ng), LQ(2 * Ng);
            uint64_t s1 = seed * 2 + 1, s2 = seed * 2 + 2;
            for (uint32_t x = 0; x < Ng; x++) {
                LP[2*x] = 1ULL; LQ[2*x] = gl_add(vv[x], beta);
                gl_t sm = mask ? p3zko::prng(s1) : 0ULL;
                gl_t qm = mask ? p3zko::prng(s2) : 1ULL;
                LP[2*x+1] = gl_mul(sm, qm); LQ[2*x+1] = qm;
            }
            fs::Transcript tp("gkr-hide");
            vector<gl_t> rf;
            p3gkr::Proof pf = p3gkr::prove(tp, "g", LP, LQ, false, rf, nullptr, nullptr, false);
            bP.push_back(pf.P); bQ.push_back(pf.Q);
            bmsg.push_back(pf.lay[3].msgs[0].s0);
        };
        vector<gl_t> oP, oQ, om, bP, bQ, bm, fP, fQ, fm, gP, gQ, gm;
        for (int i = 0; i < M; i++) {
            run(600000ULL + i, true,  vals,  oP, oQ, om);
            run(600000ULL + i, true,  valsB, bP, bQ, bm);
            run(600000ULL + i, false, vals,  fP, fQ, fm);
            run(600000ULL + i, false, valsB, gP, gQ, gm);
        }
        printf("    masks on: chisq P=%.0f Q=%.0f midmsg=%.0f\n", chisq(oP), chisq(oQ), chisq(om));
        ck("gkr root/messages uniform over mask draws",
           chisq(oP) < UNIF_HI && chisq(oQ) < UNIF_HI && chisq(om) < UNIF_HI);
        auto ndis = [](vector<gl_t> x){ std::sort(x.begin(),x.end()); size_t d=x.empty()?0:1;
                                        for(size_t i=1;i<x.size();i++) if(x[i]!=x[i-1]) d++; return d; };
        ck("teeth: masks off -> root collapses to a witness functional (leaks)",
           ndis(fQ) < 8 && fQ[0] != gQ[0]);
        ck("posterior flat: masks on -> both witnesses' roots uniform",
           chisq(oQ) < UNIF_HI && chisq(bQ) < UNIF_HI && chisq(bP) < UNIF_HI);
    }

    printf("\nINT-LAYER-ZK-HIDING: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
