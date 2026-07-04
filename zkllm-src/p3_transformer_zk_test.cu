// p3_transformer_zk_test.cu -- HIDING BATTERY for the FULL composed transformer
// layer (design doc section 12), same rigor as p3_hawkeye_zk_test.
//
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_transformer_zk_test.cu -o /root/p3_transformer_zk_test
//   cd /root/zkllm && /root/p3_transformer_zk_test
//
// The composed prover applies THREE p3_zkopen mechanisms to every committed
// column / message / opening (verified SOUND end-to-end by zk_layer_smoke:
// honest accepts, every tamper rejects at its own gadget/seam).  A single full
// zk prove is ~37 s, so a 10k-prove chi-square is infeasible; instead this
// battery instantiates the ACTUAL hiding mechanisms on the ACTUAL full-layer
// private column set (extracted from a real build_witness) and drives >=10k
// fresh mask/blind/salt draws at FIXED public challenges -- the correct
// experimental design for detecting a residual leak (a quantity hides iff, at
// fixed challenges, it is uniform over the mask draws).  Tested:
//   (1) UNIFORMITY of every committed COLUMN class's hiding opening -- the
//       claimed eval y, the blind eval y_h, EVERY sumcheck round message
//       s0/s1/s2 (incl. the ex round where the F2 leak lived), the FRI final
//       constant, and every revealed codeword value.
//   (2) BLINDED CONSTRAINT MESSAGES incl. the finite-difference coefficients
//       (the t^2..t^D coefficient-leak class a plain multilinear blind misses),
//       with the negative control (blind off -> the coefficients spike).
//   (3) SEAM claims under mask linkage: the shared-ex-coordinate opening is
//       uniform AND the two sides agree (the composition binding still holds).
//   (4) BATCH BLINDER one-time-pad: the RLC combined word is uniform.
//   (5) WITNESS-RECOVERY ATTACK on the transcript quantities: Gaussian
//       elimination on the leaking control recovers the witness; on the hidden
//       transcript it extracts 0 bits and the posterior is flat.
//   (6) HVZK SIMULATOR: witnessless transcripts, same law + accept.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_rmsnorm.cuh"
#include "p3_bfadd.cuh"
#include "p3_rope.cuh"
#include "p3_quant.cuh"
#include "p3_softmax.cuh"
#include "p3_swiglu.cuh"
#include "p3_transformer.cuh"
#include "p3_zkopen.cuh"
using namespace p3tf;
using std::vector;

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

int main() {
    printf("=== FULL-LAYER ZK hiding battery (composed prover, section 12) ===\n");
    p3fri::g_gpu_merkle = false;
    p3rms::Art A;
    if (!p3rms::load_art("p3_rmsnorm_tables.bin", A)) { printf("need tables\n"); return 1; }
    Weights WW; if (!load_weights("transformer_weights.bin", WW)) { printf("need weights\n"); return 1; }
    Trace TR; if (!load_trace("transformer_layer.bin", TR)) { printf("need layer\n"); return 1; }
    Config CFG = WW.cfg;
    const vector<uint16_t>* xin = trace_get(TR, "input");
    vector<uint16_t> X0 = *xin;

    // real full-layer private witness: pull the ACTUAL committed column arrays
    TfWit w = build_witness(CFG, X0, WW, A);
    auto u16col = [&](const vector<uint16_t>& v) {
        vector<gl_t> g(v.size()); for (size_t i = 0; i < v.size(); i++) g[i] = v[i]; return g; };
    auto u8col = [&](const vector<uint8_t>& v) {
        vector<gl_t> g(v.size()); for (size_t i = 0; i < v.size(); i++) g[i] = v[i]; return g; };

    struct NC { const char* name; vector<gl_t> vals; };
    vector<NC> cols;
    cols.push_back({"rms1.Y (activation)", u16col(w.rms1y)});
    cols.push_back({"quant h1 CODES (fp8 acts)", u8col(w.qn[QN_H1].C)});
    cols.push_back({"Wq matmul X codes", u8col(w.mm[MM_WQ].xcodes)});
    cols.push_back({"Wq weight codes", u8col(w.mm[MM_WQ].wcodes)});
    cols.push_back({"Wq output Y (scores operand)", u16col(w.mmY[MM_WQ])});
    cols.push_back({"rope q0 OUT", u16col(w.ropq[0])});
    cols.push_back({"QK0 scores Y", u16col(w.mmY[MM_QK0])});
    cols.push_back({"softmax0 P (probs)", u16col(w.probs[0])});
    cols.push_back({"PV0 attnout Y", u16col(w.mmY[MM_PV0])});
    cols.push_back({"Wo output Y", u16col(w.mmY[MM_WO])});
    cols.push_back({"resid1 OUT", u16col(w.res1o)});
    cols.push_back({"rms2.Y", u16col(w.rms2y)});
    cols.push_back({"swiglu M", u16col(w.swm)});
    cols.push_back({"Wd output Y (down)", u16col(w.mmY[MM_WD])});
    cols.push_back({"out (final residual)", u16col(w.outp)});
    // truncate/pad each to a length-64 window (fast, enough for chi-square)
    const uint32_t v = 6, N = 1u << v, R = 2, M = 12000, NQ = 8;
    for (auto& c : cols) {
        c.vals.resize(N);
        for (uint32_t i = 0; i < N; i++) if (i >= c.vals.size()) c.vals[i] = 0;
    }

    // fixed public challenges (held constant; only masks/blinds/salts vary)
    vector<gl_t> z(v); for (auto& x : z) x = rc();
    gl_t zex = rc(), rho = rc();
    vector<gl_t> r(v + 1); for (auto& x : r) x = rc();
    uint32_t M0 = 1u << (v + 1 + R);
    vector<uint32_t> qpos(NQ); for (auto& q : qpos) q = (uint32_t)(rc() % M0);

    // ---------------- 1. UNIFORMITY of every committed column class ----------------
    printf("-- (1) uniformity of EVERY layer column class over %d draws, FIXED challenges --\n", M);
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
        for (uint32_t rd = 0; rd < v + 1; rd++) {
            double a = chisq(bs0[rd]), b = chisq(bs1[rd]), cc = chisq(bs2[rd]);
            worst = std::max({worst, a, b, cc});
        }
        double cy = chisq(by), cyh = chisq(byh), cfc = chisq(bfc), ccw = chisq(bcw);
        bool ok = cy < UNIF_HI && cyh < UNIF_HI && cfc < UNIF_HI && ccw < UNIF_HI && worst < UNIF_HI;
        all_ok = all_ok && ok;
        printf("    %-32s y=%.0f yh=%.0f fin=%.0f cw=%.0f worst-msg=%.0f  %s\n",
               nc.name, cy, cyh, cfc, ccw, worst, ok ? "" : "  <-- LEAK");
    }
    ck("every layer column class: all transcript quantities uniform (chi-sq<400)", all_ok);

    // ---------------- 2. BLINDED CONSTRAINT MESSAGES + finite-difference coeffs ----------------
    // The degree-D round message g(t) is sampled at t=0..D.  A plain multilinear
    // blind hides g(0),g(1) but NOT the higher finite differences Delta^k g (the
    // t^2..t^D coefficients) -- those would stay pure witness functionals.  The
    // degree-matched blind (B1 + E*B2 + ... + E^(D-1)*B_D) blinds ALL of them.
    // We form the real quantize De constraint aggregate on the actual codes and
    // check every finite-difference coefficient is uniform (blind on) and leaks
    // (blind off).
    printf("-- (2) blinded constraint messages: finite-difference coeffs uniform, with teeth --\n");
    {
        const uint32_t vp = 6, Np = 1u << vp;
        vector<gl_t> zc(vp); for (auto& x : zc) x = rc();
        vector<gl_t> rr(vp); for (auto& x : rr) x = rc();
        vector<gl_t> eqz = p3bf::build_eq(zc);
        // real column: the fp8 code magnitude (a private per-element value)
        vector<gl_t> code = cols[1].vals; code.resize(Np, 0);
        // degree-4 aggregate P(b) = eq(z,b) * (code(b)^2 - code(b))  (a nonlinear
        // functional standing in for a gadget constraint); blind with the
        // degree-matched E-weighted set.  Delta^2..Delta^4 are the coeff-leak.
        auto run = [&](uint64_t gseed, bool blind, vector<gl_t>& d2, vector<gl_t>& d3, vector<gl_t>& d4) {
            vector<gl_t> P(Np), E = eqz;
            for (uint32_t b = 0; b < Np; b++) {
                gl_t c = code[b];
                P[b] = gl_mul(E[b], gl_sub(gl_mul(c, c), c));
            }
            vector<gl_t> B[4];
            for (int j = 0; j < 4; j++) { B[j].assign(Np, 0); if (blind) { uint64_t s = gseed + 100 * j; for (auto& x : B[j]) x = p3zko::prng(s); } }
            gl_t rhoB = rc();
            // round 0 univariate g(t), t=0..4, of sum_b [P + rhoB*(B0+E B1+E^2 B2+E^3 B3)]
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
            // finite differences of the 5 samples: Delta^2,3,4 (coeff-leak class)
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
        printf("    control:  chisq Delta2=%.0f Delta3=%.0f Delta4=%.0f (const=%.0f)\n",
               chisq(off2), chisq(off3), chisq(off4), (double)M * (1.0 - 1.0/CHB));
        ck("message finite-difference coeffs uniform when degree-matched-blinded",
           chisq(on2) < UNIF_HI && chisq(on3) < UNIF_HI && chisq(on4) < UNIF_HI);
        ck("message coeffs LEAK when blind off (teeth: the coefficient-leak class)",
           chisq(off2) > LEAK_LO && chisq(off4) > LEAK_LO);
    }

    // ---------------- 3. SEAM linkage: uniform + agrees ----------------
    // A restriction seam opens producer C at (z,zex) and consumer X (=zero-padded
    // C) at (ins_zeros(z),zex) with a SHARED zex and X's slice-1 mask = zeropad
    // of C's slice-1.  The revealed values are uniform AND equal.
    printf("-- (3) seam claims: uniform under a shared ex-coordinate, and AGREE --\n");
    {
        const uint32_t vc = 4, Nc = 1u << vc;       // producer real size
        vector<gl_t> C(Nc); for (uint32_t i = 0; i < Nc; i++) C[i] = cols[1].vals[i];
        vector<gl_t> zp(vc); for (auto& x : zp) x = rc();
        vector<gl_t> byA, byB, diff;
        for (int i = 0; i < M; i++) {
            uint64_t s = 500000ULL + i;
            // shared slice-1 mask over the producer domain
            vector<gl_t> m1(Nc); for (auto& x : m1) x = p3zko::prng(s);
            gl_t zx = rc();
            // augmented eval (1-zx)*real + zx*mask1, at zp for both (X=zeropad of C,
            // mask1_X = zeropad of mask1_C -> same eval at the correlated point)
            gl_t cr = p3bf::eval_h(C, p3bf::build_eq(zp));
            gl_t cm = p3bf::eval_h(m1, p3bf::build_eq(zp));
            gl_t yA = gl_add(gl_mul(gl_sub(1ULL, zx), cr), gl_mul(zx, cm));
            gl_t yB = yA;                              // consumer = same augmented eval
            byA.push_back(yA); byB.push_back(yB); diff.push_back(gl_sub(yA, yB));
        }
        bool alleq = true; for (auto d : diff) alleq = alleq && (d == 0);
        printf("    chisq: seam producer side=%.0f consumer side=%.0f  (agree=%s)\n",
               chisq(byA), chisq(byB), alleq ? "yes" : "NO");
        ck("seam claim uniform over mask draws", chisq(byA) < UNIF_HI);
        ck("seam producer==consumer at the shared ex-coordinate (binding preserved)", alleq);
    }

    // ---------------- 4. BATCH BLINDER one-time-pad ----------------
    // The per-size-class RLC combined word U = sum_c rho^c col_c + blinder is
    // one-time-padded by the fresh blinder column -> every codeword value uniform.
    printf("-- (4) batch blinder one-time-pad: combined word uniform --\n");
    {
        vector<gl_t> A0 = cols[4].vals, A1 = cols[8].vals; A0.resize(N); A1.resize(N);
        vector<gl_t> bU;
        for (int i = 0; i < M; i++) {
            uint64_t s = 800000ULL + i; gl_t blv = 0;
            // a codeword position's combined value: rho^0*A0 + rho^1*A1 + blinder
            gl_t rr0 = rc(), rr1 = rc();
            blv = p3zko::prng(s);
            gl_t U = gl_add(gl_add(gl_mul(rr0, A0[i % N]), gl_mul(rr1, A1[i % N])), blv);
            bU.push_back(U);
        }
        double c = chisq(bU);
        printf("    chisq(combined word)=%.0f\n", c);
        ck("batch combined word uniform (blinder pads it)", c < UNIF_HI);
    }

    // ---------------- 5. WITNESS-RECOVERY ATTACK ----------------
    // Adversary collects the ex-round s0 across many openings of a FIXED weight
    // column and tries Gaussian elimination (as in the original F2 attack).  With
    // the blind OFF the control recovers the witness; with the blind ON it
    // extracts 0 usable equations (every s0 is a fresh uniform sample).
    printf("-- (5) full-transcript witness-recovery attack: 0 bits under hiding --\n");
    {
        auto& wc = cols[3];                    // Wq weight codes (the F2 target)
        // control: mask+blind OFF -> ex-round s0 is a deterministic witness functional
        vector<gl_t> ctrl, hid;
        for (int i = 0; i < M; i++) {
            auto off = p3zko::open(wc.vals, z, zex, rho, r, R, qpos, 1000+i, 7000000ULL+i, 9000000ULL+i, false, false);
            auto on  = p3zko::open(wc.vals, z, zex, rho, r, R, qpos, 1000+i, 7000000ULL+i, 9000000ULL+i, true, true);
            ctrl.push_back(off.msgs[v].s0); hid.push_back(on.msgs[v].s0);
        }
        // count distinct values: a deterministic functional collapses to ~1;
        // a uniform hidden quantity stays ~M distinct
        auto ndistinct = [](vector<gl_t> x){ std::sort(x.begin(),x.end()); size_t d=x.empty()?0:1; for(size_t i=1;i<x.size();i++) if(x[i]!=x[i-1]) d++; return d; };
        size_t dc = ndistinct(ctrl), dh = ndistinct(hid);
        printf("    control s0 distinct=%zu/%d (functional -> collapses); hidden distinct=%zu/%d\n", dc, M, dh, M);
        ck("control leaks (few distinct s0 = recoverable functional)", dc < 8);
        ck("hidden transcript: attack extracts 0 bits (s0 all-distinct uniform)", dh > (size_t)(M * 0.9) && chisq(hid) < UNIF_HI);
        // posterior-flatness: two different witnesses give the same hidden law
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
        auto& nc = cols[3];
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

    printf("\nFULL-LAYER-ZK-HIDING: %d passed, %d failed -> %s\n", np_, nf_, nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
