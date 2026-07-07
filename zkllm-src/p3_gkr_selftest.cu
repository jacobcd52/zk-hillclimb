// p3_gkr_selftest.cu -- standalone battery for the GKR fractional-sum tree.
//
//   nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp p3_gkr_selftest.cu -o p3_gkr_selftest
//
// Checks, per size L: (1) honest prove/verify accepts, both last-layer modes;
// (2) the root fraction equals the direct rational sum (P/Q == sum p_i/q_i);
// (3) the final leaf claims equal the true MLE evaluations at rfin;
// (4) tampers reject: root P, root Q, a mid-layer message, a terminal claim,
//     a truncated layer;  (5) a WRONG-LEAF forgery (prover honest procedure on
//     leaves that differ in one entry) yields leaf claims that differ from the
//     true leaves' MLE (the caller's binding check would catch it).
#include <cstdio>
#include <cstdint>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_gkr.cuh"
#include "fs_transcript.hpp"
using std::vector;

static int np_ = 0, nf_ = 0;
static void ck(const char* n, bool c) {
    printf("  [%s] %s\n", c ? "PASS" : "FAIL", n); if (c) np_++; else nf_++;
}
static uint64_t RS = 0x1234567;
static gl_t rc() { RS = RS * 6364136223846793005ULL + 1; uint64_t z = RS; z ^= z >> 31; return z % GL_P; }

int main() {
    printf("=== p3_gkr fractional-sum tree selftest ===\n");
    for (uint32_t L : {1u, 2u, 5u, 10u, 14u}) {
        size_t N = (size_t)1 << L;
        vector<gl_t> p(N), q(N);
        for (size_t i = 0; i < N; i++) { p[i] = rc(); do { q[i] = rc(); } while (!q[i]); }
        // direct rational sum
        gl_t S = 0;
        for (size_t i = 0; i < N; i++) S = gl_add(S, gl_mul(p[i], gl_inv(q[i])));
        char nm[128];

        for (int cl = 0; cl < 2; cl++) {
            fs::Transcript tp("gkr-t");
            vector<gl_t> rf; gl_t cp = 0, cq = 0;
            p3gkr::Proof pf = p3gkr::prove(tp, "g", p, q, cl, rf, &cp, &cq);
            snprintf(nm, sizeof nm, "L=%u cl=%d root fraction == direct sum", L, cl);
            ck(nm, gl_mul(S, pf.Q) == pf.P && pf.Q != 0);
            fs::Transcript tv("gkr-t");
            vector<gl_t> rfv; gl_t o4[4]; const char* why = nullptr;
            bool ok = p3gkr::verify(tv, "g", L, pf, cl, rfv, o4, &why);
            snprintf(nm, sizeof nm, "L=%u cl=%d honest verify accepts", L, cl);
            ck(nm, ok && rfv == rf);
            // leaf-claim truth
            bool claims_ok;
            if (cl) {
                gl_t tp_ = p3bf::eval_h(p, p3bf::build_eq(rf));
                gl_t tq_ = p3bf::eval_h(q, p3bf::build_eq(rf));
                claims_ok = (o4[0] == tp_ && o4[2] == tq_ && cp == tp_ && cq == tq_);
            } else {
                // four claims at (0,rf),(1,rf): split leaf arrays by LSB
                vector<gl_t> p0(N/2), p1(N/2), q0(N/2), q1(N/2);
                for (size_t i = 0; i < N/2; i++) { p0[i]=p[2*i]; p1[i]=p[2*i+1]; q0[i]=q[2*i]; q1[i]=q[2*i+1]; }
                vector<gl_t> eq = p3bf::build_eq(rf);
                claims_ok = (o4[0] == p3bf::eval_h(p0, eq) && o4[1] == p3bf::eval_h(p1, eq) &&
                             o4[2] == p3bf::eval_h(q0, eq) && o4[3] == p3bf::eval_h(q1, eq));
            }
            snprintf(nm, sizeof nm, "L=%u cl=%d leaf claims == true MLE evals", L, cl);
            ck(nm, claims_ok);

            // tampers
            auto rejects = [&](p3gkr::Proof t) {
                fs::Transcript tv2("gkr-t");
                vector<gl_t> rf2; gl_t o2[4]; const char* w2 = nullptr;
                return !p3gkr::verify(tv2, "g", L, t, cl, rf2, o2, &w2);
            };
            { auto t = pf; t.P = gl_add(t.P, 1ULL);
              snprintf(nm, sizeof nm, "L=%u cl=%d tampered root P rejects", L, cl); ck(nm, rejects(t)); }
            { auto t = pf; t.Q = gl_add(t.Q, 1ULL);
              snprintf(nm, sizeof nm, "L=%u cl=%d tampered root Q rejects", L, cl); ck(nm, rejects(t)); }
            { auto t = pf; t.lay.back().q1 = gl_add(t.lay.back().q1, 1ULL);
              fs::Transcript tv2("gkr-t");
              vector<gl_t> rf2; gl_t o2[4]; const char* w2 = nullptr;
              bool acc = p3gkr::verify(tv2, "g", L, t, cl, rf2, o2, &w2);
              // a tampered LAST terminal shifts the returned claims (caught by the
              // caller's binding); intermediate consistency itself must still hold
              // only if the tamper is on the last layer -- either reject or a
              // different q-claim comes back
              bool caught = !acc || (cl ? o2[2] != o4[2] : o2[3] != o4[3]);
              snprintf(nm, sizeof nm, "L=%u cl=%d tampered terminal claim caught", L, cl); ck(nm, caught); }
            if (L >= 2) {
                auto t = pf; t.lay[1].msgs[0].s1 = gl_add(t.lay[1].msgs[0].s1, 1ULL);
                snprintf(nm, sizeof nm, "L=%u cl=%d tampered layer message rejects", L, cl); ck(nm, rejects(t));
            }
            { auto t = pf; t.lay.pop_back();
              snprintf(nm, sizeof nm, "L=%u cl=%d truncated proof rejects", L, cl); ck(nm, rejects(t)); }
            // wrong-leaf forgery: honest procedure on a leaf set differing in one
            // entry -> the returned leaf claims differ from the TRUE leaves' MLE
            {
                vector<gl_t> p2 = p; p2[N/3] = gl_add(p2[N/3], 5ULL);
                fs::Transcript tf("gkr-t");
                vector<gl_t> rff; gl_t fcp = 0, fcq = 0;
                p3gkr::Proof f = p3gkr::prove(tf, "g", p2, q, cl, rff, &fcp, &fcq);
                fs::Transcript tv2("gkr-t");
                vector<gl_t> rf2; gl_t o2[4]; const char* w2 = nullptr;
                bool acc = p3gkr::verify(tv2, "g", L, f, cl, rf2, o2, &w2);
                bool differs;
                if (cl) differs = (o2[0] != p3bf::eval_h(p, p3bf::build_eq(rf2)));
                else {
                    vector<gl_t> pt0(N/2), pt1(N/2);
                    for (size_t i = 0; i < N/2; i++) { pt0[i]=p[2*i]; pt1[i]=p[2*i+1]; }
                    vector<gl_t> eq = p3bf::build_eq(rf2);
                    differs = (o2[0] != p3bf::eval_h(pt0, eq)) || (o2[1] != p3bf::eval_h(pt1, eq));
                }
                snprintf(nm, sizeof nm, "L=%u cl=%d wrong-leaf forgery shifts the leaf claim (binding catches)", L, cl);
                ck(nm, acc && differs);
            }
        }
    }
    printf("\nGKR-SELFTEST: %d passed, %d failed -> %s\n", np_, nf_, nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
