// P3 logUp lookup argument over Goldilocks, committed via Basefold (hash PCS).
//
// Ports the logUp LOGIC of the curve-based zkob_lookup.cuh onto the p3 stack:
// no G1/IPA anywhere -- witness, multiplicity and helper-inverse columns are
// Basefold-committed, the rational-sum identity is enforced by two batched
// cubic sumchecks, and terminals are Basefold openings.
//
// Statement: every row i of the c-column witness (W_0[i],..,W_{c-1}[i]) is a
// row of the PUBLIC fixed table (T_0[j],..,T_{c-1}[j]).  Rows are gamma-combined
// (A_i = sum_k g^k W_k[i], Tc_j = sum_k g^k T_k[j]); logUp identity with a
// random beta:
//        sum_i 1/(A_i+beta)  ==  sum_j cnt_j/(Tc_j+beta)   ( = S )
// Prover commits cnt (multiplicities), hA_i = 1/(A_i+beta), hT_j = cnt_j/(Tc_j+beta).
// Both sides then reduce to sumchecks batched with an eq-weighted zero-check
// binding the helper columns to their defining relations:
//   A side: sum_b [ lamA*hA(b) + eq(zA,b)*( hA(b)*(A(b)+beta) - 1     ) ] = lamA*S
//   T side: sum_b [ lamT*hT(b) + eq(zT,b)*( hT(b)*(Tc(b)+beta) - cnt(b)) ] = lamT*S
// The T-side table value Tc~(rT) is recomputed by the VERIFIER from the public
// table (no table commitment needed -- "fixed tables").
//
// Padding: N and M must be powers of two; the caller pads witness rows with a
// legitimate table row (padding rows are counted in cnt -- harmless).
// Challenges are base-field (the whole p3 stack's stated GL2 upgrade applies).
// Non-ZK: masking of hA/hT/openings is the increment-2 hardening, as in p3pfc.
#pragma once
#include <cstdint>
#include <cstdio>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_batchopen.cuh"
#include "p3_scgpu.cuh"
#include "fs_transcript.hpp"

namespace p3lu {

// device functor of the batched cubic sumcheck term:
// cols = [H, V, E, D], par = [lam, beta]:  lam*H + E*(H*(V+beta) - D)
struct FLuGpu {
    static __device__ gl_t eval(const gl_t* c, const gl_t* p) {
        return gl_add(gl_mul(p[0], c[0]),
                      gl_mul(c[2], gl_sub(gl_mul(c[0], gl_add(c[1], p[1])), c[3])));
    }
};

// ---- committed column: values + Basefold root + codeword (kept for opening) ----
struct Col {
    std::vector<gl_t> v;
    p3fri::Hash root;
    std::vector<gl_t> cw;
};
static inline Col commit_col(std::vector<gl_t> vals, uint32_t R, bool gpu = true) {
    Col c; c.v = std::move(vals);
    c.root = gpu ? p3bf::commit_gpu(c.v, R, c.cw) : p3bf::commit(c.v, R, c.cw);
    return c;
}
// commit without materializing the host codeword (deferred/batched openings)
static inline Col commit_col_nc(std::vector<gl_t> vals, uint32_t R) {
    Col c; c.v = std::move(vals);
    c.root = p3bf::commit_gpu_rootonly(c.v, R);
    return c;
}

// ---- public fixed table: c columns of length M=2^m, plus a binding hash ----
struct Table {
    std::vector<std::vector<gl_t>> cols;
    p3fri::Hash id;
};
static inline Table make_table(std::vector<std::vector<gl_t>> cols) {
    Table t; t.cols = std::move(cols);
    fs::Sha256Ctx cx; fs::sha256_init(cx);
    uint64_t nc = t.cols.size(), M = t.cols[0].size();
    fs::sha256_update(cx, &nc, 8); fs::sha256_update(cx, &M, 8);
    for (auto& col : t.cols) fs::sha256_update(cx, col.data(), col.size() * sizeof(gl_t));
    fs::sha256_final(cx, t.id.data());
    return t;
}

struct Msg4 { gl_t s0, s1, s2, s3; };      // cubic sumcheck round message
struct LookupProof {
    uint32_t n = 0, m = 0, c = 0;          // log2 witness rows, log2 table rows, #cols
    p3fri::Hash root_cnt, root_hA, root_hT;
    gl_t S = 0;                            // claimed common rational sum
    std::vector<Msg4> msgsA, msgsT;
    p3bf::EvalProof open_hA;               // at rA
    std::vector<p3bf::EvalProof> open_W;   // each COMMITTED witness column at rA
    std::vector<gl_t> y_virt;              // claimed evals of VIRTUAL columns at rA
    p3bf::EvalProof open_hT, open_cnt;     // at rT
    // deferred-opening mode (prove_v with a p3bo ledger): claimed evals replace
    // the per-instance EvalProofs; the caller's batched opening backs them.
    std::vector<gl_t> yW;                  // committed witness cols at rA
    gl_t y_hA_c = 0, y_hT_c = 0, y_cnt_c = 0;
};

static inline gl_t chal(fs::Transcript& tr) {
    uint8_t b[32]; tr.challenge_bytes(b);
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v % GL_P;
}
static inline std::vector<gl_t> chal_vec(fs::Transcript& tr, uint32_t n) {
    std::vector<gl_t> v(n); for (auto& x : v) x = chal(tr); return v;
}
static inline gl_t gl_neg(gl_t x) { return gl_sub(0ULL, x); }
// Lagrange evaluation of the degree-3 poly through (0,s0)..(3,s3) at t.
static inline gl_t cubic_eval(gl_t s0, gl_t s1, gl_t s2, gl_t s3, gl_t t) {
    gl_t inv2 = gl_inv(2ULL), inv6 = gl_inv(6ULL);
    gl_t t1 = gl_sub(t, 1ULL), t2 = gl_sub(t, 2ULL), t3 = gl_sub(t, 3ULL);
    gl_t L0 = gl_neg(gl_mul(gl_mul(gl_mul(t1, t2), t3), inv6));
    gl_t L1 = gl_mul(gl_mul(gl_mul(t, t2), t3), inv2);
    gl_t L2 = gl_neg(gl_mul(gl_mul(gl_mul(t, t1), t3), inv2));
    gl_t L3 = gl_mul(gl_mul(gl_mul(t, t1), t2), inv6);
    return gl_add(gl_add(gl_mul(s0, L0), gl_mul(s1, L1)),
                  gl_add(gl_mul(s2, L2), gl_mul(s3, L3)));
}
static inline void bind_lsb(std::vector<gl_t>& f, gl_t a) {
    uint32_t h = (uint32_t)f.size() / 2; std::vector<gl_t> nf(h);
    for (uint32_t i = 0; i < h; i++)
        nf[i] = gl_add(f[2*i], gl_mul(a, gl_sub(f[2*i+1], f[2*i])));
    f = nf;
}
static inline uint32_t ilog2(size_t n) { uint32_t l = 0; while ((1ull << l) < n) l++; return l; }

// out[i] = 1/(A[i]+beta) for all i -- Montgomery batch inversion: one gl_inv
// plus 3 muls/element instead of N Fermat inversions (~64 sqmuls each).
static inline std::vector<gl_t> inv_all_add(const std::vector<gl_t>& A, gl_t beta) {
    size_t N = A.size();
    std::vector<gl_t> d(N), pre(N);
    gl_t acc = 1ULL;
    for (size_t i = 0; i < N; i++) {
        gl_t di = gl_add(A[i], beta);
        if (di == 0) throw std::runtime_error("p3lu: zero denom (resample beta)");
        d[i] = di; pre[i] = acc; acc = gl_mul(acc, di);
    }
    gl_t inv = gl_inv(acc);
    std::vector<gl_t> out(N);
    for (size_t i = N; i-- > 0;) { out[i] = gl_mul(pre[i], inv); inv = gl_mul(inv, d[i]); }
    return out;
}

// gamma-combine c columns row-wise: out[i] = sum_k g^k cols[k][i]
static inline std::vector<gl_t> combine(const std::vector<const std::vector<gl_t>*>& cols, gl_t g) {
    size_t N = cols[0]->size();
    std::vector<gl_t> out(N, 0);
    gl_t gk = 1ULL;
    for (auto* col : cols) {
        for (size_t i = 0; i < N; i++) out[i] = gl_add(out[i], gl_mul(gk, (*col)[i]));
        gk = gl_mul(gk, g);
    }
    return out;
}

// One batched cubic sumcheck:  sum_b [ lam*H(b) + E(b)*( H(b)*(V(b)+beta) - D(b) ) ]
// (A side: D = all-ones;  T side: D = cnt).  Binds LSB-first; returns challenges.
static inline std::vector<gl_t> sc_prove(std::vector<gl_t> H, std::vector<gl_t> V,
                                         std::vector<gl_t> E, std::vector<gl_t> D,
                                         gl_t lam, gl_t beta,
                                         fs::Transcript& tr, const char* tag,
                                         std::vector<Msg4>& msgs) {
    uint32_t v = ilog2(H.size());
    std::vector<gl_t> r;
    for (uint32_t rd = 0; rd < v; rd++) {
        uint32_t half = (uint32_t)H.size() / 2;
        gl_t s[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i < half; i++) {
            gl_t h = H[2*i], dh = gl_sub(H[2*i+1], H[2*i]);
            gl_t x = V[2*i], dx = gl_sub(V[2*i+1], V[2*i]);
            gl_t e = E[2*i], de = gl_sub(E[2*i+1], E[2*i]);
            gl_t d = D[2*i], dd = gl_sub(D[2*i+1], D[2*i]);
            for (int t = 0; t < 4; t++) {
                gl_t val = gl_add(gl_mul(lam, h),
                                  gl_mul(e, gl_sub(gl_mul(h, gl_add(x, beta)), d)));
                s[t] = gl_add(s[t], val);
                h = gl_add(h, dh); x = gl_add(x, dx); e = gl_add(e, de); d = gl_add(d, dd);
            }
        }
        Msg4 msg{s[0], s[1], s[2], s[3]};
        msgs.push_back(msg); tr.absorb(tag, &msg, sizeof msg);
        gl_t a = chal(tr); r.push_back(a);
        bind_lsb(H, a); bind_lsb(V, a); bind_lsb(E, a); bind_lsb(D, a);
    }
    return r;
}

// verifier side of one batched cubic sumcheck message chain
static inline bool sc_verify(const std::vector<Msg4>& msgs, uint32_t v, gl_t claim0,
                             fs::Transcript& tr, const char* tag,
                             std::vector<gl_t>& r_out, gl_t& claim_out) {
    if (msgs.size() != v) return false;
    gl_t claim = claim0;
    for (uint32_t rd = 0; rd < v; rd++) {
        const Msg4& m = msgs[rd];
        if (gl_add(m.s0, m.s1) != claim) return false;
        tr.absorb(tag, &m, sizeof m);
        gl_t a = chal(tr); r_out.push_back(a);
        claim = cubic_eval(m.s0, m.s1, m.s2, m.s3, a);
    }
    claim_out = claim;
    return true;
}

// ---------------- prover ----------------
// W: committed witness columns (all length N=2^n).  idx: prover hint, row i of
// the witness is table row idx[i] (only completeness depends on it).
// strict=false lets the selftest run the honest PROCEDURE on an inconsistent
// witness (the verifier must catch it).
// hA_tamper (selftest only): (index, delta) pairs added to hA AFTER computing the
// honest inverses -- a sum-preserving pair forgery must be caught by the
// eq-weighted zero-check, not the sum.
static inline LookupProof prove(fs::Transcript& tr, const std::vector<const Col*>& W,
                                const std::vector<uint32_t>& idx, const Table& T,
                                uint32_t R, uint32_t Q, const std::string& label,
                                bool gpu = true, bool strict = true,
                                const std::vector<std::pair<size_t, gl_t>>* hA_tamper = nullptr) {
    LookupProof pf;
    size_t N = W[0]->v.size(), M = T.cols[0].size();
    pf.n = ilog2(N); pf.m = ilog2(M); pf.c = (uint32_t)W.size();
    if ((1ull << pf.n) != N || (1ull << pf.m) != M) throw std::runtime_error("p3lu: pow2");
    if (T.cols.size() != W.size()) throw std::runtime_error("p3lu: col count");

    tr.absorb("lu-tab", T.id.data(), 32);
    for (auto* w : W) tr.absorb("lu-W", w->root.data(), 32);

    // multiplicities
    std::vector<gl_t> cnt(M, 0);
    for (size_t i = 0; i < N; i++) {
        if (idx[i] >= M) throw std::runtime_error("p3lu: idx range");
        cnt[idx[i]] = gl_add(cnt[idx[i]], 1ULL);
    }
    Col Ccnt = commit_col(cnt, R, gpu);
    pf.root_cnt = Ccnt.root; tr.absorb("lu-cnt", pf.root_cnt.data(), 32);

    gl_t gamma = chal(tr), beta = chal(tr);

    std::vector<const std::vector<gl_t>*> wp, tp;
    for (auto* w : W) wp.push_back(&w->v);
    for (auto& t : T.cols) tp.push_back(&t);
    std::vector<gl_t> A = combine(wp, gamma), Tc = combine(tp, gamma);

    std::vector<gl_t> hA(N), hT(M);
    for (size_t i = 0; i < N; i++) {
        gl_t d = gl_add(A[i], beta);
        if (d == 0) throw std::runtime_error("p3lu: zero denom (resample beta)");
        hA[i] = gl_inv(d);
    }
    for (size_t j = 0; j < M; j++) {
        gl_t d = gl_add(Tc[j], beta);
        if (d == 0) throw std::runtime_error("p3lu: zero denom (resample beta)");
        hT[j] = gl_mul(cnt[j], gl_inv(d));
    }
    if (hA_tamper) for (auto& [i, d] : *hA_tamper) hA[i] = gl_add(hA[i], d);
    Col ChA = commit_col(hA, R, gpu), ChT = commit_col(hT, R, gpu);
    pf.root_hA = ChA.root; pf.root_hT = ChT.root;
    tr.absorb("lu-hA", pf.root_hA.data(), 32); tr.absorb("lu-hT", pf.root_hT.data(), 32);

    gl_t SA = 0, ST = 0;
    for (auto x : hA) SA = gl_add(SA, x);
    for (auto x : hT) ST = gl_add(ST, x);
    if (strict && SA != ST) throw std::runtime_error("p3lu: witness not in table");
    pf.S = SA; tr.absorb("lu-S", &pf.S, sizeof pf.S);

    // A side
    std::vector<gl_t> zA = chal_vec(tr, pf.n); gl_t lamA = chal(tr);
    std::vector<gl_t> rA = sc_prove(hA, A, p3bf::build_eq(zA), std::vector<gl_t>(N, 1ULL),
                                    lamA, beta, tr, "lu-scA", pf.msgsA);
    gl_t y_hA = p3bf::eval_h(ChA.v, p3bf::build_eq(rA));
    pf.open_hA = p3bf::prove_eval(ChA.v, rA, y_hA, R, Q, ChA.cw, label + "-hA");
    for (uint32_t k = 0; k < pf.c; k++) {
        gl_t y = p3bf::eval_h(W[k]->v, p3bf::build_eq(rA));
        pf.open_W.push_back(p3bf::prove_eval(W[k]->v, rA, y, R, Q, W[k]->cw,
                                             label + "-W" + std::to_string(k)));
    }

    // T side
    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> rT = sc_prove(hT, Tc, p3bf::build_eq(zT), cnt,
                                    lamT, beta, tr, "lu-scT", pf.msgsT);
    gl_t y_hT = p3bf::eval_h(ChT.v, p3bf::build_eq(rT));
    gl_t y_cnt = p3bf::eval_h(Ccnt.v, p3bf::build_eq(rT));
    pf.open_hT = p3bf::prove_eval(ChT.v, rT, y_hT, R, Q, ChT.cw, label + "-hT");
    pf.open_cnt = p3bf::prove_eval(Ccnt.v, rT, y_cnt, R, Q, Ccnt.cw, label + "-cnt");

    // chain the opened terminals into the caller's transcript (composition binding)
    tr.absorb("lu-ends", &pf.open_hA.y, sizeof(gl_t));
    tr.absorb("lu-ends", &pf.open_hT.y, sizeof(gl_t));
    tr.absorb("lu-ends", &pf.open_cnt.y, sizeof(gl_t));
    return pf;
}

// ---------------- verifier ----------------
// W_roots: the caller-known commitments of the witness columns.  Q_pub/R_pub are
// PUBLIC FRI parameters (never read from the proof -- p3pfc red-team CRITICAL-1).
// On success, y_W_out (if non-null) receives the opened W_k(rA) values and rA_out
// the opening point, so callers can reuse the bound witness evaluations.
static inline bool verify(fs::Transcript& tr, const std::vector<p3fri::Hash>& W_roots,
                          const Table& T, const LookupProof& pf,
                          uint32_t Q_pub, uint32_t R_pub, const std::string& label,
                          const char** why = nullptr,
                          std::vector<gl_t>* y_W_out = nullptr,
                          std::vector<gl_t>* rA_out = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    size_t M = T.cols[0].size();
    if (pf.c != W_roots.size() || pf.c != T.cols.size()) return fail("col count");
    if ((1ull << pf.m) != M) return fail("table size");
    if (pf.open_W.size() != pf.c) return fail("open_W count");
    const uint32_t Q_MIN = 20;
    if (Q_pub < Q_MIN || R_pub < 1) return fail("insecure params");
    auto chkpar = [&](const p3bf::EvalProof& e, uint32_t logN_exp) {
        return e.Q == Q_pub && e.R == R_pub && e.logN == logN_exp;
    };

    tr.absorb("lu-tab", T.id.data(), 32);
    for (auto& r : W_roots) tr.absorb("lu-W", r.data(), 32);
    tr.absorb("lu-cnt", pf.root_cnt.data(), 32);
    gl_t gamma = chal(tr), beta = chal(tr);
    tr.absorb("lu-hA", pf.root_hA.data(), 32); tr.absorb("lu-hT", pf.root_hT.data(), 32);
    tr.absorb("lu-S", &pf.S, sizeof pf.S);

    // A side
    std::vector<gl_t> zA = chal_vec(tr, pf.n); gl_t lamA = chal(tr);
    std::vector<gl_t> rA; gl_t claimA;
    if (!sc_verify(pf.msgsA, pf.n, gl_mul(lamA, pf.S), tr, "lu-scA", rA, claimA))
        return fail("sumcheck claim A");
    if (pf.open_hA.roots.empty() || !(pf.open_hA.roots[0] == pf.root_hA) || pf.open_hA.z != rA)
        return fail("hA opening bind");
    if (!chkpar(pf.open_hA, pf.n)) return fail("hA opening params");
    if (!p3bf::verify_eval(pf.open_hA, label + "-hA", why)) return false;
    gl_t A_r = 0, gk = 1ULL;
    for (uint32_t k = 0; k < pf.c; k++) {
        const auto& op = pf.open_W[k];
        if (op.roots.empty() || !(op.roots[0] == W_roots[k]) || op.z != rA)
            return fail("W opening bind");
        if (!chkpar(op, pf.n)) return fail("W opening params");
        if (!p3bf::verify_eval(op, label + "-W" + std::to_string(k), why)) return false;
        A_r = gl_add(A_r, gl_mul(gk, op.y)); gk = gl_mul(gk, gamma);
    }
    gl_t eqA = p3bf::eq_point(rA, zA);
    gl_t y_hA = pf.open_hA.y;
    gl_t endA = gl_add(gl_mul(lamA, y_hA),
                       gl_mul(eqA, gl_sub(gl_mul(y_hA, gl_add(A_r, beta)), 1ULL)));
    if (claimA != endA) return fail("A terminal");

    // T side
    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> rT; gl_t claimT;
    if (!sc_verify(pf.msgsT, pf.m, gl_mul(lamT, pf.S), tr, "lu-scT", rT, claimT))
        return fail("sumcheck claim T");
    if (pf.open_hT.roots.empty() || !(pf.open_hT.roots[0] == pf.root_hT) || pf.open_hT.z != rT)
        return fail("hT opening bind");
    if (pf.open_cnt.roots.empty() || !(pf.open_cnt.roots[0] == pf.root_cnt) || pf.open_cnt.z != rT)
        return fail("cnt opening bind");
    if (!chkpar(pf.open_hT, pf.m) || !chkpar(pf.open_cnt, pf.m)) return fail("T opening params");
    if (!p3bf::verify_eval(pf.open_hT, label + "-hT", why)) return false;
    if (!p3bf::verify_eval(pf.open_cnt, label + "-cnt", why)) return false;
    // verifier evaluates the PUBLIC combined table at rT itself
    std::vector<gl_t> eqT = p3bf::build_eq(rT);
    gl_t Tc_r = 0; gk = 1ULL;
    for (auto& col : T.cols) {
        gl_t v = 0;
        for (size_t j = 0; j < M; j++) v = gl_add(v, gl_mul(col[j], eqT[j]));
        Tc_r = gl_add(Tc_r, gl_mul(gk, v)); gk = gl_mul(gk, gamma);
    }
    gl_t y_hT = pf.open_hT.y, y_cnt = pf.open_cnt.y;
    gl_t eqTr = p3bf::eq_point(rT, zT);
    gl_t endT = gl_add(gl_mul(lamT, y_hT),
                       gl_mul(eqTr, gl_sub(gl_mul(y_hT, gl_add(Tc_r, beta)), y_cnt)));
    if (claimT != endT) return fail("T terminal");

    tr.absorb("lu-ends", &pf.open_hA.y, sizeof(gl_t));
    tr.absorb("lu-ends", &pf.open_hT.y, sizeof(gl_t));
    tr.absorb("lu-ends", &pf.open_cnt.y, sizeof(gl_t));
    if (y_W_out) { y_W_out->clear(); for (auto& op : pf.open_W) y_W_out->push_back(op.y); }
    if (rA_out) *rA_out = rA;
    if (why) *why = "ok";
    return true;
}

// ---------------- virtual-column variant ----------------
// Some witness columns of a lookup can be VIRTUAL: not committed as their own
// Basefold column because they are broadcasts/bit-permutations of an already
// committed base polynomial (e.g. the per-product fp8 code a(p) = X(b,k) is
// constant in n, so a~(rA) = X~(rearranged rA)).  The prover supplies their
// value arrays; the proof carries their claimed MLE evaluations y_virt at the
// lookup point rA; the CALLER is responsible for binding each y_virt to the
// base commitment (a Basefold opening of the base at the rearranged point).
// verify_v returns rA and y_virt for exactly that purpose -- a caller that
// ignores them gets NO soundness for the virtual columns.
struct LuColSpec { const Col* com; const std::vector<gl_t>* virt; };
static inline LuColSpec LC(const Col* c) { return {c, nullptr}; }
static inline LuColSpec LV(const std::vector<gl_t>* v) { return {nullptr, v}; }

static inline LookupProof prove_v(fs::Transcript& tr, const std::vector<LuColSpec>& W,
                                  const std::vector<uint32_t>& idx, const Table& T,
                                  uint32_t R, uint32_t Q, const std::string& label,
                                  bool gpu = true, bool strict = true,
                                  std::vector<gl_t>* rA_out = nullptr,
                                  p3bo::PLedger* lg = nullptr, std::deque<Col>* keep = nullptr) {
    LookupProof pf;
    const std::vector<gl_t>* wv0 = W[0].com ? &W[0].com->v : W[0].virt;
    size_t N = wv0->size(), M = T.cols[0].size();
    pf.n = ilog2(N); pf.m = ilog2(M); pf.c = (uint32_t)W.size();
    if ((1ull << pf.n) != N || (1ull << pf.m) != M) throw std::runtime_error("p3lu: pow2");
    if (T.cols.size() != W.size()) throw std::runtime_error("p3lu: col count");

    tr.absorb("lu-tab", T.id.data(), 32);
    for (auto& w : W) {
        if (w.com) tr.absorb("lu-W", w.com->root.data(), 32);
        else       tr.absorb("lu-Wv", "virt", 4);   // binding lives with the caller
    }

    std::vector<gl_t> cnt(M, 0);
    for (size_t i = 0; i < N; i++) {
        if (idx[i] >= M) throw std::runtime_error("p3lu: idx range");
        cnt[idx[i]] = gl_add(cnt[idx[i]], 1ULL);
    }
    Col Ccnt = lg ? commit_col_nc(cnt, R) : commit_col(cnt, R, gpu);
    pf.root_cnt = Ccnt.root; tr.absorb("lu-cnt", pf.root_cnt.data(), 32);

    gl_t gamma = chal(tr), beta = chal(tr);

    std::vector<const std::vector<gl_t>*> wp, tp;
    for (auto& w : W) wp.push_back(w.com ? &w.com->v : w.virt);
    for (auto& t : T.cols) tp.push_back(&t);
    std::vector<gl_t> A = combine(wp, gamma), Tc = combine(tp, gamma);

    std::vector<gl_t> hA = inv_all_add(A, beta), hT = inv_all_add(Tc, beta);
    for (size_t j = 0; j < M; j++) hT[j] = gl_mul(cnt[j], hT[j]);
    Col ChA = lg ? commit_col_nc(hA, R) : commit_col(hA, R, gpu);
    Col ChT = lg ? commit_col_nc(hT, R) : commit_col(hT, R, gpu);
    pf.root_hA = ChA.root; pf.root_hT = ChT.root;
    tr.absorb("lu-hA", pf.root_hA.data(), 32); tr.absorb("lu-hT", pf.root_hT.data(), 32);

    gl_t SA = 0, ST = 0;
    for (auto x : hA) SA = gl_add(SA, x);
    for (auto x : hT) ST = gl_add(ST, x);
    if (strict && SA != ST) throw std::runtime_error("p3lu: witness not in table: " + label);
    pf.S = SA; tr.absorb("lu-S", &pf.S, sizeof pf.S);

    bool big = gpu && pf.n >= 16;
    std::vector<gl_t> zA = chal_vec(tr, pf.n); gl_t lamA = chal(tr);
    std::vector<gl_t> rA;
    if (big) {   // device-resident cubic sumcheck (identical messages)
        gl_t *dH, *dV, *dE, *dD, *dz;
        cudaMallocAsync(&dH, (size_t)N * 8, 0);
        cudaMemcpy(dH, hA.data(), (size_t)N * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dV, (size_t)N * 8, 0);
        cudaMemcpy(dV, A.data(), (size_t)N * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dz, (size_t)pf.n * 8, 0);
        cudaMemcpy(dz, zA.data(), (size_t)pf.n * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dE, (size_t)N * 8, 0);
        p3bf::p3bf_eq_kernel<<<((uint32_t)N + 255) / 256, 256>>>(dz, dE, pf.n, (uint32_t)N);
        cudaFreeAsync(dz, 0);
        cudaMallocAsync(&dD, (size_t)N * 8, 0);
        p3sg::p3sg_fill_kernel<<<((uint32_t)N + 255) / 256, 256>>>(dD, 1ULL, (uint32_t)N);
        std::vector<gl_t*> dc = {dH, dV, dE, dD};
        gl_t par2[2] = {lamA, beta};
        rA = p3sg::sc_prove_gpu<FLuGpu, Msg4, 4, 4>(tr, "lu-scA", dc, (uint32_t)N,
                                                    par2, 2, pf.msgsA);
        for (auto p : dc) cudaFreeAsync(p, 0);
    } else {
        rA = sc_prove(hA, A, p3bf::build_eq(zA), std::vector<gl_t>(N, 1ULL),
                      lamA, beta, tr, "lu-scA", pf.msgsA);
    }
    std::vector<gl_t> eqrA;
    if (!big) eqrA = p3bf::build_eq(rA);
    auto evA = [&](const std::vector<gl_t>& col) {
        return big ? p3bf::eval_h_gpu(col, rA) : p3bf::eval_h(col, eqrA);
    };
    gl_t y_hA = evA(ChA.v);
    if (lg) {
        pf.y_hA_c = y_hA;
        keep->push_back(std::move(ChA));
        lg->add(&keep->back().v, pf.root_hA, rA, y_hA);
    } else {
        pf.open_hA = p3bf::prove_eval(ChA.v, rA, y_hA, R, Q, ChA.cw, label + "-hA");
    }
    for (uint32_t k = 0; k < pf.c; k++) {
        gl_t y = evA(*wp[k]);
        if (W[k].com) {
            if (lg) {
                pf.yW.push_back(y);
                tr.absorb("lu-yW", &y, sizeof(gl_t));
                lg->add(&W[k].com->v, W[k].com->root, rA, y);
            } else {
                pf.open_W.push_back(p3bf::prove_eval(W[k].com->v, rA, y, R, Q, W[k].com->cw,
                                                     label + "-W" + std::to_string(k)));
            }
        } else {
            pf.y_virt.push_back(y);
            tr.absorb("lu-yv", &y, sizeof(gl_t));
        }
    }

    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> rT = sc_prove(hT, Tc, p3bf::build_eq(zT), cnt,
                                    lamT, beta, tr, "lu-scT", pf.msgsT);
    std::vector<gl_t> eqrT = p3bf::build_eq(rT);
    gl_t y_hT = p3bf::eval_h(ChT.v, eqrT);
    gl_t y_cnt = p3bf::eval_h(Ccnt.v, eqrT);
    if (lg) {
        pf.y_hT_c = y_hT; pf.y_cnt_c = y_cnt;
        keep->push_back(std::move(ChT));
        lg->add(&keep->back().v, pf.root_hT, rT, y_hT);
        keep->push_back(std::move(Ccnt));
        lg->add(&keep->back().v, pf.root_cnt, rT, y_cnt);
    } else {
        pf.open_hT = p3bf::prove_eval(ChT.v, rT, y_hT, R, Q, ChT.cw, label + "-hT");
        pf.open_cnt = p3bf::prove_eval(Ccnt.v, rT, y_cnt, R, Q, Ccnt.cw, label + "-cnt");
    }

    tr.absorb("lu-ends", &y_hA, sizeof(gl_t));
    tr.absorb("lu-ends", &y_hT, sizeof(gl_t));
    tr.absorb("lu-ends", &y_cnt, sizeof(gl_t));
    if (rA_out) *rA_out = rA;
    return pf;
}

// W_roots: entry k is the commitment of column k, or nullptr if column k is
// virtual.  On success *rA_out / *y_virt_out give the point and the claimed
// virtual-column evaluations THE CALLER MUST BIND to the base commitments.
static inline bool verify_v(fs::Transcript& tr, const std::vector<const p3fri::Hash*>& W_roots,
                            const Table& T, const LookupProof& pf,
                            uint32_t Q_pub, uint32_t R_pub, const std::string& label,
                            const char** why,
                            std::vector<gl_t>* rA_out, std::vector<gl_t>* y_virt_out,
                            std::vector<gl_t>* y_W_out = nullptr,
                            p3bo::VLedger* vlg = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    size_t M = T.cols[0].size();
    size_t n_virt = 0; for (auto* r : W_roots) if (!r) n_virt++;
    if (pf.c != W_roots.size() || pf.c != T.cols.size()) return fail("col count");
    if ((1ull << pf.m) != M) return fail("table size");
    if (vlg) { if (!pf.open_W.empty() || pf.yW.size() != pf.c - n_virt) return fail("yW count"); }
    else if (pf.open_W.size() != pf.c - n_virt) return fail("open_W count");
    if (pf.y_virt.size() != n_virt) return fail("y_virt count");
    const uint32_t Q_MIN = 20;
    if (Q_pub < Q_MIN || R_pub < 1) return fail("insecure params");
    auto chkpar = [&](const p3bf::EvalProof& e, uint32_t logN_exp) {
        return e.Q == Q_pub && e.R == R_pub && e.logN == logN_exp;
    };

    tr.absorb("lu-tab", T.id.data(), 32);
    for (auto* r : W_roots) {
        if (r) tr.absorb("lu-W", r->data(), 32);
        else   tr.absorb("lu-Wv", "virt", 4);
    }
    tr.absorb("lu-cnt", pf.root_cnt.data(), 32);
    gl_t gamma = chal(tr), beta = chal(tr);
    tr.absorb("lu-hA", pf.root_hA.data(), 32); tr.absorb("lu-hT", pf.root_hT.data(), 32);
    tr.absorb("lu-S", &pf.S, sizeof pf.S);

    std::vector<gl_t> zA = chal_vec(tr, pf.n); gl_t lamA = chal(tr);
    std::vector<gl_t> rA; gl_t claimA;
    if (!sc_verify(pf.msgsA, pf.n, gl_mul(lamA, pf.S), tr, "lu-scA", rA, claimA))
        return fail("sumcheck claim A");
    gl_t y_hA;
    if (vlg) {
        y_hA = pf.y_hA_c;
        vlg->add(pf.root_hA, rA, y_hA);
    } else {
        if (pf.open_hA.roots.empty() || !(pf.open_hA.roots[0] == pf.root_hA) || pf.open_hA.z != rA)
            return fail("hA opening bind");
        if (!chkpar(pf.open_hA, pf.n)) return fail("hA opening params");
        if (!p3bf::verify_eval(pf.open_hA, label + "-hA", why)) return false;
        y_hA = pf.open_hA.y;
    }
    gl_t A_r = 0, gk = 1ULL;
    size_t iw = 0, iv = 0;
    if (y_W_out) y_W_out->clear();
    for (uint32_t k = 0; k < pf.c; k++) {
        gl_t yk;
        if (W_roots[k]) {
            if (vlg) {
                yk = pf.yW[iw++];
                tr.absorb("lu-yW", &yk, sizeof(gl_t));
                vlg->add(*W_roots[k], rA, yk);
            } else {
                const auto& op = pf.open_W[iw++];
                if (op.roots.empty() || !(op.roots[0] == *W_roots[k]) || op.z != rA)
                    return fail("W opening bind");
                if (!chkpar(op, pf.n)) return fail("W opening params");
                if (!p3bf::verify_eval(op, label + "-W" + std::to_string(k), why)) return false;
                yk = op.y;
            }
        } else {
            yk = pf.y_virt[iv++];
            tr.absorb("lu-yv", &yk, sizeof(gl_t));
        }
        if (y_W_out) y_W_out->push_back(yk);
        A_r = gl_add(A_r, gl_mul(gk, yk)); gk = gl_mul(gk, gamma);
    }
    gl_t eqA = p3bf::eq_point(rA, zA);
    gl_t endA = gl_add(gl_mul(lamA, y_hA),
                       gl_mul(eqA, gl_sub(gl_mul(y_hA, gl_add(A_r, beta)), 1ULL)));
    if (claimA != endA) return fail("A terminal");

    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> rT; gl_t claimT;
    if (!sc_verify(pf.msgsT, pf.m, gl_mul(lamT, pf.S), tr, "lu-scT", rT, claimT))
        return fail("sumcheck claim T");
    gl_t y_hT, y_cnt;
    if (vlg) {
        y_hT = pf.y_hT_c; y_cnt = pf.y_cnt_c;
        vlg->add(pf.root_hT, rT, y_hT);
        vlg->add(pf.root_cnt, rT, y_cnt);
    } else {
        if (pf.open_hT.roots.empty() || !(pf.open_hT.roots[0] == pf.root_hT) || pf.open_hT.z != rT)
            return fail("hT opening bind");
        if (pf.open_cnt.roots.empty() || !(pf.open_cnt.roots[0] == pf.root_cnt) || pf.open_cnt.z != rT)
            return fail("cnt opening bind");
        if (!chkpar(pf.open_hT, pf.m) || !chkpar(pf.open_cnt, pf.m)) return fail("T opening params");
        if (!p3bf::verify_eval(pf.open_hT, label + "-hT", why)) return false;
        if (!p3bf::verify_eval(pf.open_cnt, label + "-cnt", why)) return false;
        y_hT = pf.open_hT.y; y_cnt = pf.open_cnt.y;
    }
    std::vector<gl_t> eqT = p3bf::build_eq(rT);
    gl_t Tc_r = 0; gk = 1ULL;
    for (auto& col : T.cols) {
        gl_t v = 0;
        for (size_t j = 0; j < M; j++) v = gl_add(v, gl_mul(col[j], eqT[j]));
        Tc_r = gl_add(Tc_r, gl_mul(gk, v)); gk = gl_mul(gk, gamma);
    }
    gl_t eqTr = p3bf::eq_point(rT, zT);
    gl_t endT = gl_add(gl_mul(lamT, y_hT),
                       gl_mul(eqTr, gl_sub(gl_mul(y_hT, gl_add(Tc_r, beta)), y_cnt)));
    if (claimT != endT) return fail("T terminal");

    tr.absorb("lu-ends", &y_hA, sizeof(gl_t));
    tr.absorb("lu-ends", &y_hT, sizeof(gl_t));
    tr.absorb("lu-ends", &y_cnt, sizeof(gl_t));
    if (rA_out) *rA_out = rA;
    if (y_virt_out) *y_virt_out = pf.y_virt;
    if (why) *why = "ok";
    return true;
}

} // namespace p3lu
