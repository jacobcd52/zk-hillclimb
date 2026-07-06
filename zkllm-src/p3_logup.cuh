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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_zkc.cuh"
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

// perf counters (profiling only; no protocol effect)
struct LuStats { double ms = 0, commit_ms = 0, sc_ms = 0, cnt_ms = 0, inv_ms = 0,
                 ev_ms = 0, scg_ms = 0; long calls = 0, commits = 0; };
static LuStats g_lustats;
static inline double lu_now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---- committed column: values + Basefold root + codeword (kept for opening) ----
// zk mode: v is the AUGMENTED array [real | mask] (p3_zkc mechanism 1), vreal
// the log2 of the real prefix, sseed the salt seed of the hiding Merkle leaves.
struct Col {
    std::vector<gl_t> v;
    p3fri::Hash root;
    std::vector<gl_t> cw;
    uint32_t vreal = 0;
    uint64_t sseed = 0;
};
static inline Col commit_col(std::vector<gl_t> vals, uint32_t R, bool gpu = true) {
    Col c; c.v = std::move(vals);
    c.root = gpu ? p3bf::commit_gpu(c.v, R, c.cw) : p3bf::commit(c.v, R, c.cw);
    return c;
}
// commit without materializing the host codeword (deferred/batched openings);
// zkmask (zk mode only) supplies a LINKED mask region (seam / row-sum bindings)
static inline Col commit_col_nc(std::vector<gl_t> vals, uint32_t R,
                                const std::vector<gl_t>* zkmask = nullptr) {
    struct Tm { double t0; Tm() : t0(lu_now_ms()) {}
                ~Tm() { g_lustats.commit_ms += lu_now_ms() - t0; g_lustats.commits++; } } tm_;
    Col c;
    if (p3zkc::G.on) {
        uint32_t vr = 0; while ((1ull << vr) < vals.size()) vr++;
        c.vreal = vr;
        c.v = p3zkc::augment(vals, zkmask ? *zkmask : p3zkc::fresh_mask(vr));
        c.sseed = p3zkc::next_seed();
        c.root = p3zkc::salted_commit_root(c.v, R, c.sseed);
        return c;
    }
    c.v = std::move(vals);
    { uint32_t vr = 0; while ((1ull << vr) < c.v.size()) vr++; c.vreal = vr; }
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

// witness-column spec of one lookup: committed column or VIRTUAL array (the
// caller binds virtual evals to a base commitment itself)
struct LuColSpec { const Col* com; const std::vector<gl_t>* virt; };
static inline LuColSpec LC(const Col* c) { return {c, nullptr}; }
static inline LuColSpec LV(const std::vector<gl_t>* v) { return {nullptr, v}; }

// ---------------- deferred lookup obligations (R3b instance merging) ----------------
// Gadgets DEFER lookups into the shared context queue; the ledger owner (the
// standalone gadget or the composed layer) flushes ONCE: obligations grouped by
// (table id, log2 rows), each group proven as ONE logUp instance over the
// stacked member domain -- one cnt/hA/hT (+2x3 zk blinds) per GROUP instead of
// per lookup.  Merged layout u = i | (j<<n) | (ex<<(n+g)) (member row i, member
// j, zk ex slice): pad members j in [k,2^g) hold table row 0.  Every member's
// witness claims land at the shared point pm = (rA[0..n) || rA[n+g..)), whose
// SHAPE equals a standalone lookup's point in both modes, so per-gadget virtual
// binding code is drop-in (registered as a bind callback, run at flush).
struct XCtx;
struct VCtx;
// prover bind: absorb/ledger extra claims at pm; returned values ride the
// GroupProof (mem[j].extra) for the verifier-side bind to check.
using PBind = std::function<std::vector<gl_t>(fs::Transcript&, XCtx&,
                  const std::vector<gl_t>& pm, const std::vector<gl_t>& yv)>;
using VBind = std::function<bool(fs::Transcript&, VCtx&, const std::vector<gl_t>& pm,
                  const std::vector<gl_t>& yv, const std::vector<gl_t>& extra,
                  const char** why)>;
struct LuObl { std::vector<LuColSpec> W; const std::vector<uint32_t>* idx;
               const Table* tab; std::string label; PBind bind; };
struct VLuObl { std::vector<const p3fri::Hash*> roots; const Table* tab; uint32_t n;
                std::string label; VBind bind; };

// Shared-ledger composition context (design doc section 11.6 R3): a caller
// (e.g. the composed transformer layer) passes ONE XCtx through every gadget
// prover so all opening obligations land in a single PLedger and every
// committed column stays alive until the single end-of-layer batched-opening
// pass.  Gadgets given an XCtx skip their internal p3bo batches; the verifier
// mirror is VCtx (ledger + deferred lookup queue).
struct XCtx {
    p3bo::PLedger lg;                          // shared opening obligations
    std::deque<Col> keep;                      // lookup helper columns
    std::deque<std::vector<Col>> colvecs;      // gadget witness-column arenas
    std::vector<LuObl> luq;                    // deferred lookup obligations
    std::deque<std::vector<gl_t>> varena;      // deferred virtual-column arrays
    std::vector<Col>& vec(size_t n) { colvecs.emplace_back(n); return colvecs.back(); }
    std::vector<gl_t>& varr(std::vector<gl_t> v) { varena.push_back(std::move(v)); return varena.back(); }
};
struct VCtx {
    p3bo::VLedger vlg;
    std::vector<VLuObl> luq;
};
static inline void defer_v(XCtx& xc, std::vector<LuColSpec> W, const std::vector<uint32_t>& idx,
                           const Table& T, std::string label, PBind bind = nullptr) {
    xc.luq.push_back({std::move(W), &idx, &T, std::move(label), std::move(bind)});
}
static inline void vdefer_v(VCtx& v, std::vector<const p3fri::Hash*> roots, const Table& T,
                            uint32_t n, std::string label, VBind bind = nullptr) {
    v.luq.push_back({std::move(roots), &T, n, std::move(label), std::move(bind)});
}

struct Msg4 { gl_t s0, s1, s2, s3; };      // cubic sumcheck round message
struct LookupProof {
    uint32_t n = 0, m = 0, c = 0;          // log2 witness rows, log2 table rows, #cols
    p3fri::Hash root_cnt, root_hA, root_hT;
    gl_t S = 0;                            // claimed common rational sum (zk: blinded S')
    p3zkc::Blind blA, blT;                 // zk: Libra blinds of the two chains
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

// one MERGED lookup instance (a flush group): per-member claims at the shared
// point pm; helper commitments and both sumcheck chains amortized over members
struct LuMember { std::vector<gl_t> yW, y_virt, extra; };
struct GroupProof {
    uint32_t n = 0, m = 0, c = 0, k = 0;   // member log-rows, table log, #cols, #members
    p3fri::Hash root_cnt, root_hA, root_hT;
    gl_t S = 0;                            // (zk: blinded S')
    p3zkc::Blind blA, blT;
    std::vector<Msg4> msgsA, msgsT;
    std::vector<LuMember> mem;             // size k, member (registration) order
    gl_t y_hA_c = 0, y_hT_c = 0, y_cnt_c = 0;
};
static inline size_t sz_group(const GroupProof& pf) {
    size_t s = 16 + 3 * 32 + 8 + (pf.msgsA.size() + pf.msgsT.size()) * 32 + 24;
    if (p3zkc::G.on) s += 2 * (3 * 32 + 8 + 3 * 8);
    for (auto& m : pf.mem)
        s += (m.yW.size() + m.y_virt.size() + m.extra.size()) * 8;
    return s;
}

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
// zk: B = 3 committed blind columns and rho add the degree-matched Libra term
// rho*( B0(b) + E(b)*B1(b) + E(b)^2*B2(b) ) -- round degrees 1,2,3, spanning
// every coefficient of the cubic message (p3_zkc mechanism 2).
static inline std::vector<gl_t> sc_prove(std::vector<gl_t> H, std::vector<gl_t> V,
                                         std::vector<gl_t> E, std::vector<gl_t> D,
                                         gl_t lam, gl_t beta,
                                         fs::Transcript& tr, const char* tag,
                                         std::vector<Msg4>& msgs,
                                         std::vector<gl_t>* B = nullptr, gl_t rho = 0) {
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
            gl_t b0 = 0, db0 = 0, b1 = 0, db1 = 0, b2 = 0, db2 = 0;
            if (B) {
                b0 = B[0][2*i]; db0 = gl_sub(B[0][2*i+1], b0);
                b1 = B[1][2*i]; db1 = gl_sub(B[1][2*i+1], b1);
                b2 = B[2][2*i]; db2 = gl_sub(B[2][2*i+1], b2);
            }
            for (int t = 0; t < 4; t++) {
                gl_t val = gl_add(gl_mul(lam, h),
                                  gl_mul(e, gl_sub(gl_mul(h, gl_add(x, beta)), d)));
                if (B) val = gl_add(val, gl_mul(rho,
                                gl_add(b0, gl_mul(e, gl_add(b1, gl_mul(e, b2))))));
                s[t] = gl_add(s[t], val);
                h = gl_add(h, dh); x = gl_add(x, dx); e = gl_add(e, de); d = gl_add(d, dd);
                if (B) { b0 = gl_add(b0, db0); b1 = gl_add(b1, db1); b2 = gl_add(b2, db2); }
            }
        }
        Msg4 msg{s[0], s[1], s[2], s[3]};
        msgs.push_back(msg); tr.absorb(tag, &msg, sizeof msg);
        gl_t a = chal(tr); r.push_back(a);
        bind_lsb(H, a); bind_lsb(V, a); bind_lsb(E, a); bind_lsb(D, a);
        if (B) { bind_lsb(B[0], a); bind_lsb(B[1], a); bind_lsb(B[2], a); }
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
static inline LookupProof prove_v(fs::Transcript& tr, const std::vector<LuColSpec>& W,
                                  const std::vector<uint32_t>& idx, const Table& T,
                                  uint32_t R, uint32_t Q, const std::string& label,
                                  bool gpu = true, bool strict = true,
                                  std::vector<gl_t>* rA_out = nullptr,
                                  p3bo::PLedger* lg = nullptr, std::deque<Col>* keep = nullptr) {
    struct Tm { double t0; Tm() : t0(lu_now_ms()) {}
                ~Tm() { g_lustats.ms += lu_now_ms() - t0; g_lustats.calls++; } } tm_;
    LookupProof pf;
    const bool zk = p3zkc::G.on;
    if (zk && !lg) throw std::runtime_error("p3lu: zk requires the deferred ledger");
    const std::vector<gl_t>* wv0 = W[0].com ? &W[0].com->v : W[0].virt;
    size_t N = wv0->size(), M = T.cols[0].size();       // zk: N = AUGMENTED length
    size_t Nreal = idx.size();
    pf.n = ilog2(Nreal); pf.m = ilog2(M); pf.c = (uint32_t)W.size();
    if ((1ull << pf.n) != Nreal || (1ull << pf.m) != M) throw std::runtime_error("p3lu: pow2");
    if (zk) { if (N != ((size_t)1 << p3zkc::vfull(pf.n))) throw std::runtime_error("p3lu: zk aug size"); }
    else if (N != Nreal) throw std::runtime_error("p3lu: idx size");
    if (T.cols.size() != W.size()) throw std::runtime_error("p3lu: col count");

    tr.absorb("lu-tab", T.id.data(), 32);
    for (auto& w : W) {
        if (w.com) tr.absorb("lu-W", w.com->root.data(), 32);
        else       tr.absorb("lu-Wv", "virt", 4);   // binding lives with the caller
    }

    std::vector<gl_t> cnt(M, 0);
    for (size_t i = 0; i < Nreal; i++) {
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

    // helper inverses on the REAL region only; zk masks are free uniform (the
    // zero-check part is killed on mask rows by the eq((z||0),.) weight) except
    // hT's, whose sum is fixed so both chains share the blinded S'.
    double t_inv = lu_now_ms();
    std::vector<gl_t> Areal(A.begin(), A.begin() + Nreal);
    std::vector<gl_t> hA = inv_all_add(Areal, beta), hT = inv_all_add(Tc, beta);
    g_lustats.inv_ms += lu_now_ms() - t_inv;
    for (size_t j = 0; j < M; j++) hT[j] = gl_mul(cnt[j], hT[j]);
    gl_t SA = 0, ST = 0;
    for (auto x : hA) SA = gl_add(SA, x);
    for (auto x : hT) ST = gl_add(ST, x);
    if (strict && SA != ST) throw std::runtime_error("p3lu: witness not in table: " + label);

    Col ChA, ChT;
    gl_t Sp = SA;                                        // published sum (zk: blinded S')
    if (zk) {
        ChA = commit_col_nc(hA, R);                      // fresh uniform mask
        Sp = 0; for (auto x : ChA.v) Sp = gl_add(Sp, x); // S' = S_A + m,  m = mask sum
        gl_t m = gl_sub(Sp, SA);                         // the A-side mask sum
        // hT's mask sum must equal m (NOT absorb S_A-S_T): then the hT-aug sums
        // to S_T + m, which equals S' iff S_A == S_T -- so the sum-equality that
        // detects a witness-not-in-table tamper is PRESERVED, only its value hid.
        std::vector<gl_t> mT = p3zkc::fresh_mask(pf.m);
        gl_t ms = 0; for (auto x : mT) ms = gl_add(ms, x);
        mT.back() = gl_add(mT.back(), gl_sub(m, ms));
        ChT = commit_col_nc(hT, R, &mT);
    } else {
        ChA = lg ? commit_col_nc(hA, R) : commit_col(hA, R, gpu);
        ChT = lg ? commit_col_nc(hT, R) : commit_col(hT, R, gpu);
    }
    pf.root_hA = ChA.root; pf.root_hT = ChT.root;
    tr.absorb("lu-hA", pf.root_hA.data(), 32); tr.absorb("lu-hT", pf.root_hT.data(), 32);

    pf.S = Sp; tr.absorb("lu-S", &pf.S, sizeof pf.S);

    // zk: commit the Libra blind columns of both chains BEFORE the chains'
    // challenges.  H = sum_b Blind(b) is published AFTER z (E depends on z)
    // but BEFORE rho: a lie about H must satisfy lam*(S-S_true) = rho*(H-H*)
    // for a rho drawn afterwards -- Schwartz-Zippel kills it.
    std::vector<Col> BA(3), BT(3);
    gl_t rhoA = 0, rhoT = 0;
    if (zk) {
        for (int j = 0; j < 3; j++) {
            std::vector<gl_t> m;
            BA[j] = commit_col_nc(p3zkc::blind_col(pf.n, m), R, &m);
            pf.blA.rt[j] = BA[j].root; tr.absorb("lu-bA", BA[j].root.data(), 32);
        }
        pf.blA.nb = 3;
        for (int j = 0; j < 3; j++) {
            std::vector<gl_t> m;
            BT[j] = commit_col_nc(p3zkc::blind_col(pf.m, m), R, &m);
            pf.blT.rt[j] = BT[j].root; tr.absorb("lu-bT", BT[j].root.data(), 32);
        }
        pf.blT.nb = 3;
    }

    bool big = gpu && pf.n >= 16 && !zk;
    std::vector<gl_t> zA = chal_vec(tr, pf.n); gl_t lamA = chal(tr);
    std::vector<gl_t> zAf = p3zkc::zpt(zA);
    if (zk) {
        // full blind sum: sum_b [B0 + E*B1 + E^2*B2] with E public -- absorbed
        // before rho, after zA (a deterministic public function, not forgeable)
        std::vector<gl_t> E = p3bf::build_eq(zAf);
        gl_t HA = 0;
        for (size_t b = 0; b < N; b++)
            HA = gl_add(HA, gl_add(BA[0].v[b],
                     gl_mul(E[b], gl_add(BA[1].v[b], gl_mul(E[b], BA[2].v[b])))));
        pf.blA.H = HA;
        tr.absorb("lu-HA", &pf.blA.H, sizeof(gl_t));
        rhoA = chal(tr);
    }
    std::vector<gl_t> rA;
    double t_sc = lu_now_ms();
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
    } else if (zk) {
        std::vector<gl_t> Bcols[3] = {BA[0].v, BA[1].v, BA[2].v};
        rA = sc_prove(ChA.v, A, p3bf::build_eq(zAf), std::vector<gl_t>(N, 1ULL),
                      lamA, beta, tr, "lu-scA", pf.msgsA, Bcols, rhoA);
    } else {
        rA = sc_prove(hA, A, p3bf::build_eq(zA), std::vector<gl_t>(N, 1ULL),
                      lamA, beta, tr, "lu-scA", pf.msgsA);
    }
    g_lustats.sc_ms += lu_now_ms() - t_sc;
    std::vector<gl_t> eqrA;
    if (!big) eqrA = p3bf::build_eq(rA);
    auto evA = [&](const std::vector<gl_t>& col) {
        return big ? p3bf::eval_h_gpu(col, rA) : p3bf::eval_h(col, eqrA);
    };
    gl_t y_hA = evA(ChA.v);
    if (lg) {
        pf.y_hA_c = y_hA;
        uint64_t ss = ChA.sseed;
        keep->push_back(std::move(ChA));
        lg->add(&keep->back().v, pf.root_hA, rA, y_hA, ss);
        if (zk) for (int j = 0; j < 3; j++) {
            pf.blA.yB[j] = evA(BA[j].v);
            tr.absorb("lu-ybA", &pf.blA.yB[j], sizeof(gl_t));
            uint64_t s2 = BA[j].sseed;
            keep->push_back(std::move(BA[j]));
            lg->add(&keep->back().v, pf.blA.rt[j], rA, pf.blA.yB[j], s2);
        }
    } else {
        pf.open_hA = p3bf::prove_eval(ChA.v, rA, y_hA, R, Q, ChA.cw, label + "-hA");
    }
    for (uint32_t k = 0; k < pf.c; k++) {
        gl_t y = evA(*wp[k]);
        if (W[k].com) {
            if (lg) {
                pf.yW.push_back(y);
                tr.absorb("lu-yW", &y, sizeof(gl_t));
                lg->add(&W[k].com->v, W[k].com->root, rA, y, W[k].com->sseed);
            } else {
                pf.open_W.push_back(p3bf::prove_eval(W[k].com->v, rA, y, R, Q, W[k].com->cw,
                                                     label + "-W" + std::to_string(k)));
            }
        } else {
            pf.y_virt.push_back(y);
            tr.absorb("lu-yv", &y, sizeof(gl_t));
        }
    }

    double t_scT = lu_now_ms();
    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> zTf = p3zkc::zpt(zT);
    std::vector<gl_t> rT;
    if (zk) {
        std::vector<gl_t> E = p3bf::build_eq(zTf);
        gl_t HT = 0;
        size_t Maug = ChT.v.size();
        for (size_t b = 0; b < Maug; b++)
            HT = gl_add(HT, gl_add(BT[0].v[b],
                     gl_mul(E[b], gl_add(BT[1].v[b], gl_mul(E[b], BT[2].v[b])))));
        pf.blT.H = HT;
        tr.absorb("lu-HT", &pf.blT.H, sizeof(gl_t));
        rhoT = chal(tr);
        std::vector<gl_t> TcA = Tc; TcA.resize(Maug, 0);       // zero-extended public table
        std::vector<gl_t> Bcols[3] = {BT[0].v, BT[1].v, BT[2].v};
        rT = sc_prove(ChT.v, TcA, E, Ccnt.v, lamT, beta, tr, "lu-scT", pf.msgsT,
                      Bcols, rhoT);
    } else {
        rT = sc_prove(hT, Tc, p3bf::build_eq(zT), cnt,
                      lamT, beta, tr, "lu-scT", pf.msgsT);
    }
    g_lustats.scg_ms += lu_now_ms() - t_scT;
    std::vector<gl_t> eqrT = p3bf::build_eq(rT);
    gl_t y_hT = p3bf::eval_h(ChT.v, eqrT);
    gl_t y_cnt = p3bf::eval_h(Ccnt.v, eqrT);
    if (lg) {
        pf.y_hT_c = y_hT; pf.y_cnt_c = y_cnt;
        uint64_t ssT = ChT.sseed, ssC = Ccnt.sseed;
        keep->push_back(std::move(ChT));
        lg->add(&keep->back().v, pf.root_hT, rT, y_hT, ssT);
        keep->push_back(std::move(Ccnt));
        lg->add(&keep->back().v, pf.root_cnt, rT, y_cnt, ssC);
        if (zk) for (int j = 0; j < 3; j++) {
            pf.blT.yB[j] = p3bf::eval_h(BT[j].v, eqrT);
            tr.absorb("lu-ybT", &pf.blT.yB[j], sizeof(gl_t));
            uint64_t s2 = BT[j].sseed;
            keep->push_back(std::move(BT[j]));
            lg->add(&keep->back().v, pf.blT.rt[j], rT, pf.blT.yB[j], s2);
        }
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

    const bool zk = p3zkc::G.on;
    if (zk && !vlg) return fail("zk requires deferred ledger");
    if (zk) {
        if (pf.blA.nb != 3 || pf.blT.nb != 3) return fail("zk blind count");
        for (int j = 0; j < 3; j++) tr.absorb("lu-bA", pf.blA.rt[j].data(), 32);
        for (int j = 0; j < 3; j++) tr.absorb("lu-bT", pf.blT.rt[j].data(), 32);
    }

    std::vector<gl_t> zA = chal_vec(tr, pf.n); gl_t lamA = chal(tr);
    std::vector<gl_t> zAf = p3zkc::zpt(zA);
    gl_t rhoA = 0;
    if (zk) { gl_t H = pf.blA.H; tr.absorb("lu-HA", &H, sizeof(gl_t)); rhoA = chal(tr); }
    std::vector<gl_t> rA; gl_t claimA;
    gl_t claimA0 = gl_add(gl_mul(lamA, pf.S), gl_mul(rhoA, pf.blA.H));
    if (!sc_verify(pf.msgsA, zk ? p3zkc::vfull(pf.n) : pf.n, claimA0, tr, "lu-scA", rA, claimA))
        return fail("sumcheck claim A");
    gl_t y_hA;
    if (vlg) {
        y_hA = pf.y_hA_c;
        vlg->add(pf.root_hA, rA, y_hA);
        if (zk) for (int j = 0; j < 3; j++) {
            gl_t yb = pf.blA.yB[j];
            tr.absorb("lu-ybA", &yb, sizeof(gl_t));
            vlg->add(pf.blA.rt[j], rA, yb);
        }
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
    gl_t eqA = p3bf::eq_point(rA, zk ? zAf : zA);
    gl_t endA = gl_add(gl_mul(lamA, y_hA),
                       gl_mul(eqA, gl_sub(gl_mul(y_hA, gl_add(A_r, beta)), 1ULL)));
    if (zk) endA = gl_add(endA, gl_mul(rhoA,
                     gl_add(pf.blA.yB[0], gl_mul(eqA, gl_add(pf.blA.yB[1],
                            gl_mul(eqA, pf.blA.yB[2]))))));
    if (claimA != endA) return fail("A terminal");

    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> zTf = p3zkc::zpt(zT);
    gl_t rhoT = 0;
    if (zk) { gl_t H = pf.blT.H; tr.absorb("lu-HT", &H, sizeof(gl_t)); rhoT = chal(tr); }
    std::vector<gl_t> rT; gl_t claimT;
    gl_t claimT0 = gl_add(gl_mul(lamT, pf.S), gl_mul(rhoT, pf.blT.H));
    if (!sc_verify(pf.msgsT, zk ? p3zkc::vfull(pf.m) : pf.m, claimT0, tr, "lu-scT", rT, claimT))
        return fail("sumcheck claim T");
    gl_t y_hT, y_cnt;
    if (vlg) {
        y_hT = pf.y_hT_c; y_cnt = pf.y_cnt_c;
        vlg->add(pf.root_hT, rT, y_hT);
        vlg->add(pf.root_cnt, rT, y_cnt);
        if (zk) for (int j = 0; j < 3; j++) {
            gl_t yb = pf.blT.yB[j];
            tr.absorb("lu-ybT", &yb, sizeof(gl_t));
            vlg->add(pf.blT.rt[j], rT, yb);
        }
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
    // zk: rT spans the augmented domain; build_eq(rT)[j] for j < M already
    // carries the (1-r_ex) factors of the zero-extended public table
    std::vector<gl_t> eqT = p3bf::build_eq(rT);
    gl_t Tc_r = 0; gk = 1ULL;
    for (auto& col : T.cols) {
        gl_t v = 0;
        for (size_t j = 0; j < M; j++) v = gl_add(v, gl_mul(col[j], eqT[j]));
        Tc_r = gl_add(Tc_r, gl_mul(gk, v)); gk = gl_mul(gk, gamma);
    }
    gl_t eqTr = p3bf::eq_point(rT, zk ? zTf : zT);
    gl_t endT = gl_add(gl_mul(lamT, y_hT),
                       gl_mul(eqTr, gl_sub(gl_mul(y_hT, gl_add(Tc_r, beta)), y_cnt)));
    if (zk) endT = gl_add(endT, gl_mul(rhoT,
                     gl_add(pf.blT.yB[0], gl_mul(eqTr, gl_add(pf.blT.yB[1],
                            gl_mul(eqTr, pf.blT.yB[2]))))));
    if (claimT != endT) return fail("T terminal");

    tr.absorb("lu-ends", &y_hA, sizeof(gl_t));
    tr.absorb("lu-ends", &y_hT, sizeof(gl_t));
    tr.absorb("lu-ends", &y_cnt, sizeof(gl_t));
    if (rA_out) *rA_out = rA;
    if (y_virt_out) *y_virt_out = pf.y_virt;
    if (why) *why = "ok";
    return true;
}

// ==================== grouped (merged) lookup instances ====================
// One logUp instance for k same-(table, log-rows) lookups.  Merged index
// u = i | (j<<n) | (ex<<(n+g)), g = ceil log2 k, ex = the zk mask coordinates
// (width E = e_of(n), the MEMBER policy width -- so the shared claim point
// pm = (rA[0..n) || rA[n+g..)) has exactly a standalone lookup's shape).
// Pad members j in [k, 2^g) hold table row 0 on every row (their multiplicity
// is added to cnt[idx of row 0] -- the canonical row-0 index is 0).

static inline gl_t eq_bits(const std::vector<gl_t>& r, uint32_t off, uint32_t g, uint32_t j) {
    gl_t e = 1ULL;
    for (uint32_t t = 0; t < g; t++)
        e = gl_mul(e, (j >> t) & 1 ? r[off + t] : gl_sub(1ULL, r[off + t]));
    return e;
}

static inline GroupProof prove_group(fs::Transcript& tr, XCtx& xc,
                                     const std::vector<size_t>& mi, uint32_t n,
                                     uint32_t R, uint32_t Q, bool strict, bool gpu) {
    struct Tm { double t0; Tm() : t0(lu_now_ms()) {}
                ~Tm() { g_lustats.ms += lu_now_ms() - t0; g_lustats.calls++; } } tm_;
    const bool zk = p3zkc::G.on;
    const Table& T = *xc.luq[mi[0]].tab;
    GroupProof pf;
    const uint32_t k = (uint32_t)mi.size();
    uint32_t g = 0; while ((1u << g) < k) g++;
    const uint32_t kp = 1u << g;
    const size_t N = (size_t)1 << n, M = T.cols[0].size();
    const uint32_t m = ilog2(M), c = (uint32_t)T.cols.size();
    const uint32_t E = zk ? p3zkc::e_of(n) : 0;
    const size_t NA = (size_t)1 << (n + E);              // member array length
    const size_t NR = (size_t)1 << (n + g);              // merged real rows
    const size_t NM = (size_t)1 << (n + g + E);          // merged full rows
    pf.n = n; pf.m = m; pf.c = c; pf.k = k;
    pf.mem.resize(k);

    tr.absorb("lug-tab", T.id.data(), 32);
    uint32_t hdr[4] = {n, g, k, c};
    tr.absorb("lug-dims", hdr, sizeof hdr);
    for (size_t j : mi) {
        auto& W = xc.luq[j].W;
        if ((uint32_t)W.size() != c) throw std::runtime_error("p3lu: group col count");
        for (auto& w : W) {
            const std::vector<gl_t>* v = w.com ? &w.com->v : w.virt;
            if (v->size() != NA) throw std::runtime_error("p3lu: group member size");
            if (w.com) tr.absorb("lug-W", w.com->root.data(), 32);
            else       tr.absorb("lug-Wv", "virt", 4);
        }
    }

    // merged multiplicities (pad rows = table row 0)
    std::vector<gl_t> cnt(M, 0);
    for (size_t j : mi) {
        const auto& I = *xc.luq[j].idx;
        if (I.size() != N) throw std::runtime_error("p3lu: group idx size");
        for (size_t i = 0; i < N; i++) {
            if (I[i] >= M) throw std::runtime_error("p3lu: idx range");
            cnt[I[i]] = gl_add(cnt[I[i]], 1ULL);
        }
    }
    cnt[0] = gl_add(cnt[0], ((uint64_t)(kp - k) << n) % GL_P);
    Col Ccnt = commit_col_nc(cnt, R);
    pf.root_cnt = Ccnt.root; tr.absorb("lug-cnt", pf.root_cnt.data(), 32);

    gl_t gamma = chal(tr), beta = chal(tr);

    std::vector<gl_t> gp(c); { gl_t w = 1ULL; for (uint32_t t = 0; t < c; t++) { gp[t] = w; w = gl_mul(w, gamma); } }
    gl_t a0 = 0; for (uint32_t t = 0; t < c; t++) a0 = gl_add(a0, gl_mul(gp[t], T.cols[t][0]));

    // merged combined witness A over the full merged domain
    std::vector<gl_t> Am(NM);
    for (uint32_t ex = 0; ex < (1u << E); ex++)
        for (uint32_t j = 0; j < kp; j++) {
            gl_t* dst = Am.data() + (((size_t)ex << (n + g)) | ((size_t)j << n));
            if (j >= k) { for (size_t i = 0; i < N; i++) dst[i] = a0; continue; }
            auto& W = xc.luq[mi[j]].W;
            for (size_t i = 0; i < N; i++) dst[i] = 0;
            for (uint32_t t = 0; t < c; t++) {
                const std::vector<gl_t>& v = W[t].com ? W[t].com->v : *W[t].virt;
                const gl_t* src = v.data() + ((size_t)ex << n);
                for (size_t i = 0; i < N; i++)
                    dst[i] = gl_add(dst[i], gl_mul(gp[t], src[i]));
            }
        }

    // helper columns
    double t_inv = lu_now_ms();
    std::vector<gl_t> Areal(Am.begin(), Am.begin() + NR);
    std::vector<gl_t> hA = inv_all_add(Areal, beta);
    std::vector<const std::vector<gl_t>*> tp;
    for (auto& t : T.cols) tp.push_back(&t);
    std::vector<gl_t> Tc = combine(tp, gamma);
    std::vector<gl_t> hT = inv_all_add(Tc, beta);
    for (size_t j = 0; j < M; j++) hT[j] = gl_mul(cnt[j], hT[j]);
    g_lustats.inv_ms += lu_now_ms() - t_inv;
    gl_t SA = 0, ST = 0;
    for (auto x : hA) SA = gl_add(SA, x);
    for (auto x : hT) ST = gl_add(ST, x);
    if (strict && SA != ST)
        throw std::runtime_error("p3lu: witness not in table: " + xc.luq[mi[0]].label);

    Col ChA, ChT;
    gl_t Sp = SA;
    if (zk) {
        // merged hA: real prefix NR computed, mask tail (E member-policy slices)
        // fresh uniform -- committed manually (the canonical e_of(n+g) policy
        // would give a different width than the member ex coordinates)
        hA.resize(NM, 0);
        if (p3zkc::G.mask_on) {
            uint64_t s = p3zkc::next_seed();
            for (size_t i = NR; i < NM; i++) hA[i] = p3zkc::zprng(s);
        }
        ChA.v = std::move(hA); ChA.vreal = n + g;
        ChA.sseed = p3zkc::next_seed();
        ChA.root = p3zkc::salted_commit_root(ChA.v, R, ChA.sseed);
        Sp = 0; for (auto x : ChA.v) Sp = gl_add(Sp, x);
        gl_t mval = gl_sub(Sp, SA);
        std::vector<gl_t> mT = p3zkc::fresh_mask(pf.m);
        gl_t ms = 0; for (auto x : mT) ms = gl_add(ms, x);
        mT.back() = gl_add(mT.back(), gl_sub(mval, ms));
        ChT = commit_col_nc(hT, R, &mT);
    } else {
        ChA = commit_col_nc(hA, R);
        ChT = commit_col_nc(hT, R);
    }
    pf.root_hA = ChA.root; pf.root_hT = ChT.root;
    tr.absorb("lug-hA", pf.root_hA.data(), 32); tr.absorb("lug-hT", pf.root_hT.data(), 32);
    pf.S = Sp; tr.absorb("lug-S", &pf.S, sizeof pf.S);

    // zk Libra blinds: A side over the merged full domain (manual length),
    // T side over the canonical augmented table domain
    std::vector<Col> BA(3), BT(3);
    gl_t rhoA = 0, rhoT = 0;
    if (zk) {
        for (int j = 0; j < 3; j++) {
            std::vector<gl_t> b(NM, 0);
            if (p3zkc::G.blind_on) { uint64_t s = p3zkc::next_seed(); for (auto& x : b) x = p3zkc::zprng(s); }
            BA[j].v = std::move(b); BA[j].vreal = n + g;
            BA[j].sseed = p3zkc::next_seed();
            BA[j].root = p3zkc::salted_commit_root(BA[j].v, R, BA[j].sseed);
            pf.blA.rt[j] = BA[j].root; tr.absorb("lug-bA", BA[j].root.data(), 32);
        }
        pf.blA.nb = 3;
        for (int j = 0; j < 3; j++) {
            std::vector<gl_t> mk;
            BT[j] = commit_col_nc(p3zkc::blind_col(pf.m, mk), R, &mk);
            pf.blT.rt[j] = BT[j].root; tr.absorb("lug-bT", BT[j].root.data(), 32);
        }
        pf.blT.nb = 3;
    }

    // ---- A-side sumcheck over the merged domain ----
    std::vector<gl_t> zA = chal_vec(tr, n + g); gl_t lamA = chal(tr);
    std::vector<gl_t> zAf = zA; zAf.resize(n + g + E, 0);
    if (zk) {
        std::vector<gl_t> Ez = p3bf::build_eq(zAf);
        gl_t HA = 0;
        for (size_t b = 0; b < NM; b++)
            HA = gl_add(HA, gl_add(BA[0].v[b],
                     gl_mul(Ez[b], gl_add(BA[1].v[b], gl_mul(Ez[b], BA[2].v[b])))));
        pf.blA.H = HA;
        tr.absorb("lug-HA", &pf.blA.H, sizeof(gl_t));
        rhoA = chal(tr);
    }
    double t_sc = lu_now_ms();
    bool big = gpu && (n + g) >= 16 && !zk;
    std::vector<gl_t> rA;
    if (big) {
        gl_t *dH, *dV, *dE, *dD, *dz;
        cudaMallocAsync(&dH, (size_t)NM * 8, 0);
        cudaMemcpy(dH, ChA.v.data(), (size_t)NM * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dV, (size_t)NM * 8, 0);
        cudaMemcpy(dV, Am.data(), (size_t)NM * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dz, (size_t)(n + g) * 8, 0);
        cudaMemcpy(dz, zA.data(), (size_t)(n + g) * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dE, (size_t)NM * 8, 0);
        p3bf::p3bf_eq_kernel<<<((uint32_t)NM + 255) / 256, 256>>>(dz, dE, n + g, (uint32_t)NM);
        cudaFreeAsync(dz, 0);
        cudaMallocAsync(&dD, (size_t)NM * 8, 0);
        p3sg::p3sg_fill_kernel<<<((uint32_t)NM + 255) / 256, 256>>>(dD, 1ULL, (uint32_t)NM);
        std::vector<gl_t*> dc = {dH, dV, dE, dD};
        gl_t par2[2] = {lamA, beta};
        rA = p3sg::sc_prove_gpu<FLuGpu, Msg4, 4, 4>(tr, "lug-scA", dc, (uint32_t)NM,
                                                    par2, 2, pf.msgsA);
        for (auto p : dc) cudaFreeAsync(p, 0);
    } else if (zk) {
        std::vector<gl_t> Bcols[3] = {BA[0].v, BA[1].v, BA[2].v};
        rA = sc_prove(ChA.v, Am, p3bf::build_eq(zAf), std::vector<gl_t>(NM, 1ULL),
                      lamA, beta, tr, "lug-scA", pf.msgsA, Bcols, rhoA);
    } else {
        rA = sc_prove(ChA.v, Am, p3bf::build_eq(zA), std::vector<gl_t>(NM, 1ULL),
                      lamA, beta, tr, "lug-scA", pf.msgsA);
    }
    g_lustats.sc_ms += lu_now_ms() - t_sc;

    // shared member point pm = (rA[0..n) || rA[n+g..))
    std::vector<gl_t> pm(rA.begin(), rA.begin() + n);
    pm.insert(pm.end(), rA.begin() + n + g, rA.end());
    std::vector<gl_t> eqrA, eqpm;
    if (!big) eqrA = p3bf::build_eq(rA);
    eqpm = p3bf::build_eq(pm);
    auto evA = [&](const std::vector<gl_t>& col) {
        return big ? p3bf::eval_h_gpu(col, rA) : p3bf::eval_h(col, eqrA);
    };
    gl_t y_hA = evA(ChA.v);
    pf.y_hA_c = y_hA;
    {
        uint64_t ss = ChA.sseed;
        xc.keep.push_back(std::move(ChA));
        xc.lg.add(&xc.keep.back().v, pf.root_hA, rA, y_hA, ss);
    }
    if (zk) for (int j = 0; j < 3; j++) {
        pf.blA.yB[j] = p3bf::eval_h(BA[j].v, eqrA);
        tr.absorb("lug-ybA", &pf.blA.yB[j], sizeof(gl_t));
        uint64_t s2 = BA[j].sseed;
        xc.keep.push_back(std::move(BA[j]));
        xc.lg.add(&xc.keep.back().v, pf.blA.rt[j], rA, pf.blA.yB[j], s2);
    }
    // per-member witness claims at pm (+ the member's bind callback)
    for (uint32_t j = 0; j < k; j++) {
        auto& W = xc.luq[mi[j]].W;
        for (uint32_t t = 0; t < c; t++) {
            const std::vector<gl_t>& v = W[t].com ? W[t].com->v : *W[t].virt;
            gl_t y = p3bf::eval_h(v, eqpm);
            if (W[t].com) {
                pf.mem[j].yW.push_back(y);
                tr.absorb("lug-yW", &y, sizeof(gl_t));
                xc.lg.add(&W[t].com->v, W[t].com->root, pm, y, W[t].com->sseed);
            } else {
                pf.mem[j].y_virt.push_back(y);
                tr.absorb("lug-yv", &y, sizeof(gl_t));
            }
        }
        if (xc.luq[mi[j]].bind)
            pf.mem[j].extra = xc.luq[mi[j]].bind(tr, xc, pm, pf.mem[j].y_virt);
    }

    // ---- T side (unchanged semantics, once per group) ----
    double t_scT = lu_now_ms();
    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> zTf = p3zkc::zpt(zT);
    std::vector<gl_t> rT;
    if (zk) {
        std::vector<gl_t> Ez = p3bf::build_eq(zTf);
        gl_t HT = 0;
        size_t Maug = ChT.v.size();
        for (size_t b = 0; b < Maug; b++)
            HT = gl_add(HT, gl_add(BT[0].v[b],
                     gl_mul(Ez[b], gl_add(BT[1].v[b], gl_mul(Ez[b], BT[2].v[b])))));
        pf.blT.H = HT;
        tr.absorb("lug-HT", &pf.blT.H, sizeof(gl_t));
        rhoT = chal(tr);
        std::vector<gl_t> TcA = Tc; TcA.resize(Maug, 0);
        std::vector<gl_t> Bcols[3] = {BT[0].v, BT[1].v, BT[2].v};
        rT = sc_prove(ChT.v, TcA, Ez, Ccnt.v, lamT, beta, tr, "lug-scT", pf.msgsT,
                      Bcols, rhoT);
    } else if (gpu && m >= 14) {
        // device-resident T-side sumcheck for the big tables (byte-identical
        // messages to the host loop, as the A-side big path)
        gl_t *dH, *dV, *dE, *dD, *dz;
        cudaMallocAsync(&dH, (size_t)M * 8, 0);
        cudaMemcpy(dH, hT.data(), (size_t)M * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dV, (size_t)M * 8, 0);
        cudaMemcpy(dV, Tc.data(), (size_t)M * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dz, (size_t)m * 8, 0);
        cudaMemcpy(dz, zT.data(), (size_t)m * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dE, (size_t)M * 8, 0);
        p3bf::p3bf_eq_kernel<<<((uint32_t)M + 255) / 256, 256>>>(dz, dE, m, (uint32_t)M);
        cudaFreeAsync(dz, 0);
        cudaMallocAsync(&dD, (size_t)M * 8, 0);
        cudaMemcpy(dD, cnt.data(), (size_t)M * 8, cudaMemcpyHostToDevice);
        std::vector<gl_t*> dc = {dH, dV, dE, dD};
        gl_t par2[2] = {lamT, beta};
        rT = p3sg::sc_prove_gpu<FLuGpu, Msg4, 4, 4>(tr, "lug-scT", dc, (uint32_t)M,
                                                    par2, 2, pf.msgsT);
        for (auto p : dc) cudaFreeAsync(p, 0);
    } else {
        rT = sc_prove(hT, Tc, p3bf::build_eq(zT), cnt,
                      lamT, beta, tr, "lug-scT", pf.msgsT);
    }
    g_lustats.scg_ms += lu_now_ms() - t_scT;
    std::vector<gl_t> eqrT = p3bf::build_eq(rT);
    gl_t y_hT = p3bf::eval_h(ChT.v, eqrT);
    gl_t y_cnt = p3bf::eval_h(Ccnt.v, eqrT);
    pf.y_hT_c = y_hT; pf.y_cnt_c = y_cnt;
    {
        uint64_t ssT = ChT.sseed, ssC = Ccnt.sseed;
        xc.keep.push_back(std::move(ChT));
        xc.lg.add(&xc.keep.back().v, pf.root_hT, rT, y_hT, ssT);
        xc.keep.push_back(std::move(Ccnt));
        xc.lg.add(&xc.keep.back().v, pf.root_cnt, rT, y_cnt, ssC);
    }
    if (zk) for (int j = 0; j < 3; j++) {
        pf.blT.yB[j] = p3bf::eval_h(BT[j].v, eqrT);
        tr.absorb("lug-ybT", &pf.blT.yB[j], sizeof(gl_t));
        uint64_t s2 = BT[j].sseed;
        xc.keep.push_back(std::move(BT[j]));
        xc.lg.add(&xc.keep.back().v, pf.blT.rt[j], rT, pf.blT.yB[j], s2);
    }
    tr.absorb("lug-ends", &y_hA, sizeof(gl_t));
    tr.absorb("lug-ends", &y_hT, sizeof(gl_t));
    tr.absorb("lug-ends", &y_cnt, sizeof(gl_t));
    return pf;
}

// group the queued obligations by (table id, log-rows), first-appearance order
template <typename OBL>
static inline void lu_groups(const std::vector<OBL>& q,
                             std::function<uint32_t(const OBL&)> n_of,
                             std::vector<std::vector<size_t>>& members,
                             std::vector<uint32_t>& gn) {
    std::vector<p3fri::Hash> gid;
    for (size_t i = 0; i < q.size(); i++) {
        uint32_t n = n_of(q[i]);
        size_t gi = 0;
        for (; gi < gid.size(); gi++) if (gid[gi] == q[i].tab->id && gn[gi] == n) break;
        if (gi == gid.size()) { gid.push_back(q[i].tab->id); gn.push_back(n); members.push_back({}); }
        members[gi].push_back(i);
    }
}

static inline std::vector<GroupProof> lu_flush(fs::Transcript& tr, XCtx& xc,
                                               uint32_t R, uint32_t Q, bool strict,
                                               bool gpu = true) {
    std::vector<std::vector<size_t>> members; std::vector<uint32_t> gn;
    lu_groups<LuObl>(xc.luq, [](const LuObl& o) { return ilog2(o.idx->size()); }, members, gn);
    std::vector<GroupProof> out;
    for (size_t gi = 0; gi < members.size(); gi++)
        out.push_back(prove_group(tr, xc, members[gi], gn[gi], R, Q, strict, gpu));
    xc.luq.clear();
    return out;
}

static inline bool verify_group(fs::Transcript& tr, VCtx& V, const std::vector<size_t>& mi,
                                uint32_t n, const GroupProof& pf,
                                uint32_t Q_pub, uint32_t R_pub, const char** why) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    const bool zk = p3zkc::G.on;
    const Table& T = *V.luq[mi[0]].tab;
    const size_t M = T.cols[0].size();
    const uint32_t k = (uint32_t)mi.size();
    uint32_t g = 0; while ((1u << g) < k) g++;
    const uint32_t kp = 1u << g;
    const uint32_t m = ilog2(M), c = (uint32_t)T.cols.size();
    const uint32_t E = zk ? p3zkc::e_of(n) : 0;
    const uint32_t Q_MIN = 20;
    if (Q_pub < Q_MIN || R_pub < 1) return fail("insecure params");
    if (pf.n != n || pf.m != m || pf.c != c || pf.k != k) return fail("group dims");
    if (pf.mem.size() != k) return fail("group member count");

    tr.absorb("lug-tab", T.id.data(), 32);
    uint32_t hdr[4] = {n, g, k, c};
    tr.absorb("lug-dims", hdr, sizeof hdr);
    for (size_t j : mi) {
        auto& roots = V.luq[j].roots;
        if ((uint32_t)roots.size() != c) return fail("group root count");
        for (auto* r : roots) {
            if (r) tr.absorb("lug-W", r->data(), 32);
            else   tr.absorb("lug-Wv", "virt", 4);
        }
    }
    tr.absorb("lug-cnt", pf.root_cnt.data(), 32);
    gl_t gamma = chal(tr), beta = chal(tr);
    tr.absorb("lug-hA", pf.root_hA.data(), 32); tr.absorb("lug-hT", pf.root_hT.data(), 32);
    tr.absorb("lug-S", &pf.S, sizeof pf.S);
    if (zk) {
        if (pf.blA.nb != 3 || pf.blT.nb != 3) return fail("zk blind count");
        for (int j = 0; j < 3; j++) tr.absorb("lug-bA", pf.blA.rt[j].data(), 32);
        for (int j = 0; j < 3; j++) tr.absorb("lug-bT", pf.blT.rt[j].data(), 32);
    }

    std::vector<gl_t> gp(c); { gl_t w = 1ULL; for (uint32_t t = 0; t < c; t++) { gp[t] = w; w = gl_mul(w, gamma); } }
    gl_t a0 = 0; for (uint32_t t = 0; t < c; t++) a0 = gl_add(a0, gl_mul(gp[t], T.cols[t][0]));

    std::vector<gl_t> zA = chal_vec(tr, n + g); gl_t lamA = chal(tr);
    std::vector<gl_t> zAf = zA; zAf.resize(n + g + E, 0);
    gl_t rhoA = 0;
    if (zk) { gl_t H = pf.blA.H; tr.absorb("lug-HA", &H, sizeof(gl_t)); rhoA = chal(tr); }
    std::vector<gl_t> rA; gl_t claimA;
    gl_t claimA0 = gl_add(gl_mul(lamA, pf.S), gl_mul(rhoA, pf.blA.H));
    if (!sc_verify(pf.msgsA, n + g + E, claimA0, tr, "lug-scA", rA, claimA))
        return fail("group sumcheck A");
    gl_t y_hA = pf.y_hA_c;
    V.vlg.add(pf.root_hA, rA, y_hA);
    if (zk) for (int j = 0; j < 3; j++) {
        gl_t yb = pf.blA.yB[j];
        tr.absorb("lug-ybA", &yb, sizeof(gl_t));
        V.vlg.add(pf.blA.rt[j], rA, yb);
    }
    std::vector<gl_t> pm(rA.begin(), rA.begin() + n);
    pm.insert(pm.end(), rA.begin() + n + g, rA.end());
    gl_t A_r = 0;
    for (uint32_t j = 0; j < k; j++) {
        auto& roots = V.luq[mi[j]].roots;
        size_t iw = 0, iv = 0;
        gl_t yj = 0;
        for (uint32_t t = 0; t < c; t++) {
            gl_t y;
            if (roots[t]) {
                if (iw >= pf.mem[j].yW.size()) return fail("group yW count");
                y = pf.mem[j].yW[iw++];
                tr.absorb("lug-yW", &y, sizeof(gl_t));
                V.vlg.add(*roots[t], pm, y);
            } else {
                if (iv >= pf.mem[j].y_virt.size()) return fail("group y_virt count");
                y = pf.mem[j].y_virt[iv++];
                tr.absorb("lug-yv", &y, sizeof(gl_t));
            }
            yj = gl_add(yj, gl_mul(gp[t], y));
        }
        if (iw != pf.mem[j].yW.size() || iv != pf.mem[j].y_virt.size())
            return fail("group claim count");
        A_r = gl_add(A_r, gl_mul(eq_bits(rA, n, g, j), yj));
        if (V.luq[mi[j]].bind &&
            !V.luq[mi[j]].bind(tr, V, pm, pf.mem[j].y_virt, pf.mem[j].extra, why))
            return false;
    }
    for (uint32_t j = k; j < kp; j++)
        A_r = gl_add(A_r, gl_mul(eq_bits(rA, n, g, j), a0));
    gl_t eqA = p3bf::eq_point(rA, zAf);
    gl_t endA = gl_add(gl_mul(lamA, y_hA),
                       gl_mul(eqA, gl_sub(gl_mul(y_hA, gl_add(A_r, beta)), 1ULL)));
    if (zk) endA = gl_add(endA, gl_mul(rhoA,
                     gl_add(pf.blA.yB[0], gl_mul(eqA, gl_add(pf.blA.yB[1],
                            gl_mul(eqA, pf.blA.yB[2]))))));
    if (claimA != endA) return fail("group A terminal");

    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> zTf = p3zkc::zpt(zT);
    gl_t rhoT = 0;
    if (zk) { gl_t H = pf.blT.H; tr.absorb("lug-HT", &H, sizeof(gl_t)); rhoT = chal(tr); }
    std::vector<gl_t> rT; gl_t claimT;
    gl_t claimT0 = gl_add(gl_mul(lamT, pf.S), gl_mul(rhoT, pf.blT.H));
    if (!sc_verify(pf.msgsT, zk ? p3zkc::vfull(pf.m) : pf.m, claimT0, tr, "lug-scT", rT, claimT))
        return fail("group sumcheck T");
    gl_t y_hT = pf.y_hT_c, y_cnt = pf.y_cnt_c;
    V.vlg.add(pf.root_hT, rT, y_hT);
    V.vlg.add(pf.root_cnt, rT, y_cnt);
    if (zk) for (int j = 0; j < 3; j++) {
        gl_t yb = pf.blT.yB[j];
        tr.absorb("lug-ybT", &yb, sizeof(gl_t));
        V.vlg.add(pf.blT.rt[j], rT, yb);
    }
    std::vector<gl_t> eqT = p3bf::build_eq(rT);
    gl_t Tc_r = 0;
    for (uint32_t t = 0; t < c; t++) {
        gl_t v = 0;
        for (size_t j = 0; j < M; j++) v = gl_add(v, gl_mul(T.cols[t][j], eqT[j]));
        Tc_r = gl_add(Tc_r, gl_mul(gp[t], v));
    }
    gl_t eqTr = p3bf::eq_point(rT, zk ? zTf : zT);
    gl_t endT = gl_add(gl_mul(lamT, y_hT),
                       gl_mul(eqTr, gl_sub(gl_mul(y_hT, gl_add(Tc_r, beta)), y_cnt)));
    if (zk) endT = gl_add(endT, gl_mul(rhoT,
                     gl_add(pf.blT.yB[0], gl_mul(eqTr, gl_add(pf.blT.yB[1],
                            gl_mul(eqTr, pf.blT.yB[2]))))));
    if (claimT != endT) return fail("group T terminal");

    tr.absorb("lug-ends", &y_hA, sizeof(gl_t));
    tr.absorb("lug-ends", &y_hT, sizeof(gl_t));
    tr.absorb("lug-ends", &y_cnt, sizeof(gl_t));
    return true;
}

static inline bool lu_verify_flush(fs::Transcript& tr, VCtx& V,
                                   const std::vector<GroupProof>& gps,
                                   uint32_t Q_pub, uint32_t R_pub, const char** why) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    std::vector<std::vector<size_t>> members; std::vector<uint32_t> gn;
    lu_groups<VLuObl>(V.luq, [](const VLuObl& o) { return o.n; }, members, gn);
    if (gps.size() != members.size()) return fail("lookup group count");
    for (size_t gi = 0; gi < members.size(); gi++)
        if (!verify_group(tr, V, members[gi], gn[gi], gps[gi], Q_pub, R_pub, why))
            return false;
    V.luq.clear();
    return true;
}

} // namespace p3lu
