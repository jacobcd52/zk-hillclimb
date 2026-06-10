// COORDINATOR-BUILT (do not let submission agents edit).
// Toy-scale ground truth for witness-free verification of zkLLM's logUp-style
// lookup argument (tLookup phase1 + phase2), used by zkrelu / zksoftmax /
// rescaling for range checks.
//
// KEY UPSTREAM FACT: tLookup::prove's `proof` parameter is NEVER written.
// Upstream serializes nothing -- every round poly is recomputed from the
// witness and checked internally. So the prover driver here replicates the
// phase1/phase2 recursion (calling the REAL upstream step-poly functions and
// reduce kernels) and serializes each round poly as FOUR EVALUATIONS
// p(0..3) (round polys are degree 3: deg-2 product term x deg-1 eq factor).
//
// Statement proven (logUp): for witness S (size D), table T (size N, PUBLIC),
// multiplicities m, auxiliary A = 1/(S+beta), B = 1/(T+beta):
//   claim_0 = alpha + alpha^2  anchors
//     alpha   * sum_x eq(u,x) A(x)(S(x)+beta)            (A is correct inverse)
//   + alpha^2 * (N/D) * sum-style term for B(T+beta)     (B is correct inverse)
//   + [ sum_x A(x) - (N/D) sum_j m(j) B(j) ]             (logUp identity, via C)
// with C = alpha^2 - sum(B*m) folded in as a per-round constant C/2^r.
//
// Witness-free verifier, given {alpha, beta, u, v (challenges), round evals,
// terminal claims A_f, S_f, m_f}:
//   cur = alpha + alpha^2                       (RECOMPUTED, not taken from prover)
//   phase1 rounds r=0..|v1|-1: v_r = v1[n1-1-r]; phase2: v_r = v2[n2-1-r]
//     check cur == p(0)+p(1); cur' = Lagrange4(p(0..3))(v_r)
//     Lagrange4: L0=(v-1)(v-2)(v-3)*(-INV6), L1=v(v-2)(v-3)*INV2,
//                L2=v(v-1)(v-3)*(-INV2),     L3=v(v-1)(v-2)*INV6
//   alpha_acc   = alpha   * prod_{ALL rounds}    eq(u[logD-1-k], v_k)
//   alphasq_acc = alpha^2 * prod_{PHASE2 rounds} eq(u[logD-1-k], v_k)
//   B_f, T_f: RECOMPUTED by the verifier from the public table + beta, folded
//     with the upstream front/back-half reduce over v2 (end-first).
//   terminal: cur == alpha_acc*A_f*(S_f+beta)
//                  + inv_ratio*alphasq_acc*B_f*(T_f+beta)
//                  + A_f - inv_ratio*m_f*B_f          (inv_ratio = N/D)
//   A_f, m_f are commitment-opening obligations (vrf_toy_open.cu); S_f chains
//   to the layer data. eq(u,v) = 2uv - (u+v) + 1, all integer-view arithmetic.
#include "tlookup.cuh"
#include "polynomial.cuh"
#include <random>
#include <array>
#include <iostream>
using namespace std;

// ---- upstream internals (defined in tlookup.cu, external linkage) ----
Polynomial tLookup_phase1_step_poly(const FrTensor& A, const FrTensor& S,
    const Fr_t& alpha, const Fr_t& beta, const Fr_t& C, const vector<Fr_t>& u);
Polynomial tLookup_phase2_step_poly(const FrTensor& A, const FrTensor& S, const FrTensor& B,
    const FrTensor& T, const FrTensor& m, const Fr_t& alpha_, const Fr_t& beta,
    const Fr_t& inv_size_ratio, const Fr_t& alpha_sq, const vector<Fr_t>& u);
Fr_t tLookup_phase1(const Fr_t& claim, const FrTensor& A, const FrTensor& S, const FrTensor& B,
    const FrTensor& T, const FrTensor& m, const Fr_t& alpha, const Fr_t& beta, const Fr_t& C,
    const Fr_t& inv_size_ratio, const Fr_t& alpha_sq,
    const vector<Fr_t>& u, const vector<Fr_t>& v1, const vector<Fr_t>& v2);
KERNEL void tlookup_inv_kernel(Fr_t* in_data, Fr_t beta, Fr_t* out_data, uint N);
KERNEL void tLookup_phase1_reduce_kernel(const Fr_t* A_data, const Fr_t* S_data,
    Fr_t* new_A_data, Fr_t* new_S_data, Fr_t v, uint N_out);
KERNEL void tLookup_phase2_reduce_kernel(const Fr_t* A_data, const Fr_t* S_data,
    const Fr_t* B_data, const Fr_t* T_data, const Fr_t* m_data,
    Fr_t* new_A_data, Fr_t* new_S_data, Fr_t* new_B_data, Fr_t* new_T_data, Fr_t* new_m_data,
    Fr_t v, uint N_out);

// ---- verifier-side helpers (same integer-view conventions as the kernels) ----
KERNEL void k_scalar_op(Fr_t a, Fr_t b, int op, GLOBAL Fr_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    if (op == 0) *out = blstrs__scalar__Scalar_add(a, b);
    if (op == 1) *out = blstrs__scalar__Scalar_sub(a, b);
    if (op == 2) *out = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a, b));
}
// front/back-half scalar fold, matches the upstream reduce kernels:
// new[i] = a[i] + mont(v)*(a[i+N_out] - a[i])
KERNEL void k_fr_fold(GLOBAL Fr_t* a, Fr_t v, GLOBAL Fr_t* out, uint N_out) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;
    out[gid] = blstrs__scalar__Scalar_add(a[gid],
        blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(v),
            blstrs__scalar__Scalar_sub(a[gid + N_out], a[gid])));
}

static Fr_t h_scalar(Fr_t a, Fr_t b, int op) {
    Fr_t *d, h; cudaMalloc(&d, sizeof(Fr_t));
    k_scalar_op<<<1,1>>>(a, b, op, d);
    cudaMemcpy(&h, d, sizeof(Fr_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
static bool fr_eq(const Fr_t& a, const Fr_t& b) {
    for (int i = 0; i < 8; i++) if (a.val[i] != b.val[i]) return false;
    return true;
}

static const Fr_t F_ZERO  = {0, 0, 0, 0, 0, 0, 0, 0};
static const Fr_t F_ONE   = {1, 0, 0, 0, 0, 0, 0, 0};
static const Fr_t F_TWO   = {2, 0, 0, 0, 0, 0, 0, 0};
static const Fr_t F_THREE = {3, 0, 0, 0, 0, 0, 0, 0};
static const Fr_t F_SIX   = {6, 0, 0, 0, 0, 0, 0, 0};
static const Fr_t F_INV2  = {2147483649u, 2147483647u, 2147429887u, 2849952257u,
                             80800770u, 429714436u, 2496577188u, 972477353u};

// eq(u,v) = 2uv - (u+v) + 1 (matches upstream eqEvalKernel), integer view
static Fr_t my_eq(const Fr_t& u, const Fr_t& v) {
    Fr_t uv = h_scalar(u, v, 2);
    Fr_t e = h_scalar(h_scalar(uv, uv, 0), h_scalar(u, v, 0), 1);
    return h_scalar(e, F_ONE, 0);
}

// Lagrange-4 from {p(0),p(1),p(2),p(3)} at v; exact for degree <= 3.
static Fr_t lagrange4(const array<Fr_t,4>& p, const Fr_t& v, const Fr_t& inv6) {
    Fr_t vm1 = h_scalar(v, F_ONE, 1), vm2 = h_scalar(v, F_TWO, 1), vm3 = h_scalar(v, F_THREE, 1);
    Fr_t l0 = h_scalar(F_ZERO, h_scalar(h_scalar(h_scalar(vm1, vm2, 2), vm3, 2), inv6, 2), 1); // -(v-1)(v-2)(v-3)/6
    Fr_t l1 = h_scalar(h_scalar(h_scalar(v, vm2, 2), vm3, 2), F_INV2, 2);                      //  v(v-2)(v-3)/2
    Fr_t l2 = h_scalar(F_ZERO, h_scalar(h_scalar(h_scalar(v, vm1, 2), vm3, 2), F_INV2, 2), 1); // -v(v-1)(v-3)/2
    Fr_t l3 = h_scalar(h_scalar(h_scalar(v, vm1, 2), vm2, 2), inv6, 2);                        //  v(v-1)(v-2)/6
    Fr_t acc = h_scalar(p[0], l0, 2);
    acc = h_scalar(acc, h_scalar(p[1], l1, 2), 0);
    acc = h_scalar(acc, h_scalar(p[2], l2, 2), 0);
    acc = h_scalar(acc, h_scalar(p[3], l3, 2), 0);
    return acc;
}

struct LookupTranscript {
    vector<array<Fr_t,4>> evs;   // round polys as evaluations at 0,1,2,3
    Fr_t A_f, S_f, B_f, T_f, m_f; // terminal folded values (witness side)
    Fr_t final_claim;
    bool prover_ok;              // upstream-style internal round checks
    int max_degree;
};

// Replicates upstream tLookup_phase1/phase2 recursion, serializing round polys.
// Uses the REAL upstream step-poly functions and reduce kernels.
static Fr_t my_phase2(Fr_t claim, const FrTensor& A, const FrTensor& S, const FrTensor& B,
    const FrTensor& T, const FrTensor& m, Fr_t alpha_, const Fr_t& beta, const Fr_t& inv_ratio,
    Fr_t alpha_sq, vector<Fr_t> u, vector<Fr_t> v2, LookupTranscript& tr)
{
    if (!v2.size()) {
        tr.A_f = A(0); tr.S_f = S(0); tr.B_f = B(0); tr.T_f = T(0); tr.m_f = m(0);
        return claim;
    }
    auto p = tLookup_phase2_step_poly(A, S, B, T, m, alpha_, beta, inv_ratio, alpha_sq, u);
    tr.max_degree = max(tr.max_degree, p.getDegree());
    tr.evs.push_back({p(F_ZERO), p(F_ONE), p(F_TWO), p(F_THREE)});
    if (!fr_eq(claim, h_scalar(p(F_ZERO), p(F_ONE), 0))) tr.prover_ok = false;
    uint N_out = m.size >> 1;
    FrTensor nA(N_out), nS(N_out), nB(N_out), nT(N_out), nm(N_out);
    tLookup_phase2_reduce_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, S.gpu_data, B.gpu_data, T.gpu_data, m.gpu_data,
        nA.gpu_data, nS.gpu_data, nB.gpu_data, nT.gpu_data, nm.gpu_data, v2.back(), N_out);
    cudaDeviceSynchronize();
    Fr_t eqv = Polynomial::eq(u.back(), v2.back());
    Fr_t next = p(v2.back());
    u.pop_back(); v2.pop_back();
    return my_phase2(next, nA, nS, nB, nT, nm, alpha_ * eqv, beta, inv_ratio,
                     alpha_sq * eqv, u, v2, tr);
}

static Fr_t my_phase1(Fr_t claim, const FrTensor& A, const FrTensor& S, const FrTensor& B,
    const FrTensor& T, const FrTensor& m, Fr_t alpha_, const Fr_t& beta, Fr_t C,
    const Fr_t& inv_ratio, const Fr_t& alpha_sq, vector<Fr_t> u, vector<Fr_t> v1,
    const vector<Fr_t>& v2, LookupTranscript& tr)
{
    if (!v1.size()) return my_phase2(claim, A, S, B, T, m, alpha_, beta, inv_ratio, alpha_sq, u, v2, tr);
    auto p = tLookup_phase1_step_poly(A, S, alpha_, beta, C, u);
    tr.max_degree = max(tr.max_degree, p.getDegree());
    tr.evs.push_back({p(F_ZERO), p(F_ONE), p(F_TWO), p(F_THREE)});
    if (!fr_eq(claim, h_scalar(p(F_ZERO), p(F_ONE), 0))) tr.prover_ok = false;
    uint N_out = A.size >> 1;
    FrTensor nA(N_out), nS(N_out);
    tLookup_phase1_reduce_kernel<<<(A.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, S.gpu_data, nA.gpu_data, nS.gpu_data, v1.back(), N_out);
    cudaDeviceSynchronize();
    Fr_t eqv = Polynomial::eq(u.back(), v1.back());
    Fr_t next = p(v1.back());
    u.pop_back(); v1.pop_back();
    return my_phase1(next, nA, nS, B, T, m, alpha_ * eqv, beta, h_scalar(C, F_INV2, 2),
                     inv_ratio, alpha_sq, u, v1, v2, tr);
}

// Witness-free verifier. Inputs: public (alpha, beta, u, v, low, len, D, N) +
// transcript (round evals, A_f, S_f, m_f). B_f, T_f are recomputed here.
static bool verify_lookup(const Fr_t& alpha, const Fr_t& beta,
    const vector<Fr_t>& u, const vector<Fr_t>& v1, const vector<Fr_t>& v2,
    uint D, uint N, const FrTensor& public_table,
    const vector<array<Fr_t,4>>& evs, const Fr_t& A_f, const Fr_t& S_f, const Fr_t& m_f,
    const Fr_t& inv6, bool quiet = false)
{
    const uint n1 = v1.size(), n2 = v2.size(), logD = u.size();
    bool ok = true;
    if (evs.size() != n1 + n2) { if(!quiet) cout << "  bad round count" << endl; return false; }

    Fr_t cur = h_scalar(alpha, h_scalar(alpha, alpha, 2), 0);  // alpha + alpha^2, recomputed
    Fr_t alpha_acc = alpha, alphasq_acc = h_scalar(alpha, alpha, 2);
    for (uint k = 0; k < n1 + n2; k++) {
        Fr_t v_k = (k < n1) ? v1[n1 - 1 - k] : v2[n2 - 1 - (k - n1)];
        Fr_t u_k = u[logD - 1 - k];
        if (!fr_eq(cur, h_scalar(evs[k][0], evs[k][1], 0))) {
            if(!quiet) cout << "  round " << k << ": claim != p(0)+p(1)  FAIL" << endl;
            ok = false;
        }
        cur = lagrange4(evs[k], v_k, inv6);
        Fr_t eqv = my_eq(u_k, v_k);
        alpha_acc = h_scalar(alpha_acc, eqv, 2);
        if (k >= n1) alphasq_acc = h_scalar(alphasq_acc, eqv, 2);
    }
    // recompute B and T terminal evals from the PUBLIC table
    FrTensor B_pub(N);
    {   // const_cast: upstream kernel takes non-const in_data but only reads it
        tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            const_cast<Fr_t*>(public_table.gpu_data), beta, B_pub.gpu_data, N);
        cudaDeviceSynchronize();
    }
    Fr_t *dB, *dT, *dtmp;
    cudaMalloc(&dB, N * sizeof(Fr_t)); cudaMalloc(&dT, N * sizeof(Fr_t)); cudaMalloc(&dtmp, N * sizeof(Fr_t));
    cudaMemcpy(dB, B_pub.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dT, public_table.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    uint sz = N;
    for (uint k = 0; k < n2; k++) {       // fold over v2, end-first
        Fr_t v_k = v2[n2 - 1 - k];
        uint nsz = sz >> 1;
        k_fr_fold<<<(nsz + 31) / 32, 32>>>(dB, v_k, dtmp, nsz); cudaDeviceSynchronize();
        std::swap(dB, dtmp);
        k_fr_fold<<<(nsz + 31) / 32, 32>>>(dT, v_k, dtmp, nsz); cudaDeviceSynchronize();
        std::swap(dT, dtmp);
        sz = nsz;
    }
    Fr_t B_f, T_f;
    cudaMemcpy(&B_f, dB, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&T_f, dT, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(dB); cudaFree(dT); cudaFree(dtmp);

    // inv_ratio = N/D recomputed: N * inv(D)
    Fr_t inv_ratio = h_scalar({N,0,0,0,0,0,0,0}, inv({D,0,0,0,0,0,0,0}), 2);

    // terminal: alpha_acc*A_f*(S_f+b) + inv_ratio*alphasq_acc*B_f*(T_f+b) + A_f - inv_ratio*m_f*B_f
    Fr_t t1 = h_scalar(alpha_acc, h_scalar(A_f, h_scalar(S_f, beta, 0), 2), 2);
    Fr_t t2 = h_scalar(inv_ratio, h_scalar(alphasq_acc, h_scalar(B_f, h_scalar(T_f, beta, 0), 2), 2), 2);
    Fr_t t4 = h_scalar(inv_ratio, h_scalar(m_f, B_f, 2), 2);
    Fr_t rhs = h_scalar(h_scalar(h_scalar(t1, t2, 0), A_f, 0), t4, 1);
    if (!fr_eq(cur, rhs)) {
        if(!quiet) cout << "  terminal check FAIL" << endl;
        ok = false;
    }
    return ok;
}

static bool run_case(uint D, uint N, int low, uint len, const Fr_t& inv6) {
    cout << "=== case D=" << D << " N=" << N << " low=" << low << " len=" << len << " ===" << endl;
    if ((1u << ceilLog2(len)) != N) { cout << "bad case params" << endl; return false; }

    // witness: D values in [low, low+len)
    tLookupRange tl(low, len);
    mt19937 rng(12345 + D + N);
    vector<int> vals(D);
    for (auto& x : vals) x = low + (int)(rng() % len);
    FrTensor S(D, vals.data());
    FrTensor m = tl.prep(S);

    auto ab = random_vec(2);
    Fr_t alpha = ab[0], beta = ab[1];
    auto u = random_vec(ceilLog2(D)), v = random_vec(ceilLog2(D));
    vector<Fr_t> v1 = {v.begin(), v.begin() + ceilLog2(D / N)};
    vector<Fr_t> v2 = {v.begin() + ceilLog2(D / N), v.end()};

    FrTensor A(D), B(N);
    tlookup_inv_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(S.gpu_data, beta, A.gpu_data, D);
    cudaDeviceSynchronize();
    tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(tl.table.gpu_data, beta, B.gpu_data, N);
    cudaDeviceSynchronize();

    Fr_t alpha_sq = alpha * alpha;
    Fr_t C = alpha_sq - (B * m).sum();
    Fr_t claim = alpha + alpha_sq;
    Fr_t inv_ratio = Fr_t{N,0,0,0,0,0,0,0} / Fr_t{D,0,0,0,0,0,0,0};

    // ground truth: the REAL upstream recursion (throws internally if broken)
    Fr_t upstream_final = tLookup_phase1(claim, A, S, B, tl.table, m,
        alpha, beta, C, inv_ratio, alpha_sq, u, v1, v2);

    // my serializing replication
    LookupTranscript tr; tr.prover_ok = true; tr.max_degree = 0;
    Fr_t my_final = my_phase1(claim, A, S, B, tl.table, m, alpha, beta, C,
        inv_ratio, alpha_sq, u, v1, v2, tr);
    tr.final_claim = my_final;
    bool repl_ok = fr_eq(my_final, upstream_final) && tr.prover_ok;
    cout << "replication == upstream final, internal checks: " << (repl_ok ? "YES" : "NO")
         << "  rounds: " << tr.evs.size() << "  max round-poly degree: " << tr.max_degree << endl;

    // witness-free verify
    bool vok = verify_lookup(alpha, beta, u, v1, v2, D, N, tl.table,
                             tr.evs, tr.A_f, tr.S_f, tr.m_f, inv6);
    cout << "witness-free verify: " << (vok ? "PASS" : "FAIL") << endl;

    // forgeries
    auto evs_bad = tr.evs;
    evs_bad[0][1] = h_scalar(evs_bad[0][1], F_ONE, 0);
    bool f1 = !verify_lookup(alpha, beta, u, v1, v2, D, N, tl.table,
                             evs_bad, tr.A_f, tr.S_f, tr.m_f, inv6, true);
    bool f2 = !verify_lookup(alpha, beta, u, v1, v2, D, N, tl.table,
                             tr.evs, tr.A_f, tr.S_f, h_scalar(tr.m_f, F_ONE, 0), inv6, true);
    bool f3 = !verify_lookup(alpha, beta, u, v1, v2, D, N, tl.table,
                             tr.evs, h_scalar(tr.A_f, F_ONE, 0), tr.S_f, tr.m_f, inv6, true);
    cout << "forged round poly rejected: " << (f1 ? "YES" : "NO(!!)")
         << "  forged m_f rejected: " << (f2 ? "YES" : "NO(!!)")
         << "  forged A_f rejected: " << (f3 ? "YES" : "NO(!!)") << endl;

    // SEMANTIC forgery: one out-of-range value in S, honest-procedure proof.
    // (m from the clamped values; A = pointwise inverse of the bad S.)
    vector<int> vals_bad(vals);
    vals_bad[D/2] = low + (int)len;     // outside table (padding repeats low+len-1)
    FrTensor S_bad(D, vals_bad.data());
    FrTensor A_bad(D);
    tlookup_inv_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(S_bad.gpu_data, beta, A_bad.gpu_data, D);
    cudaDeviceSynchronize();
    Fr_t C_bad = alpha_sq - (B * m).sum();
    LookupTranscript trb; trb.prover_ok = true; trb.max_degree = 0;
    my_phase1(claim, A_bad, S_bad, B, tl.table, m, alpha, beta, C_bad,
              inv_ratio, alpha_sq, u, v1, v2, trb);
    bool f4 = !verify_lookup(alpha, beta, u, v1, v2, D, N, tl.table,
                             trb.evs, trb.A_f, trb.S_f, trb.m_f, inv6, true);
    cout << "out-of-range S rejected: " << (f4 ? "YES" : "NO(!!)") << endl;

    bool all = repl_ok && vok && f1 && f2 && f3 && f4;
    cout << (all ? "CASE PASS" : "CASE FAIL") << endl;
    return all;
}

int main() {
    // INV2 / INV6 self-checks (integer view)
    Fr_t inv6 = inv(F_SIX);
    if (!fr_eq(h_scalar(F_INV2, F_TWO, 2), F_ONE) || !fr_eq(h_scalar(inv6, F_SIX, 2), F_ONE)) {
        cout << "INV2/INV6 SELF-CHECK FAILED" << endl; return 2;
    }
    cout << "INV2/INV6 self-checks passed" << endl;
    bool a = run_case(32, 8, -3, 5, inv6);   // 2 phase1 + 3 phase2 rounds, clamped table pad
    bool b = run_case(16, 16, 0, 16, inv6);  // v1 empty: pure phase2
    bool c = run_case(64, 4, -2, 3, inv6);   // 4 phase1 + 2 phase2 rounds
    cout << ((a && b && c) ? "TOY-LOOKUP-VERIFY: ALL PASS" : "TOY-LOOKUP-VERIFY: FAIL") << endl;
    return (a && b && c) ? 0 : 1;
}
