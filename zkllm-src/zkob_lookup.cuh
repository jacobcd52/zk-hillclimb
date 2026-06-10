// COORDINATOR-BUILT (do not let submission agents edit).
// Shared machinery for the obligation drivers (zkob_rescale, zkob_glu, ...):
//   - logUp lookup FS recursion (fs_phase1/fs_phase2, pinned in vrf_toy_lookup §9)
//   - generic row-committed IPA openings (open_prove/open_verify, §10)
//   - my_eq / lagrange4 / k_fr_fold / k_bump helpers
//   - LookupProof serialization, int64 tensor loader
// Extracted VERBATIM from the validated zkob_rescale.cu (selftest-protected);
// any edit here requires rerunning EVERY driver's selftest.
#ifndef ZKOB_LOOKUP_CUH
#define ZKOB_LOOKUP_CUH
#include "vrf_common.cuh"
#include "tlookup.cuh"
#include <array>
// ---- upstream internals (tlookup.cu / rescaling.cu, external linkage) ----
Polynomial tLookup_phase1_step_poly(const FrTensor& A, const FrTensor& S,
    const Fr_t& alpha, const Fr_t& beta, const Fr_t& C, const vector<Fr_t>& u);
Polynomial tLookup_phase2_step_poly(const FrTensor& A, const FrTensor& S, const FrTensor& B,
    const FrTensor& T, const FrTensor& m, const Fr_t& alpha_, const Fr_t& beta,
    const Fr_t& inv_size_ratio, const Fr_t& alpha_sq, const vector<Fr_t>& u);
KERNEL void tlookup_inv_kernel(Fr_t* in_data, Fr_t beta, Fr_t* out_data, uint N);
KERNEL void tLookup_phase1_reduce_kernel(const Fr_t* A_data, const Fr_t* S_data,
    Fr_t* new_A_data, Fr_t* new_S_data, Fr_t v, uint N_out);
KERNEL void tLookup_phase2_reduce_kernel(const Fr_t* A_data, const Fr_t* S_data,
    const Fr_t* B_data, const Fr_t* T_data, const Fr_t* m_data,
    Fr_t* new_A_data, Fr_t* new_S_data, Fr_t* new_B_data, Fr_t* new_T_data, Fr_t* new_m_data,
    Fr_t v, uint N_out);
KERNEL void rescaling_kernel(Fr_t* in_ptr, Fr_t* out_ptr, Fr_t* rem_ptr,
    long scaling_factor, uint N);

// ---- local helpers ----
static const Fr_t F_THREE = {3, 0, 0, 0, 0, 0, 0, 0};
static const Fr_t F_SIX   = {6, 0, 0, 0, 0, 0, 0, 0};

// front/back-half scalar fold (upstream reduce orientation): binds current MSB
KERNEL void k_fr_fold(GLOBAL Fr_t* a, Fr_t v, GLOBAL Fr_t* out, uint N_out) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;
    out[gid] = blstrs__scalar__Scalar_add(a[gid],
        blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(v),
            blstrs__scalar__Scalar_sub(a[gid + N_out], a[gid])));
}
// NOTE (-dlto MISCOMPILE, 2026-06-10): a batched per-row affine-check kernel
// (one straight-line G1Jacobian_mul + two adds + minus + z-limb test) returned
// "not equal" for ALL rows on honest inputs under the zkLLM build flags, even
// though the identical algebra via the 1-thread helpers (h_mul/h_add/g1_eq)
// passes. Yet another shape the LTO bug eats — the verifier below therefore
// does the affine link with the proven 1-thread helpers, row by row.
// bump one element (used only by the selftest's semantic forgery)
KERNEL void k_bump(GLOBAL Fr_t* a, uint idx, Fr_t d, int sub) {
    if (GET_GLOBAL_ID() > 0) return;
    a[idx] = sub ? blstrs__scalar__Scalar_sub(a[idx], d)
                 : blstrs__scalar__Scalar_add(a[idx], d);
}

// eq(u,v) = 2uv - (u+v) + 1, integer view (matches upstream eqEvalKernel)
static Fr_t my_eq(const Fr_t& u, const Fr_t& v) {
    Fr_t uv = h_scalar(u, v, 2);
    Fr_t e = h_scalar(h_scalar(uv, uv, 0), h_scalar(u, v, 0), 1);
    return h_scalar(e, F_ONE, 0);
}
// Lagrange-4 from {p(0..3)} at v; exact for degree <= 3 (pinned §9)
static Fr_t lagrange4(const array<Fr_t,4>& p, const Fr_t& v, const Fr_t& inv6) {
    Fr_t vm1 = h_scalar(v, F_ONE, 1), vm2 = h_scalar(v, F_TWO, 1), vm3 = h_scalar(v, F_THREE, 1);
    Fr_t l0 = h_scalar(F_ZERO, h_scalar(h_scalar(h_scalar(vm1, vm2, 2), vm3, 2), inv6, 2), 1);
    Fr_t l1 = h_scalar(h_scalar(h_scalar(v, vm2, 2), vm3, 2), F_INV2, 2);
    Fr_t l2 = h_scalar(F_ZERO, h_scalar(h_scalar(h_scalar(v, vm1, 2), vm3, 2), F_INV2, 2), 1);
    Fr_t l3 = h_scalar(h_scalar(h_scalar(v, vm1, 2), vm2, 2), inv6, 2);
    Fr_t acc = h_scalar(p[0], l0, 2);
    acc = h_scalar(acc, h_scalar(p[1], l1, 2), 0);
    acc = h_scalar(acc, h_scalar(p[2], l2, 2), 0);
    acc = h_scalar(acc, h_scalar(p[3], l3, 2), 0);
    return acc;
}

struct LookupProof {
    vector<Fr_t> ev;          // 4 per round: p(0), p(1), p(2), p(3)
    Fr_t A_f, S_f, m_f;       // serialized terminals (opening obligations)
    Fr_t B_f, T_f;            // NOT serialized; verifier recomputes from table
};
static void write_lookup(const string& path, const LookupProof& p) {
    FILE* f = open_or_die(path, "wb");
    write_pod_vec(f, p.ev);
    fwrite(&p.A_f, sizeof(Fr_t), 1, f);
    fwrite(&p.S_f, sizeof(Fr_t), 1, f);
    fwrite(&p.m_f, sizeof(Fr_t), 1, f);
    fclose(f);
}
static LookupProof read_lookup(const string& path) {
    FILE* f = open_or_die(path, "rb");
    LookupProof p;
    p.ev = read_pod_vec<Fr_t>(f);
    if (fread(&p.A_f, sizeof(Fr_t), 1, f) != 1 ||
        fread(&p.S_f, sizeof(Fr_t), 1, f) != 1 ||
        fread(&p.m_f, sizeof(Fr_t), 1, f) != 1)
        throw runtime_error("read_lookup: terminals");
    fclose(f);
    return p;
}

// int64 loader done host-side: upstream FrTensor::from_long_bin is BUGGY
// (allocates/loads sizeof(int)*size bytes for 8-byte data -> garbage tensor).
// The (uint, const long*) ctor is correct, so fread here and hand it the buffer.
static FrTensor load_long_tensor(const string& path, uint expect) {
    FILE* f = open_or_die(path, "rb");
    vector<long> buf(expect);
    if (fread(buf.data(), sizeof(long), expect, f) != expect)
        throw runtime_error("short read / wrong dims: " + path);
    fclose(f);
    return FrTensor(expect, buf.data());
}

// ---- generic IPA opening of a flat pow2 tensor committed row-wise (G gens) ----
// flat index = row * G + col, so u_pt[0..logG) are column bits, rest row bits.
static void open_prove(const FrTensor& padded, uint G, const Commitment& gen,
                       const G1Jacobian_t& Q, const vector<Fr_t>& u_pt,
                       const string& path, fs::Transcript& tr) {
    const uint logG = ceilLog2(G);
    vector<Fr_t> u_col(u_pt.begin(), u_pt.begin() + logG);
    vector<Fr_t> u_row(u_pt.begin() + logG, u_pt.end());
    if (u_row.empty()) {
        write_ipa(path, ipa_prove(padded.gpu_data, me_weights(u_col), gen.gpu_data, Q, G, tr));
    } else {
        FrTensor a = padded.partial_me(u_row, G);
        write_ipa(path, ipa_prove(a.gpu_data, me_weights(u_col), gen.gpu_data, Q, G, tr));
    }
}
static bool open_verify(const G1TensorJacobian& com, const Commitment& gen, uint G,
                        const G1Jacobian_t& Q, const vector<Fr_t>& u_pt, const Fr_t& eval,
                        const string& path, fs::Transcript& tr) {
    const uint logG = ceilLog2(G);
    if (u_pt.size() < logG) return false;
    if (com.size != (1u << (u_pt.size() - logG))) return false;
    vector<Fr_t> u_col(u_pt.begin(), u_pt.begin() + logG);
    vector<Fr_t> u_row(u_pt.begin() + logG, u_pt.end());
    G1Jacobian_t C0 = fold_chain(com.gpu_data, com.size, u_row, 0);
    G1Jacobian_t P0 = h_add(C0, h_mul(Q, eval));
    return ipa_verify(gen.gpu_data, G, Q, P0, u_col, read_ipa(path), tr);
}

// ---- FS-per-round lookup recursion (toy my_phase1/my_phase2 + transcript) ----
// strict=false lets the selftest's semantic forgery run the honest PROCEDURE
// on an inconsistent witness without throwing (the VERIFIER must catch it).
static Fr_t fs_phase2(Fr_t claim, const FrTensor& A, const FrTensor& S, const FrTensor& Bv,
    const FrTensor& T, const FrTensor& m, Fr_t alpha_, const Fr_t& beta, const Fr_t& inv_ratio,
    Fr_t alpha_sq, vector<Fr_t> u, fs::Transcript& tr, vector<Fr_t>& ws,
    LookupProof& out, bool strict)
{
    if (u.empty()) {
        out.A_f = A(0); out.S_f = S(0); out.B_f = Bv(0); out.T_f = T(0); out.m_f = m(0);
        return claim;
    }
    auto p = tLookup_phase2_step_poly(A, S, Bv, T, m, alpha_, beta, inv_ratio, alpha_sq, u);
    Fr_t e0 = p(F_ZERO), e1 = p(F_ONE), e2 = p(F_TWO), e3 = p(F_THREE);
    if (strict && !fr_eq(claim, h_scalar(e0, e1, 0)))
        throw runtime_error("phase2 round inconsistency (witness bug)");
    out.ev.push_back(e0); out.ev.push_back(e1); out.ev.push_back(e2); out.ev.push_back(e3);
    absorb_fr(tr, "p0", e0); absorb_fr(tr, "p1", e1);
    absorb_fr(tr, "p2", e2); absorb_fr(tr, "p3", e3);
    Fr_t v = fs_challenge_fr(tr); ws.push_back(v);
    uint N_out = m.size >> 1;
    FrTensor nA(N_out), nS(N_out), nB(N_out), nT(N_out), nm(N_out);
    tLookup_phase2_reduce_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, S.gpu_data, Bv.gpu_data, T.gpu_data, m.gpu_data,
        nA.gpu_data, nS.gpu_data, nB.gpu_data, nT.gpu_data, nm.gpu_data, v, N_out);
    cudaDeviceSynchronize();
    Fr_t eqv = Polynomial::eq(u.back(), v);
    Fr_t next = p(v);
    u.pop_back();
    return fs_phase2(next, nA, nS, nB, nT, nm, alpha_ * eqv, beta, inv_ratio,
                     alpha_sq * eqv, u, tr, ws, out, strict);
}
static Fr_t fs_phase1(Fr_t claim, const FrTensor& A, const FrTensor& S, const FrTensor& Bv,
    const FrTensor& T, const FrTensor& m, Fr_t alpha_, const Fr_t& beta, Fr_t Cc,
    const Fr_t& inv_ratio, const Fr_t& alpha_sq, vector<Fr_t> u, fs::Transcript& tr,
    vector<Fr_t>& ws, LookupProof& out, bool strict)
{
    if (A.size == m.size)
        return fs_phase2(claim, A, S, Bv, T, m, alpha_, beta, inv_ratio, alpha_sq,
                         u, tr, ws, out, strict);
    auto p = tLookup_phase1_step_poly(A, S, alpha_, beta, Cc, u);
    Fr_t e0 = p(F_ZERO), e1 = p(F_ONE), e2 = p(F_TWO), e3 = p(F_THREE);
    if (strict && !fr_eq(claim, h_scalar(e0, e1, 0)))
        throw runtime_error("phase1 round inconsistency (witness bug)");
    out.ev.push_back(e0); out.ev.push_back(e1); out.ev.push_back(e2); out.ev.push_back(e3);
    absorb_fr(tr, "p0", e0); absorb_fr(tr, "p1", e1);
    absorb_fr(tr, "p2", e2); absorb_fr(tr, "p3", e3);
    Fr_t v = fs_challenge_fr(tr); ws.push_back(v);
    uint N_out = A.size >> 1;
    FrTensor nA(N_out), nS(N_out);
    tLookup_phase1_reduce_kernel<<<(A.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, S.gpu_data, nA.gpu_data, nS.gpu_data, v, N_out);
    cudaDeviceSynchronize();
    Fr_t eqv = Polynomial::eq(u.back(), v);
    Fr_t next = p(v);
    u.pop_back();
    return fs_phase1(next, nA, nS, Bv, T, m, alpha_ * eqv, beta, h_scalar(Cc, F_INV2, 2),
                     inv_ratio, alpha_sq, u, tr, ws, out, strict);
}

// ---- Fr-only kernels (G1 kernels are -dlto miscompile bait; Fr are proven) ----
// eq-tensor doubling: bit i is the NEW top bit (LSB-first me_weights order)
KERNEL void k_eq_expand(GLOBAL Fr_t* in, Fr_t c, GLOBAL Fr_t* out, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    Fr_t cm = blstrs__scalar__Scalar_mont(c);
    Fr_t one_minus = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ONE, cm); // mont(1-c)
    out[gid]     = blstrs__scalar__Scalar_mul(one_minus, in[gid]);
    out[gid + n] = blstrs__scalar__Scalar_mul(cm, in[gid]);
}
// hadamard round poly: p(t) = sum_i E_t[i]*S_t[i]*U_t[i] for t = 0..3, where
// X_t[i] = X[i] + t*(X[i+h]-X[i]) (front/back orientation, binds current MSB)
KERNEL void k_hp3_step(GLOBAL Fr_t* E, GLOBAL Fr_t* S, GLOBAL Fr_t* U,
                       GLOBAL Fr_t* o0, GLOBAL Fr_t* o1, GLOBAL Fr_t* o2, GLOBAL Fr_t* o3,
                       uint h) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= h) return;
    Fr_t e = E[gid], de = blstrs__scalar__Scalar_sub(E[gid + h], E[gid]);
    Fr_t s = S[gid], ds = blstrs__scalar__Scalar_sub(S[gid + h], S[gid]);
    Fr_t u = U[gid], du = blstrs__scalar__Scalar_sub(U[gid + h], U[gid]);
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        Fr_t es = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(e),
                                             blstrs__scalar__Scalar_mont(s));
        Fr_t esu = blstrs__scalar__Scalar_mul(es, u);
        if (t == 0) o0[gid] = esu;
        else if (t == 1) o1[gid] = esu;
        else if (t == 2) o2[gid] = esu;
        else o3[gid] = esu;
        e = blstrs__scalar__Scalar_add(e, de);
        s = blstrs__scalar__Scalar_add(s, ds);
        u = blstrs__scalar__Scalar_add(u, du);
    }
}

static FrTensor load_int32_tensor(const string& path, uint expect) {
    FILE* f = open_or_die(path, "rb");
    vector<int> buf(expect);
    if (fread(buf.data(), sizeof(int), expect, f) != expect)
        throw runtime_error("short read / wrong dims: " + path);
    fclose(f);
    return FrTensor(expect, buf.data());
}

// eq tensor E[b] = prod_i (b>>i&1 ? u[i] : 1-u[i]), built on device by doubling
static FrTensor build_eq_tensor(const vector<Fr_t>& u) {
    const uint logD = u.size();
    FrTensor E(1u << logD);
    cudaMemset(E.gpu_data, 0, sizeof(Fr_t) * E.size);
    k_bump<<<1,1>>>(E.gpu_data, 0, F_ONE, 0);
    cudaDeviceSynchronize();
    Fr_t* cur; Fr_t* nxt;
    cudaMalloc(&cur, sizeof(Fr_t) << logD);
    cudaMalloc(&nxt, sizeof(Fr_t) << logD);
    cudaMemcpy(cur, E.gpu_data, sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    uint n = 1;
    for (uint i = 0; i < logD; i++) {
        k_eq_expand<<<(n + 255) / 256, 256>>>(cur, u[i], nxt, n);
        cudaDeviceSynchronize();
        std::swap(cur, nxt);
        n <<= 1;
    }
    cudaMemcpy(E.gpu_data, cur, sizeof(Fr_t) * E.size, cudaMemcpyDeviceToDevice);
    cudaFree(cur); cudaFree(nxt);
    return E;
}

struct HadamardProof {
    Fr_t claim_H;
    vector<Fr_t> ev;          // 4 per round: p(0..3)
    Fr_t S_f2, U_f2;          // terminal opening obligations
};
static void write_hp(const string& path, const HadamardProof& p) {
    FILE* f = open_or_die(path, "wb");
    fwrite(&p.claim_H, sizeof(Fr_t), 1, f);
    write_pod_vec(f, p.ev);
    fwrite(&p.S_f2, sizeof(Fr_t), 1, f);
    fwrite(&p.U_f2, sizeof(Fr_t), 1, f);
    fclose(f);
}
static HadamardProof read_hp(const string& path) {
    FILE* f = open_or_die(path, "rb");
    HadamardProof p;
    if (fread(&p.claim_H, sizeof(Fr_t), 1, f) != 1) throw runtime_error("read_hp");
    p.ev = read_pod_vec<Fr_t>(f);
    if (fread(&p.S_f2, sizeof(Fr_t), 1, f) != 1 ||
        fread(&p.U_f2, sizeof(Fr_t), 1, f) != 1) throw runtime_error("read_hp terminals");
    fclose(f);
    return p;
}

// FS-per-round hadamard sumcheck recursion (prover side)
static void fs_hadamard(Fr_t claim, FrTensor& E, FrTensor& S, FrTensor& U,
                        fs::Transcript& tr, vector<Fr_t>& ws, HadamardProof& out,
                        bool strict) {
    if (E.size == 1) { out.S_f2 = S(0); out.U_f2 = U(0); return; }
    const uint h = E.size >> 1;
    FrTensor o0(h), o1(h), o2(h), o3(h);
    k_hp3_step<<<(h + 255) / 256, 256>>>(E.gpu_data, S.gpu_data, U.gpu_data,
        o0.gpu_data, o1.gpu_data, o2.gpu_data, o3.gpu_data, h);
    cudaDeviceSynchronize();
    array<Fr_t,4> e = {o0.sum(), o1.sum(), o2.sum(), o3.sum()};
    if (strict && !fr_eq(claim, h_scalar(e[0], e[1], 0)))
        throw runtime_error("hadamard round inconsistency (witness bug)");
    for (int i = 0; i < 4; i++) out.ev.push_back(e[i]);
    absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
    absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
    Fr_t w = fs_challenge_fr(tr); ws.push_back(w);
    FrTensor nE(h), nS(h), nU(h);
    k_fr_fold<<<(h + 255) / 256, 256>>>(E.gpu_data, w, nE.gpu_data, h);
    k_fr_fold<<<(h + 255) / 256, 256>>>(S.gpu_data, w, nS.gpu_data, h);
    k_fr_fold<<<(h + 255) / 256, 256>>>(U.gpu_data, w, nU.gpu_data, h);
    cudaDeviceSynchronize();
    Fr_t next = lagrange4(e, w, inv(F_SIX));
    fs_hadamard(next, nE, nS, nU, tr, ws, out, strict);
}



// ---- degree-4 sumcheck machinery (bracket proofs: eq * T * T * M) ----
// row poly p(t) = sum_i E_t[i]*A_t[i]*B_t[i]*C_t[i], t = 0..4
KERNEL void k_hp4_step(GLOBAL Fr_t* E, GLOBAL Fr_t* A, GLOBAL Fr_t* B, GLOBAL Fr_t* C,
                       GLOBAL Fr_t* o0, GLOBAL Fr_t* o1, GLOBAL Fr_t* o2,
                       GLOBAL Fr_t* o3, GLOBAL Fr_t* o4, uint h) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= h) return;
    Fr_t e = E[gid], de = blstrs__scalar__Scalar_sub(E[gid + h], E[gid]);
    Fr_t a = A[gid], da = blstrs__scalar__Scalar_sub(A[gid + h], A[gid]);
    Fr_t b = B[gid], db = blstrs__scalar__Scalar_sub(B[gid + h], B[gid]);
    Fr_t c = C[gid], dc = blstrs__scalar__Scalar_sub(C[gid + h], C[gid]);
    #pragma unroll
    for (int t = 0; t < 5; t++) {
        // mont-ify all factors except ONE (same rule as k_hp3_step):
        // ea = mont(e)*mont(a) = eaR;  bc = mont(b)*c = bc plain;  v = ea*bc = eabc plain
        Fr_t ea = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(e),
                                             blstrs__scalar__Scalar_mont(a));
        Fr_t bc = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(b), c);
        Fr_t v = blstrs__scalar__Scalar_mul(ea, bc);
        if (t == 0) o0[gid] = v;
        else if (t == 1) o1[gid] = v;
        else if (t == 2) o2[gid] = v;
        else if (t == 3) o3[gid] = v;
        else o4[gid] = v;
        e = blstrs__scalar__Scalar_add(e, de);
        a = blstrs__scalar__Scalar_add(a, da);
        b = blstrs__scalar__Scalar_add(b, db);
        c = blstrs__scalar__Scalar_add(c, dc);
    }
}
// broadcast a row vector across columns: out[s*Cpad + j] = in[s]
KERNEL void k_bcast_rows(GLOBAL Fr_t* in, GLOBAL Fr_t* out, uint Cpad, uint D) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= D) return;
    out[gid] = in[gid / Cpad];
}
// Lagrange-5 from {p(0..4)} at v; exact for degree <= 4
static Fr_t lagrange5(const array<Fr_t,5>& p, const Fr_t& v) {
    static const Fr_t F_FOUR = {4, 0, 0, 0, 0, 0, 0, 0};
    static const Fr_t F_24 = {24, 0, 0, 0, 0, 0, 0, 0};
    Fr_t inv6 = inv(F_SIX), inv24 = inv(F_24), inv4 = inv(F_FOUR);
    Fr_t m1 = h_scalar(v, F_ONE, 1), m2 = h_scalar(v, F_TWO, 1),
         m3 = h_scalar(v, F_THREE, 1), m4 = h_scalar(v, F_FOUR, 1);
    auto mul3 = [](const Fr_t& a, const Fr_t& b, const Fr_t& c) {
        return h_scalar(h_scalar(a, b, 2), c, 2); };
    Fr_t l0 = h_scalar(mul3(m1, m2, m3), h_scalar(m4, inv24, 2), 2);             // /24
    Fr_t l1 = h_scalar(F_ZERO, h_scalar(mul3(v, m2, m3), h_scalar(m4, inv6, 2), 2), 1);  // /-6
    Fr_t l2 = h_scalar(mul3(v, m1, m3), h_scalar(m4, inv4, 2), 2);               // /4
    Fr_t l3 = h_scalar(F_ZERO, h_scalar(mul3(v, m1, m2), h_scalar(m4, inv6, 2), 2), 1);  // /-6
    Fr_t l4 = h_scalar(mul3(v, m1, m2), h_scalar(m3, inv24, 2), 2);              // /24
    Fr_t acc = h_scalar(p[0], l0, 2);
    acc = h_scalar(acc, h_scalar(p[1], l1, 2), 0);
    acc = h_scalar(acc, h_scalar(p[2], l2, 2), 0);
    acc = h_scalar(acc, h_scalar(p[3], l3, 2), 0);
    acc = h_scalar(acc, h_scalar(p[4], l4, 2), 0);
    return acc;
}
struct QuarticProof {
    Fr_t claim0;              // initial claim (context-defined)
    vector<Fr_t> ev;          // 5 per round: p(0..4)
    Fr_t A_f, B_f, C_f;       // terminal opening obligations (E_f verifier-computed)
};
static void write_qp(const string& path, const QuarticProof& p) {
    FILE* f = open_or_die(path, "wb");
    fwrite(&p.claim0, sizeof(Fr_t), 1, f);
    write_pod_vec(f, p.ev);
    fwrite(&p.A_f, sizeof(Fr_t), 1, f);
    fwrite(&p.B_f, sizeof(Fr_t), 1, f);
    fwrite(&p.C_f, sizeof(Fr_t), 1, f);
    fclose(f);
}
static QuarticProof read_qp(const string& path) {
    FILE* f = open_or_die(path, "rb");
    QuarticProof p;
    if (fread(&p.claim0, sizeof(Fr_t), 1, f) != 1) throw runtime_error("read_qp");
    p.ev = read_pod_vec<Fr_t>(f);
    if (fread(&p.A_f, sizeof(Fr_t), 1, f) != 1 ||
        fread(&p.B_f, sizeof(Fr_t), 1, f) != 1 ||
        fread(&p.C_f, sizeof(Fr_t), 1, f) != 1) throw runtime_error("read_qp terminals");
    fclose(f);
    return p;
}
// FS-per-round degree-4 sumcheck recursion (prover side); tag disambiguates
// multiple instances on one transcript
static void fs_quartic(Fr_t claim, FrTensor& E, FrTensor& A, FrTensor& B, FrTensor& C,
                       fs::Transcript& tr, const string& tag, vector<Fr_t>& ws,
                       QuarticProof& out, bool strict) {
    if (E.size == 1) { out.A_f = A(0); out.B_f = B(0); out.C_f = C(0); return; }
    const uint h = E.size >> 1;
    FrTensor o0(h), o1(h), o2(h), o3(h), o4(h);
    k_hp4_step<<<(h + 255) / 256, 256>>>(E.gpu_data, A.gpu_data, B.gpu_data, C.gpu_data,
        o0.gpu_data, o1.gpu_data, o2.gpu_data, o3.gpu_data, o4.gpu_data, h);
    cudaDeviceSynchronize();
    array<Fr_t,5> e = {o0.sum(), o1.sum(), o2.sum(), o3.sum(), o4.sum()};
    if (strict && !fr_eq(claim, h_scalar(e[0], e[1], 0)))
        throw runtime_error("quartic round inconsistency (witness bug): " + tag);
    for (int i = 0; i < 5; i++) out.ev.push_back(e[i]);
    for (int i = 0; i < 5; i++) absorb_fr(tr, (tag + "p" + to_string(i)).c_str(), e[i]);
    Fr_t w = fs_challenge_fr(tr); ws.push_back(w);
    FrTensor nE(h), nA(h), nB(h), nC(h);
    k_fr_fold<<<(h + 255) / 256, 256>>>(E.gpu_data, w, nE.gpu_data, h);
    k_fr_fold<<<(h + 255) / 256, 256>>>(A.gpu_data, w, nA.gpu_data, h);
    k_fr_fold<<<(h + 255) / 256, 256>>>(B.gpu_data, w, nB.gpu_data, h);
    k_fr_fold<<<(h + 255) / 256, 256>>>(C.gpu_data, w, nC.gpu_data, h);
    cudaDeviceSynchronize();
    fs_quartic(lagrange5(e, w), nE, nA, nB, nC, tr, tag, ws, out, strict);
}

#endif // ZKOB_LOOKUP_CUH
