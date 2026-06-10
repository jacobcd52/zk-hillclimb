// Real driver for ONE rmsnorm obligation: binds the inverse-RMS advice R to X
// within +-1 integer tolerance, closing the unbound rms_inv_temp covert channel
// (<= log2(3) bits/row covert freedom, documented as the measured floor).
//
// Pipeline semantics (m68-pipeline.py): X int32 (B x C, scale 2^16),
// R = round(2^16 / sqrt(mean(X_real^2) + eps)) (B), g int32 (C, registered).
// W = R x g outer (scale 2^32) -> rescale 2^16 -> W_ -> Y = W_ .* X (scale 2^32).
// M[s] = sum_j X[s,j]^2 + C_eps, C_eps = round(eps * C * 2^32) (u64 CLI arg).
// Exact R satisfies (R-1)^2 * M <= 2^64 * C <= (R+1)^2 * M.
//
// Proof obligations (one transcript, 17 IPA openings):
//  1. LIMB RANGE LOOKUP: P1 = 2^64 C - (R-1)^2 M >= 0, P2 = (R+1)^2 M - 2^64 C >= 0,
//     both < 2^80. L = limb matrix, n_rows = max(16, 65536/B_pad) rows x B_pad cols
//     (rows 0-4 = P1 16-bit limbs, 5-9 = P2 limbs, rest zero), committed with gen_B;
//     logUp vs tLookupRange(0, 65536).
//  2. HOMOMORPHIC AFFINE LIMB LINKS (host h_mul/h_add/g1_eq, no openings):
//     com_P1 == sum_{i<5} 2^{16i} com_L_row[i]; com_P2 == sum 2^{16i} com_L_row[5+i].
//  3. SS SUMCHECK (binds M to X): claim = M~(u_b) - C_eps = sum eq.X.X over logD vars
//     with E_bcast = bcast_rows(eq_tensor(u_b)); fs_hadamard with S = U = X_pad.
//     Verifier eq factor only for rounds k < logB (row bits are the MSBs).
//     Requires S_f2 == U_f2; openings: X at reverse(ws), M~(u_b) vs com_M.
//  4. BRACKET QUARTICS (fs_quartic, tags "q1"/"q2", Lagrange-5):
//     claim_q1 = 2^64 C - P1~(u_b2) over (E2, T1, T1, M), T1 = R-1;
//     claim_q2 = P2~(u_b2) + 2^64 C over (E2, T2, T2, M), T2 = R+1.
//     No commitment for T1/T2: verifier opens R at pt1 expecting q1.A_f + 1 and at
//     pt2 expecting q2.A_f - 1; requires A_f == B_f. Openings: P1@u_b2, P2@u_b2,
//     R@pt1, M@pt1, R@pt2, M@pt2.
//  5. OUTER PRODUCT W = R x g via MLE factorization (no sumcheck): absorb
//     val_R = R~(u_b3), val_g = g~(u_c3), val_W = W~(u_pt3); check
//     val_W == val_R * val_g; three openings (g vs the REGISTERED com_g).
//     u_pt3 = concat(u_c3 [low logC bits], u_b3 [high]). Needs B == B_pad (g is
//     zero-padded so W_pad = R (x) g_pad exactly on the whole grid).
//  6. INTERNAL RESCALE W_ = rescale(W, 2^16) on the UNPADDED tensor then padded
//     (byte-identical with the separate zkob_rescale run); driver commits com_W_.
//  7. HADAMARD Y = W_ .* X (glu part 2). Openings Y@u_h, W_@pt, X@pt.
// Chain outputs: W.i64 and Y.i64, UNPADDED B x C int64.
//
// FS schedule (one transcript, seed = run_seed:obligation_id):
//   absorb B, C, C_eps(lo,hi), com_X, com_g, com_R, com_M, com_P1, com_P2,
//   com_L, com_m_L, com_W, com_W_, com_Y -> beta -> com_A_L -> alpha -> u_L
//   -> limb lookup rounds + terminals + 3 openings -> u_b -> ev_M
//   -> SS rounds + terminals + 2 openings -> u_b2 -> ev_P1, ev_P2
//   -> q1 rounds, q2 rounds, terminals + 6 openings -> u_b3, u_c3
//   -> val_R/val_g/val_W + 3 openings -> u_h -> claim_Y
//   -> hadamard rounds + terminals + 3 openings
//
// Usage:
//   zkob_rmsnorm prove  <obdir> <seed> <X-int32> <R-int32> <g-int32> <B> <C>
//                       <C_eps-u64> <gen_C> <gen_B> <q> [W-i64-out Y-i64-out]
//   zkob_rmsnorm verify <obdir> <seed> <B> <C> <C_eps> <com_g-path>
//                       <gen_C> <gen_B> <q>
//   zkob_rmsnorm selftest
#include "zkob_lookup.cuh"
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <sys/stat.h>
using namespace std;

typedef unsigned __int128 u128;
typedef __int128 i128;

// plain (non-Montgomery) Fr from a u128 value (< p, so direct limbs)
static Fr_t fr_from_u128(u128 v) {
    return {(uint32_t)v, (uint32_t)(v >> 32), (uint32_t)(v >> 64), (uint32_t)(v >> 96),
            0, 0, 0, 0};
}
// limb weights 2^{16i}, i = 0..4
static const Fr_t W16[5] = {
    {1, 0, 0, 0, 0, 0, 0, 0}, {65536, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0}, {0, 65536, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 0, 0}};

static const uint LOOKUP_N = 65536;   // 16-bit limb range table

// ---------------- prove ----------------
// evil (selftest only; honest procedure run on an inconsistent witness):
//  1: R[idx] += 2, P1 stored mod p (negative wraps), limbs = low 80 bits
//     -> AFFINE LINK on P1 must reject (everything else stays consistent)
//  2: R[idx] += 2, limbs honest-truncated, P1 = limb reconstruction
//     -> quartic q1 round 0 must reject (q2 stays consistent)
//  3: M[idx] += 1, brackets recomputed from the new M -> SS round 0 must reject
//  4: Y[idx] += 1 -> hadamard round 0 must reject
//  5: W[idx] += 1 -> outer product point check must reject
//  6: R[idx] -= 2, P2 stored mod p (negative wraps), limbs = low 80 bits
//     -> AFFINE LINK on P2 must reject (mirror of 1)
//  7: R[idx] -= 2, limbs honest-truncated, P2 = limb reconstruction
//     -> quartic q2 round 0 must reject (q1 stays consistent; mirror of 2)
//  8: L[0,idx] += 2^16 with a borrow L[1,idx] -= 1 (P1 value unchanged, so the
//     affine links and quartics stay consistent; m_L committed from the honest
//     limbs) -> only the limb range lookup can reject
static void prove(const string& obdir, const string& seed,
                  const vector<int>& Xh, const vector<int>& Rh, const vector<int>& gh,
                  uint B, uint C, unsigned long long C_eps,
                  const Commitment& gen_C, const Commitment& gen_B,
                  const G1Jacobian_t& Q,
                  const string& w_out, const string& y_out,
                  int evil = 0, uint evil_idx = 0) {
    const uint B_pad = 1u << ceilLog2(B), C_pad = 1u << ceilLog2(C);
    const uint D = B_pad * C_pad;
    const uint logD = ceilLog2(D), logB = ceilLog2(B_pad), logC = ceilLog2(C_pad);
    const uint N = LOOKUP_N;
    const uint n_rows = max(16u, N / B_pad), D_L = n_rows * B_pad;
    const uint logDL = ceilLog2(D_L), n1 = ceilLog2(D_L / N);
    if (B != B_pad) throw runtime_error("B must equal B_pad (power of two)");
    if (C >= (1u << 16)) throw runtime_error("C must be < 2^16");
    if (gen_C.size != C_pad) throw runtime_error("gen_C size != C_pad");
    if (gen_B.size != B_pad) throw runtime_error("gen_B size != B_pad");
    if (B_pad > N || N > D_L || D_L % N)
        throw runtime_error("limb lookup layout needs B_pad <= N <= D_L, N | D_L");
    if (Xh.size() != (size_t)B * C || Rh.size() != B || gh.size() != C)
        throw runtime_error("input dims");

    // ---- host bracket math in __int128 ----
    vector<int> Rmod(Rh);
    if (evil == 1 || evil == 2) Rmod[evil_idx] += 2;
    if (evil == 6 || evil == 7) Rmod[evil_idx] -= 2;
    const u128 U1 = 1;
    const u128 MASK80 = (U1 << 80) - 1;
    const i128 c64C = ((i128)C) << 64;
    vector<Fr_t> Mfr(B), P1fr(B), P2fr(B);
    vector<int> Lh((size_t)D_L, 0);
    for (uint s = 0; s < B; s++) {
        u128 M = C_eps;
        for (uint j = 0; j < C; j++)
            M += (u128)((long)Xh[s*C+j] * (long)Xh[s*C+j]);
        if (evil == 3 && s == evil_idx) M += 1;
        if (M >= (U1 << 62)) throw runtime_error("M >= 2^62");
        Mfr[s] = fr_from_u128(M);
        i128 r = Rmod[s];
        i128 V1 = c64C - (r - 1) * (r - 1) * (i128)M;
        i128 V2 = (r + 1) * (r + 1) * (i128)M - c64C;
        if (evil == 0 || evil == 3 || evil == 4 || evil == 5 || evil == 8) {
            if (V1 < 0 || V2 < 0)
                throw runtime_error("advice R out of +-1 tolerance (bracket violated)");
            if ((u128)V1 >= (U1 << 80) || (u128)V2 >= (U1 << 80))
                throw runtime_error("bracket residual >= 2^80");
        }
        if (evil == 6 || evil == 7) {     // R-2: V2 goes negative, V1 must stay in range
            if (V1 < 0 || (u128)V1 >= (U1 << 80))
                throw runtime_error("P1 out of [0, 2^80) (selftest evil setup broken)");
        } else if (V2 < 0 || (u128)V2 >= (U1 << 80))
            throw runtime_error("P2 out of [0, 2^80) (selftest evil setup broken)");
        u128 t1 = (u128)V1 & MASK80;      // honest: == V1
        u128 t2 = (u128)V2 & MASK80;      // honest: == V2
        for (uint i = 0; i < 5; i++) {
            Lh[(size_t)i * B_pad + s] = (int)((t1 >> (16 * i)) & 0xFFFF);
            Lh[(size_t)(5 + i) * B_pad + s] = (int)((t2 >> (16 * i)) & 0xFFFF);
        }
        if (evil == 2)      P1fr[s] = fr_from_u128(t1);             // reconstruction
        else if (V1 < 0)    P1fr[s] = h_scalar(F_ZERO, fr_from_u128((u128)(-V1)), 1);
        else                P1fr[s] = fr_from_u128((u128)V1);
        if (evil == 7)      P2fr[s] = fr_from_u128(t2);             // reconstruction
        else if (V2 < 0)    P2fr[s] = h_scalar(F_ZERO, fr_from_u128((u128)(-V2)), 1);
        else                P2fr[s] = fr_from_u128(t2);
    }
    // evil 8: out-of-range limb with a compensating borrow — P1's value (and so
    // the affine link, the quartics and every commitment but com_L) is UNCHANGED;
    // only the range lookup stands between this forgery and ACCEPT.
    vector<int> Lh_honest;
    if (evil == 8) {
        Lh_honest = Lh;
        if (Lh[(size_t)B_pad + evil_idx] < 1)
            throw runtime_error("evil 8 setup: L[1,idx] == 0 (pick another idx)");
        Lh[evil_idx] += 65536;                 // row 0: limb leaves [0, 2^16)
        Lh[(size_t)B_pad + evil_idx] -= 1;     // row 1 borrow: same P1 value
    }

    // ---- tensors (all PLAIN Fr values) ----
    FrTensor X(B * C, Xh.data());
    FrTensor X_pad = X.pad({B, C});
    FrTensor R_t(B, Rmod.data());
    vector<int> t1h(B), t2h(B);
    for (uint s = 0; s < B; s++) { t1h[s] = Rmod[s] - 1; t2h[s] = Rmod[s] + 1; }
    FrTensor T1(B, t1h.data()), T2(B, t2h.data());
    FrTensor M_t(B, Mfr.data());
    FrTensor P1_t(B, P1fr.data()), P2_t(B, P2fr.data());
    FrTensor L(D_L, Lh.data());
    FrTensor g_t(C, gh.data());
    FrTensor g_pad = g_t.pad({C});

    // W = R x g (host, exact int64), chain file = the UNPADDED tensor
    vector<long> Wh((size_t)B * C);
    for (uint s = 0; s < B; s++)
        for (uint j = 0; j < C; j++)
            Wh[(size_t)s*C+j] = (long)Rmod[s] * (long)gh[j];
    if (evil == 5) Wh[evil_idx] += 1;
    if (!w_out.empty()) {
        FILE* f = open_or_die(w_out, "wb");
        fwrite(Wh.data(), sizeof(long), (size_t)B * C, f); fclose(f);
    }
    FrTensor W_t(B * C, Wh.data());
    FrTensor W_pad = W_t.pad({B, C});

    // internal rescale W_ = rescale(W, 2^16) on the UNPADDED tensor, then pad
    FrTensor Wr(B * C), Wrem(B * C);
    rescaling_kernel<<<(B*C + 255) / 256, 256>>>(
        W_t.gpu_data, Wr.gpu_data, Wrem.gpu_data, (long)(1L << 16), B * C);
    cudaDeviceSynchronize();
    FrTensor Wr_pad = Wr.pad({B, C});

    FrTensor Y_pad = Wr_pad * X_pad;
    if (evil == 4) {
        k_bump<<<1,1>>>(Y_pad.gpu_data, evil_idx, F_ONE, 0);
        cudaDeviceSynchronize();
    }
    if (!y_out.empty()) {
        Y_pad.save_long(y_out);
        if (C != C_pad) {   // strip the column padding (B == B_pad already)
            vector<long> padded(D);
            FILE* f = open_or_die(y_out, "rb");
            if (fread(padded.data(), sizeof(long), D, f) != D)
                throw runtime_error("chain strip: short read");
            fclose(f);
            f = open_or_die(y_out, "wb");
            for (uint b = 0; b < B; b++)
                fwrite(padded.data() + (size_t)b * C_pad, sizeof(long), C, f);
            fclose(f);
        }
    }

    // limb multiplicities vs the public range table (evil 8: from the HONEST
    // limbs — the forged 65536 is outside the table, where prep's unchecked
    // atomicAdd would write out of bounds; committing the honest multiplicities
    // is also the forging prover's best move)
    tLookupRange tl(0, N);
    FrTensor m_L = (evil == 8) ? tl.prep(FrTensor(D_L, Lh_honest.data()))
                               : tl.prep(L);

    // ---- commitments ----
    G1TensorJacobian com_X  = gen_C.commit(X_pad);
    G1TensorJacobian com_g  = gen_C.commit(g_pad);
    G1TensorJacobian com_R  = gen_B.commit(R_t);
    G1TensorJacobian com_M  = gen_B.commit(M_t);
    G1TensorJacobian com_P1 = gen_B.commit(P1_t);
    G1TensorJacobian com_P2 = gen_B.commit(P2_t);
    G1TensorJacobian com_L  = gen_B.commit(L);
    G1TensorJacobian com_mL = gen_B.commit(m_L);
    G1TensorJacobian com_W  = gen_C.commit(W_pad);
    G1TensorJacobian com_Wr = gen_C.commit(Wr_pad);
    G1TensorJacobian com_Y  = gen_C.commit(Y_pad);
    com_X.save(obdir + "/com_X.bin");
    com_g.save(obdir + "/com_g.bin");
    com_R.save(obdir + "/com_R.bin");
    com_M.save(obdir + "/com_M.bin");
    com_P1.save(obdir + "/com_P1.bin");
    com_P2.save(obdir + "/com_P2.bin");
    com_L.save(obdir + "/com_L.bin");
    com_mL.save(obdir + "/com_m_L.bin");
    com_W.save(obdir + "/com_W.bin");
    com_Wr.save(obdir + "/com_Wr.bin");
    com_Y.save(obdir + "/com_Y.bin");
    { FILE* f = open_or_die(obdir + "/dims.bin", "wb");
      uint32_t d[2] = {B, C}; uint64_t e = C_eps;
      fwrite(d, sizeof(uint32_t), 2, f); fwrite(&e, sizeof(uint64_t), 1, f); fclose(f); }

    // ---- transcript ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C);
    absorb_u32(tr, "C_eps_lo", (uint32_t)C_eps);
    absorb_u32(tr, "C_eps_hi", (uint32_t)(C_eps >> 32));
    absorb_g1_tensor(tr, "com_X", com_X);
    absorb_g1_tensor(tr, "com_g", com_g);
    absorb_g1_tensor(tr, "com_R", com_R);
    absorb_g1_tensor(tr, "com_M", com_M);
    absorb_g1_tensor(tr, "com_P1", com_P1);
    absorb_g1_tensor(tr, "com_P2", com_P2);
    absorb_g1_tensor(tr, "com_L", com_L);
    absorb_g1_tensor(tr, "com_m_L", com_mL);
    absorb_g1_tensor(tr, "com_W", com_W);
    absorb_g1_tensor(tr, "com_W_", com_Wr);
    absorb_g1_tensor(tr, "com_Y", com_Y);
    Fr_t beta = fs_challenge_fr(tr);

    // ---- (1) limb range lookup ----
    FrTensor A_L(D_L);
    tlookup_inv_kernel<<<(D_L+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        L.gpu_data, beta, A_L.gpu_data, D_L);
    cudaDeviceSynchronize();
    G1TensorJacobian com_AL = gen_B.commit(A_L);
    com_AL.save(obdir + "/com_A_L.bin");
    absorb_g1_tensor(tr, "com_A_L", com_AL);
    Fr_t alpha = fs_challenge_fr(tr);
    auto u_L = fs_challenge_vec(tr, logDL);

    FrTensor Bv(N);
    tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        tl.table.gpu_data, beta, Bv.gpu_data, N);
    cudaDeviceSynchronize();
    Fr_t alpha_sq = alpha * alpha;
    Fr_t Cc = alpha_sq - (Bv * m_L).sum();
    Fr_t claim = alpha + alpha_sq;
    Fr_t inv_ratio = Fr_t{N,0,0,0,0,0,0,0} / Fr_t{D_L,0,0,0,0,0,0,0};

    LookupProof pf;
    vector<Fr_t> wsl;
    fs_phase1(claim, A_L, L, Bv, tl.table, m_L, alpha, beta, Cc,
              inv_ratio, alpha_sq, u_L, tr, wsl, pf, evil != 8);
    write_lookup(obdir + "/lookup.bin", pf);
    absorb_fr(tr, "A_f", pf.A_f); absorb_fr(tr, "S_f", pf.S_f); absorb_fr(tr, "m_f", pf.m_f);

    vector<Fr_t> u_ptL(wsl.rbegin(), wsl.rend());
    vector<Fr_t> u_mL(wsl.rbegin(), wsl.rend() - n1);
    if (evil == 0) {   // convention sanity: fold terminals == ME evaluations
        vector<Fr_t> u_col(u_ptL.begin(), u_ptL.begin() + logB);
        vector<Fr_t> u_row(u_ptL.begin() + logB, u_ptL.end());
        if (!fr_eq(pf.A_f, A_L.multi_dim_me({u_row, u_col}, {n_rows, B_pad})) ||
            !fr_eq(pf.S_f, L.multi_dim_me({u_row, u_col}, {n_rows, B_pad})))
            throw runtime_error("lookup terminal != multi_dim_me (convention bug)");
    }
    open_prove(A_L, B_pad, gen_B, Q, u_ptL, obdir + "/ipa_AL.bin", tr);
    open_prove(L,   B_pad, gen_B, Q, u_ptL, obdir + "/ipa_L.bin", tr);
    open_prove(m_L, B_pad, gen_B, Q, u_mL,  obdir + "/ipa_mL.bin", tr);

    // ---- (3) SS sumcheck: M~(u_b) - C_eps = sum eq . X . X ----
    auto u_b = fs_challenge_vec(tr, logB);
    HadamardProof ss;
    ss.claim_H = M_t.multi_dim_me({u_b}, {B_pad});            // ev_M
    absorb_fr(tr, "ev_M", ss.claim_H);
    Fr_t C_eps_fr = {(uint32_t)C_eps, (uint32_t)(C_eps >> 32), 0, 0, 0, 0, 0, 0};
    Fr_t claim_ss = h_scalar(ss.claim_H, C_eps_fr, 1);
    FrTensor E_row = build_eq_tensor(u_b);
    FrTensor E_full(D);
    k_bcast_rows<<<(D + 255) / 256, 256>>>(E_row.gpu_data, E_full.gpu_data, C_pad, D);
    cudaDeviceSynchronize();
    vector<Fr_t> ws_ss;
    {
        FrTensor S2(X_pad), U2(X_pad);
        fs_hadamard(claim_ss, E_full, S2, U2, tr, ws_ss, ss, evil != 3);
    }
    absorb_fr(tr, "S_f2", ss.S_f2); absorb_fr(tr, "U_f2", ss.U_f2);
    write_hp(obdir + "/hpss.bin", ss);
    vector<Fr_t> u_ss(ws_ss.rbegin(), ws_ss.rend());
    if (evil == 0) {
        vector<Fr_t> u_col(u_ss.begin(), u_ss.begin() + logC);
        vector<Fr_t> u_row(u_ss.begin() + logC, u_ss.end());
        if (!fr_eq(ss.S_f2, X_pad.multi_dim_me({u_row, u_col}, {B_pad, C_pad})) ||
            !fr_eq(ss.S_f2, ss.U_f2))
            throw runtime_error("SS terminal != multi_dim_me (convention bug)");
    }
    open_prove(X_pad, C_pad, gen_C, Q, u_ss, obdir + "/ipa_X_ss.bin", tr);
    open_prove(M_t,   B_pad, gen_B, Q, u_b,  obdir + "/ipa_M.bin", tr);

    // ---- (4) bracket quartics ----
    auto u_b2 = fs_challenge_vec(tr, logB);
    QuarticProof qp1, qp2;
    qp1.claim0 = P1_t.multi_dim_me({u_b2}, {B_pad});          // ev_P1
    qp2.claim0 = P2_t.multi_dim_me({u_b2}, {B_pad});          // ev_P2
    absorb_fr(tr, "ev_P1", qp1.claim0);
    absorb_fr(tr, "ev_P2", qp2.claim0);
    Fr_t c64C_fr = {0, 0, C, 0, 0, 0, 0, 0};                  // 2^64 * C
    Fr_t claim_q1 = h_scalar(c64C_fr, qp1.claim0, 1);
    Fr_t claim_q2 = h_scalar(qp2.claim0, c64C_fr, 0);
    vector<Fr_t> ws_q1, ws_q2;
    {
        FrTensor E2 = build_eq_tensor(u_b2);
        FrTensor Aq(T1), Bq(T1), Cq(M_t);
        fs_quartic(claim_q1, E2, Aq, Bq, Cq, tr, "q1", ws_q1, qp1, evil != 2);
    }
    {
        FrTensor E2 = build_eq_tensor(u_b2);
        FrTensor Aq(T2), Bq(T2), Cq(M_t);
        fs_quartic(claim_q2, E2, Aq, Bq, Cq, tr, "q2", ws_q2, qp2, evil != 7);
    }
    absorb_fr(tr, "q1A_f", qp1.A_f); absorb_fr(tr, "q1B_f", qp1.B_f);
    absorb_fr(tr, "q1C_f", qp1.C_f);
    absorb_fr(tr, "q2A_f", qp2.A_f); absorb_fr(tr, "q2B_f", qp2.B_f);
    absorb_fr(tr, "q2C_f", qp2.C_f);
    write_qp(obdir + "/qp1.bin", qp1);
    write_qp(obdir + "/qp2.bin", qp2);
    vector<Fr_t> pt1(ws_q1.rbegin(), ws_q1.rend());
    vector<Fr_t> pt2(ws_q2.rbegin(), ws_q2.rend());
    if (evil == 0) {
        if (!fr_eq(qp1.A_f, qp1.B_f) || !fr_eq(qp2.A_f, qp2.B_f))
            throw runtime_error("quartic A_f != B_f (convention bug)");
        if (!fr_eq(h_scalar(qp1.A_f, F_ONE, 0), R_t.multi_dim_me({pt1}, {B_pad})) ||
            !fr_eq(qp1.C_f, M_t.multi_dim_me({pt1}, {B_pad})) ||
            !fr_eq(h_scalar(qp2.A_f, F_ONE, 1), R_t.multi_dim_me({pt2}, {B_pad})) ||
            !fr_eq(qp2.C_f, M_t.multi_dim_me({pt2}, {B_pad})))
            throw runtime_error("quartic terminal != multi_dim_me (convention bug)");
    }
    open_prove(P1_t, B_pad, gen_B, Q, u_b2, obdir + "/ipa_P1.bin", tr);
    open_prove(P2_t, B_pad, gen_B, Q, u_b2, obdir + "/ipa_P2.bin", tr);
    open_prove(R_t,  B_pad, gen_B, Q, pt1,  obdir + "/ipa_R_q1.bin", tr);
    open_prove(M_t,  B_pad, gen_B, Q, pt1,  obdir + "/ipa_M_q1.bin", tr);
    open_prove(R_t,  B_pad, gen_B, Q, pt2,  obdir + "/ipa_R_q2.bin", tr);
    open_prove(M_t,  B_pad, gen_B, Q, pt2,  obdir + "/ipa_M_q2.bin", tr);

    // ---- (5) outer product W = R x g via MLE factorization ----
    auto u_b3 = fs_challenge_vec(tr, logB);
    auto u_c3 = fs_challenge_vec(tr, logC);
    Fr_t val_R = R_t.multi_dim_me({u_b3}, {B_pad});
    Fr_t val_g = g_pad.multi_dim_me({u_c3}, {C_pad});
    Fr_t val_W = W_pad.multi_dim_me({u_b3, u_c3}, {B_pad, C_pad});
    if (evil != 5 && !fr_eq(val_W, h_scalar(val_R, val_g, 2)))
        throw runtime_error("outer product val_W != val_R*val_g (convention bug)");
    absorb_fr(tr, "val_R", val_R); absorb_fr(tr, "val_g", val_g);
    absorb_fr(tr, "val_W", val_W);
    { FILE* f = open_or_die(obdir + "/outer.bin", "wb");
      fwrite(&val_R, sizeof(Fr_t), 1, f); fwrite(&val_g, sizeof(Fr_t), 1, f);
      fwrite(&val_W, sizeof(Fr_t), 1, f); fclose(f); }
    vector<Fr_t> u_pt3(u_c3);
    u_pt3.insert(u_pt3.end(), u_b3.begin(), u_b3.end());
    open_prove(R_t,   B_pad, gen_B, Q, u_b3,  obdir + "/ipa_R_o.bin", tr);
    open_prove(g_pad, C_pad, gen_C, Q, u_c3,  obdir + "/ipa_g.bin", tr);
    open_prove(W_pad, C_pad, gen_C, Q, u_pt3, obdir + "/ipa_W.bin", tr);

    // ---- (7) hadamard Y = W_ .* X ----
    auto u_h = fs_challenge_vec(tr, logD);
    HadamardProof hp;
    {
        vector<Fr_t> uh_col(u_h.begin(), u_h.begin() + logC);
        vector<Fr_t> uh_row(u_h.begin() + logC, u_h.end());
        hp.claim_H = Y_pad.multi_dim_me({uh_row, uh_col}, {B_pad, C_pad});
    }
    absorb_fr(tr, "claim_Y", hp.claim_H);
    FrTensor Eh = build_eq_tensor(u_h);
    vector<Fr_t> wsh;
    {
        FrTensor Sh(Wr_pad), Uh(X_pad);
        fs_hadamard(hp.claim_H, Eh, Sh, Uh, tr, wsh, hp, evil != 4);
    }
    absorb_fr(tr, "S_f2h", hp.S_f2); absorb_fr(tr, "U_f2h", hp.U_f2);
    write_hp(obdir + "/hp.bin", hp);
    vector<Fr_t> u_pth(wsh.rbegin(), wsh.rend());
    if (evil == 0) {
        vector<Fr_t> u_col(u_pth.begin(), u_pth.begin() + logC);
        vector<Fr_t> u_row(u_pth.begin() + logC, u_pth.end());
        if (!fr_eq(hp.S_f2, Wr_pad.multi_dim_me({u_row, u_col}, {B_pad, C_pad})) ||
            !fr_eq(hp.U_f2, X_pad.multi_dim_me({u_row, u_col}, {B_pad, C_pad})))
            throw runtime_error("hadamard terminal != multi_dim_me (convention bug)");
    }
    open_prove(Y_pad,  C_pad, gen_C, Q, u_h,   obdir + "/ipa_Y.bin", tr);
    open_prove(Wr_pad, C_pad, gen_C, Q, u_pth, obdir + "/ipa_Wr.bin", tr);
    open_prove(X_pad,  C_pad, gen_C, Q, u_pth, obdir + "/ipa_X_h.bin", tr);
    cout << "PROVED rmsnorm obligation -> " << obdir << endl;
}

// ---------------- verify (witness-free) ----------------
#define RJ(msg) do { ostringstream oss_; oss_ << msg; \
    cout << "REJECT: " << oss_.str() << endl; \
    if (reason) *reason = oss_.str(); return false; } while (0)

static bool verify(const string& obdir, const string& seed, uint B, uint C,
                   unsigned long long C_eps, const string& com_g_path,
                   const Commitment& gen_C, const Commitment& gen_B,
                   const G1Jacobian_t& Q, string* reason = nullptr) {
    const uint B_pad = 1u << ceilLog2(B), C_pad = 1u << ceilLog2(C);
    const uint D = B_pad * C_pad;
    const uint logD = ceilLog2(D), logB = ceilLog2(B_pad), logC = ceilLog2(C_pad);
    const uint N = LOOKUP_N;
    const uint n_rows = max(16u, N / B_pad), D_L = n_rows * B_pad;
    const uint logDL = ceilLog2(D_L), n1 = ceilLog2(D_L / N), n2 = ceilLog2(N);
    if (B != B_pad || C >= (1u << 16) || gen_C.size != C_pad || gen_B.size != B_pad ||
        B_pad > N || N > D_L || D_L % N)
        RJ("bad layout params");
    { FILE* f = open_or_die(obdir + "/dims.bin", "rb");
      uint32_t d[2]; uint64_t e;
      if (fread(d, sizeof(uint32_t), 2, f) != 2 || fread(&e, sizeof(uint64_t), 1, f) != 1)
          { fclose(f); RJ("dims.bin short read"); }
      fclose(f);
      if (d[0] != B || d[1] != C || e != C_eps) RJ("dims.bin mismatch"); }

    G1TensorJacobian com_X (obdir + "/com_X.bin");
    G1TensorJacobian com_g (com_g_path);                     // REGISTERED commitment
    G1TensorJacobian com_R (obdir + "/com_R.bin");
    G1TensorJacobian com_M (obdir + "/com_M.bin");
    G1TensorJacobian com_P1(obdir + "/com_P1.bin");
    G1TensorJacobian com_P2(obdir + "/com_P2.bin");
    G1TensorJacobian com_L (obdir + "/com_L.bin");
    G1TensorJacobian com_mL(obdir + "/com_m_L.bin");
    G1TensorJacobian com_W (obdir + "/com_W.bin");
    G1TensorJacobian com_Wr(obdir + "/com_Wr.bin");
    G1TensorJacobian com_Y (obdir + "/com_Y.bin");
    G1TensorJacobian com_AL(obdir + "/com_A_L.bin");
    if (com_X.size != B_pad || com_W.size != B_pad || com_Wr.size != B_pad ||
        com_Y.size != B_pad || com_g.size != 1 || com_R.size != 1 ||
        com_M.size != 1 || com_P1.size != 1 || com_P2.size != 1 ||
        com_L.size != n_rows || com_AL.size != n_rows || com_mL.size != N / B_pad)
        RJ("commitment row counts");

    LookupProof pf = read_lookup(obdir + "/lookup.bin");
    HadamardProof ss = read_hp(obdir + "/hpss.bin");
    QuarticProof qp1 = read_qp(obdir + "/qp1.bin");
    QuarticProof qp2 = read_qp(obdir + "/qp2.bin");
    HadamardProof hp = read_hp(obdir + "/hp.bin");
    Fr_t val_R, val_g, val_W;
    { FILE* f = open_or_die(obdir + "/outer.bin", "rb");
      if (fread(&val_R, sizeof(Fr_t), 1, f) != 1 ||
          fread(&val_g, sizeof(Fr_t), 1, f) != 1 ||
          fread(&val_W, sizeof(Fr_t), 1, f) != 1) { fclose(f); RJ("outer.bin short read"); }
      fclose(f); }
    if (pf.ev.size() != 4 * logDL) RJ("limb lookup round count");
    if (ss.ev.size() != 4 * logD) RJ("SS round count");
    if (qp1.ev.size() != 5 * logB || qp2.ev.size() != 5 * logB) RJ("quartic round count");
    if (hp.ev.size() != 4 * logD) RJ("hadamard round count");

    // ---- (2) homomorphic affine limb links (1-thread helpers; no openings) ----
    {
        vector<G1Jacobian_t> hL(10), hp1(1), hp2(1);
        cudaMemcpy(hL.data(), com_L.gpu_data, 10 * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hp1.data(), com_P1.gpu_data, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hp2.data(), com_P2.gpu_data, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        G1Jacobian_t acc1 = h_mul(hL[0], W16[0]);
        for (uint i = 1; i < 5; i++) acc1 = h_add(acc1, h_mul(hL[i], W16[i]));
        if (!g1_eq(acc1, hp1[0])) RJ("affine link P1 != sum 2^16i * com_L_row[i]");
        G1Jacobian_t acc2 = h_mul(hL[5], W16[0]);
        for (uint i = 1; i < 5; i++) acc2 = h_add(acc2, h_mul(hL[5 + i], W16[i]));
        if (!g1_eq(acc2, hp2[0])) RJ("affine link P2 != sum 2^16i * com_L_row[5+i]");
    }

    // ---- transcript replay ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C);
    absorb_u32(tr, "C_eps_lo", (uint32_t)C_eps);
    absorb_u32(tr, "C_eps_hi", (uint32_t)(C_eps >> 32));
    absorb_g1_tensor(tr, "com_X", com_X);
    absorb_g1_tensor(tr, "com_g", com_g);
    absorb_g1_tensor(tr, "com_R", com_R);
    absorb_g1_tensor(tr, "com_M", com_M);
    absorb_g1_tensor(tr, "com_P1", com_P1);
    absorb_g1_tensor(tr, "com_P2", com_P2);
    absorb_g1_tensor(tr, "com_L", com_L);
    absorb_g1_tensor(tr, "com_m_L", com_mL);
    absorb_g1_tensor(tr, "com_W", com_W);
    absorb_g1_tensor(tr, "com_W_", com_Wr);
    absorb_g1_tensor(tr, "com_Y", com_Y);
    Fr_t beta = fs_challenge_fr(tr);
    absorb_g1_tensor(tr, "com_A_L", com_AL);
    Fr_t alpha = fs_challenge_fr(tr);
    auto u_L = fs_challenge_vec(tr, logDL);

    // ---- (1) limb lookup: anchor recomputed, Lagrange-4 chain ----
    const Fr_t inv6 = inv(F_SIX);
    Fr_t cur = h_scalar(alpha, h_scalar(alpha, alpha, 2), 0);    // alpha + alpha^2
    Fr_t alpha_acc = alpha, alphasq_acc = h_scalar(alpha, alpha, 2);
    vector<Fr_t> wsl;
    for (uint k = 0; k < logDL; k++) {
        array<Fr_t,4> e = {pf.ev[4*k], pf.ev[4*k+1], pf.ev[4*k+2], pf.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("limb lookup round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "p0", e[0]); absorb_fr(tr, "p1", e[1]);
        absorb_fr(tr, "p2", e[2]); absorb_fr(tr, "p3", e[3]);
        Fr_t w = fs_challenge_fr(tr); wsl.push_back(w);
        cur = lagrange4(e, w, inv6);
        Fr_t eqv = my_eq(u_L[logDL - 1 - k], w);
        alpha_acc = h_scalar(alpha_acc, eqv, 2);
        if (k >= n1) alphasq_acc = h_scalar(alphasq_acc, eqv, 2);
    }
    // B_f, T_f recomputed from the PUBLIC range table
    Fr_t B_f, T_f;
    {
        tLookupRange tl(0, N);
        FrTensor B_pub(N);
        tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            tl.table.gpu_data, beta, B_pub.gpu_data, N);
        cudaDeviceSynchronize();
        Fr_t *dB, *dT, *dtmp;
        cudaMalloc(&dB, N * sizeof(Fr_t)); cudaMalloc(&dT, N * sizeof(Fr_t));
        cudaMalloc(&dtmp, N * sizeof(Fr_t));
        cudaMemcpy(dB, B_pub.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dT, tl.table.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
        uint sz = N;
        for (uint k = 0; k < n2; k++) {
            uint nsz = sz >> 1;
            k_fr_fold<<<(nsz + 31) / 32, 32>>>(dB, wsl[n1 + k], dtmp, nsz);
            cudaDeviceSynchronize(); std::swap(dB, dtmp);
            k_fr_fold<<<(nsz + 31) / 32, 32>>>(dT, wsl[n1 + k], dtmp, nsz);
            cudaDeviceSynchronize(); std::swap(dT, dtmp);
            sz = nsz;
        }
        cudaMemcpy(&B_f, dB, sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&T_f, dT, sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaFree(dB); cudaFree(dT); cudaFree(dtmp);
    }
    {
        Fr_t inv_ratio = h_scalar({N,0,0,0,0,0,0,0}, inv({D_L,0,0,0,0,0,0,0}), 2);
        Fr_t t1 = h_scalar(alpha_acc, h_scalar(pf.A_f, h_scalar(pf.S_f, beta, 0), 2), 2);
        Fr_t t2 = h_scalar(inv_ratio, h_scalar(alphasq_acc, h_scalar(B_f, h_scalar(T_f, beta, 0), 2), 2), 2);
        Fr_t t4 = h_scalar(inv_ratio, h_scalar(pf.m_f, B_f, 2), 2);
        Fr_t rhs = h_scalar(h_scalar(h_scalar(t1, t2, 0), pf.A_f, 0), t4, 1);
        if (!fr_eq(cur, rhs)) RJ("limb lookup terminal identity");
    }
    absorb_fr(tr, "A_f", pf.A_f); absorb_fr(tr, "S_f", pf.S_f); absorb_fr(tr, "m_f", pf.m_f);
    vector<Fr_t> u_ptL(wsl.rbegin(), wsl.rend());
    vector<Fr_t> u_mL(wsl.rbegin(), wsl.rend() - n1);
    if (!open_verify(com_AL, gen_B, B_pad, Q, u_ptL, pf.A_f, obdir + "/ipa_AL.bin", tr))
        RJ("IPA opening of A_f vs com_A_L");
    if (!open_verify(com_L, gen_B, B_pad, Q, u_ptL, pf.S_f, obdir + "/ipa_L.bin", tr))
        RJ("IPA opening of S_f vs com_L");
    if (!open_verify(com_mL, gen_B, B_pad, Q, u_mL, pf.m_f, obdir + "/ipa_mL.bin", tr))
        RJ("IPA opening of m_f vs com_m_L");

    // ---- (3) SS sumcheck ----
    auto u_b = fs_challenge_vec(tr, logB);
    absorb_fr(tr, "ev_M", ss.claim_H);
    Fr_t C_eps_fr = {(uint32_t)C_eps, (uint32_t)(C_eps >> 32), 0, 0, 0, 0, 0, 0};
    Fr_t curs = h_scalar(ss.claim_H, C_eps_fr, 1);
    Fr_t eq_acc = F_ONE;
    vector<Fr_t> ws_ss;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {ss.ev[4*k], ss.ev[4*k+1], ss.ev[4*k+2], ss.ev[4*k+3]};
        if (!fr_eq(curs, h_scalar(e[0], e[1], 0)))
            RJ("SS round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_ss.push_back(w);
        curs = lagrange4(e, w, inv6);
        if (k < logB)   // row bits are the MSBs; column rounds carry eq factor 1
            eq_acc = h_scalar(eq_acc, my_eq(u_b[logB - 1 - k], w), 2);
    }
    if (!fr_eq(ss.S_f2, ss.U_f2)) RJ("SS S_f2 != U_f2");
    if (!fr_eq(curs, h_scalar(eq_acc, h_scalar(ss.S_f2, ss.U_f2, 2), 2)))
        RJ("SS terminal identity");
    absorb_fr(tr, "S_f2", ss.S_f2); absorb_fr(tr, "U_f2", ss.U_f2);
    vector<Fr_t> u_ss(ws_ss.rbegin(), ws_ss.rend());
    if (!open_verify(com_X, gen_C, C_pad, Q, u_ss, ss.S_f2, obdir + "/ipa_X_ss.bin", tr))
        RJ("IPA opening of X (SS) vs com_X");
    if (!open_verify(com_M, gen_B, B_pad, Q, u_b, ss.claim_H, obdir + "/ipa_M.bin", tr))
        RJ("IPA opening of ev_M vs com_M");

    // ---- (4) bracket quartics ----
    auto u_b2 = fs_challenge_vec(tr, logB);
    absorb_fr(tr, "ev_P1", qp1.claim0);
    absorb_fr(tr, "ev_P2", qp2.claim0);
    Fr_t c64C_fr = {0, 0, C, 0, 0, 0, 0, 0};
    vector<Fr_t> ws_q1, ws_q2;
    {
        Fr_t curq = h_scalar(c64C_fr, qp1.claim0, 1);   // 2^64 C - ev_P1
        Fr_t eq1 = F_ONE;
        for (uint k = 0; k < logB; k++) {
            array<Fr_t,5> e = {qp1.ev[5*k], qp1.ev[5*k+1], qp1.ev[5*k+2],
                               qp1.ev[5*k+3], qp1.ev[5*k+4]};
            if (!fr_eq(curq, h_scalar(e[0], e[1], 0)))
                RJ("q1 round " << k << " p(0)+p(1) != claim");
            for (int i = 0; i < 5; i++)
                absorb_fr(tr, ("q1p" + to_string(i)).c_str(), e[i]);
            Fr_t w = fs_challenge_fr(tr); ws_q1.push_back(w);
            curq = lagrange5(e, w);
            eq1 = h_scalar(eq1, my_eq(u_b2[logB - 1 - k], w), 2);
        }
        if (!fr_eq(qp1.A_f, qp1.B_f)) RJ("q1 A_f != B_f");
        Fr_t rhs = h_scalar(eq1, h_scalar(h_scalar(qp1.A_f, qp1.B_f, 2), qp1.C_f, 2), 2);
        if (!fr_eq(curq, rhs)) RJ("q1 terminal identity");
    }
    {
        Fr_t curq = h_scalar(qp2.claim0, c64C_fr, 0);   // ev_P2 + 2^64 C
        Fr_t eq2 = F_ONE;
        for (uint k = 0; k < logB; k++) {
            array<Fr_t,5> e = {qp2.ev[5*k], qp2.ev[5*k+1], qp2.ev[5*k+2],
                               qp2.ev[5*k+3], qp2.ev[5*k+4]};
            if (!fr_eq(curq, h_scalar(e[0], e[1], 0)))
                RJ("q2 round " << k << " p(0)+p(1) != claim");
            for (int i = 0; i < 5; i++)
                absorb_fr(tr, ("q2p" + to_string(i)).c_str(), e[i]);
            Fr_t w = fs_challenge_fr(tr); ws_q2.push_back(w);
            curq = lagrange5(e, w);
            eq2 = h_scalar(eq2, my_eq(u_b2[logB - 1 - k], w), 2);
        }
        if (!fr_eq(qp2.A_f, qp2.B_f)) RJ("q2 A_f != B_f");
        Fr_t rhs = h_scalar(eq2, h_scalar(h_scalar(qp2.A_f, qp2.B_f, 2), qp2.C_f, 2), 2);
        if (!fr_eq(curq, rhs)) RJ("q2 terminal identity");
    }
    absorb_fr(tr, "q1A_f", qp1.A_f); absorb_fr(tr, "q1B_f", qp1.B_f);
    absorb_fr(tr, "q1C_f", qp1.C_f);
    absorb_fr(tr, "q2A_f", qp2.A_f); absorb_fr(tr, "q2B_f", qp2.B_f);
    absorb_fr(tr, "q2C_f", qp2.C_f);
    vector<Fr_t> pt1(ws_q1.rbegin(), ws_q1.rend());
    vector<Fr_t> pt2(ws_q2.rbegin(), ws_q2.rend());
    if (!open_verify(com_P1, gen_B, B_pad, Q, u_b2, qp1.claim0, obdir + "/ipa_P1.bin", tr))
        RJ("IPA opening of ev_P1 vs com_P1");
    if (!open_verify(com_P2, gen_B, B_pad, Q, u_b2, qp2.claim0, obdir + "/ipa_P2.bin", tr))
        RJ("IPA opening of ev_P2 vs com_P2");
    // MLE of the constant 1 is 1: R~(pt1) must equal T1~(pt1) + 1, etc.
    if (!open_verify(com_R, gen_B, B_pad, Q, pt1, h_scalar(qp1.A_f, F_ONE, 0),
                     obdir + "/ipa_R_q1.bin", tr))
        RJ("IPA opening of R@pt1 (expect q1.A_f + 1) vs com_R");
    if (!open_verify(com_M, gen_B, B_pad, Q, pt1, qp1.C_f, obdir + "/ipa_M_q1.bin", tr))
        RJ("IPA opening of M@pt1 vs com_M");
    if (!open_verify(com_R, gen_B, B_pad, Q, pt2, h_scalar(qp2.A_f, F_ONE, 1),
                     obdir + "/ipa_R_q2.bin", tr))
        RJ("IPA opening of R@pt2 (expect q2.A_f - 1) vs com_R");
    if (!open_verify(com_M, gen_B, B_pad, Q, pt2, qp2.C_f, obdir + "/ipa_M_q2.bin", tr))
        RJ("IPA opening of M@pt2 vs com_M");

    // ---- (5) outer product ----
    auto u_b3 = fs_challenge_vec(tr, logB);
    auto u_c3 = fs_challenge_vec(tr, logC);
    absorb_fr(tr, "val_R", val_R); absorb_fr(tr, "val_g", val_g);
    absorb_fr(tr, "val_W", val_W);
    if (!fr_eq(val_W, h_scalar(val_R, val_g, 2)))
        RJ("outer product check val_W != val_R * val_g");
    vector<Fr_t> u_pt3(u_c3);
    u_pt3.insert(u_pt3.end(), u_b3.begin(), u_b3.end());
    if (!open_verify(com_R, gen_B, B_pad, Q, u_b3, val_R, obdir + "/ipa_R_o.bin", tr))
        RJ("IPA opening of val_R vs com_R");
    if (!open_verify(com_g, gen_C, C_pad, Q, u_c3, val_g, obdir + "/ipa_g.bin", tr))
        RJ("IPA opening of val_g vs registered com_g");
    if (!open_verify(com_W, gen_C, C_pad, Q, u_pt3, val_W, obdir + "/ipa_W.bin", tr))
        RJ("IPA opening of val_W vs com_W");

    // ---- (7) hadamard Y = W_ .* X ----
    auto u_h = fs_challenge_vec(tr, logD);
    absorb_fr(tr, "claim_Y", hp.claim_H);
    Fr_t curh = hp.claim_H, eq_h = F_ONE;
    vector<Fr_t> wsh;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp.ev[4*k], hp.ev[4*k+1], hp.ev[4*k+2], hp.ev[4*k+3]};
        if (!fr_eq(curh, h_scalar(e[0], e[1], 0)))
            RJ("hadamard round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); wsh.push_back(w);
        curh = lagrange4(e, w, inv6);
        eq_h = h_scalar(eq_h, my_eq(u_h[logD - 1 - k], w), 2);
    }
    if (!fr_eq(curh, h_scalar(eq_h, h_scalar(hp.S_f2, hp.U_f2, 2), 2)))
        RJ("hadamard terminal identity");
    absorb_fr(tr, "S_f2h", hp.S_f2); absorb_fr(tr, "U_f2h", hp.U_f2);
    vector<Fr_t> u_pth(wsh.rbegin(), wsh.rend());
    if (!open_verify(com_Y, gen_C, C_pad, Q, u_h, hp.claim_H, obdir + "/ipa_Y.bin", tr))
        RJ("IPA opening of claim_Y vs com_Y");
    if (!open_verify(com_Wr, gen_C, C_pad, Q, u_pth, hp.S_f2, obdir + "/ipa_Wr.bin", tr))
        RJ("IPA opening of W_ terminal vs com_W_");
    if (!open_verify(com_X, gen_C, C_pad, Q, u_pth, hp.U_f2, obdir + "/ipa_X_h.bin", tr))
        RJ("IPA opening of X (hadamard) vs com_X");
    cout << "ACCEPT" << endl;
    return true;
}

// ---------------- selftest ----------------
static void tamper_byte(const string& path, long offset, int delta) {
    FILE* f = open_or_die(path, "rb+");
    fseek(f, offset, offset >= 0 ? SEEK_SET : SEEK_END);
    int c = fgetc(f);
    fseek(f, -1, SEEK_CUR);
    fputc((c + delta) & 0xff, f);
    fclose(f);
}
static long file_size(const string& path) {
    struct stat st;
    if (stat(path.c_str(), &st)) return 0;
    return (long)st.st_size;
}
// largest r with r^2 * M <= 2^64 * C (exact, __int128)
static int exact_R(u128 M, uint C) {
    u128 c64C = ((u128)C) << 64;
    long r = (long)sqrt((double)(c64C / M));
    if (r < 1) r = 1;
    while ((u128)(r + 1) * (u128)(r + 1) * M <= c64C) r++;
    while (r > 0 && (u128)r * (u128)r * M > c64C) r--;
    if (r < 1 || r > 0x7fffffffL) throw runtime_error("exact_R out of int32 range");
    return (int)r;
}
static void make_inputs(uint B, uint C, unsigned long long C_eps, int xmag, int gmag,
                        vector<int>& Xh, vector<int>& Rh, vector<int>& gh) {
    Xh.resize((size_t)B * C); Rh.resize(B); gh.resize(C);
    for (size_t i = 0; i < Xh.size(); i++) Xh[i] = (rand() % (2 * xmag)) - xmag;
    for (uint j = 0; j < C; j++) gh[j] = (rand() % (2 * gmag)) - gmag;
    for (uint s = 0; s < B; s++) {
        u128 M = C_eps;
        for (uint j = 0; j < C; j++)
            M += (u128)((long)Xh[s*C+j] * (long)Xh[s*C+j]);
        Rh[s] = exact_R(M, C);
    }
}

static const char* PROOF_FILES[] = {
    "dims.bin", "lookup.bin", "hpss.bin", "qp1.bin", "qp2.bin", "outer.bin", "hp.bin",
    "com_X.bin", "com_g.bin", "com_R.bin", "com_M.bin", "com_P1.bin", "com_P2.bin",
    "com_L.bin", "com_m_L.bin", "com_A_L.bin", "com_W.bin", "com_Wr.bin", "com_Y.bin",
    "ipa_AL.bin", "ipa_L.bin", "ipa_mL.bin", "ipa_X_ss.bin", "ipa_M.bin",
    "ipa_P1.bin", "ipa_P2.bin", "ipa_R_q1.bin", "ipa_M_q1.bin", "ipa_R_q2.bin",
    "ipa_M_q2.bin", "ipa_R_o.bin", "ipa_g.bin", "ipa_W.bin", "ipa_Y.bin",
    "ipa_Wr.bin", "ipa_X_h.bin"};
static long tamper_offset(const string& f) {
    if (f == "dims.bin") return 0;
    if (f == "lookup.bin") return 4 + 32;           // round-0 p(1)
    if (f.substr(0, 4) == "ipa_") return -32;       // a_final
    if (f.substr(0, 4) == "com_") return 24;        // first point, x limbs
    return 4;                                       // proof files: first Fr body byte
}

static bool selftest_small() {
    const uint B = 8, C = 5;
    const unsigned long long C_eps = 21475;   // round(1e-6 * 5 * 2^32)
    cout << "==== selftest small case B=" << B << " C=" << C
         << " C_eps=" << C_eps << " ====" << endl;
    const uint B_pad = 1u << ceilLog2(B), C_pad = 1u << ceilLog2(C);
    srand(4242);
    vector<int> Xh, Rh, gh;
    make_inputs(B, C, C_eps, 1 << 12, 1 << 8, Xh, Rh, gh);
    Commitment gen_C = Commitment::random(C_pad);
    Commitment gen_B = Commitment::random(B_pad);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_rmsnorm_ob";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:rmsnorm";
    string com_g_path = obdir + "/com_g.bin";
    bool all = true;

    prove(obdir, seed, Xh, Rh, gh, B, C, C_eps, gen_C, gen_B, Q,
          "/tmp/zkob_rmsnorm_W.i64.bin", "/tmp/zkob_rmsnorm_Y.i64.bin");
    bool honest = verify(obdir, seed, B, C, C_eps, com_g_path, gen_C, gen_B, Q);
    cout << (honest ? "PASS" : "FAIL") << ": honest small case "
         << (honest ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && honest;

    // semantic evil modes: each must be rejected by EXACTLY the named check
    struct Evil { int mode; uint idx; const char* expect; const char* what; };
    vector<Evil> evils = {
        {1, 3, "affine link P1", "R+2, P1 mod p, limbs truncated"},
        {2, 3, "q1 round 0",     "R+2, P1 = limb reconstruction"},
        {3, 2, "SS round 0",     "M[i]+1, brackets recomputed"},
        {4, 5, "hadamard round 0", "Y[i]+1"},
        {5, 6, "outer product",  "W[i]+1"},
        {6, 3, "affine link P2", "R-2, P2 mod p, limbs truncated"},
        {7, 3, "q2 round 0",     "R-2, P2 = limb reconstruction"},
        {8, 3, "limb lookup round 0", "L[0,i]+2^16 with borrow, P1 unchanged"},
    };
    string evdir = "/tmp/zkob_rmsnorm_evil";
    mkdir(evdir.c_str(), 0755);
    for (auto& ev : evils) {
        prove(evdir, seed, Xh, Rh, gh, B, C, C_eps, gen_C, gen_B, Q, "", "",
              ev.mode, ev.idx);
        string reason;
        bool rejected = !verify(evdir, seed, B, C, C_eps, evdir + "/com_g.bin",
                                gen_C, gen_B, Q, &reason);
        bool right = rejected && reason.find(ev.expect) != string::npos;
        cout << (right ? "PASS" : "FAIL") << ": evil=" << ev.mode << " (" << ev.what
             << ") rejected by [" << (rejected ? reason : string("NOT REJECTED"))
             << "], expected [" << ev.expect << "]" << endl;
        all = all && right;
    }

    // byte tampers on every proof / commitment / ipa file (tamper, verify, restore)
    for (const char* fn : PROOF_FILES) {
        long off = tamper_offset(fn);
        tamper_byte(obdir + "/" + fn, off, +1);
        bool rejected = !verify(obdir, seed, B, C, C_eps, com_g_path, gen_C, gen_B, Q);
        tamper_byte(obdir + "/" + fn, off, -1);
        cout << (rejected ? "PASS" : "FAIL") << ": byte tamper " << fn << "@" << off
             << " rejected: " << (rejected ? "YES" : "NO(!!)") << endl;
        all = all && rejected;
    }
    bool restored = verify(obdir, seed, B, C, C_eps, com_g_path, gen_C, gen_B, Q);
    cout << (restored ? "PASS" : "FAIL") << ": restored verify "
         << (restored ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && restored;
    cout << (all ? "SMALL CASE: PASS" : "SMALL CASE: FAIL") << endl;
    return all;
}

static bool selftest_real() {
    const uint B = 1024, C = 768;
    // llama-68m rms_norm_eps = 1e-6 (model config, checked):
    // C_eps = round(1e-6 * 768 * 2^32) = 3298535
    const unsigned long long C_eps = 3298535ULL;
    cout << "==== selftest real-scale case B=" << B << " C=" << C
         << " C_eps=" << C_eps << " ====" << endl;
    const string genpath = "/tmp/gen1024.bin";
    if (file_size(genpath) != 1024L * (long)sizeof(G1Jacobian_t)) {
        cout << "generating " << genpath << " via ppgen..." << endl;
        if (system(("./ppgen 1024 " + genpath).c_str()))
            throw runtime_error("ppgen failed");
    }
    Commitment gen(genpath);                       // C_pad = B_pad = 1024
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    srand(20260610);
    vector<int> Xh, Rh, gh;
    make_inputs(B, C, C_eps, 1 << 16, 1 << 15, Xh, Rh, gh);
    string obdir = "/tmp/zkob_rmsnorm_real";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:rmsnorm:real";
    bool all = true;

    auto t0 = chrono::steady_clock::now();
    prove(obdir, seed, Xh, Rh, gh, B, C, C_eps, gen, gen, Q,
          "/tmp/zkob_rmsnorm_real_W.i64.bin", "/tmp/zkob_rmsnorm_real_Y.i64.bin");
    auto t1 = chrono::steady_clock::now();
    bool honest = verify(obdir, seed, B, C, C_eps, obdir + "/com_g.bin", gen, gen, Q);
    auto t2 = chrono::steady_clock::now();
    double prove_s = chrono::duration<double>(t1 - t0).count();
    double verify_s = chrono::duration<double>(t2 - t1).count();
    long bytes = 0;
    for (const char* fn : PROOF_FILES) bytes += file_size(obdir + "/" + string(fn));
    cout << (honest ? "PASS" : "FAIL") << ": real-scale honest "
         << (honest ? "ACCEPT" : "REJECT(!!)") << "  prove " << prove_s
         << " s, verify " << verify_s << " s, proof+commitments " << bytes
         << " bytes" << endl;
    all = all && honest;

    tamper_byte(obdir + "/lookup.bin", 4 + 32, +1);
    bool rejected = !verify(obdir, seed, B, C, C_eps, obdir + "/com_g.bin", gen, gen, Q);
    tamper_byte(obdir + "/lookup.bin", 4 + 32, -1);
    cout << (rejected ? "PASS" : "FAIL") << ": real-scale byte tamper rejected: "
         << (rejected ? "YES" : "NO(!!)") << endl;
    all = all && rejected;
    cout << (all ? "REAL-SCALE CASE: PASS" : "REAL-SCALE CASE: FAIL") << endl;
    return all;
}

static vector<int> load_i32(const string& path, uint expect) {
    FILE* f = open_or_die(path, "rb");
    vector<int> v(expect);
    if (fread(v.data(), sizeof(int), expect, f) != expect)
        throw runtime_error("short read / wrong dims: " + path);
    fclose(f);
    return v;
}

int main(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    if (mode == "selftest") {
        bool a = selftest_small();
        bool b = selftest_real();
        cout << ((a && b) ? "ZKOB-RMSNORM SELFTEST: ALL PASS"
                          : "ZKOB-RMSNORM SELFTEST: FAIL") << endl;
        return (a && b) ? 0 : 1;
    }
    if (mode == "prove" && (argc == 13 || argc == 15)) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[7]), C = stoi(argv[8]);
        unsigned long long C_eps = stoull(argv[9]);
        vector<int> Xh = load_i32(argv[4], B * C);
        vector<int> Rh = load_i32(argv[5], B);
        vector<int> gh = load_i32(argv[6], C);
        Commitment gen_C(argv[10]), gen_B(argv[11]), qg(argv[12]);
        prove(obdir, seed, Xh, Rh, gh, B, C, C_eps, gen_C, gen_B, qg(0),
              argc == 15 ? argv[13] : "", argc == 15 ? argv[14] : "");
        return 0;
    }
    if (mode == "verify" && argc == 11) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[4]), C = stoi(argv[5]);
        unsigned long long C_eps = stoull(argv[6]);
        Commitment gen_C(argv[8]), gen_B(argv[9]), qg(argv[10]);
        return verify(obdir, seed, B, C, C_eps, argv[7], gen_C, gen_B, qg(0)) ? 0 : 1;
    }
    cerr << "usage: zkob_rmsnorm selftest\n"
         << "       zkob_rmsnorm prove  <obdir> <seed> <X-int32> <R-int32> <g-int32> <B> <C> <C_eps-u64> <gen_C> <gen_B> <q> [W-i64-out Y-i64-out]\n"
         << "       zkob_rmsnorm verify <obdir> <seed> <B> <C> <C_eps> <com_g-path> <gen_C> <gen_B> <q>" << endl;
    return 2;
}
