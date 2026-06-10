// Real driver for ONE softmax obligation (layer{l}.attn.softmax.h{hh}), per
// SOFTMAX_DESIGN.md (DESIGN FINAL 2026-06-10). Binds, for the B x NCOL score
// grid z_ (int32, scale 2^9, chained from scores_rescale10):
//   (R1) z_[i,j] in [LOW_E, LOW_E+LEN_E) and E[i,j] = X_E[z_[i,j] - LOW_E]
//        -- glu-style mapping lookup on comb = z_ + r*E vs table + r*mapped;
//        the verifier forms com_comb = com_z + r*com_E homomorphically.
//   (R2) S[i] = sum_{j<=i} E[i,j]
//        -- row-sum sumcheck: ev_S = S~(u_b) = sum_b W_rs(b)*E(b)*1(b) with
//        W_rs = bcast_rows(eq(u_b)) .* MK (MK = public causal mask, j<=i);
//        verifier recomputes the public weight fold itself, requires U_f2 == 1.
//   (R3) r1 = 2^17*MK*E + S_bcast - 2*P*S_bcast in [0, 2S) per entry
//        <=> P = round_half_up(2^16*MK*E/S) exactly, P[masked] = 0.
//        Bound by: V1 (c1 = MLE of MK.*E at u_r), V2 (c2 = MLE of P.*S_bcast
//        at u_r; S_bcast never committed -- its terminal U_f2 opens against
//        com_S at the row-bit suffix of pt2), 4 plane openings of the limb
//        tensor L (planes: r1 lo, r1 hi, r2 lo, r2 hi; r2 = 2S-1-r1) + S_id,
//        the limb range lookup (L vs tLookupRange(0, LEN_R)), and the two
//        plain-field identities
//          (I1) 2^17*c1 + S_id - 2*c2 == v00 + LEN_R*v10   "bracket r1 identity"
//          (I2) r~1 + r~2 + 1 == 2*S_id                    "bracket sum identity"
// Zero prover advice; all committed tensors are deterministic in (z_, MK, X_E).
//
// Layout: B == NCOL required (both pow2); one generator set gen (size NCOL)
// commits every tensor row-wise, no padding anywhere. Flat grid index =
// i*NCOL + j; L flat index = plane*B*NCOL + i*NCOL + j (plane = 2 MSBs).
// Exp lookup: D = B*NCOL, N = LEN_E, NCOL <= N <= D, N | D.
// Limb lookup: D_L = 4*D, N = LEN_R, NCOL <= N <= D_L, N | D_L.
//
// FS schedule (one transcript, seed = run_seed:obligation_id) -- see
// SOFTMAX_DESIGN.md section 5; every challenge derived only after absorbing
// what it binds; each IPA absorbs its own L/R points round-by-round.
//
// Usage:
//   zkob_softmax prove  <obdir> <seed> <z-int32.bin> <B> <NCOL> <LOW_E> <LEN_E>
//                       <expmap-int32.bin> <LEN_R> <gen.bin> <q.bin> [P-int32-out.bin]
//   zkob_softmax verify <obdir> <seed> <B> <NCOL> <LOW_E> <LEN_E>
//                       <expmap-int32.bin> <LEN_R> <gen.bin> <q.bin>
//   zkob_softmax selftest
#include "zkob_lookup.cuh"
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <sys/stat.h>
using namespace std;

static_assert(sizeof(long) == 8, "host math needs 64-bit long");

static const uint LOG_OUT = 16;   // P at scale 2^16 (pinned)

// load the LEN_E mapped exp-table values (int32). The (LOW_E, LEN_E) range is
// implicit, glu table convention -- but do NOT port glu's mapped(0)==0 check:
// there is no padding here, so no (0, mapped(0)) table row is ever fabricated
// (mapped(0) = X_E[-LOW_E] = 65536 for the real table).
static vector<int> load_expmap(const string& path, uint len) {
    if (len != (1u << ceilLog2(len))) throw runtime_error("LEN_E must be pow2");
    FILE* f = open_or_die(path, "rb");
    vector<int> buf(len);
    if (fread(buf.data(), sizeof(int), len, f) != len)
        throw runtime_error("expmap table: short read");
    fclose(f);
    return buf;
}

static void layout_guards(uint B, uint NCOL, uint LEN_E, uint LEN_R, uint gen_size) {
    const uint D = B * NCOL, DL = 4 * D;
    if (B != NCOL) throw runtime_error("B != NCOL");
    if (B != (1u << ceilLog2(B))) throw runtime_error("B not a power of two");
    if (gen_size != NCOL) throw runtime_error("generator size != NCOL");
    if (LEN_E != (1u << ceilLog2(LEN_E)) || LEN_R != (1u << ceilLog2(LEN_R)))
        throw runtime_error("table lengths must be pow2");
    if (NCOL > LEN_E || LEN_E > D || D % LEN_E)
        throw runtime_error("exp lookup layout needs NCOL <= LEN_E <= D, LEN_E | D");
    if (NCOL > LEN_R || LEN_R > DL || DL % LEN_R)
        throw runtime_error("limb lookup layout needs NCOL <= LEN_R <= 4D, LEN_R | 4D");
}

// ---------------- prove ----------------
// evil (selftest only; honest PROCEDURE on an inconsistent witness):
//  1: E[i,j] += 1 at an UNMASKED (i,j); S, P, r1/r2, limbs recomputed from the
//     evil E -> only the exp lookup can reject (round 0).
//  2: P[i,j] += 1 at a MASKED (i,j); true r1 = -S < 0; r1 limbs = mod-LEN_R^2
//     truncation, r2 limbs = mod-LEN_R^2 of 2S-1-(truncated r1); all
//     recursions self-consistent -> "bracket r1 identity" (I1) rejects.
//  3: P[i,j] -= 1 at an unmasked diagonal entry with honest P >= 1;
//     r1' = r1 + 2S < LEN_R^2 representable, limbs honest for r1'; r2 limbs =
//     mod truncation of the (negative) 2S-1-r1' -> I1 holds exactly, the limb
//     lookup passes -> only "bracket sum identity" (I2) rejects.
//  4: S[row] += 1; P, r1/r2, limbs recomputed from the evil S -> row-sum
//     round 0 rejects (ev_S no longer equals sum W_rs.*E).
//  5: V2 run with a corrupted broadcast buffer Sb[idx] += 1 (P, S, limbs all
//     honest) -> "IPA opening of V2 U_f2 vs com_S" rejects.
//  6: L[0,idx] += LEN_R with a compensating borrow L[1,idx] -= 1 (out-of-range
//     lo limb; the reconstructed r1 -- hence I1/I2, every opening and every
//     commitment except com_L/com_A_L -- stays consistent; m_L committed from
//     the HONEST limbs) -> only the limb range lookup can reject (round 0).
static void prove(const string& obdir, const string& seed,
                  const vector<int>& zh, uint B, uint NCOL, int LOW_E, uint LEN_E,
                  const vector<int>& maph, uint LEN_R,
                  const Commitment& gen, const G1Jacobian_t& Q,
                  const string& p_out,
                  int evil = 0, uint evil_i = 0, uint evil_j = 0) {
    const uint D = B * NCOL, DL = 4 * D;
    const uint logB = ceilLog2(B), logC = ceilLog2(NCOL);
    const uint logD = logB + logC, logDL = logD + 2;
    const uint n1_E = ceilLog2(D / LEN_E), n1_L = ceilLog2(DL / LEN_R);
    layout_guards(B, NCOL, LEN_E, LEN_R, gen.size);
    if (zh.size() != (size_t)D || maph.size() != (size_t)LEN_E)
        throw runtime_error("input dims");
    const long long LENR2 = (long long)LEN_R * (long long)LEN_R;

    // ---- host integer chain (long long; all bounds < 2^63 per design §6) ----
    vector<long> Eh(D), Sh(B), Ph(D), Lh(DL);
    for (uint i = 0; i < B; i++)
        for (uint j = 0; j < NCOL; j++) {
            long long z = zh[(size_t)i * NCOL + j];
            if (z < (long long)LOW_E || z >= (long long)LOW_E + (long long)LEN_E)
                throw runtime_error("z_ out of table domain [LOW_E, LOW_E+LEN_E)");
            Eh[(size_t)i * NCOL + j] = maph[z - LOW_E];
        }
    if (evil == 1) {
        if (evil_j > evil_i) throw runtime_error("evil 1 setup: idx must be unmasked");
        Eh[(size_t)evil_i * NCOL + evil_j] += 1;
    }
    for (uint i = 0; i < B; i++) {
        long long s = 0;
        for (uint j = 0; j <= i && j < NCOL; j++) s += Eh[(size_t)i * NCOL + j];
        Sh[i] = s;
    }
    if (evil == 4) Sh[evil_i] += 1;
    for (uint i = 0; i < B; i++)
        if (Sh[i] < 1) throw runtime_error("S[i] < 1 (table floor violated)");
    for (uint i = 0; i < B; i++)
        for (uint j = 0; j < NCOL; j++) {
            const size_t idx = (size_t)i * NCOL + j;
            const bool mk = (j <= i);
            const long long S = Sh[i];
            const long long num = (mk ? ((long long)Eh[idx] << (LOG_OUT + 1)) : 0LL) + S;
            long long P = num / (2 * S);          // floor; round-half-up of 2^16*E/S
            const bool ev_here = (i == evil_i && j == evil_j);
            if (evil == 2 && ev_here) {
                if (mk) throw runtime_error("evil 2 setup: idx must be masked");
                P += 1;
            }
            if (evil == 3 && ev_here) {
                if (!mk || i != j) throw runtime_error("evil 3 setup: pick a diagonal idx");
                if (P < 1) throw runtime_error("evil 3 setup: honest P < 1");
                P -= 1;
            }
            const long long r1 = num - 2 * P * S;            // true bracket value
            long long r1c, r2c;
            if ((evil == 2 || evil == 3) && ev_here) {
                r1c = ((r1 % LENR2) + LENR2) % LENR2;        // mod-LEN_R^2 truncation
                long long r2t = 2 * S - 1 - r1c;
                r2c = ((r2t % LENR2) + LENR2) % LENR2;
            } else {
                if (r1 < 0 || r1 >= 2 * S)
                    throw runtime_error("r1 out of [0, 2S) (P not per spec)");
                if (r1 >= LENR2 || 2 * S - 1 - r1 >= LENR2)
                    throw runtime_error("bracket residual >= LEN_R^2");
                r1c = r1;
                r2c = 2 * S - 1 - r1;
            }
            Ph[idx] = P;
            Lh[0 * (size_t)D + idx] = r1c % LEN_R;
            Lh[1 * (size_t)D + idx] = r1c / LEN_R;
            Lh[2 * (size_t)D + idx] = r2c % LEN_R;
            Lh[3 * (size_t)D + idx] = r2c / LEN_R;
        }
    // evil 6: out-of-range limb with a compensating borrow -- the reconstructed
    // r1 (and so I1/I2 and every commitment except com_L/com_A_L) is UNCHANGED;
    // only the limb range lookup stands between this forgery and ACCEPT.
    vector<long> Lh_honest;
    if (evil == 6) {
        size_t idx = (size_t)evil_i * NCOL + evil_j;
        if (Lh[(size_t)D + idx] < 1)      // need r1 >= LEN_R: scan for a hi limb >= 1
            for (idx = 0; idx < D && Lh[(size_t)D + idx] < 1; idx++);
        if (idx >= (size_t)D)
            throw runtime_error("evil 6 setup: no entry with r1 >= LEN_R");
        Lh_honest = Lh;
        Lh[idx] += (long)LEN_R;           // plane 0: r1 lo limb leaves [0, LEN_R)
        Lh[(size_t)D + idx] -= 1;         // plane 1 borrow: same r1 value
    }
    if (!p_out.empty()) {       // chain file: unpadded B x NCOL int32, scale 2^16
        vector<int> pi(D);
        for (size_t k = 0; k < D; k++) pi[k] = (int)Ph[k];
        FILE* f = open_or_die(p_out, "wb");
        fwrite(pi.data(), sizeof(int), D, f); fclose(f);
    }

    // ---- tensors (all PLAIN Fr values) ----
    FrTensor z_t(D, zh.data());
    FrTensor E_t(D, Eh.data());
    FrTensor S_t(B, Sh.data());
    FrTensor P_t(D, Ph.data());
    FrTensor L_t(DL, Lh.data());
    vector<int> mkh(D), onesh(D, 1);
    for (uint i = 0; i < B; i++)
        for (uint j = 0; j < NCOL; j++) mkh[(size_t)i * NCOL + j] = (j <= i) ? 1 : 0;
    FrTensor MK_t(D, mkh.data());
    FrTensor ones_t(D, onesh.data());
    FrTensor Sb(D);                               // broadcast grid, NEVER committed
    k_bcast_rows<<<(D + 255) / 256, 256>>>(S_t.gpu_data, Sb.gpu_data, NCOL, D);
    cudaDeviceSynchronize();
    if (evil == 5) {
        k_bump<<<1,1>>>(Sb.gpu_data, evil_i * NCOL + evil_j, F_ONE, 0);
        cudaDeviceSynchronize();
    }

    FrTensor mapped_t(LEN_E, maph.data());
    tLookupRangeMapping tlE(LOW_E, LEN_E, mapped_t);
    FrTensor m_E = tlE.prep(z_t);                 // multiplicities of the z indices
    tLookupRange tlR(0, LEN_R);
    // limb multiplicities vs the public range table (evil 6: from the HONEST
    // limbs -- the forged value is outside the table, where prep's unchecked
    // atomicAdd would write out of bounds; committing the honest multiplicities
    // is also the forging prover's best move)
    FrTensor m_L = (evil == 6) ? tlR.prep(FrTensor(DL, Lh_honest.data()))
                               : tlR.prep(L_t);

    // ---- commitments (single gen, no padding) ----
    G1TensorJacobian com_z  = gen.commit(z_t);
    G1TensorJacobian com_E  = gen.commit(E_t);
    G1TensorJacobian com_P  = gen.commit(P_t);
    G1TensorJacobian com_S  = gen.commit(S_t);
    G1TensorJacobian com_L  = gen.commit(L_t);
    G1TensorJacobian com_mE = gen.commit(m_E);
    G1TensorJacobian com_mL = gen.commit(m_L);
    com_z.save(obdir + "/com_z.bin");
    com_E.save(obdir + "/com_E.bin");
    com_P.save(obdir + "/com_P.bin");
    com_S.save(obdir + "/com_S.bin");
    com_L.save(obdir + "/com_L.bin");
    com_mE.save(obdir + "/com_m_E.bin");
    com_mL.save(obdir + "/com_m_L.bin");
    { FILE* f = open_or_die(obdir + "/dims.bin", "wb");
      uint32_t d0[2] = {B, NCOL}; int32_t lo = LOW_E;
      uint32_t d1[3] = {LEN_E, LEN_R, LOG_OUT};
      fwrite(d0, sizeof(uint32_t), 2, f); fwrite(&lo, sizeof(int32_t), 1, f);
      fwrite(d1, sizeof(uint32_t), 3, f); fclose(f); }

    // ---- transcript ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "NCOL", NCOL);
    absorb_u32(tr, "LOW_E", (uint32_t)LOW_E); absorb_u32(tr, "LEN_E", LEN_E);
    absorb_u32(tr, "LEN_R", LEN_R); absorb_u32(tr, "LOG_OUT", LOG_OUT);
    absorb_g1_tensor(tr, "com_z", com_z);
    absorb_g1_tensor(tr, "com_E", com_E);
    absorb_g1_tensor(tr, "com_P", com_P);
    absorb_g1_tensor(tr, "com_S", com_S);
    absorb_g1_tensor(tr, "com_L", com_L);
    absorb_g1_tensor(tr, "com_m_E", com_mE);
    absorb_g1_tensor(tr, "com_m_L", com_mL);
    Fr_t r = fs_challenge_fr(tr);                 // exp pair combiner
    Fr_t beta_E = fs_challenge_fr(tr);

    // ---- obligation 1: exp mapping lookup ----
    FrTensor comb = z_t + E_t * r;
    FrTensor T_comb = tlE.table + tlE.mapped_vals * r;
    FrTensor A_E(D);
    tlookup_inv_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        comb.gpu_data, beta_E, A_E.gpu_data, D);
    cudaDeviceSynchronize();
    G1TensorJacobian com_AE = gen.commit(A_E);
    com_AE.save(obdir + "/com_A_E.bin");
    absorb_g1_tensor(tr, "com_A_E", com_AE);
    Fr_t alpha_E = fs_challenge_fr(tr);
    auto u_E = fs_challenge_vec(tr, logD);

    LookupProof pfE;
    vector<Fr_t> ws_E;
    {
        FrTensor BvE(LEN_E);
        tlookup_inv_kernel<<<(LEN_E+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            T_comb.gpu_data, beta_E, BvE.gpu_data, LEN_E);
        cudaDeviceSynchronize();
        Fr_t alpha_sq = alpha_E * alpha_E;
        Fr_t Cc = alpha_sq - (BvE * m_E).sum();
        Fr_t claim = alpha_E + alpha_sq;
        Fr_t inv_ratio = Fr_t{LEN_E,0,0,0,0,0,0,0} / Fr_t{D,0,0,0,0,0,0,0};
        fs_phase1(claim, A_E, comb, BvE, T_comb, m_E, alpha_E, beta_E, Cc,
                  inv_ratio, alpha_sq, u_E, tr, ws_E, pfE, evil != 1);
    }
    write_lookup(obdir + "/lookup_E.bin", pfE);
    absorb_fr(tr, "A_f", pfE.A_f); absorb_fr(tr, "S_f", pfE.S_f); absorb_fr(tr, "m_f", pfE.m_f);
    vector<Fr_t> u_ptE(ws_E.rbegin(), ws_E.rend());
    vector<Fr_t> u_mE(ws_E.rbegin(), ws_E.rend() - n1_E);
    if (evil == 0) {        // convention sanity: fold terminals == ME evaluations
        vector<Fr_t> u_col(u_ptE.begin(), u_ptE.begin() + logC);
        vector<Fr_t> u_row(u_ptE.begin() + logC, u_ptE.end());
        if (!fr_eq(pfE.A_f, A_E.multi_dim_me({u_row, u_col}, {B, NCOL})) ||
            !fr_eq(pfE.S_f, comb.multi_dim_me({u_row, u_col}, {B, NCOL})))
            throw runtime_error("exp lookup terminal != multi_dim_me (convention bug)");
    }
    open_prove(A_E,  NCOL, gen, Q, u_ptE, obdir + "/ipa_A_E.bin", tr);
    open_prove(comb, NCOL, gen, Q, u_ptE, obdir + "/ipa_comb.bin", tr);
    open_prove(m_E,  NCOL, gen, Q, u_mE,  obdir + "/ipa_m_E.bin", tr);

    // ---- obligation 2: limb range lookup ----
    Fr_t beta_L = fs_challenge_fr(tr);
    FrTensor A_L(DL);
    tlookup_inv_kernel<<<(DL+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        L_t.gpu_data, beta_L, A_L.gpu_data, DL);
    cudaDeviceSynchronize();
    G1TensorJacobian com_AL = gen.commit(A_L);
    com_AL.save(obdir + "/com_A_L.bin");
    absorb_g1_tensor(tr, "com_A_L", com_AL);
    Fr_t alpha_L = fs_challenge_fr(tr);
    auto u_L = fs_challenge_vec(tr, logDL);

    LookupProof pfL;
    vector<Fr_t> ws_L;
    {
        FrTensor BvL(LEN_R);
        tlookup_inv_kernel<<<(LEN_R+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            tlR.table.gpu_data, beta_L, BvL.gpu_data, LEN_R);
        cudaDeviceSynchronize();
        Fr_t alpha_sq = alpha_L * alpha_L;
        Fr_t Cc = alpha_sq - (BvL * m_L).sum();
        Fr_t claim = alpha_L + alpha_sq;
        Fr_t inv_ratio = Fr_t{LEN_R,0,0,0,0,0,0,0} / Fr_t{DL,0,0,0,0,0,0,0};
        fs_phase1(claim, A_L, L_t, BvL, tlR.table, m_L, alpha_L, beta_L, Cc,
                  inv_ratio, alpha_sq, u_L, tr, ws_L, pfL, evil != 6);
    }
    write_lookup(obdir + "/lookup_L.bin", pfL);
    absorb_fr(tr, "A_f", pfL.A_f); absorb_fr(tr, "S_f", pfL.S_f); absorb_fr(tr, "m_f", pfL.m_f);
    vector<Fr_t> u_ptL(ws_L.rbegin(), ws_L.rend());
    vector<Fr_t> u_mL(ws_L.rbegin(), ws_L.rend() - n1_L);
    if (evil == 0) {
        vector<Fr_t> u_col(u_ptL.begin(), u_ptL.begin() + logC);
        vector<Fr_t> u_row(u_ptL.begin() + logC, u_ptL.end());
        if (!fr_eq(pfL.A_f, A_L.multi_dim_me({u_row, u_col}, {4 * B, NCOL})) ||
            !fr_eq(pfL.S_f, L_t.multi_dim_me({u_row, u_col}, {4 * B, NCOL})))
            throw runtime_error("limb lookup terminal != multi_dim_me (convention bug)");
    }
    open_prove(A_L, NCOL, gen, Q, u_ptL, obdir + "/ipa_A_L.bin", tr);
    open_prove(L_t, NCOL, gen, Q, u_ptL, obdir + "/ipa_L_lk.bin", tr);
    open_prove(m_L, NCOL, gen, Q, u_mL,  obdir + "/ipa_m_L.bin", tr);

    // ---- obligation 3: row-sum sumcheck (R2) ----
    auto u_b = fs_challenge_vec(tr, logB);
    HadamardProof hp_rs;
    hp_rs.claim_H = S_t.multi_dim_me({u_b}, {B});             // ev_S
    absorb_fr(tr, "ev_S", hp_rs.claim_H);
    vector<Fr_t> ws_rs;
    {
        FrTensor eq_b = build_eq_tensor(u_b);
        FrTensor W_rs(D), Wtmp(D);
        k_bcast_rows<<<(D + 255) / 256, 256>>>(eq_b.gpu_data, Wtmp.gpu_data, NCOL, D);
        cudaDeviceSynchronize();
        k_fr_emul<<<(D + 63) / 64, 64>>>(Wtmp.gpu_data, MK_t.gpu_data, W_rs.gpu_data, D);
        cudaDeviceSynchronize();
        FrTensor Sc(E_t), Uc(ones_t);
        fs_hadamard(hp_rs.claim_H, W_rs, Sc, Uc, tr, ws_rs, hp_rs, evil != 4);
    }
    absorb_fr(tr, "S_f2", hp_rs.S_f2); absorb_fr(tr, "U_f2", hp_rs.U_f2);
    write_hp(obdir + "/hp_rs.bin", hp_rs);
    vector<Fr_t> pt_rs(ws_rs.rbegin(), ws_rs.rend());
    if (evil == 0) {
        vector<Fr_t> u_col(pt_rs.begin(), pt_rs.begin() + logC);
        vector<Fr_t> u_row(pt_rs.begin() + logC, pt_rs.end());
        if (!fr_eq(hp_rs.S_f2, E_t.multi_dim_me({u_row, u_col}, {B, NCOL})) ||
            !fr_eq(hp_rs.U_f2, F_ONE))
            throw runtime_error("row-sum terminal != multi_dim_me / 1 (convention bug)");
    }
    open_prove(E_t, NCOL, gen, Q, pt_rs, obdir + "/ipa_E_rs.bin", tr);
    open_prove(S_t, NCOL, gen, Q, u_b,   obdir + "/ipa_S_rs.bin", tr);

    // ---- obligation 4: bracket sumcheck V1 (MK.*E at u_r) ----
    auto u_r = fs_challenge_vec(tr, logD);
    vector<Fr_t> ur_col(u_r.begin(), u_r.begin() + logC);
    vector<Fr_t> ur_row(u_r.begin() + logC, u_r.end());
    HadamardProof hp_v1;
    {
        FrTensor MKE = MK_t * E_t;
        hp_v1.claim_H = MKE.multi_dim_me({ur_row, ur_col}, {B, NCOL});   // c1
    }
    absorb_fr(tr, "c1", hp_v1.claim_H);
    vector<Fr_t> ws_v1;
    {
        FrTensor eq_r = build_eq_tensor(u_r);
        FrTensor W1(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(eq_r.gpu_data, MK_t.gpu_data, W1.gpu_data, D);
        cudaDeviceSynchronize();
        FrTensor Sc(E_t), Uc(ones_t);
        fs_hadamard(hp_v1.claim_H, W1, Sc, Uc, tr, ws_v1, hp_v1, true);
    }
    absorb_fr(tr, "S_f2", hp_v1.S_f2); absorb_fr(tr, "U_f2", hp_v1.U_f2);
    write_hp(obdir + "/hp_v1.bin", hp_v1);
    vector<Fr_t> pt1(ws_v1.rbegin(), ws_v1.rend());
    if (evil == 0) {
        vector<Fr_t> u_col(pt1.begin(), pt1.begin() + logC);
        vector<Fr_t> u_row(pt1.begin() + logC, pt1.end());
        if (!fr_eq(hp_v1.S_f2, E_t.multi_dim_me({u_row, u_col}, {B, NCOL})) ||
            !fr_eq(hp_v1.U_f2, F_ONE))
            throw runtime_error("V1 terminal != multi_dim_me / 1 (convention bug)");
    }
    open_prove(E_t, NCOL, gen, Q, pt1, obdir + "/ipa_E_v1.bin", tr);

    // ---- obligation 5: bracket sumcheck V2 (P.*S_bcast at u_r) ----
    HadamardProof hp_v2;
    {
        FrTensor PSb = P_t * Sb;
        hp_v2.claim_H = PSb.multi_dim_me({ur_row, ur_col}, {B, NCOL});   // c2
    }
    absorb_fr(tr, "c2", hp_v2.claim_H);
    vector<Fr_t> ws_v2;
    {
        FrTensor eq_r = build_eq_tensor(u_r);
        FrTensor Sc(P_t), Uc(Sb);
        fs_hadamard(hp_v2.claim_H, eq_r, Sc, Uc, tr, ws_v2, hp_v2, true);
    }
    absorb_fr(tr, "S_f2", hp_v2.S_f2); absorb_fr(tr, "U_f2", hp_v2.U_f2);
    write_hp(obdir + "/hp_v2.bin", hp_v2);
    vector<Fr_t> pt2(ws_v2.rbegin(), ws_v2.rend());
    vector<Fr_t> pt2_rows(pt2.begin() + logC, pt2.end());
    if (evil == 0) {
        vector<Fr_t> u_col(pt2.begin(), pt2.begin() + logC);
        if (!fr_eq(hp_v2.S_f2, P_t.multi_dim_me({pt2_rows, u_col}, {B, NCOL})) ||
            !fr_eq(hp_v2.U_f2, Sb.multi_dim_me({pt2_rows, u_col}, {B, NCOL})) ||
            !fr_eq(hp_v2.U_f2, S_t.multi_dim_me({pt2_rows}, {B})))
            throw runtime_error("V2 terminal != multi_dim_me / bcast suffix (convention bug)");
    }
    open_prove(P_t, NCOL, gen, Q, pt2,      obdir + "/ipa_P_v2.bin", tr);
    open_prove(S_t, NCOL, gen, Q, pt2_rows, obdir + "/ipa_S_v2.bin", tr);

    // ---- obligation 6: residual reconstruction (L plane openings + S_id) ----
    Fr_t lv[5];
    {
        const char* labels[4] = {"v00", "v10", "v01", "v11"};
        for (uint p = 0; p < 4; p++) {
            vector<Fr_t> u_pt(u_r);
            u_pt.push_back((p & 1) ? F_ONE : F_ZERO);
            u_pt.push_back((p >> 1) ? F_ONE : F_ZERO);
            vector<Fr_t> u_row2(u_pt.begin() + logC, u_pt.end());
            lv[p] = L_t.multi_dim_me({u_row2, ur_col}, {4 * B, NCOL});
            absorb_fr(tr, labels[p], lv[p]);
        }
        if (evil == 0) {    // convention sanity: the {4B, NCOL} opening of L at
                            // (u_r, plane bits) == the ME of that plane's D-slice
            for (uint p = 0; p < 4; p++) {
                FrTensor plane(D, Lh.data() + (size_t)p * D);
                if (!fr_eq(lv[p], plane.multi_dim_me({ur_row, ur_col}, {B, NCOL})))
                    throw runtime_error("L plane opening != plane-slice multi_dim_me (convention bug)");
            }
        }
        static const char* fns[4] = {"/ipa_L00.bin", "/ipa_L10.bin",
                                     "/ipa_L01.bin", "/ipa_L11.bin"};
        for (uint p = 0; p < 4; p++) {
            vector<Fr_t> u_pt(u_r);
            u_pt.push_back((p & 1) ? F_ONE : F_ZERO);
            u_pt.push_back((p >> 1) ? F_ONE : F_ZERO);
            open_prove(L_t, NCOL, gen, Q, u_pt, obdir + fns[p], tr);
        }
    }
    lv[4] = S_t.multi_dim_me({ur_row}, {B});                  // S_id
    if (evil == 0 &&    // convention sanity: S_id == MLE of the broadcast at u_r
        !fr_eq(lv[4], Sb.multi_dim_me({ur_row, ur_col}, {B, NCOL})))
        throw runtime_error("S_id != broadcast multi_dim_me (convention bug)");
    absorb_fr(tr, "S_id", lv[4]);
    { FILE* f = open_or_die(obdir + "/lvals.bin", "wb");
      fwrite(lv, sizeof(Fr_t), 5, f); fclose(f); }
    open_prove(S_t, NCOL, gen, Q, ur_row, obdir + "/ipa_S_id.bin", tr);
    cout << "PROVED softmax obligation -> " << obdir << endl;
}

// ---------------- verify (witness-free) ----------------
#define RJ(msg) do { ostringstream oss_; oss_ << msg; \
    cout << "REJECT: " << oss_.str() << endl; \
    if (reason) *reason = oss_.str(); return false; } while (0)

// fold a public device tensor over the round challenges (k_fr_fold chain)
static Fr_t fold_public(const FrTensor& W, const vector<Fr_t>& ws) {
    uint sz = W.size;
    Fr_t *da, *db;
    cudaMalloc(&da, sz * sizeof(Fr_t));
    cudaMalloc(&db, (sz / 2 + 1) * sizeof(Fr_t));
    cudaMemcpy(da, W.gpu_data, sz * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    for (auto& w : ws) {
        uint nsz = sz >> 1;
        k_fr_fold<<<(nsz + 31) / 32, 32>>>(da, w, db, nsz);
        cudaDeviceSynchronize();
        std::swap(da, db); sz = nsz;
    }
    Fr_t out;
    cudaMemcpy(&out, da, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db);
    return out;
}

static bool verify(const string& obdir, const string& seed,
                   uint B, uint NCOL, int LOW_E, uint LEN_E,
                   const vector<int>& maph, uint LEN_R,
                   const Commitment& gen, const G1Jacobian_t& Q,
                   string* reason = nullptr) {
    const uint D = B * NCOL, DL = 4 * D;
    const uint logB = ceilLog2(B), logC = ceilLog2(NCOL);
    const uint logD = logB + logC, logDL = logD + 2;
    const uint n1_E = ceilLog2(D / LEN_E), n2_E = ceilLog2(LEN_E);
    const uint n1_L = ceilLog2(DL / LEN_R), n2_L = ceilLog2(LEN_R);
    if (B != NCOL || B != (1u << logB) || gen.size != NCOL ||
        LEN_E != (1u << n2_E) || LEN_R != (1u << n2_L) ||
        NCOL > LEN_E || LEN_E > D || D % LEN_E ||
        NCOL > LEN_R || LEN_R > DL || DL % LEN_R)
        RJ("bad layout params");
    { FILE* f = open_or_die(obdir + "/dims.bin", "rb");
      uint32_t d0[2]; int32_t lo; uint32_t d1[3];
      if (fread(d0, sizeof(uint32_t), 2, f) != 2 ||
          fread(&lo, sizeof(int32_t), 1, f) != 1 ||
          fread(d1, sizeof(uint32_t), 3, f) != 3) { fclose(f); RJ("dims.bin short read"); }
      fclose(f);
      if (d0[0] != B || d0[1] != NCOL || lo != LOW_E ||
          d1[0] != LEN_E || d1[1] != LEN_R || d1[2] != LOG_OUT)
          RJ("dims.bin mismatch"); }

    G1TensorJacobian com_z (obdir + "/com_z.bin");
    G1TensorJacobian com_E (obdir + "/com_E.bin");
    G1TensorJacobian com_P (obdir + "/com_P.bin");
    G1TensorJacobian com_S (obdir + "/com_S.bin");
    G1TensorJacobian com_L (obdir + "/com_L.bin");
    G1TensorJacobian com_mE(obdir + "/com_m_E.bin");
    G1TensorJacobian com_mL(obdir + "/com_m_L.bin");
    G1TensorJacobian com_AE(obdir + "/com_A_E.bin");
    G1TensorJacobian com_AL(obdir + "/com_A_L.bin");
    if (com_z.size != B || com_E.size != B || com_P.size != B ||
        com_S.size != 1 || com_L.size != 4 * B || com_AE.size != B ||
        com_AL.size != 4 * B || com_mE.size != LEN_E / NCOL ||
        com_mL.size != LEN_R / NCOL)
        RJ("commitment row counts");

    LookupProof pfE = read_lookup(obdir + "/lookup_E.bin");
    LookupProof pfL = read_lookup(obdir + "/lookup_L.bin");
    HadamardProof hp_rs = read_hp(obdir + "/hp_rs.bin");
    HadamardProof hp_v1 = read_hp(obdir + "/hp_v1.bin");
    HadamardProof hp_v2 = read_hp(obdir + "/hp_v2.bin");
    Fr_t lv[5];
    { FILE* f = open_or_die(obdir + "/lvals.bin", "rb");
      if (fread(lv, sizeof(Fr_t), 5, f) != 5) { fclose(f); RJ("lvals.bin short read"); }
      fclose(f); }
    if (pfE.ev.size() != 4 * logD) RJ("exp lookup round count");
    if (pfL.ev.size() != 4 * logDL) RJ("limb lookup round count");
    if (hp_rs.ev.size() != 4 * logD || hp_v1.ev.size() != 4 * logD ||
        hp_v2.ev.size() != 4 * logD)
        RJ("hadamard round count");

    // public causal mask (derived from B, never committed)
    vector<int> mkh(D);
    for (uint i = 0; i < B; i++)
        for (uint j = 0; j < NCOL; j++) mkh[(size_t)i * NCOL + j] = (j <= i) ? 1 : 0;
    FrTensor MK_t(D, mkh.data());

    // ---- transcript replay ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "NCOL", NCOL);
    absorb_u32(tr, "LOW_E", (uint32_t)LOW_E); absorb_u32(tr, "LEN_E", LEN_E);
    absorb_u32(tr, "LEN_R", LEN_R); absorb_u32(tr, "LOG_OUT", LOG_OUT);
    absorb_g1_tensor(tr, "com_z", com_z);
    absorb_g1_tensor(tr, "com_E", com_E);
    absorb_g1_tensor(tr, "com_P", com_P);
    absorb_g1_tensor(tr, "com_S", com_S);
    absorb_g1_tensor(tr, "com_L", com_L);
    absorb_g1_tensor(tr, "com_m_E", com_mE);
    absorb_g1_tensor(tr, "com_m_L", com_mL);
    Fr_t r = fs_challenge_fr(tr);
    Fr_t beta_E = fs_challenge_fr(tr);
    absorb_g1_tensor(tr, "com_A_E", com_AE);
    Fr_t alpha_E = fs_challenge_fr(tr);
    auto u_E = fs_challenge_vec(tr, logD);

    // combined commitment com_z + r*com_E, formed homomorphically (1-thread
    // helpers; batched G1 kernels are -dlto miscompile bait)
    vector<G1Jacobian_t> hcomb(B);
    {
        vector<G1Jacobian_t> hz(B), he(B);
        cudaMemcpy(hz.data(), com_z.gpu_data, B * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(he.data(), com_E.gpu_data, B * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        for (uint j = 0; j < B; j++) hcomb[j] = h_add(hz[j], h_mul(he[j], r));
    }
    G1TensorJacobian com_comb(B, hcomb.data());

    const Fr_t inv6 = inv(F_SIX);

    // ---- obligation 1: exp lookup rounds ----
    Fr_t cur = h_scalar(alpha_E, h_scalar(alpha_E, alpha_E, 2), 0);   // alpha + alpha^2
    Fr_t alpha_acc = alpha_E, alphasq_acc = h_scalar(alpha_E, alpha_E, 2);
    vector<Fr_t> ws_E;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {pfE.ev[4*k], pfE.ev[4*k+1], pfE.ev[4*k+2], pfE.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("exp lookup round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "p0", e[0]); absorb_fr(tr, "p1", e[1]);
        absorb_fr(tr, "p2", e[2]); absorb_fr(tr, "p3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_E.push_back(w);
        cur = lagrange4(e, w, inv6);
        Fr_t eqv = my_eq(u_E[logD - 1 - k], w);
        alpha_acc = h_scalar(alpha_acc, eqv, 2);
        if (k >= n1_E) alphasq_acc = h_scalar(alphasq_acc, eqv, 2);
    }
    // B_f, T_f recomputed from the PUBLIC combined table
    Fr_t B_f, T_f;
    {
        FrTensor mapped_t(LEN_E, maph.data());
        tLookupRangeMapping tlE(LOW_E, LEN_E, mapped_t);
        FrTensor T_comb = tlE.table + tlE.mapped_vals * r;
        FrTensor B_pub(LEN_E);
        tlookup_inv_kernel<<<(LEN_E+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            T_comb.gpu_data, beta_E, B_pub.gpu_data, LEN_E);
        cudaDeviceSynchronize();
        vector<Fr_t> ws2(ws_E.begin() + n1_E, ws_E.end());
        B_f = fold_public(B_pub, ws2);
        T_f = fold_public(T_comb, ws2);
    }
    {
        Fr_t inv_ratio = h_scalar({LEN_E,0,0,0,0,0,0,0}, inv({D,0,0,0,0,0,0,0}), 2);
        Fr_t t1 = h_scalar(alpha_acc, h_scalar(pfE.A_f, h_scalar(pfE.S_f, beta_E, 0), 2), 2);
        Fr_t t2 = h_scalar(inv_ratio, h_scalar(alphasq_acc, h_scalar(B_f, h_scalar(T_f, beta_E, 0), 2), 2), 2);
        Fr_t t4 = h_scalar(inv_ratio, h_scalar(pfE.m_f, B_f, 2), 2);
        Fr_t rhs = h_scalar(h_scalar(h_scalar(t1, t2, 0), pfE.A_f, 0), t4, 1);
        if (!fr_eq(cur, rhs)) RJ("exp lookup terminal identity");
    }
    absorb_fr(tr, "A_f", pfE.A_f); absorb_fr(tr, "S_f", pfE.S_f); absorb_fr(tr, "m_f", pfE.m_f);
    vector<Fr_t> u_ptE(ws_E.rbegin(), ws_E.rend());
    vector<Fr_t> u_mE(ws_E.rbegin(), ws_E.rend() - n1_E);
    if (!open_verify(com_AE, gen, NCOL, Q, u_ptE, pfE.A_f, obdir + "/ipa_A_E.bin", tr))
        RJ("IPA opening of A_f vs com_A_E");
    if (!open_verify(com_comb, gen, NCOL, Q, u_ptE, pfE.S_f, obdir + "/ipa_comb.bin", tr))
        RJ("IPA opening of S_f vs com_z + r*com_E");
    if (!open_verify(com_mE, gen, NCOL, Q, u_mE, pfE.m_f, obdir + "/ipa_m_E.bin", tr))
        RJ("IPA opening of m_f vs com_m_E");

    // ---- obligation 2: limb lookup rounds ----
    Fr_t beta_L = fs_challenge_fr(tr);
    absorb_g1_tensor(tr, "com_A_L", com_AL);
    Fr_t alpha_L = fs_challenge_fr(tr);
    auto u_L = fs_challenge_vec(tr, logDL);
    cur = h_scalar(alpha_L, h_scalar(alpha_L, alpha_L, 2), 0);
    alpha_acc = alpha_L; alphasq_acc = h_scalar(alpha_L, alpha_L, 2);
    vector<Fr_t> ws_L;
    for (uint k = 0; k < logDL; k++) {
        array<Fr_t,4> e = {pfL.ev[4*k], pfL.ev[4*k+1], pfL.ev[4*k+2], pfL.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("limb lookup round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "p0", e[0]); absorb_fr(tr, "p1", e[1]);
        absorb_fr(tr, "p2", e[2]); absorb_fr(tr, "p3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_L.push_back(w);
        cur = lagrange4(e, w, inv6);
        Fr_t eqv = my_eq(u_L[logDL - 1 - k], w);
        alpha_acc = h_scalar(alpha_acc, eqv, 2);
        if (k >= n1_L) alphasq_acc = h_scalar(alphasq_acc, eqv, 2);
    }
    {
        tLookupRange tlR(0, LEN_R);
        FrTensor B_pub(LEN_R);
        tlookup_inv_kernel<<<(LEN_R+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            tlR.table.gpu_data, beta_L, B_pub.gpu_data, LEN_R);
        cudaDeviceSynchronize();
        vector<Fr_t> ws2(ws_L.begin() + n1_L, ws_L.end());
        B_f = fold_public(B_pub, ws2);
        T_f = fold_public(tlR.table, ws2);
    }
    {
        Fr_t inv_ratio = h_scalar({LEN_R,0,0,0,0,0,0,0}, inv({DL,0,0,0,0,0,0,0}), 2);
        Fr_t t1 = h_scalar(alpha_acc, h_scalar(pfL.A_f, h_scalar(pfL.S_f, beta_L, 0), 2), 2);
        Fr_t t2 = h_scalar(inv_ratio, h_scalar(alphasq_acc, h_scalar(B_f, h_scalar(T_f, beta_L, 0), 2), 2), 2);
        Fr_t t4 = h_scalar(inv_ratio, h_scalar(pfL.m_f, B_f, 2), 2);
        Fr_t rhs = h_scalar(h_scalar(h_scalar(t1, t2, 0), pfL.A_f, 0), t4, 1);
        if (!fr_eq(cur, rhs)) RJ("limb lookup terminal identity");
    }
    absorb_fr(tr, "A_f", pfL.A_f); absorb_fr(tr, "S_f", pfL.S_f); absorb_fr(tr, "m_f", pfL.m_f);
    vector<Fr_t> u_ptL(ws_L.rbegin(), ws_L.rend());
    vector<Fr_t> u_mL(ws_L.rbegin(), ws_L.rend() - n1_L);
    if (!open_verify(com_AL, gen, NCOL, Q, u_ptL, pfL.A_f, obdir + "/ipa_A_L.bin", tr))
        RJ("IPA opening of A_f vs com_A_L");
    if (!open_verify(com_L, gen, NCOL, Q, u_ptL, pfL.S_f, obdir + "/ipa_L_lk.bin", tr))
        RJ("IPA opening of S_f vs com_L");
    if (!open_verify(com_mL, gen, NCOL, Q, u_mL, pfL.m_f, obdir + "/ipa_m_L.bin", tr))
        RJ("IPA opening of m_f vs com_m_L");

    // ---- obligation 3: row-sum sumcheck ----
    auto u_b = fs_challenge_vec(tr, logB);
    absorb_fr(tr, "ev_S", hp_rs.claim_H);
    cur = hp_rs.claim_H;
    vector<Fr_t> ws_rs;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_rs.ev[4*k], hp_rs.ev[4*k+1], hp_rs.ev[4*k+2], hp_rs.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("row-sum round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_rs.push_back(w);
        cur = lagrange4(e, w, inv6);
    }
    if (!fr_eq(hp_rs.U_f2, F_ONE)) RJ("row-sum U_f2 != 1");
    {
        // recompute the public weight fold: W_rs = bcast_rows(eq(u_b)) .* MK
        FrTensor eq_b = build_eq_tensor(u_b);
        FrTensor Wtmp(D), W_rs(D);
        k_bcast_rows<<<(D + 255) / 256, 256>>>(eq_b.gpu_data, Wtmp.gpu_data, NCOL, D);
        cudaDeviceSynchronize();
        k_fr_emul<<<(D + 63) / 64, 64>>>(Wtmp.gpu_data, MK_t.gpu_data, W_rs.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t W_f = fold_public(W_rs, ws_rs);
        if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp_rs.S_f2, hp_rs.U_f2, 2), 2)))
            RJ("row-sum terminal identity");
    }
    absorb_fr(tr, "S_f2", hp_rs.S_f2); absorb_fr(tr, "U_f2", hp_rs.U_f2);
    vector<Fr_t> pt_rs(ws_rs.rbegin(), ws_rs.rend());
    if (!open_verify(com_E, gen, NCOL, Q, pt_rs, hp_rs.S_f2, obdir + "/ipa_E_rs.bin", tr))
        RJ("IPA opening of E (row-sum) vs com_E");
    if (!open_verify(com_S, gen, NCOL, Q, u_b, hp_rs.claim_H, obdir + "/ipa_S_rs.bin", tr))
        RJ("IPA opening of ev_S vs com_S");

    // ---- obligation 4: bracket sumcheck V1 ----
    auto u_r = fs_challenge_vec(tr, logD);
    vector<Fr_t> ur_row(u_r.begin() + logC, u_r.end());
    absorb_fr(tr, "c1", hp_v1.claim_H);
    cur = hp_v1.claim_H;
    vector<Fr_t> ws_v1;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_v1.ev[4*k], hp_v1.ev[4*k+1], hp_v1.ev[4*k+2], hp_v1.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("V1 round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_v1.push_back(w);
        cur = lagrange4(e, w, inv6);
    }
    if (!fr_eq(hp_v1.U_f2, F_ONE)) RJ("V1 U_f2 != 1");
    {
        FrTensor eq_r = build_eq_tensor(u_r);
        FrTensor W1(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(eq_r.gpu_data, MK_t.gpu_data, W1.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t W_f = fold_public(W1, ws_v1);
        if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp_v1.S_f2, hp_v1.U_f2, 2), 2)))
            RJ("V1 terminal identity");
    }
    absorb_fr(tr, "S_f2", hp_v1.S_f2); absorb_fr(tr, "U_f2", hp_v1.U_f2);
    vector<Fr_t> pt1(ws_v1.rbegin(), ws_v1.rend());
    if (!open_verify(com_E, gen, NCOL, Q, pt1, hp_v1.S_f2, obdir + "/ipa_E_v1.bin", tr))
        RJ("IPA opening of E (V1) vs com_E");

    // ---- obligation 5: bracket sumcheck V2 (pure eq weight) ----
    absorb_fr(tr, "c2", hp_v2.claim_H);
    cur = hp_v2.claim_H;
    Fr_t eq_acc = F_ONE;
    vector<Fr_t> ws_v2;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_v2.ev[4*k], hp_v2.ev[4*k+1], hp_v2.ev[4*k+2], hp_v2.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("V2 round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_v2.push_back(w);
        cur = lagrange4(e, w, inv6);
        eq_acc = h_scalar(eq_acc, my_eq(u_r[logD - 1 - k], w), 2);
    }
    if (!fr_eq(cur, h_scalar(eq_acc, h_scalar(hp_v2.S_f2, hp_v2.U_f2, 2), 2)))
        RJ("V2 terminal identity");
    absorb_fr(tr, "S_f2", hp_v2.S_f2); absorb_fr(tr, "U_f2", hp_v2.U_f2);
    vector<Fr_t> pt2(ws_v2.rbegin(), ws_v2.rend());
    vector<Fr_t> pt2_rows(pt2.begin() + logC, pt2.end());
    if (!open_verify(com_P, gen, NCOL, Q, pt2, hp_v2.S_f2, obdir + "/ipa_P_v2.bin", tr))
        RJ("IPA opening of P (V2) vs com_P");
    // the never-committed broadcast tensor is pinned to com_S here: its MLE at
    // pt2 equals the row vector's MLE at the row-bit suffix of pt2
    if (!open_verify(com_S, gen, NCOL, Q, pt2_rows, hp_v2.U_f2, obdir + "/ipa_S_v2.bin", tr))
        RJ("IPA opening of V2 U_f2 vs com_S");

    // ---- obligation 6: L plane openings + S_id + identities ----
    {
        const char* labels[4] = {"v00", "v10", "v01", "v11"};
        for (uint p = 0; p < 4; p++) absorb_fr(tr, labels[p], lv[p]);
        static const char* fns[4] = {"/ipa_L00.bin", "/ipa_L10.bin",
                                     "/ipa_L01.bin", "/ipa_L11.bin"};
        static const char* names[4] = {"L00", "L10", "L01", "L11"};
        for (uint p = 0; p < 4; p++) {
            vector<Fr_t> u_pt(u_r);
            u_pt.push_back((p & 1) ? F_ONE : F_ZERO);
            u_pt.push_back((p >> 1) ? F_ONE : F_ZERO);
            if (!open_verify(com_L, gen, NCOL, Q, u_pt, lv[p], obdir + fns[p], tr))
                RJ("IPA opening of " << names[p] << " vs com_L");
        }
    }
    absorb_fr(tr, "S_id", lv[4]);
    if (!open_verify(com_S, gen, NCOL, Q, ur_row, lv[4], obdir + "/ipa_S_id.bin", tr))
        RJ("IPA opening of S_id vs com_S");

    // plain-field bracket identities (Schwartz-Zippel at u_r)
    const Fr_t F_LENR = {LEN_R, 0, 0, 0, 0, 0, 0, 0};
    const Fr_t F_2P17 = {1u << (LOG_OUT + 1), 0, 0, 0, 0, 0, 0, 0};
    Fr_t rt1 = h_scalar(lv[0], h_scalar(F_LENR, lv[1], 2), 0);   // v00 + LEN_R*v10
    Fr_t rt2 = h_scalar(lv[2], h_scalar(F_LENR, lv[3], 2), 0);   // v01 + LEN_R*v11
    {
        Fr_t lhs = h_scalar(F_2P17, hp_v1.claim_H, 2);           // 2^17*c1
        lhs = h_scalar(lhs, lv[4], 0);                           // + S_id
        lhs = h_scalar(lhs, h_scalar(F_TWO, hp_v2.claim_H, 2), 1); // - 2*c2
        if (!fr_eq(lhs, rt1)) RJ("bracket r1 identity (I1)");
    }
    {
        Fr_t lhs = h_scalar(h_scalar(rt1, rt2, 0), F_ONE, 0);    // r~1 + r~2 + 1
        Fr_t rhs = h_scalar(F_TWO, lv[4], 2);                    // 2*S_id
        if (!fr_eq(lhs, rhs)) RJ("bracket sum identity (I2)");
    }
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

static const char* PROOF_FILES[] = {
    "dims.bin",
    "com_z.bin", "com_E.bin", "com_P.bin", "com_S.bin", "com_L.bin",
    "com_m_E.bin", "com_m_L.bin", "com_A_E.bin", "com_A_L.bin",
    "lookup_E.bin", "lookup_L.bin",
    "hp_rs.bin", "hp_v1.bin", "hp_v2.bin",
    "lvals.bin",
    "ipa_A_E.bin", "ipa_comb.bin", "ipa_m_E.bin",
    "ipa_A_L.bin", "ipa_L_lk.bin", "ipa_m_L.bin",
    "ipa_E_rs.bin", "ipa_S_rs.bin",
    "ipa_E_v1.bin", "ipa_P_v2.bin", "ipa_S_v2.bin",
    "ipa_L00.bin", "ipa_L10.bin", "ipa_L01.bin", "ipa_L11.bin", "ipa_S_id.bin"};
static long tamper_offset(const string& f) {
    if (f == "dims.bin") return 0;
    if (f == "lvals.bin") return 4;                 // v00 second byte
    if (f.substr(0, 4) == "ipa_") return -32;       // a_final
    if (f.substr(0, 4) == "com_") return 24;        // first point, x limbs
    return 4 + 32;                                  // lookup/hp: round-0 evaluation
}

static bool selftest_case(uint B, int LOW_E, uint LEN_E, uint LEN_R) {
    const uint NCOL = B, D = B * NCOL;
    cout << "==== selftest case B=NCOL=" << B << " LOW_E=" << LOW_E
         << " LEN_E=" << LEN_E << " LEN_R=" << LEN_R
         << " (n1_E=" << ceilLog2(D / LEN_E) << ", n1_L=" << ceilLog2(4 * D / LEN_R)
         << ") ====" << endl;
    // toy mapped table mapped[k] = k + 1 (>= 1 keeps S >= 1); driver is
    // table-agnostic like glu
    vector<int> maph(LEN_E);
    for (uint k = 0; k < LEN_E; k++) maph[k] = (int)k + 1;
    srand(4242 + B);
    vector<int> zh(D);
    for (uint k = 0; k < D; k++) zh[k] = LOW_E + (rand() % (int)LEN_E);
    Commitment gen = Commitment::random(NCOL);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_softmax_ob";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:softmax";
    bool all = true;

    prove(obdir, seed, zh, B, NCOL, LOW_E, LEN_E, maph, LEN_R, gen, Q,
          "/tmp/zkob_softmax_P.i32.bin");
    bool honest = verify(obdir, seed, B, NCOL, LOW_E, LEN_E, maph, LEN_R, gen, Q);
    cout << (honest ? "PASS" : "FAIL") << ": honest case "
         << (honest ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && honest;

    // semantic evil modes: each rejected by EXACTLY the named check
    struct Evil { int mode; uint i, j; const char* expect; const char* what; };
    vector<Evil> evils = {
        {1, 2, 1, "exp lookup round 0",
                  "E[2,1]+=1 (unmasked), S/P/limbs recomputed"},
        {2, 0, 1, "bracket r1 identity",
                  "P[0,1]+=1 (MASKED), limbs = mod-LEN_R^2 truncation"},
        {3, 1, 1, "bracket sum identity",
                  "P[1,1]-=1 (diagonal, honest P>=1), r1' = r1+2S limbs honest"},
        {4, 1, 0, "row-sum round 0",
                  "S[1]+=1, P/limbs recomputed"},
        {5, 2, 0, "IPA opening of V2 U_f2 vs com_S",
                  "V2 broadcast buffer Sb[2,0]+=1, all commitments honest"},
        {6, 3, 1, "limb lookup round 0",
                  "L[0,idx]+=LEN_R, L[1,idx]-=1 (r1 value unchanged, m_L honest)"},
    };
    string evdir = "/tmp/zkob_softmax_evil";
    mkdir(evdir.c_str(), 0755);
    for (auto& ev : evils) {
        prove(evdir, seed, zh, B, NCOL, LOW_E, LEN_E, maph, LEN_R, gen, Q, "",
              ev.mode, ev.i, ev.j);
        string reason;
        bool rejected = !verify(evdir, seed, B, NCOL, LOW_E, LEN_E, maph, LEN_R,
                                gen, Q, &reason);
        bool right = rejected && reason.find(ev.expect) != string::npos;
        cout << (right ? "PASS" : "FAIL") << ": evil=" << ev.mode << " (" << ev.what
             << ") rejected by [" << (rejected ? reason : string("NOT REJECTED"))
             << "], expected [" << ev.expect << "]" << endl;
        all = all && right;
    }

    // byte tampers on every proof file (tamper, verify must reject, restore)
    for (const char* fn : PROOF_FILES) {
        long off = tamper_offset(fn);
        tamper_byte(obdir + "/" + fn, off, +1);
        bool rejected = !verify(obdir, seed, B, NCOL, LOW_E, LEN_E, maph, LEN_R, gen, Q);
        tamper_byte(obdir + "/" + fn, off, -1);
        cout << (rejected ? "PASS" : "FAIL") << ": byte tamper " << fn << "@" << off
             << " rejected: " << (rejected ? "YES" : "NO(!!)") << endl;
        all = all && rejected;
    }
    bool restored = verify(obdir, seed, B, NCOL, LOW_E, LEN_E, maph, LEN_R, gen, Q);
    cout << (restored ? "PASS" : "FAIL") << ": restored verify "
         << (restored ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && restored;
    cout << (all ? "CASE PASS" : "CASE FAIL") << endl;
    return all;
}

static bool selftest_real() {
    const uint B = 1024, NCOL = 1024, LEN_E = 1u << 20, LEN_R = 1u << 20;
    const int LOW_E = -(1 << 19);
    cout << "==== selftest real-scale case B=NCOL=1024 LOW_E=" << LOW_E
         << " LEN_E=2^20 LEN_R=2^20 ====" << endl;
    // real exp table: registered file if present, else generated in-driver via
    // host double exp -- NON-AUTHORITATIVE, selftest only (the registered
    // sha256 file from gen_softmax_exp_table.py is the source of truth)
    vector<int> maph(LEN_E);
    if (file_size("softmax-exp-table.bin") == (long)LEN_E * 4) {
        cout << "loading softmax-exp-table.bin" << endl;
        FILE* f = open_or_die("softmax-exp-table.bin", "rb");
        if (fread(maph.data(), sizeof(int), LEN_E, f) != LEN_E)
            throw runtime_error("exp table short read");
        fclose(f);
    } else {
        cout << "softmax-exp-table.bin not found: generating in-driver via host "
                "double exp (NON-AUTHORITATIVE, selftest only)" << endl;
        for (uint k = 0; k < LEN_E; k++)
            maph[k] = (int)llround(65536.0 * exp((double)(LOW_E + (int)k) / 65536.0));
    }
    const string genpath = "/tmp/gen1024.bin";
    if (file_size(genpath) != 1024L * (long)sizeof(G1Jacobian_t)) {
        cout << "generating " << genpath << " via ppgen..." << endl;
        if (system(("./ppgen 1024 " + genpath).c_str()))
            throw runtime_error("ppgen failed");
    }
    Commitment gen(genpath);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    // scores z_ ~ round(N(0, 2^13)) clipped to the domain (realistic envelope:
    // measured max |z_| = 277 * 2^9 ~ 1.4e5, sigma 2^13 covers the same range)
    mt19937_64 rng(20260610);
    normal_distribution<double> nd(0.0, 8192.0);
    vector<int> zh((size_t)B * NCOL);
    for (size_t k = 0; k < zh.size(); k++) {
        long long v = llround(nd(rng));
        if (v < LOW_E) v = LOW_E;
        if (v > LOW_E + (long long)LEN_E - 1) v = LOW_E + (long long)LEN_E - 1;
        zh[k] = (int)v;
    }
    string obdir = "/tmp/zkob_softmax_real";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:softmax:real";
    bool all = true;

    auto t0 = chrono::steady_clock::now();
    prove(obdir, seed, zh, B, NCOL, LOW_E, LEN_E, maph, LEN_R, gen, Q,
          "/tmp/zkob_softmax_real_P.i32.bin");
    auto t1 = chrono::steady_clock::now();
    bool honest = verify(obdir, seed, B, NCOL, LOW_E, LEN_E, maph, LEN_R, gen, Q);
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

    tamper_byte(obdir + "/lookup_E.bin", 4 + 32, +1);
    bool rejected = !verify(obdir, seed, B, NCOL, LOW_E, LEN_E, maph, LEN_R, gen, Q);
    tamper_byte(obdir + "/lookup_E.bin", 4 + 32, -1);
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
        bool a = selftest_case(8,  -8,  16, 32);   // n1_E = 2, n1_L = 3
        bool b = selftest_case(4,  -8,  16, 32);   // n1_E = 0 (pure phase2)
        bool c = selftest_case(16, -32, 64, 64);   // bigger grid
        bool d = selftest_real();
        bool ok = a && b && c && d;
        cout << (ok ? "ZKOB-SOFTMAX SELFTEST: ALL PASS"
                    : "ZKOB-SOFTMAX SELFTEST: FAIL") << endl;
        return ok ? 0 : 1;
    }
    if (mode == "prove" && (argc == 13 || argc == 14)) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[5]), NCOL = stoi(argv[6]);
        int LOW_E = stoi(argv[7]); uint LEN_E = (uint)stoul(argv[8]);
        uint LEN_R = (uint)stoul(argv[10]);
        vector<int> zh = load_i32(argv[4], B * NCOL);
        vector<int> maph = load_expmap(argv[9], LEN_E);
        Commitment gen(argv[11]), qg(argv[12]);
        prove(obdir, seed, zh, B, NCOL, LOW_E, LEN_E, maph, LEN_R, gen, qg(0),
              argc == 14 ? argv[13] : "");
        return 0;
    }
    if (mode == "verify" && argc == 12) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[4]), NCOL = stoi(argv[5]);
        int LOW_E = stoi(argv[6]); uint LEN_E = (uint)stoul(argv[7]);
        uint LEN_R = (uint)stoul(argv[9]);
        vector<int> maph = load_expmap(argv[8], LEN_E);
        Commitment gen(argv[10]), qg(argv[11]);
        return verify(obdir, seed, B, NCOL, LOW_E, LEN_E, maph, LEN_R, gen, qg(0)) ? 0 : 1;
    }
    cerr << "usage: zkob_softmax selftest\n"
         << "       zkob_softmax prove  <obdir> <seed> <z-int32> <B> <NCOL> <LOW_E> <LEN_E> <expmap-int32> <LEN_R> <gen> <q> [P-int32-out]\n"
         << "       zkob_softmax verify <obdir> <seed> <B> <NCOL> <LOW_E> <LEN_E> <expmap-int32> <LEN_R> <gen> <q>" << endl;
    return 2;
}
