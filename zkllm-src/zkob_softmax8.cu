// Real driver for ONE temperature-8 softmax obligation
// (layer{l}.attn.softmax.h{hh}, faithful-arch chain), per
// STAGE3_FAITHFUL_DESIGN.md section 4.3 (DESIGN FINAL 2026-06-11). Structure
// copied from the validated zkob_softmax.cu (temp-128); deltas per 4.3.
// Binds, for the B x NCOL score grid z_ (int32, scale 2^9, chained from
// scores_rescale10) and the per-row allowed max mx (B int32, scale 2^9,
// chained from zkob_rowmax causal -- edge RM2 byte-pins com_mx):
//   (R0) Dm[i,j] = MK[i,j]*(z_[i,j] - mx[i]) + (1 - MK[i,j])*SENT
//        (MK = public causal mask j<=i; SENT = LOW8+LEN8-1 = +1, the pinned
//        masked-position sentinel). Bound by the Dm-binding block: cD1 =
//        MLE of MK.*z_ at u_d, cD2 = MLE of MK.*mx_bcast at u_d (mx_bcast
//        never committed; its terminal opens com_mx at the row-bit suffix),
//        vDm = Dm~(u_d) opened vs com_Dm, and the plain-field identity
//          D~m(u_d) == cD1 - cD2 + SENT*(1 - k_MK),  k_MK = sum_b eq(u_d,b)MK(b)
//        (k_MK computed by the verifier itself; "Dm identity").
//   (R1) E[i,j] = X_E8[Dm[i,j] - LOW8] -- glu-style mapping lookup on
//        comb = Dm + r*E vs table + r*mapped; com_comb = com_Dm + r*com_E
//        homomorphically. By the sentinel row, E = 0 at masked positions BY
//        THE TABLE -- MK disappears from every downstream constraint.
//   (R2) S[i] = sum_j E[i,j] (no mask weight) -- row-sum sumcheck with the
//        PURE broadcast eq weight: verifier uses the rmsnorm eq_acc shortcut
//        (row-bit rounds only) instead of rebuild+fold.
//   (R3) r1 = 2^17*E + S_bcast - 2*P*S_bcast in [0, 2S) per entry
//        <=> P = round_half_up(2^16*E/S) exactly, P[masked] = 0 (E = 0 there).
//        Bound by V1 (c1 = MLE of E at u_r, pure eq weight), V2 (c2 = MLE of
//        P.*S_bcast at u_r, verbatim softmax 4.5), 4 plane openings of the
//        2x14-bit limb tensor L (r1 lo/hi, r2 lo/hi; r2 = 2S-1-r1) + S_id,
//        the limb range lookup (L vs tLookupRange(0, LEN_R8), LEN_R8 = 2^14
//        at real scale -- S <= 2^26 so r1, r2 < 2^27 < LEN_R8^2), and
//          (I1) 2^17*c1 + S_id - 2*c2 == v00 + LEN_R8*v10   "bracket r1 identity"
//          (I2) r~1 + r~2 + 1 == 2*S_id                     "bracket sum identity"
// SOUNDNESS INTERLOCK (audit MINOR-3): dropping the MK factor that softmax
// carried in the row-sum/V1/bracket weights is sound ONLY because masked E is
// exactly 0, which is enforced by the conjunction of THREE verifier checks --
// the Dm identity (Dm = SENT at masked), the mapping lookup (E = X_E8[Dm-LOW8])
// and the sentinel check (table[SENT] = 0; verify()-side, selftest-pinned).
// Any edit to the Dm block, the table loading, or the registered table
// semantics must re-establish masked-E=0 before trusting the MK-free weights.
// Zero prover advice; all committed tensors are deterministic in
// (z_, mx, MK, X_E8). z_/mx domains are NOT proof-bound here (delta from
// softmax R1): the chained rowmax instance on the same com_z/com_mx proves
// mx = allowed row max (so allowed Dm <= 0, never SENT); standalone, this
// driver guards them with honest-prover throws only (design 4.3 composition
// soundness note, the MINOR-5 class).
//
// Layout: B == NCOL required (both pow2); one generator set gen (size NCOL)
// commits every tensor row-wise, no padding anywhere; mx is ONE row of B
// values under the same gen (B == NCOL). Flat grid index = i*NCOL + j; L flat
// index = plane*B*NCOL + i*NCOL + j (plane = 2 MSBs).
// Exp lookup: D = B*NCOL, N = LEN8, NCOL <= N <= D, N | D (real: N = D = 2^20,
// pure phase2). Limb lookup: D_L = 4*D, N = LEN_R8, NCOL <= N <= D_L, N | D_L.
//
// FS schedule (one transcript, seed = run_seed:obligation_id) -- design 4.3
// schedule block: preamble u32s + 9 base commitments -> r, beta_E ->
// com_A_E8 -> alpha_E, u_E -> exp lookup + 3 IPAs -> u_d -> cD1 block ->
// cD2 block -> vDm + IPA(Dm) -> beta_L -> com_A_L -> alpha_L, u_L -> limb
// lookup + 3 IPAs -> u_b -> ev_S -> row-sum + 2 IPAs -> u_r -> c1/V1 + IPA ->
// c2/V2 + 2 IPAs -> v00..v11 + 4 plane IPAs -> S_id + IPA. 19 IPA openings.
// Verifier-only: I1, I2, the Dm identity.
//
// Usage:
//   zkob_softmax8 prove  <obdir> <seed> <z-int32.bin> <mx-int32.bin> <B> <NCOL>
//                        <LOW8> <LEN8> <expmap8-int32.bin> <LEN_R8> <gen.bin>
//                        <q.bin> [P-int32-out.bin]
//   zkob_softmax8 verify <obdir> <seed> <B> <NCOL> <LOW8> <LEN8>
//                        <expmap8-int32.bin> <LEN_R8> <gen.bin> <q.bin>
//   zkob_softmax8 selftest
// The driver does NOT mkdir the obdir. SENT = LOW8 + LEN8 - 1 is derived; the
// table's last entry must be 0 (the sentinel check, replacing glu's mapped(0)
// check). P chain file: unpadded B x NCOL int32, scale 2^16.
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
static const long long ENV = 1LL << 19;   // z_/mx honest envelope (design 4.3)

// load the LEN8 mapped exp-table values (int32) and run the sentinel check:
// the table's LAST entry (v = SENT) must map to 0 -- the pinned masked row.
static vector<int> load_expmap8(const string& path, uint len) {
    if (len != (1u << ceilLog2(len))) throw runtime_error("LEN8 must be pow2");
    FILE* f = open_or_die(path, "rb");
    vector<int> buf(len);
    if (fread(buf.data(), sizeof(int), len, f) != len)
        throw runtime_error("expmap8 table: short read");
    fclose(f);
    return buf;
}

static void layout_guards(uint B, uint NCOL, uint LEN8, uint LEN_R8, uint gen_size) {
    const uint D = B * NCOL, DL = 4 * D;
    if (B != NCOL) throw runtime_error("B != NCOL");
    if (B != (1u << ceilLog2(B))) throw runtime_error("B not a power of two");
    if (gen_size != NCOL) throw runtime_error("generator size != NCOL");
    if (LEN8 != (1u << ceilLog2(LEN8)) || LEN_R8 != (1u << ceilLog2(LEN_R8)))
        throw runtime_error("table lengths must be pow2");
    if (NCOL > LEN8 || LEN8 > D || D % LEN8)
        throw runtime_error("exp lookup layout needs NCOL <= LEN8 <= D, LEN8 | D");
    if (NCOL > LEN_R8 || LEN_R8 > DL || DL % LEN_R8)
        throw runtime_error("limb lookup layout needs NCOL <= LEN_R8 <= 4D, LEN_R8 | 4D");
}

// ---------------- prove ----------------
// evil (selftest only; honest PROCEDURE on an inconsistent witness):
//  1: E[i,j] += 1 at an ALLOWED (i,j); S, P, r1/r2, limbs recomputed from the
//     evil E -> only the exp lookup can reject (round 0).
//  2: Dm[i,j] := SENT at an ALLOWED (i,j) (E = 0 there by the table; S, P,
//     limbs recomputed consistently -- the "silently dropped probability"
//     forgery). Lookup sees a valid (SENT, 0) row; row-sum/brackets are
//     consistent with the evil E; cD1/cD2 are MLEs of the TRUE masked diffs
//     -> only the "Dm identity" can reject. THE certifying evil for R0.
//  3: P[i,j] += 1 at a MASKED (i,j); true r1 = -S < 0; r1 limbs = mod-LEN_R8^2
//     truncation, r2 limbs likewise -> "bracket r1 identity" (I1) rejects.
//  4: P[i,j] -= 1 at an unmasked diagonal entry with honest P >= 1;
//     r1' = r1 + 2S representable, limbs honest for r1'; r2 limbs = mod
//     truncation of the (negative) 2S-1-r1' -> I1 holds exactly, the limb
//     lookup passes -> only "bracket sum identity" (I2) rejects.
//  5: S[row] += 1; P, r1/r2, limbs recomputed from the evil S -> row-sum
//     round 0 rejects (ev_S no longer equals sum W_rs.*E).
//  6: V2 run with a corrupted broadcast buffer Sb[idx] += 1 (P, S, limbs all
//     honest) -> "IPA opening of V2 U_f2 vs com_S" rejects.
//  7: cD2 run with a corrupted broadcast buffer mxb[idx] += 1 at a MASKED idx
//     (claim absorbed from that run, so the sumcheck is self-consistent and,
//     the idx being masked, cD2's VALUE is unchanged -> the Dm identity stays
//     clean) -> "IPA opening of cD2 terminal vs com_mx" is the sole catcher.
//  8: L[0,idx] += LEN_R8 with a compensating borrow L[1,idx] -= 1 (out-of-
//     range lo limb; the reconstructed r1 -- hence I1/I2, every opening and
//     every commitment except com_L/com_A_L -- stays consistent; m_L committed
//     from the HONEST limbs) -> only the limb range lookup can reject (round 0).
static void prove(const string& obdir, const string& seed,
                  const vector<int>& zh, const vector<int>& mxh,
                  uint B, uint NCOL, int LOW8, uint LEN8,
                  const vector<int>& maph, uint LEN_R8,
                  const Commitment& gen, const G1Jacobian_t& Q,
                  const string& p_out,
                  int evil = 0, uint evil_i = 0, uint evil_j = 0) {
    const uint D = B * NCOL, DL = 4 * D;
    const uint logB = ceilLog2(B), logC = ceilLog2(NCOL);
    const uint logD = logB + logC, logDL = logD + 2;
    const uint n1_E = ceilLog2(D / LEN8), n1_L = ceilLog2(DL / LEN_R8);
    layout_guards(B, NCOL, LEN8, LEN_R8, gen.size);
    if (zh.size() != (size_t)D || mxh.size() != (size_t)B ||
        maph.size() != (size_t)LEN8)
        throw runtime_error("input dims");
    if (maph[LEN8 - 1] != 0)
        throw runtime_error("table sentinel (last entry) != 0");
    const long long SENT = (long long)LOW8 + (long long)LEN8 - 1;
    const long long LENR2 = (long long)LEN_R8 * (long long)LEN_R8;

    // ---- host integer chain (long long; all bounds < 2^63 per design 4.3) ----
    vector<long> Dmh(D), Eh(D), Sh(B), Ph(D), Lh(DL);
    for (uint i = 0; i < B; i++) {
        if (mxh[i] <= -ENV || mxh[i] >= ENV)
            throw runtime_error("mx outside the +-2^19 envelope");
        for (uint j = 0; j < NCOL; j++) {
            const size_t idx = (size_t)i * NCOL + j;
            const long long z = zh[idx];
            if (z <= -ENV || z >= ENV)
                throw runtime_error("z_ outside the +-2^19 envelope");
            if (j <= i) {
                const long long diff = z - (long long)mxh[i];
                if (diff > 0)
                    throw runtime_error("allowed diff > 0 (mx below the allowed row max)");
                if (diff < (long long)LOW8)
                    throw runtime_error("allowed diff below LOW8 (design 6.7 corner)");
                Dmh[idx] = diff;
            } else Dmh[idx] = SENT;
        }
    }
    if (evil == 2) {
        if (evil_j > evil_i) throw runtime_error("evil 2 setup: idx must be allowed");
        Dmh[(size_t)evil_i * NCOL + evil_j] = SENT;
    }
    for (size_t k = 0; k < D; k++) Eh[k] = maph[Dmh[k] - (long long)LOW8];
    if (evil == 1) {
        if (evil_j > evil_i) throw runtime_error("evil 1 setup: idx must be allowed");
        Eh[(size_t)evil_i * NCOL + evil_j] += 1;
    }
    for (uint i = 0; i < B; i++) {
        long long s = 0;
        for (uint j = 0; j < NCOL; j++) s += Eh[(size_t)i * NCOL + j];
        Sh[i] = s;
    }
    if (evil == 5) Sh[evil_i] += 1;
    for (uint i = 0; i < B; i++)
        if (Sh[i] < 1) throw runtime_error("S[i] < 1 (table floor violated)");
    for (uint i = 0; i < B; i++)
        for (uint j = 0; j < NCOL; j++) {
            const size_t idx = (size_t)i * NCOL + j;
            const bool mk = (j <= i);
            const long long S = Sh[i];
            const long long num = ((long long)Eh[idx] << (LOG_OUT + 1)) + S;
            long long P = num / (2 * S);          // floor; round-half-up of 2^16*E/S
            const bool ev_here = (i == evil_i && j == evil_j);
            if (evil == 3 && ev_here) {
                if (mk) throw runtime_error("evil 3 setup: idx must be masked");
                P += 1;
            }
            if (evil == 4 && ev_here) {
                if (!mk || i != j) throw runtime_error("evil 4 setup: pick a diagonal idx");
                if (P < 1) throw runtime_error("evil 4 setup: honest P < 1");
                P -= 1;
            }
            const long long r1 = num - 2 * P * S;            // true bracket value
            long long r1c, r2c;
            if ((evil == 3 || evil == 4) && ev_here) {
                r1c = ((r1 % LENR2) + LENR2) % LENR2;        // mod-LEN_R8^2 truncation
                long long r2t = 2 * S - 1 - r1c;
                r2c = ((r2t % LENR2) + LENR2) % LENR2;
            } else {
                if (r1 < 0 || r1 >= 2 * S)
                    throw runtime_error("r1 out of [0, 2S) (P not per spec)");
                if (r1 >= LENR2 || 2 * S - 1 - r1 >= LENR2)
                    throw runtime_error("bracket residual >= LEN_R8^2");
                r1c = r1;
                r2c = 2 * S - 1 - r1;
            }
            Ph[idx] = P;
            Lh[0 * (size_t)D + idx] = r1c % LEN_R8;
            Lh[1 * (size_t)D + idx] = r1c / LEN_R8;
            Lh[2 * (size_t)D + idx] = r2c % LEN_R8;
            Lh[3 * (size_t)D + idx] = r2c / LEN_R8;
        }
    // evil 8: out-of-range limb with a compensating borrow -- the reconstructed
    // r1 (and so I1/I2 and every commitment except com_L/com_A_L) is UNCHANGED;
    // only the limb range lookup stands between this forgery and ACCEPT.
    vector<long> Lh_honest;
    if (evil == 8) {
        size_t idx = (size_t)evil_i * NCOL + evil_j;
        if (Lh[(size_t)D + idx] < 1)      // need r1 >= LEN_R8: scan for a hi limb >= 1
            for (idx = 0; idx < D && Lh[(size_t)D + idx] < 1; idx++);
        if (idx >= (size_t)D)
            throw runtime_error("evil 8 setup: no entry with r1 >= LEN_R8");
        Lh_honest = Lh;
        Lh[idx] += (long)LEN_R8;          // plane 0: r1 lo limb leaves [0, LEN_R8)
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
    FrTensor mx_t(B, mxh.data());
    FrTensor Dm_t(D, Dmh.data());
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
    if (evil == 6) {
        k_bump<<<1,1>>>(Sb.gpu_data, evil_i * NCOL + evil_j, F_ONE, 0);
        cudaDeviceSynchronize();
    }
    FrTensor mxb(D);                              // mx broadcast, NEVER committed
    k_bcast_rows<<<(D + 255) / 256, 256>>>(mx_t.gpu_data, mxb.gpu_data, NCOL, D);
    cudaDeviceSynchronize();
    if (evil == 7) {
        if (evil_j <= evil_i) throw runtime_error("evil 7 setup: idx must be masked");
        k_bump<<<1,1>>>(mxb.gpu_data, evil_i * NCOL + evil_j, F_ONE, 0);
        cudaDeviceSynchronize();
    }

    FrTensor mapped_t(LEN8, maph.data());
    tLookupRangeMapping tlE(LOW8, LEN8, mapped_t);
    FrTensor m_E8 = tlE.prep(Dm_t);               // multiplicities of the Dm indices
    tLookupRange tlR(0, LEN_R8);
    // limb multiplicities vs the public range table (evil 8: from the HONEST
    // limbs -- the forged value is outside the table, where prep's unchecked
    // atomicAdd would write out of bounds; committing the honest multiplicities
    // is also the forging prover's best move)
    FrTensor m_L = (evil == 8) ? tlR.prep(FrTensor(DL, Lh_honest.data()))
                               : tlR.prep(L_t);

    // ---- commitments (single gen, no padding; mx = ONE row of B values) ----
    G1TensorJacobian com_z  = gen.commit(z_t);
    G1TensorJacobian com_mx = gen.commit(mx_t);
    G1TensorJacobian com_Dm = gen.commit(Dm_t);
    G1TensorJacobian com_E  = gen.commit(E_t);
    G1TensorJacobian com_P  = gen.commit(P_t);
    G1TensorJacobian com_S  = gen.commit(S_t);
    G1TensorJacobian com_L  = gen.commit(L_t);
    G1TensorJacobian com_mE = gen.commit(m_E8);
    G1TensorJacobian com_mL = gen.commit(m_L);
    com_z.save(obdir + "/com_z.bin");
    com_mx.save(obdir + "/com_mx.bin");
    com_Dm.save(obdir + "/com_Dm.bin");
    com_E.save(obdir + "/com_E.bin");
    com_P.save(obdir + "/com_P.bin");
    com_S.save(obdir + "/com_S.bin");
    com_L.save(obdir + "/com_L.bin");
    com_mE.save(obdir + "/com_m_E8.bin");
    com_mL.save(obdir + "/com_m_L.bin");
    { FILE* f = open_or_die(obdir + "/dims.bin", "wb");
      uint32_t d0[2] = {B, NCOL}; int32_t lo = LOW8;
      uint32_t d1[3] = {LEN8, LEN_R8, LOG_OUT}; int32_t se = (int32_t)SENT;
      fwrite(d0, sizeof(uint32_t), 2, f); fwrite(&lo, sizeof(int32_t), 1, f);
      fwrite(d1, sizeof(uint32_t), 3, f); fwrite(&se, sizeof(int32_t), 1, f);
      fclose(f); }

    // ---- transcript ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "NCOL", NCOL);
    absorb_u32(tr, "LOW8", (uint32_t)LOW8); absorb_u32(tr, "LEN8", LEN8);
    absorb_u32(tr, "LEN_R8", LEN_R8); absorb_u32(tr, "LOG_OUT", LOG_OUT);
    absorb_u32(tr, "SENT", (uint32_t)(int32_t)SENT);
    absorb_g1_tensor(tr, "com_z", com_z);
    absorb_g1_tensor(tr, "com_mx", com_mx);
    absorb_g1_tensor(tr, "com_Dm", com_Dm);
    absorb_g1_tensor(tr, "com_E", com_E);
    absorb_g1_tensor(tr, "com_P", com_P);
    absorb_g1_tensor(tr, "com_S", com_S);
    absorb_g1_tensor(tr, "com_L", com_L);
    absorb_g1_tensor(tr, "com_m_E8", com_mE);
    absorb_g1_tensor(tr, "com_m_L", com_mL);
    Fr_t r = fs_challenge_fr(tr);                 // exp pair combiner
    Fr_t beta_E = fs_challenge_fr(tr);

    // ---- obligation 1: exp mapping lookup (comb = Dm + r*E) ----
    FrTensor comb = Dm_t + E_t * r;
    FrTensor T_comb = tlE.table + tlE.mapped_vals * r;
    FrTensor A_E(D);
    tlookup_inv_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        comb.gpu_data, beta_E, A_E.gpu_data, D);
    cudaDeviceSynchronize();
    G1TensorJacobian com_AE = gen.commit(A_E);
    com_AE.save(obdir + "/com_A_E8.bin");
    absorb_g1_tensor(tr, "com_A_E8", com_AE);
    Fr_t alpha_E = fs_challenge_fr(tr);
    auto u_E = fs_challenge_vec(tr, logD);

    LookupProof pfE;
    vector<Fr_t> ws_E;
    {
        FrTensor BvE(LEN8);
        tlookup_inv_kernel<<<(LEN8+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            T_comb.gpu_data, beta_E, BvE.gpu_data, LEN8);
        cudaDeviceSynchronize();
        Fr_t alpha_sq = alpha_E * alpha_E;
        Fr_t Cc = alpha_sq - (BvE * m_E8).sum();
        Fr_t claim = alpha_E + alpha_sq;
        Fr_t inv_ratio = Fr_t{LEN8,0,0,0,0,0,0,0} / Fr_t{D,0,0,0,0,0,0,0};
        fs_phase1(claim, A_E, comb, BvE, T_comb, m_E8, alpha_E, beta_E, Cc,
                  inv_ratio, alpha_sq, u_E, tr, ws_E, pfE, evil != 1);
    }
    write_lookup(obdir + "/lookup_E8.bin", pfE);
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
    open_prove(A_E,  NCOL, gen, Q, u_ptE, obdir + "/ipa_A_E8.bin", tr);
    open_prove(comb, NCOL, gen, Q, u_ptE, obdir + "/ipa_comb.bin", tr);
    open_prove(m_E8, NCOL, gen, Q, u_mE,  obdir + "/ipa_m_E8.bin", tr);

    // ---- obligation 2: Dm-binding block (NEW vs softmax; design 4.3) ----
    auto u_d = fs_challenge_vec(tr, logD);
    vector<Fr_t> ud_col(u_d.begin(), u_d.begin() + logC);
    vector<Fr_t> ud_row(u_d.begin() + logC, u_d.end());
    // cD1: weight eq(u_d).*MK on z_
    HadamardProof hp_cd1;
    {
        FrTensor MKz = MK_t * z_t;
        hp_cd1.claim_H = MKz.multi_dim_me({ud_row, ud_col}, {B, NCOL});
    }
    absorb_fr(tr, "cD1", hp_cd1.claim_H);
    vector<Fr_t> ws_cd1;
    {
        FrTensor eq_d = build_eq_tensor(u_d);
        FrTensor W(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(eq_d.gpu_data, MK_t.gpu_data, W.gpu_data, D);
        cudaDeviceSynchronize();
        FrTensor Sc(z_t), Uc(ones_t);
        fs_hadamard(hp_cd1.claim_H, W, Sc, Uc, tr, ws_cd1, hp_cd1, true);
    }
    absorb_fr(tr, "S_f2", hp_cd1.S_f2); absorb_fr(tr, "U_f2", hp_cd1.U_f2);
    write_hp(obdir + "/hp_cD1.bin", hp_cd1);
    vector<Fr_t> pt_cd1(ws_cd1.rbegin(), ws_cd1.rend());
    if (evil == 0) {
        vector<Fr_t> u_col(pt_cd1.begin(), pt_cd1.begin() + logC);
        vector<Fr_t> u_row(pt_cd1.begin() + logC, pt_cd1.end());
        if (!fr_eq(hp_cd1.S_f2, z_t.multi_dim_me({u_row, u_col}, {B, NCOL})) ||
            !fr_eq(hp_cd1.U_f2, F_ONE))
            throw runtime_error("cD1 terminal != multi_dim_me / 1 (convention bug)");
    }
    open_prove(z_t, NCOL, gen, Q, pt_cd1, obdir + "/ipa_z_cD1.bin", tr);

    // cD2: same weight on the never-committed broadcast mxb
    HadamardProof hp_cd2;
    {
        FrTensor MKm = MK_t * mxb;
        hp_cd2.claim_H = MKm.multi_dim_me({ud_row, ud_col}, {B, NCOL});
    }
    absorb_fr(tr, "cD2", hp_cd2.claim_H);
    vector<Fr_t> ws_cd2;
    {
        FrTensor eq_d = build_eq_tensor(u_d);
        FrTensor W(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(eq_d.gpu_data, MK_t.gpu_data, W.gpu_data, D);
        cudaDeviceSynchronize();
        FrTensor Sc(mxb), Uc(ones_t);
        fs_hadamard(hp_cd2.claim_H, W, Sc, Uc, tr, ws_cd2, hp_cd2, true);
    }
    absorb_fr(tr, "S_f2", hp_cd2.S_f2); absorb_fr(tr, "U_f2", hp_cd2.U_f2);
    write_hp(obdir + "/hp_cD2.bin", hp_cd2);
    vector<Fr_t> pt_cd2(ws_cd2.rbegin(), ws_cd2.rend());
    vector<Fr_t> pt_cd2_rows(pt_cd2.begin() + logC, pt_cd2.end());
    if (evil == 0) {
        vector<Fr_t> u_col(pt_cd2.begin(), pt_cd2.begin() + logC);
        if (!fr_eq(hp_cd2.S_f2, mxb.multi_dim_me({pt_cd2_rows, u_col}, {B, NCOL})) ||
            !fr_eq(hp_cd2.S_f2, mx_t.multi_dim_me({pt_cd2_rows}, {B})) ||
            !fr_eq(hp_cd2.U_f2, F_ONE))
            throw runtime_error("cD2 terminal != multi_dim_me / bcast suffix (convention bug)");
    }
    // the never-committed broadcast is pinned to com_mx: its MLE at pt_cd2
    // equals the row vector's MLE at the row-bit suffix
    open_prove(mx_t, B, gen, Q, pt_cd2_rows, obdir + "/ipa_mx_cD2.bin", tr);

    // vDm: opening of com_Dm at u_d (the Dm identity's LHS)
    Fr_t vDm = Dm_t.multi_dim_me({ud_row, ud_col}, {B, NCOL});
    absorb_fr(tr, "vDm", vDm);
    { FILE* f = open_or_die(obdir + "/vdm.bin", "wb");
      fwrite(&vDm, sizeof(Fr_t), 1, f); fclose(f); }
    open_prove(Dm_t, NCOL, gen, Q, u_d, obdir + "/ipa_Dm.bin", tr);

    // ---- obligation 3: limb range lookup ----
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
        FrTensor BvL(LEN_R8);
        tlookup_inv_kernel<<<(LEN_R8+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            tlR.table.gpu_data, beta_L, BvL.gpu_data, LEN_R8);
        cudaDeviceSynchronize();
        Fr_t alpha_sq = alpha_L * alpha_L;
        Fr_t Cc = alpha_sq - (BvL * m_L).sum();
        Fr_t claim = alpha_L + alpha_sq;
        Fr_t inv_ratio = Fr_t{LEN_R8,0,0,0,0,0,0,0} / Fr_t{DL,0,0,0,0,0,0,0};
        fs_phase1(claim, A_L, L_t, BvL, tlR.table, m_L, alpha_L, beta_L, Cc,
                  inv_ratio, alpha_sq, u_L, tr, ws_L, pfL, evil != 8);
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

    // ---- obligation 4: row-sum sumcheck (R2; PURE broadcast eq weight) ----
    auto u_b = fs_challenge_vec(tr, logB);
    HadamardProof hp_rs;
    hp_rs.claim_H = S_t.multi_dim_me({u_b}, {B});             // ev_S
    absorb_fr(tr, "ev_S", hp_rs.claim_H);
    vector<Fr_t> ws_rs;
    {
        FrTensor eq_b = build_eq_tensor(u_b);
        FrTensor W_rs(D);
        k_bcast_rows<<<(D + 255) / 256, 256>>>(eq_b.gpu_data, W_rs.gpu_data, NCOL, D);
        cudaDeviceSynchronize();
        FrTensor Sc(E_t), Uc(ones_t);
        fs_hadamard(hp_rs.claim_H, W_rs, Sc, Uc, tr, ws_rs, hp_rs, evil != 5);
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

    // ---- obligation 5: bracket sumcheck V1 (pure eq weight on E) ----
    auto u_r = fs_challenge_vec(tr, logD);
    vector<Fr_t> ur_col(u_r.begin(), u_r.begin() + logC);
    vector<Fr_t> ur_row(u_r.begin() + logC, u_r.end());
    HadamardProof hp_v1;
    hp_v1.claim_H = E_t.multi_dim_me({ur_row, ur_col}, {B, NCOL});   // c1
    absorb_fr(tr, "c1", hp_v1.claim_H);
    vector<Fr_t> ws_v1;
    {
        FrTensor eq_r = build_eq_tensor(u_r);
        FrTensor Sc(E_t), Uc(ones_t);
        fs_hadamard(hp_v1.claim_H, eq_r, Sc, Uc, tr, ws_v1, hp_v1, true);
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

    // ---- obligation 6: bracket sumcheck V2 (P.*S_bcast at u_r) ----
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

    // ---- obligation 7: residual reconstruction (L plane openings + S_id) ----
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

    if (evil == 0) {    // convention sanity: the Dm identity holds in the field
        FrTensor eq_d = build_eq_tensor(u_d);
        FrTensor eqMK(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(eq_d.gpu_data, MK_t.gpu_data, eqMK.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t k_MK = eqMK.sum();
        Fr_t F_SENT = (SENT >= 0) ? Fr_t{(uint)SENT,0,0,0,0,0,0,0}
                                  : h_scalar(F_ZERO, Fr_t{(uint)(-SENT),0,0,0,0,0,0,0}, 1);
        Fr_t rhs = h_scalar(hp_cd1.claim_H, hp_cd2.claim_H, 1);
        rhs = h_scalar(rhs, h_scalar(F_SENT, h_scalar(F_ONE, k_MK, 1), 2), 0);
        if (!fr_eq(vDm, rhs))
            throw runtime_error("Dm identity violated on honest witness (convention bug)");
    }
    cout << "PROVED softmax8 obligation -> " << obdir << endl;
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
                   uint B, uint NCOL, int LOW8, uint LEN8,
                   const vector<int>& maph, uint LEN_R8,
                   const Commitment& gen, const G1Jacobian_t& Q,
                   string* reason = nullptr) {
    const uint D = B * NCOL, DL = 4 * D;
    const uint logB = ceilLog2(B), logC = ceilLog2(NCOL);
    const uint logD = logB + logC, logDL = logD + 2;
    const uint n1_E = ceilLog2(D / LEN8), n2_E = ceilLog2(LEN8);
    const uint n1_L = ceilLog2(DL / LEN_R8), n2_L = ceilLog2(LEN_R8);
    if (B != NCOL || B != (1u << logB) || gen.size != NCOL ||
        LEN8 != (1u << n2_E) || LEN_R8 != (1u << n2_L) ||
        NCOL > LEN8 || LEN8 > D || D % LEN8 ||
        NCOL > LEN_R8 || LEN_R8 > DL || DL % LEN_R8)
        RJ("bad layout params");
    if (maph.size() != (size_t)LEN8 || maph[LEN8 - 1] != 0)
        RJ("table sentinel (last entry) != 0");
    const long long SENT = (long long)LOW8 + (long long)LEN8 - 1;
    { FILE* f = open_or_die(obdir + "/dims.bin", "rb");
      uint32_t d0[2]; int32_t lo; uint32_t d1[3]; int32_t se;
      if (fread(d0, sizeof(uint32_t), 2, f) != 2 ||
          fread(&lo, sizeof(int32_t), 1, f) != 1 ||
          fread(d1, sizeof(uint32_t), 3, f) != 3 ||
          fread(&se, sizeof(int32_t), 1, f) != 1) { fclose(f); RJ("dims.bin short read"); }
      fclose(f);
      if (d0[0] != B || d0[1] != NCOL || lo != LOW8 ||
          d1[0] != LEN8 || d1[1] != LEN_R8 || d1[2] != LOG_OUT ||
          se != (int32_t)SENT)
          RJ("dims.bin mismatch"); }

    G1TensorJacobian com_z (obdir + "/com_z.bin");
    G1TensorJacobian com_mx(obdir + "/com_mx.bin");
    G1TensorJacobian com_Dm(obdir + "/com_Dm.bin");
    G1TensorJacobian com_E (obdir + "/com_E.bin");
    G1TensorJacobian com_P (obdir + "/com_P.bin");
    G1TensorJacobian com_S (obdir + "/com_S.bin");
    G1TensorJacobian com_L (obdir + "/com_L.bin");
    G1TensorJacobian com_mE(obdir + "/com_m_E8.bin");
    G1TensorJacobian com_mL(obdir + "/com_m_L.bin");
    G1TensorJacobian com_AE(obdir + "/com_A_E8.bin");
    G1TensorJacobian com_AL(obdir + "/com_A_L.bin");
    if (com_z.size != B || com_mx.size != 1 || com_Dm.size != B ||
        com_E.size != B || com_P.size != B || com_S.size != 1 ||
        com_L.size != 4 * B || com_AE.size != B || com_AL.size != 4 * B ||
        com_mE.size != LEN8 / NCOL || com_mL.size != LEN_R8 / NCOL)
        RJ("commitment row counts");

    LookupProof pfE = read_lookup(obdir + "/lookup_E8.bin");
    LookupProof pfL = read_lookup(obdir + "/lookup_L.bin");
    HadamardProof hp_cd1 = read_hp(obdir + "/hp_cD1.bin");
    HadamardProof hp_cd2 = read_hp(obdir + "/hp_cD2.bin");
    HadamardProof hp_rs = read_hp(obdir + "/hp_rs.bin");
    HadamardProof hp_v1 = read_hp(obdir + "/hp_v1.bin");
    HadamardProof hp_v2 = read_hp(obdir + "/hp_v2.bin");
    Fr_t vDm;
    { FILE* f = open_or_die(obdir + "/vdm.bin", "rb");
      if (fread(&vDm, sizeof(Fr_t), 1, f) != 1) { fclose(f); RJ("vdm.bin short read"); }
      fclose(f); }
    Fr_t lv[5];
    { FILE* f = open_or_die(obdir + "/lvals.bin", "rb");
      if (fread(lv, sizeof(Fr_t), 5, f) != 5) { fclose(f); RJ("lvals.bin short read"); }
      fclose(f); }
    if (pfE.ev.size() != 4 * logD) RJ("exp lookup round count");
    if (pfL.ev.size() != 4 * logDL) RJ("limb lookup round count");
    if (hp_cd1.ev.size() != 4 * logD || hp_cd2.ev.size() != 4 * logD ||
        hp_rs.ev.size() != 4 * logD || hp_v1.ev.size() != 4 * logD ||
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
    absorb_u32(tr, "LOW8", (uint32_t)LOW8); absorb_u32(tr, "LEN8", LEN8);
    absorb_u32(tr, "LEN_R8", LEN_R8); absorb_u32(tr, "LOG_OUT", LOG_OUT);
    absorb_u32(tr, "SENT", (uint32_t)(int32_t)SENT);
    absorb_g1_tensor(tr, "com_z", com_z);
    absorb_g1_tensor(tr, "com_mx", com_mx);
    absorb_g1_tensor(tr, "com_Dm", com_Dm);
    absorb_g1_tensor(tr, "com_E", com_E);
    absorb_g1_tensor(tr, "com_P", com_P);
    absorb_g1_tensor(tr, "com_S", com_S);
    absorb_g1_tensor(tr, "com_L", com_L);
    absorb_g1_tensor(tr, "com_m_E8", com_mE);
    absorb_g1_tensor(tr, "com_m_L", com_mL);
    Fr_t r = fs_challenge_fr(tr);
    Fr_t beta_E = fs_challenge_fr(tr);
    absorb_g1_tensor(tr, "com_A_E8", com_AE);
    Fr_t alpha_E = fs_challenge_fr(tr);
    auto u_E = fs_challenge_vec(tr, logD);

    // combined commitment com_Dm + r*com_E, formed homomorphically (1-thread
    // helpers; batched G1 kernels are -dlto miscompile bait)
    vector<G1Jacobian_t> hcomb(B);
    {
        vector<G1Jacobian_t> hd(B), he(B);
        cudaMemcpy(hd.data(), com_Dm.gpu_data, B * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(he.data(), com_E.gpu_data, B * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        for (uint j = 0; j < B; j++) hcomb[j] = h_add(hd[j], h_mul(he[j], r));
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
        FrTensor mapped_t(LEN8, maph.data());
        tLookupRangeMapping tlE(LOW8, LEN8, mapped_t);
        FrTensor T_comb = tlE.table + tlE.mapped_vals * r;
        FrTensor B_pub(LEN8);
        tlookup_inv_kernel<<<(LEN8+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            T_comb.gpu_data, beta_E, B_pub.gpu_data, LEN8);
        cudaDeviceSynchronize();
        vector<Fr_t> ws2(ws_E.begin() + n1_E, ws_E.end());
        B_f = fold_public(B_pub, ws2);
        T_f = fold_public(T_comb, ws2);
    }
    {
        Fr_t inv_ratio = h_scalar({LEN8,0,0,0,0,0,0,0}, inv({D,0,0,0,0,0,0,0}), 2);
        Fr_t t1 = h_scalar(alpha_acc, h_scalar(pfE.A_f, h_scalar(pfE.S_f, beta_E, 0), 2), 2);
        Fr_t t2 = h_scalar(inv_ratio, h_scalar(alphasq_acc, h_scalar(B_f, h_scalar(T_f, beta_E, 0), 2), 2), 2);
        Fr_t t4 = h_scalar(inv_ratio, h_scalar(pfE.m_f, B_f, 2), 2);
        Fr_t rhs = h_scalar(h_scalar(h_scalar(t1, t2, 0), pfE.A_f, 0), t4, 1);
        if (!fr_eq(cur, rhs)) RJ("exp lookup terminal identity");
    }
    absorb_fr(tr, "A_f", pfE.A_f); absorb_fr(tr, "S_f", pfE.S_f); absorb_fr(tr, "m_f", pfE.m_f);
    vector<Fr_t> u_ptE(ws_E.rbegin(), ws_E.rend());
    vector<Fr_t> u_mE(ws_E.rbegin(), ws_E.rend() - n1_E);
    if (!open_verify(com_AE, gen, NCOL, Q, u_ptE, pfE.A_f, obdir + "/ipa_A_E8.bin", tr))
        RJ("IPA opening of A_f vs com_A_E8");
    if (!open_verify(com_comb, gen, NCOL, Q, u_ptE, pfE.S_f, obdir + "/ipa_comb.bin", tr))
        RJ("IPA opening of S_f vs com_Dm + r*com_E");
    if (!open_verify(com_mE, gen, NCOL, Q, u_mE, pfE.m_f, obdir + "/ipa_m_E8.bin", tr))
        RJ("IPA opening of m_f vs com_m_E8");

    // ---- obligation 2: Dm-binding block ----
    auto u_d = fs_challenge_vec(tr, logD);
    // cD1
    absorb_fr(tr, "cD1", hp_cd1.claim_H);
    cur = hp_cd1.claim_H;
    vector<Fr_t> ws_cd1;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_cd1.ev[4*k], hp_cd1.ev[4*k+1], hp_cd1.ev[4*k+2], hp_cd1.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("cD1 round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_cd1.push_back(w);
        cur = lagrange4(e, w, inv6);
    }
    if (!fr_eq(hp_cd1.U_f2, F_ONE)) RJ("cD1 U_f2 != 1");
    {
        FrTensor eq_d = build_eq_tensor(u_d);
        FrTensor W(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(eq_d.gpu_data, MK_t.gpu_data, W.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t W_f = fold_public(W, ws_cd1);
        if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp_cd1.S_f2, hp_cd1.U_f2, 2), 2)))
            RJ("cD1 terminal identity");
    }
    absorb_fr(tr, "S_f2", hp_cd1.S_f2); absorb_fr(tr, "U_f2", hp_cd1.U_f2);
    vector<Fr_t> pt_cd1(ws_cd1.rbegin(), ws_cd1.rend());
    if (!open_verify(com_z, gen, NCOL, Q, pt_cd1, hp_cd1.S_f2, obdir + "/ipa_z_cD1.bin", tr))
        RJ("IPA opening of cD1 terminal vs com_z");
    // cD2
    absorb_fr(tr, "cD2", hp_cd2.claim_H);
    cur = hp_cd2.claim_H;
    vector<Fr_t> ws_cd2;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_cd2.ev[4*k], hp_cd2.ev[4*k+1], hp_cd2.ev[4*k+2], hp_cd2.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("cD2 round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_cd2.push_back(w);
        cur = lagrange4(e, w, inv6);
    }
    if (!fr_eq(hp_cd2.U_f2, F_ONE)) RJ("cD2 U_f2 != 1");
    {
        FrTensor eq_d = build_eq_tensor(u_d);
        FrTensor W(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(eq_d.gpu_data, MK_t.gpu_data, W.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t W_f = fold_public(W, ws_cd2);
        if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp_cd2.S_f2, hp_cd2.U_f2, 2), 2)))
            RJ("cD2 terminal identity");
    }
    absorb_fr(tr, "S_f2", hp_cd2.S_f2); absorb_fr(tr, "U_f2", hp_cd2.U_f2);
    vector<Fr_t> pt_cd2(ws_cd2.rbegin(), ws_cd2.rend());
    vector<Fr_t> pt_cd2_rows(pt_cd2.begin() + logC, pt_cd2.end());
    // the never-committed broadcast is pinned to com_mx here: its MLE at
    // pt_cd2 equals the row vector's MLE at the row-bit suffix
    if (!open_verify(com_mx, gen, B, Q, pt_cd2_rows, hp_cd2.S_f2,
                     obdir + "/ipa_mx_cD2.bin", tr))
        RJ("IPA opening of cD2 terminal vs com_mx");
    // vDm
    absorb_fr(tr, "vDm", vDm);
    if (!open_verify(com_Dm, gen, NCOL, Q, u_d, vDm, obdir + "/ipa_Dm.bin", tr))
        RJ("IPA opening of vDm vs com_Dm");

    // ---- obligation 3: limb lookup rounds ----
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
        tLookupRange tlR(0, LEN_R8);
        FrTensor B_pub(LEN_R8);
        tlookup_inv_kernel<<<(LEN_R8+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            tlR.table.gpu_data, beta_L, B_pub.gpu_data, LEN_R8);
        cudaDeviceSynchronize();
        vector<Fr_t> ws2(ws_L.begin() + n1_L, ws_L.end());
        B_f = fold_public(B_pub, ws2);
        T_f = fold_public(tlR.table, ws2);
    }
    {
        Fr_t inv_ratio = h_scalar({LEN_R8,0,0,0,0,0,0,0}, inv({DL,0,0,0,0,0,0,0}), 2);
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

    // ---- obligation 4: row-sum sumcheck (pure broadcast eq weight:
    //      rmsnorm eq_acc shortcut, row rounds only) ----
    auto u_b = fs_challenge_vec(tr, logB);
    absorb_fr(tr, "ev_S", hp_rs.claim_H);
    cur = hp_rs.claim_H;
    Fr_t eq_acc = F_ONE;
    vector<Fr_t> ws_rs;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_rs.ev[4*k], hp_rs.ev[4*k+1], hp_rs.ev[4*k+2], hp_rs.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("row-sum round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_rs.push_back(w);
        cur = lagrange4(e, w, inv6);
        if (k < logB)   // row bits are the MSBs; column rounds carry eq factor 1
            eq_acc = h_scalar(eq_acc, my_eq(u_b[logB - 1 - k], w), 2);
    }
    if (!fr_eq(hp_rs.U_f2, F_ONE)) RJ("row-sum U_f2 != 1");
    if (!fr_eq(cur, h_scalar(eq_acc, h_scalar(hp_rs.S_f2, hp_rs.U_f2, 2), 2)))
        RJ("row-sum terminal identity");
    absorb_fr(tr, "S_f2", hp_rs.S_f2); absorb_fr(tr, "U_f2", hp_rs.U_f2);
    vector<Fr_t> pt_rs(ws_rs.rbegin(), ws_rs.rend());
    if (!open_verify(com_E, gen, NCOL, Q, pt_rs, hp_rs.S_f2, obdir + "/ipa_E_rs.bin", tr))
        RJ("IPA opening of E (row-sum) vs com_E");
    if (!open_verify(com_S, gen, NCOL, Q, u_b, hp_rs.claim_H, obdir + "/ipa_S_rs.bin", tr))
        RJ("IPA opening of ev_S vs com_S");

    // ---- obligation 5: bracket sumcheck V1 (pure eq weight) ----
    auto u_r = fs_challenge_vec(tr, logD);
    vector<Fr_t> ur_row(u_r.begin() + logC, u_r.end());
    absorb_fr(tr, "c1", hp_v1.claim_H);
    cur = hp_v1.claim_H;
    eq_acc = F_ONE;
    vector<Fr_t> ws_v1;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_v1.ev[4*k], hp_v1.ev[4*k+1], hp_v1.ev[4*k+2], hp_v1.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("V1 round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_v1.push_back(w);
        cur = lagrange4(e, w, inv6);
        eq_acc = h_scalar(eq_acc, my_eq(u_r[logD - 1 - k], w), 2);
    }
    if (!fr_eq(hp_v1.U_f2, F_ONE)) RJ("V1 U_f2 != 1");
    if (!fr_eq(cur, h_scalar(eq_acc, h_scalar(hp_v1.S_f2, hp_v1.U_f2, 2), 2)))
        RJ("V1 terminal identity");
    absorb_fr(tr, "S_f2", hp_v1.S_f2); absorb_fr(tr, "U_f2", hp_v1.U_f2);
    vector<Fr_t> pt1(ws_v1.rbegin(), ws_v1.rend());
    if (!open_verify(com_E, gen, NCOL, Q, pt1, hp_v1.S_f2, obdir + "/ipa_E_v1.bin", tr))
        RJ("IPA opening of E (V1) vs com_E");

    // ---- obligation 6: bracket sumcheck V2 (pure eq weight) ----
    absorb_fr(tr, "c2", hp_v2.claim_H);
    cur = hp_v2.claim_H;
    eq_acc = F_ONE;
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

    // ---- obligation 7: L plane openings + S_id ----
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

    // ---- verifier-only plain-field identities (Schwartz-Zippel) ----
    const Fr_t F_LENR = {LEN_R8, 0, 0, 0, 0, 0, 0, 0};
    const Fr_t F_2P17 = {1u << (LOG_OUT + 1), 0, 0, 0, 0, 0, 0, 0};
    Fr_t rt1 = h_scalar(lv[0], h_scalar(F_LENR, lv[1], 2), 0);   // v00 + LEN_R8*v10
    Fr_t rt2 = h_scalar(lv[2], h_scalar(F_LENR, lv[3], 2), 0);   // v01 + LEN_R8*v11
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
    {   // the Dm identity: D~m(u_d) == cD1 - cD2 + SENT*(1 - k_MK), with
        // k_MK = sum_b eq(u_d,b)*MK(b) computed by the verifier itself
        FrTensor eq_d = build_eq_tensor(u_d);
        FrTensor eqMK(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(eq_d.gpu_data, MK_t.gpu_data, eqMK.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t k_MK = eqMK.sum();
        Fr_t F_SENT = (SENT >= 0) ? Fr_t{(uint)SENT,0,0,0,0,0,0,0}
                                  : h_scalar(F_ZERO, Fr_t{(uint)(-SENT),0,0,0,0,0,0,0}, 1);
        Fr_t rhs = h_scalar(hp_cd1.claim_H, hp_cd2.claim_H, 1);  // cD1 - cD2
        rhs = h_scalar(rhs, h_scalar(F_SENT, h_scalar(F_ONE, k_MK, 1), 2), 0);
        if (!fr_eq(vDm, rhs)) RJ("Dm identity");
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
static bool files_equal(const string& a, const string& b) {
    long sa = file_size(a), sb = file_size(b);
    if (sa != sb || sa == 0) return false;
    FILE* fa = open_or_die(a, "rb");
    FILE* fb = open_or_die(b, "rb");
    vector<char> ba(sa), bb(sb);
    bool ok = fread(ba.data(), 1, sa, fa) == (size_t)sa &&
              fread(bb.data(), 1, sb, fb) == (size_t)sb &&
              memcmp(ba.data(), bb.data(), sa) == 0;
    fclose(fa); fclose(fb);
    return ok;
}

static const char* PROOF_FILES[] = {
    "dims.bin",
    "com_z.bin", "com_mx.bin", "com_Dm.bin", "com_E.bin", "com_P.bin",
    "com_S.bin", "com_L.bin", "com_m_E8.bin", "com_m_L.bin",
    "com_A_E8.bin", "com_A_L.bin",
    "lookup_E8.bin", "lookup_L.bin",
    "hp_cD1.bin", "hp_cD2.bin", "hp_rs.bin", "hp_v1.bin", "hp_v2.bin",
    "vdm.bin", "lvals.bin",
    "ipa_A_E8.bin", "ipa_comb.bin", "ipa_m_E8.bin",
    "ipa_z_cD1.bin", "ipa_mx_cD2.bin", "ipa_Dm.bin",
    "ipa_A_L.bin", "ipa_L_lk.bin", "ipa_m_L.bin",
    "ipa_E_rs.bin", "ipa_S_rs.bin",
    "ipa_E_v1.bin", "ipa_P_v2.bin", "ipa_S_v2.bin",
    "ipa_L00.bin", "ipa_L10.bin", "ipa_L01.bin", "ipa_L11.bin", "ipa_S_id.bin"};
static long tamper_offset(const string& f) {
    if (f == "dims.bin") return 0;
    if (f == "lvals.bin" || f == "vdm.bin") return 4;   // second byte of an Fr
    if (f.substr(0, 4) == "ipa_") return -32;           // a_final
    if (f.substr(0, 4) == "com_") return 24;            // first point, x limbs
    return 4 + 32;                                      // lookup/hp: round-0 evaluation
}

// toy mapped table: mapped[k] = k+1 (>= 1 keeps S >= 1) EXCEPT the pinned
// sentinel last row mapped[LEN8-1] = 0 (masked positions map to E = 0)
static vector<int> toy_table(uint LEN8) {
    vector<int> maph(LEN8);
    for (uint k = 0; k < LEN8; k++) maph[k] = (int)k + 1;
    maph[LEN8 - 1] = 0;
    return maph;
}
// toy rowmax-style host max: mx[i] = max_{j<=i} z[i,j]
static vector<int> host_mx(const vector<int>& zh, uint B, uint NCOL) {
    vector<int> mx(B);
    for (uint i = 0; i < B; i++) {
        int m = zh[(size_t)i * NCOL];
        for (uint j = 1; j <= i && j < NCOL; j++)
            m = max(m, zh[(size_t)i * NCOL + j]);
        mx[i] = m;
    }
    return mx;
}

static bool selftest_case(uint B, int LOW8, uint LEN8, uint LEN_R8) {
    const uint NCOL = B, D = B * NCOL;
    cout << "==== selftest case B=NCOL=" << B << " LOW8=" << LOW8
         << " LEN8=" << LEN8 << " LEN_R8=" << LEN_R8
         << " (n1_E=" << ceilLog2(D / LEN8) << ", n1_L=" << ceilLog2(4 * D / LEN_R8)
         << ", SENT=" << (LOW8 + (int)LEN8 - 1) << ") ====" << endl;
    vector<int> maph = toy_table(LEN8);
    // toy z in [-R, R] with R = |LOW8|/2 so every allowed diff >= LOW8
    const int R = (-LOW8) / 2;
    srand(8242 + B);
    vector<int> zh(D);
    for (uint k = 0; k < D; k++) zh[k] = (rand() % (2 * R + 1)) - R;
    vector<int> mxh = host_mx(zh, B, NCOL);
    Commitment gen = Commitment::random(NCOL);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_softmax8_ob";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:softmax8";
    bool all = true;

    prove(obdir, seed, zh, mxh, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q,
          "/tmp/zkob_softmax8_P.i32.bin");
    bool honest = verify(obdir, seed, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q);
    cout << (honest ? "PASS" : "FAIL") << ": honest case "
         << (honest ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && honest;

    // semantic evil modes: each rejected by EXACTLY the named check
    struct Evil { int mode; uint i, j; const char* expect; const char* what; };
    vector<Evil> evils = {
        {1, 2, 1, "exp lookup round 0",
                  "E[2,1]+=1 (allowed), S/P/limbs recomputed"},
        {2, 2, 0, "Dm identity",
                  "Dm[2,0]:=SENT (allowed), E=0 there, S/P/limbs recomputed"},
        {3, 0, 1, "bracket r1 identity",
                  "P[0,1]+=1 (MASKED), limbs = mod-LEN_R8^2 truncation"},
        {4, 1, 1, "bracket sum identity",
                  "P[1,1]-=1 (diagonal, honest P>=1), r1' = r1+2S limbs honest"},
        {5, 1, 0, "row-sum round 0",
                  "S[1]+=1, P/limbs recomputed"},
        {6, 2, 0, "IPA opening of V2 U_f2 vs com_S",
                  "V2 broadcast buffer Sb[2,0]+=1, all commitments honest"},
        {7, 0, 1, "IPA opening of cD2 terminal vs com_mx",
                  "cD2 broadcast buffer mxb[0,1]+=1 (masked idx), cD2 from that run"},
        {8, 3, 1, "limb lookup round 0",
                  "L[0,idx]+=LEN_R8, L[1,idx]-=1 (r1 value unchanged, m_L honest)"},
    };
    string evdir = "/tmp/zkob_softmax8_evil";
    mkdir(evdir.c_str(), 0755);
    for (auto& ev : evils) {
        prove(evdir, seed, zh, mxh, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q, "",
              ev.mode, ev.i, ev.j);
        string reason;
        bool rejected = !verify(evdir, seed, B, NCOL, LOW8, LEN8, maph, LEN_R8,
                                gen, Q, &reason);
        bool right = rejected && reason.find(ev.expect) != string::npos;
        cout << (right ? "PASS" : "FAIL") << ": evil=" << ev.mode << " (" << ev.what
             << ") rejected by [" << (rejected ? reason : string("NOT REJECTED"))
             << "], expected [" << ev.expect << "]" << endl;
        all = all && right;
    }

    // verify-side sentinel pin (audit MINOR-3 interlock): an HONEST proof dir
    // verified against a public table whose last entry != 0 must be rejected
    // by EXACTLY the sentinel check -- masked E = 0 (and hence the MK-free
    // row-sum/V1/bracket weights) relies on table[SENT] == 0
    {
        vector<int> bad(maph); bad[LEN8 - 1] = 5;
        const char* expect = "table sentinel (last entry) != 0";
        string reason;
        bool rejected = !verify(obdir, seed, B, NCOL, LOW8, LEN8, bad, LEN_R8,
                                gen, Q, &reason);
        bool right = rejected && reason.find(expect) != string::npos;
        cout << (right ? "PASS" : "FAIL")
             << ": sentinel-tampered public table rejected by ["
             << (rejected ? reason : string("NOT REJECTED"))
             << "], expected [" << expect << "]" << endl;
        all = all && right;
    }

    // byte tampers on every proof file (tamper, verify must reject, restore)
    for (const char* fn : PROOF_FILES) {
        long off = tamper_offset(fn);
        tamper_byte(obdir + "/" + fn, off, +1);
        bool rejected;
        try {
            rejected = !verify(obdir, seed, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q);
        } catch (const exception& e) {
            cout << "REJECT (parse throw): " << e.what() << endl;
            rejected = true;     // fail-closed on parse throws
        }
        tamper_byte(obdir + "/" + fn, off, -1);
        cout << (rejected ? "PASS" : "FAIL") << ": byte tamper " << fn << "@" << off
             << " rejected: " << (rejected ? "YES" : "NO(!!)") << endl;
        all = all && rejected;
    }
    bool restored = verify(obdir, seed, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q);
    cout << (restored ? "PASS" : "FAIL") << ": restored verify "
         << (restored ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && restored;
    cout << (all ? "CASE PASS" : "CASE FAIL") << endl;
    return all;
}

// honest-prover guard throws (completeness guards, incl. the mx-edge cases)
static bool selftest_guards() {
    cout << "==== selftest guards (honest-prover throws) ====" << endl;
    const uint B = 8, NCOL = 8, D = 64;
    const int LOW8 = -14; const uint LEN8 = 16, LEN_R8 = 32;
    vector<int> maph = toy_table(LEN8);
    srand(777);
    vector<int> zh(D);
    for (uint k = 0; k < D; k++) zh[k] = (rand() % 15) - 7;
    vector<int> mxh = host_mx(zh, B, NCOL);
    Commitment gen = Commitment::random(NCOL);
    Commitment gen_bad = Commitment::random(NCOL / 2);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_softmax8_guard";
    mkdir(obdir.c_str(), 0755);
    bool all = true;
    auto expect_throw = [&](const char* what, const char* expect, auto fn) {
        bool ok = false; string got;
        try { fn(); } catch (const exception& e) { got = e.what();
            ok = got.find(expect) != string::npos; }
        cout << (ok ? "PASS" : "FAIL") << ": guard (" << what << ") threw ["
             << (got.empty() ? "NOTHING(!!)" : got) << "], expected [" << expect
             << "]" << endl;
        all = all && ok;
    };
    expect_throw("B != NCOL", "B != NCOL", [&]{
        vector<int> z2((size_t)8 * 4), m2(8);
        prove(obdir, "g", z2, m2, 8, 4, LOW8, LEN8, maph, LEN_R8, gen, Q, ""); });
    expect_throw("non-pow2 LEN8", "pow2", [&]{
        vector<int> t3(12, 1); t3[11] = 0;
        prove(obdir, "g", zh, mxh, B, NCOL, LOW8, 12, t3, LEN_R8, gen, Q, ""); });
    expect_throw("gen size", "generator size != NCOL", [&]{
        prove(obdir, "g", zh, mxh, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen_bad, Q, ""); });
    expect_throw("exp layout LEN8 > D", "exp lookup layout", [&]{
        vector<int> t128 = toy_table(128);
        prove(obdir, "g", zh, mxh, B, NCOL, -126, 128, t128, LEN_R8, gen, Q, ""); });
    expect_throw("limb layout LEN_R8 < NCOL", "limb lookup layout", [&]{
        prove(obdir, "g", zh, mxh, B, NCOL, LOW8, LEN8, maph, 4, gen, Q, ""); });
    expect_throw("table sentinel != 0", "sentinel", [&]{
        vector<int> bad(maph); bad[LEN8 - 1] = 5;
        prove(obdir, "g", zh, mxh, B, NCOL, LOW8, LEN8, bad, LEN_R8, gen, Q, ""); });
    expect_throw("z envelope", "z_ outside", [&]{
        vector<int> z2(zh); z2[9] = 1 << 19;   // mx kept honest-original: the z
        prove(obdir, "g", z2, mxh, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q, ""); });
    expect_throw("mx envelope", "mx outside", [&]{
        vector<int> m2(mxh); m2[3] = -(1 << 19) - 1;
        prove(obdir, "g", zh, m2, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q, ""); });
    expect_throw("mx below row max (allowed diff > 0)", "allowed diff > 0", [&]{
        vector<int> m2(mxh); m2[4] -= 1;
        prove(obdir, "g", zh, m2, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q, ""); });
    expect_throw("allowed diff below LOW8", "allowed diff below LOW8", [&]{
        vector<int> m2(mxh); m2[7] = 9;     // toy z in [-7,7]: diff <= -16 < -14
        vector<int> z2(zh); z2[(size_t)7 * NCOL] = -7;
        for (uint j = 0; j <= 7; j++) z2[(size_t)7 * NCOL + j] = min(z2[(size_t)7 * NCOL + j], 7);
        prove(obdir, "g", z2, m2, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q, ""); });
    expect_throw("S < 1 (table floor)", "S[i] < 1", [&]{
        vector<int> zero_tab(LEN8, 0);
        prove(obdir, "g", zh, mxh, B, NCOL, LOW8, LEN8, zero_tab, LEN_R8, gen, Q, ""); });
    cout << (all ? "GUARDS PASS" : "GUARDS FAIL") << endl;
    return all;
}

static bool selftest_real() {
    const uint B = 1024, NCOL = 1024, LEN8 = 1u << 20, LEN_R8 = 1u << 14;
    const int LOW8 = -(1 << 20) + 2;       // domain [-1048574, +1]; SENT = +1
    cout << "==== selftest real-scale case B=NCOL=1024 LOW8=" << LOW8
         << " LEN8=2^20 LEN_R8=2^14 (mx from a real zkob_rowmax causal run) ===="
         << endl;
    // real exp table: registered file if present, else generated in-driver via
    // host double exp -- NON-AUTHORITATIVE, selftest only (the registered
    // sha256 file from gen_softmax8_table.py is the source of truth)
    vector<int> maph(LEN8);
    if (file_size("softmax8-exp-table.bin") == (long)LEN8 * 4) {
        cout << "loading softmax8-exp-table.bin" << endl;
        FILE* f = open_or_die("softmax8-exp-table.bin", "rb");
        if (fread(maph.data(), sizeof(int), LEN8, f) != LEN8)
            throw runtime_error("exp table short read");
        fclose(f);
    } else {
        cout << "softmax8-exp-table.bin not found: generating in-driver via host "
                "double exp (NON-AUTHORITATIVE, selftest only)" << endl;
        for (uint k = 0; k < LEN8; k++) {
            const long long v = (long long)LOW8 + (long long)k;
            maph[k] = (v > 0) ? 0 : (int)llround(65536.0 * exp((double)v / 4096.0));
        }
    }
    const string genpath = "/tmp/gen1024.bin";
    if (file_size(genpath) != 1024L * (long)sizeof(G1Jacobian_t)) {
        cout << "generating " << genpath << " via ppgen..." << endl;
        if (system(("./ppgen 1024 " + genpath).c_str()))
            throw runtime_error("ppgen failed");
    }
    const string qpath = "/tmp/zkob_q1.bin";
    if (file_size(qpath) != (long)sizeof(G1Jacobian_t)) {
        cout << "generating " << qpath << " via ppgen..." << endl;
        if (system(("./ppgen 1 " + qpath).c_str()))
            throw runtime_error("ppgen failed");
    }
    Commitment gen(genpath);
    Commitment qg(qpath);
    G1Jacobian_t Q = qg(0);
    // scores z_ ~ round(N(0, 2^13)) clipped to +-(2^19-1) (the rowmax causal
    // selftest distribution; realistic envelope per SCORES_RANGE)
    mt19937_64 rng(20260611);
    normal_distribution<double> nd(0.0, 8192.0);
    vector<int> zh((size_t)B * NCOL);
    for (size_t k = 0; k < zh.size(); k++) {
        long long v = llround(nd(rng));
        if (v < -((1LL << 19) - 1)) v = -((1LL << 19) - 1);
        if (v > (1LL << 19) - 1) v = (1LL << 19) - 1;
        zh[k] = (int)v;
    }
    bool all = true;

    // ---- chain-file style: invoke zkob_rowmax (causal) to produce mx ----
    const string zpath = "/tmp/zkob_softmax8_real_z.i32.bin";
    const string mxpath = "/tmp/zkob_softmax8_real_mx.i32.bin";
    const string rmdir = "/tmp/zkob_sm8_rowmax_ob";
    { FILE* f = open_or_die(zpath, "wb");
      fwrite(zh.data(), sizeof(int), zh.size(), f); fclose(f); }
    mkdir(rmdir.c_str(), 0755);
    string rm_seed = "selftest:softmax8:real:rowmax";
    string cmd = "./zkob_rowmax prove " + rmdir + " " + rm_seed + " " + zpath +
                 " 1024 1024 causal 0 1048576 1 " + genpath + " " + genpath +
                 " " + qpath + " " + mxpath;
    cout << "invoking: " << cmd << endl;
    if (system(cmd.c_str())) throw runtime_error("zkob_rowmax prove failed");
    string vcmd = "./zkob_rowmax verify " + rmdir + " " + rm_seed +
                  " 1024 1024 causal 0 1048576 1 " + genpath + " " + genpath +
                  " " + qpath;
    bool rm_ok = system(vcmd.c_str()) == 0;
    cout << (rm_ok ? "PASS" : "FAIL") << ": chained zkob_rowmax verify "
         << (rm_ok ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && rm_ok;
    vector<int> mxh(B);
    { FILE* f = open_or_die(mxpath, "rb");
      if (fread(mxh.data(), sizeof(int), B, f) != B)
          throw runtime_error("mx chain file short read");
      fclose(f); }
    {   // sanity: the chained mx IS the allowed row max
        vector<int> ref = host_mx(zh, B, NCOL);
        bool ok = true;
        for (uint i = 0; i < B && ok; i++) ok = (ref[i] == mxh[i]);
        cout << (ok ? "PASS" : "FAIL") << ": rowmax mx chain file == host row max" << endl;
        all = all && ok;
    }

    string obdir = "/tmp/zkob_softmax8_real";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:softmax8:real";

    auto t0 = chrono::steady_clock::now();
    prove(obdir, seed, zh, mxh, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q,
          "/tmp/zkob_softmax8_real_P.i32.bin");
    auto t1 = chrono::steady_clock::now();
    bool honest = verify(obdir, seed, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q);
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

    // edge RM2 (byte): rowmax's com_mx == softmax8's com_mx (same gen, same mx)
    bool rm2 = files_equal(rmdir + "/com_mx.bin", obdir + "/com_mx.bin");
    cout << (rm2 ? "PASS" : "FAIL") << ": edge RM2 com_mx byte-identity (rowmax "
            "obdir vs softmax8 obdir)" << endl;
    all = all && rm2;
    // edge RM1-analog (byte): same z committed under the same gen in both obdirs
    bool rm1 = files_equal(rmdir + "/com_z.bin", obdir + "/com_z.bin");
    cout << (rm1 ? "PASS" : "FAIL") << ": edge SX8a/RM1 com_z byte-identity" << endl;
    all = all && rm1;

    tamper_byte(obdir + "/lookup_E8.bin", 4 + 32, +1);
    bool rejected = !verify(obdir, seed, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, Q);
    tamper_byte(obdir + "/lookup_E8.bin", 4 + 32, -1);
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
        bool a = selftest_case(8,  -14, 16, 32);   // n1_E = 2, n1_L = 3
        bool b = selftest_case(4,  -14, 16, 32);   // n1_E = 0 (pure phase2)
        bool c = selftest_case(16, -62, 64, 64);   // bigger grid
        bool g = selftest_guards();
        bool d = selftest_real();
        bool ok = a && b && c && g && d;
        cout << (ok ? "ZKOB-SOFTMAX8 SELFTEST: ALL PASS"
                    : "ZKOB-SOFTMAX8 SELFTEST: FAIL") << endl;
        return ok ? 0 : 1;
    }
    if (mode == "prove" && (argc == 14 || argc == 15)) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[6]), NCOL = stoi(argv[7]);
        int LOW8 = stoi(argv[8]); uint LEN8 = (uint)stoul(argv[9]);
        uint LEN_R8 = (uint)stoul(argv[11]);
        vector<int> zh = load_i32(argv[4], B * NCOL);
        vector<int> mxh = load_i32(argv[5], B);
        vector<int> maph = load_expmap8(argv[10], LEN8);
        Commitment gen(argv[12]), qg(argv[13]);
        prove(obdir, seed, zh, mxh, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, qg(0),
              argc == 15 ? argv[14] : "");
        return 0;
    }
    if (mode == "verify" && argc == 12) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[4]), NCOL = stoi(argv[5]);
        int LOW8 = stoi(argv[6]); uint LEN8 = (uint)stoul(argv[7]);
        uint LEN_R8 = (uint)stoul(argv[9]);
        vector<int> maph = load_expmap8(argv[8], LEN8);
        Commitment gen(argv[10]), qg(argv[11]);
        return verify(obdir, seed, B, NCOL, LOW8, LEN8, maph, LEN_R8, gen, qg(0)) ? 0 : 1;
    }
    cerr << "usage: zkob_softmax8 selftest\n"
         << "       zkob_softmax8 prove  <obdir> <seed> <z-int32> <mx-int32> <B> <NCOL> <LOW8> <LEN8> <expmap8-int32> <LEN_R8> <gen> <q> [P-int32-out]\n"
         << "       zkob_softmax8 verify <obdir> <seed> <B> <NCOL> <LOW8> <LEN8> <expmap8-int32> <LEN_R8> <gen> <q>" << endl;
    return 2;
}
