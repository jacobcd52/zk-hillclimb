// Real driver for ONE RoPE obligation (layer{l}.attn.rope.{q,k}), per
// ROPE_ATTENTION_DESIGN.md (DESIGN FINAL 2026-06-10) section 4.1 / 5.1.
// Binds, for the chained B x C int32 tensor T at scale 2^16 (padded to the
// B x C_pad grid), the exact integer RoPE relation (R-ROPE):
//   Y64[t,e] = T[t,e]*W1[t,e] + T[t,flip(e)]*W2[t,e]      (int64, scale 2^32)
//   W1[t,e]  = [e<C] *        C_tab[t, e mod HD]          (public, registered)
//   W2[t,e]  = [e<C] * sigma(e) * S_tab[t, e mod HD],  sigma(e) = (e & HD/2) ? +1 : -1
//   flip(e)  = e XOR (HD/2)    (flips column bit fb = log2(HD)-1 only)
// The single rounding (R-RND, sf 2^16) is a SEPARATE zkob_rescale run on the
// emitted Y64 chain file. Zero prover advice; ZERO new CUDA kernels (composed
// from fs_hadamard / build_eq_tensor / k_fr_emul / k_fr_fold / FS-IPA).
//
// Relation binding (one FS challenge u over the 2^logD padded grid):
//   ev := Y64~(u) == c1 + c2
//   c1 = sum_b eq(u,b)*W1(b)*T(b)     (hadamard 1: E = eq(u) .* W1, S = T,  U = 1)
//   c2 = sum_b eq(u,b)*W2(b)*Tx(b)    (hadamard 2: E = eq(u) .* W2, S = Tx, U = 1)
// with Tx(b) = T(flip-bit-fb(b)) materialized host-side, NEVER committed: the
// h2 terminal opens the SAME com_T at pt2' = pt2 with coordinate fb replaced
// by 1 - pt2[fb] (rotate_half MLE identity, design section 1.4). Both weights
// are public: the verifier rebuilds W1/W2 from its own registered table copies
// (sign folded host-side as negative ints, mod-p via the FrTensor int ctor),
// folds them itself, and REQUIRES U_f2 == 1 per hadamard (load-bearing).
//
// FS schedule (design section 5.1; labels exact):
//   absorb B, C, HD, SCALE_R(=16); com_T; com_Y64 -> u (logD vars)
//   absorb ev; absorb c1; h1 rounds ("hp0".."hp3" -> w); absorb S_f2, U_f2;
//   IPA(T)@pt1=reverse(ws1); absorb c2; h2 rounds; absorb S_f2, U_f2;
//   IPA(T)@pt2'=flipbit_fb(reverse(ws2)); IPA(Y64)@u.
//   [verifier-only: U_f2 == 1 twice; cur_i == W_f_i*S_f2_i*U_f2_i; c1+c2 == ev]
//
// Files in <obdir> (9): dims.bin, com_T.bin, com_Y64.bin, ev.bin (1 Fr_t, raw),
//   hp1.bin, hp2.bin (HadamardProof, claim_H = c1/c2), ipa_T1/ipa_T2/ipa_Y.bin
//
// Usage:
//   zkob_rope prove   <obdir> <seed> <T-int32.bin> <B> <C> <HD>
//                     <cos-int32.bin> <sin-int32.bin> <gen.bin> <q.bin> [Y64-i64-out.bin]
//                     [--claims <accdir> <obid>]
//   zkob_rope verify  <obdir> <seed> <B> <C> <HD>
//                     <cos-int32.bin> <sin-int32.bin> <gen.bin> <q.bin>
//                     [--claims <vaccdir> <obid>]
//   zkob_rope selftest
// The driver does NOT mkdir the obdir.
//
// CLAIM MODE (Stage C of the transport rebuild, flag-selected; the old
// inline-IPA tail stays compilable and is the DEFAULT): with --claims, prove
// EMITS its three terminal claims at the exact old open_prove sites —
//   T  vs com_T   at pt1            (eval hp1.S_f2)
//   T  vs com_T   at pt2' (flipped) (eval hp2.S_f2)   <- same tensor, 2nd pt
//   Y  vs com_Y64 at u              (eval ev)
// — into <accdir>/claims.bin plus witrefs and drvstate; the inline-IPA
// absorbs vanish from the transcript on BOTH sides symmetrically. Verify in
// claim mode keeps every round check, the U_f2 == 1 pins, both verifier-
// rebuilt weight terminals and the c1 + c2 == ev identity UNCHANGED, then
// recomputes the three claims from its own FS replay into <vaccdir>.
// RELOCATED LOCUS (the §5.1 relocation rule): semantic evil 2 (unpermuted
// rotate_half) used to die at "IPA opening of h2 terminal vs com_T at the
// flipped point"; in claim mode every driver-local check passes and the
// FALSE T@pt2' claim dies in the batch at round0 (BO-1a class) — pinned in
// the claim-mode selftest.
#include "zkob_lookup.cuh"
#include "zkob_claims.cuh"
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <sys/stat.h>
using namespace std;

static_assert(sizeof(long) == 8, "host math needs 64-bit long");

static const uint SCALE_R = 16;   // table scale 2^16, pinned (design section 2.1)

// layout guards (design section 6, pinned honest-prover checks)
static void layout_guards(uint B, uint C, uint HD, uint gen_size) {
    const uint C_pad = 1u << ceilLog2(C);
    if (B != (1u << ceilLog2(B))) throw runtime_error("B not a power of two");
    if (HD < 2 || HD != (1u << ceilLog2(HD))) throw runtime_error("HD must be a power of two >= 2");
    if (C % HD) throw runtime_error("HD must divide C");
    if (C_pad % HD) throw runtime_error("HD must divide C_pad");
    if (C / HD < 2) throw runtime_error("NH = C/HD must be >= 2");
    if (gen_size != C_pad) throw runtime_error("generator size != C_pad");
}

// load a B*HD int32 table file; file size must be EXACTLY 4*B*HD bytes
static vector<int> load_table(const string& path, uint B, uint HD) {
    FILE* f = open_or_die(path, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz != (long)B * HD * 4)
        { fclose(f); throw runtime_error("table file size != B*HD int32: " + path); }
    fseek(f, 0, SEEK_SET);
    vector<int> buf((size_t)B * HD);
    if (fread(buf.data(), sizeof(int), buf.size(), f) != buf.size())
        { fclose(f); throw runtime_error("table short read: " + path); }
    fclose(f);
    return buf;
}

// public weights on the padded grid (host ints; negative values mod-p via the
// FrTensor int ctor) -- the SAME formula on both prover and verifier sides
static void build_weights(vector<int>& W1i, vector<int>& W2i,
                          const vector<int>& cosT, const vector<int>& sinT,
                          uint B, uint C, uint C_pad, uint HD) {
    const uint flipv = HD / 2;
    W1i.assign((size_t)B * C_pad, 0);
    W2i.assign((size_t)B * C_pad, 0);
    for (uint t = 0; t < B; t++)
        for (uint e = 0; e < C; e++) {
            const size_t g = (size_t)t * C_pad + e;
            W1i[g] = cosT[(size_t)t * HD + (e % HD)];
            W2i[g] = ((e & flipv) ? +1 : -1) * sinT[(size_t)t * HD + (e % HD)];
        }
}

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

// ---------------- prove ----------------
// evil (selftest only; honest PROCEDURE on an inconsistent witness):
//  1: Y64[idx] += 1; ev computed from the evil Y64; hadamards honest from T
//     -> only "c1 + c2 != ev" can reject (Y IPA + both hadamards self-consistent).
//  2: hadamard 2 run with the UNPERMUTED T (Tx := T); Y64 computed consistently
//     with that wrong relation -> the flipped-point opening rejects
//     ("IPA opening of h2 terminal vs com_T at the flipped point").
//  3: bumped cos table W1[idx] += 1 used consistently (Y64, c1, h1 weight)
//     -> "h1 weight terminal" (verifier W_f rebuilt from the REGISTERED table).
//  4: h1 run with a corrupted ones-buffer U[idx] += 1, c1 computed from it,
//     everything else honest -> "h1 U_f2 != 1".
//  5: sigma sign flipped on one W2 entry (the SIN / hadamard-2 path), used
//     consistently throughout (Y64, c2, h2 E-buffer) -> "h2 weight terminal"
//     (the verifier rebuilds W2 from the REGISTERED table; audit MINOR-3).
static void prove(const string& obdir, const string& seed,
                  const vector<int>& Th, uint B, uint C, uint HD,
                  const vector<int>& cosT, const vector<int>& sinT,
                  const Commitment& gen, const G1Jacobian_t& Q,
                  const string& y_out,
                  int evil = 0, size_t evil_idx = 0,
                  const string& accdir = "", const string& obid = "") {
    const bool claim_mode = !accdir.empty();
    layout_guards(B, C, HD, gen.size);
    const uint C_pad = 1u << ceilLog2(C);
    const uint logB = ceilLog2(B), logCp = ceilLog2(C_pad), logD = logB + logCp;
    const uint D = B * C_pad;
    const uint flipv = HD / 2, fb = ceilLog2(HD) - 1;
    if (Th.size() != (size_t)B * C) throw runtime_error("input dims");

    // ---- public weights (host int64-safe; |W| <= 2^16 real, 2^7 toy) ----
    vector<int> W1i, W2i;
    build_weights(W1i, W2i, cosT, sinT, B, C, C_pad, HD);
    if (evil == 3) W1i[evil_idx] += 1;
    if (evil == 5) W2i[evil_idx] = W2i[evil_idx] ? -W2i[evil_idx] : 1;  // sign flip; 1 if the entry is 0 so the corruption is never vacuous

    // ---- padded T grid, flipped grid, exact Y64 (host int64) ----
    vector<int> Tg(D, 0), Txg(D);
    for (uint t = 0; t < B; t++)
        for (uint e = 0; e < C; e++) Tg[(size_t)t * C_pad + e] = Th[(size_t)t * C + e];
    for (uint t = 0; t < B; t++)
        for (uint e = 0; e < C_pad; e++)
            Txg[(size_t)t * C_pad + e] = Tg[(size_t)t * C_pad + (e ^ flipv)];
    if (evil == 2) Txg = Tg;                       // unpermuted rotate_half
    vector<long> Yg(D, 0);
    for (size_t g = 0; g < D; g++) {
        Yg[g] = (long)Tg[g] * W1i[g] + (long)Txg[g] * W2i[g];
        if (llabs(Yg[g]) >= (1L << 47))
            throw runtime_error("|Y64| >= 2^47 (int32 chain format after rescale at risk)");
    }
    if (evil == 1) Yg[evil_idx] += 1;
    if (!y_out.empty()) {            // chain file: UNPADDED B x C int64, scale 2^32
        vector<long> yo((size_t)B * C);
        for (uint t = 0; t < B; t++)
            for (uint e = 0; e < C; e++) yo[(size_t)t * C + e] = Yg[(size_t)t * C_pad + e];
        FILE* f = open_or_die(y_out, "wb");
        fwrite(yo.data(), sizeof(long), yo.size(), f); fclose(f);
    }

    // ---- tensors (all PLAIN Fr values) ----
    FrTensor T_t(D, Tg.data());
    FrTensor Tx_t(D, Txg.data());
    FrTensor W1_t(D, W1i.data());
    FrTensor W2_t(D, W2i.data());
    FrTensor Y_t(D, Yg.data());
    vector<int> onesh(D, 1), ones1h(D, 1);
    if (evil == 4) ones1h[evil_idx] += 1;          // corrupted U for hadamard 1 only
    FrTensor U1_t(D, ones1h.data());
    FrTensor U2_t(D, onesh.data());

    // ---- commitments (gen = C_pad gens, B rows each) ----
    G1TensorJacobian com_T = gen.commit(T_t);
    G1TensorJacobian com_Y = gen.commit(Y_t);
    com_T.save(obdir + "/com_T.bin");
    com_Y.save(obdir + "/com_Y64.bin");
    { FILE* f = open_or_die(obdir + "/dims.bin", "wb");
      uint32_t d[4] = {B, C, HD, SCALE_R};
      fwrite(d, sizeof(uint32_t), 4, f); fclose(f); }

    // ---- transcript ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C);
    absorb_u32(tr, "HD", HD); absorb_u32(tr, "SCALE_R", SCALE_R);
    absorb_g1_tensor(tr, "com_T", com_T);
    absorb_g1_tensor(tr, "com_Y64", com_Y);
    auto u = fs_challenge_vec(tr, logD);
    vector<Fr_t> u_col(u.begin(), u.begin() + logCp);
    vector<Fr_t> u_row(u.begin() + logCp, u.end());

    Fr_t ev = Y_t.multi_dim_me({u_row, u_col}, {B, C_pad});
    absorb_fr(tr, "ev", ev);
    { FILE* f = open_or_die(obdir + "/ev.bin", "wb");
      fwrite(&ev, sizeof(Fr_t), 1, f); fclose(f); }

    FrTensor E_u = build_eq_tensor(u);

    // ---- hadamard 1: E = eq(u) .* W1, S = T, U = 1 ----
    FrTensor E1 = E_u * W1_t;
    HadamardProof hp1;
    hp1.claim_H = (E1 * T_t * U1_t).sum();                          // c1
    absorb_fr(tr, "c1", hp1.claim_H);
    vector<Fr_t> ws1;
    {
        FrTensor Ec(E1), Sc(T_t), Uc(U1_t);
        fs_hadamard(hp1.claim_H, Ec, Sc, Uc, tr, ws1, hp1, true);
    }
    absorb_fr(tr, "S_f2", hp1.S_f2); absorb_fr(tr, "U_f2", hp1.U_f2);
    write_hp(obdir + "/hp1.bin", hp1);
    vector<Fr_t> pt1(ws1.rbegin(), ws1.rend());
    if (evil == 0) {        // convention sanity: fold terminals == ME evaluations
        vector<Fr_t> p_col(pt1.begin(), pt1.begin() + logCp);
        vector<Fr_t> p_row(pt1.begin() + logCp, pt1.end());
        if (!fr_eq(hp1.S_f2, T_t.multi_dim_me({p_row, p_col}, {B, C_pad})) ||
            !fr_eq(hp1.U_f2, F_ONE))
            throw runtime_error("h1 terminal != multi_dim_me / 1 (convention bug)");
    }
    auto mk_claim = [&](const string& tensor, const string& comref,
                        const vector<Fr_t>& point, const Fr_t& eval) {
        BoClaim c;
        c.id = obid + ":" + tensor;
        c.comref = comref;
        c.domain = C_pad; c.n_rows = B;
        c.point = point;
        c.eval = eval;
        claim_emit(accdir, c);
    };
    if (claim_mode) {
        // claim T@pt1 at the exact old open_prove site (witref once; the
        // second T claim shares the tensor)
        string witT = accdir + "/wit_" + obid + "_T.fr";
        T_t.save(witT);
        mk_claim("T1", obdir + "/com_T.bin", pt1, hp1.S_f2);
        witref_emit(accdir, obdir + "/com_T.bin", witT);
    } else {
        open_prove(T_t, C_pad, gen, Q, pt1, obdir + "/ipa_T1.bin", tr);
    }

    // ---- hadamard 2: E = eq(u) .* W2, S = Tx, U = 1 ----
    FrTensor E2 = E_u * W2_t;
    HadamardProof hp2;
    hp2.claim_H = (E2 * Tx_t * U2_t).sum();                         // c2
    absorb_fr(tr, "c2", hp2.claim_H);
    vector<Fr_t> ws2;
    {
        FrTensor Ec(E2), Sc(Tx_t), Uc(U2_t);
        fs_hadamard(hp2.claim_H, Ec, Sc, Uc, tr, ws2, hp2, true);
    }
    absorb_fr(tr, "S_f2", hp2.S_f2); absorb_fr(tr, "U_f2", hp2.U_f2);
    write_hp(obdir + "/hp2.bin", hp2);
    // pt2' = pt2 with coordinate fb replaced by 1 - pt2[fb] (the rotate_half
    // MLE identity, design section 1.4) -- both sides compute it themselves
    vector<Fr_t> pt2(ws2.rbegin(), ws2.rend());
    vector<Fr_t> pt2p(pt2);
    pt2p[fb] = h_scalar(F_ONE, pt2[fb], 1);
    if (evil != 2) {        // completeness guard: T~(pt2') == Tx~(pt2) == S_f2
        vector<Fr_t> p_col(pt2p.begin(), pt2p.begin() + logCp);
        vector<Fr_t> p_row(pt2p.begin() + logCp, pt2p.end());
        if (!fr_eq(hp2.S_f2, T_t.multi_dim_me({p_row, p_col}, {B, C_pad})))
            throw runtime_error("T~(pt2') != h2 S_f2 (flip convention bug)");
        if (evil == 0 && !fr_eq(hp2.U_f2, F_ONE))
            throw runtime_error("h2 U_f2 != 1 (convention bug)");
    }
    if (claim_mode) {
        mk_claim("T2", obdir + "/com_T.bin", pt2p, hp2.S_f2);
        string witY = accdir + "/wit_" + obid + "_Y.fr";
        Y_t.save(witY);
        mk_claim("Y", obdir + "/com_Y64.bin", u, ev);
        witref_emit(accdir, obdir + "/com_Y64.bin", witY);
        drvstate_emit(accdir, obid, tr);
        cout << "PROVED rope obligation (claim mode, 3 claims emitted) -> " << obdir << endl;
        return;
    }
    open_prove(T_t, C_pad, gen, Q, pt2p, obdir + "/ipa_T2.bin", tr);

    // ---- Y64 opening at u ----
    open_prove(Y_t, C_pad, gen, Q, u, obdir + "/ipa_Y.bin", tr);
    cout << "PROVED rope obligation -> " << obdir << endl;
}

// ---------------- verify (witness-free) ----------------
#define RJ(msg) do { ostringstream oss_; oss_ << msg; \
    cout << "REJECT: " << oss_.str() << endl; \
    if (reason) *reason = oss_.str(); return false; } while (0)

static bool verify(const string& obdir, const string& seed,
                   uint B, uint C, uint HD,
                   const vector<int>& cosT, const vector<int>& sinT,
                   const Commitment& gen, const G1Jacobian_t& Q,
                   string* reason = nullptr,
                   const string& vaccdir = "", const string& obid = "") {
    const bool claim_mode = !vaccdir.empty();
    BoTimer prof("rope_verify");
    const uint C_pad = 1u << ceilLog2(C);
    const uint logB = ceilLog2(B), logCp = ceilLog2(C_pad), logD = logB + logCp;
    const uint D = B * C_pad;
    const uint fb = ceilLog2(HD) - 1;
    if (B != (1u << logB) || HD < 2 || HD != (1u << ceilLog2(HD)) ||
        (C % HD) || (C_pad % HD) || C / HD < 2 || gen.size != C_pad)
        RJ("bad layout params");
    { FILE* f = open_or_die(obdir + "/dims.bin", "rb");
      uint32_t d[4];
      if (fread(d, sizeof(uint32_t), 4, f) != 4) { fclose(f); RJ("dims.bin short read"); }
      fclose(f);
      if (d[0] != B || d[1] != C || d[2] != HD || d[3] != SCALE_R)
          RJ("dims.bin mismatch"); }

    G1TensorJacobian com_T(obdir + "/com_T.bin");
    G1TensorJacobian com_Y(obdir + "/com_Y64.bin");
    if (com_T.size != B || com_Y.size != B) RJ("commitment row counts");
    HadamardProof hp1 = read_hp(obdir + "/hp1.bin");
    HadamardProof hp2 = read_hp(obdir + "/hp2.bin");
    if (hp1.ev.size() != 4 * logD || hp2.ev.size() != 4 * logD)
        RJ("hadamard round count");
    Fr_t ev;
    { FILE* f = open_or_die(obdir + "/ev.bin", "rb");
      if (fread(&ev, sizeof(Fr_t), 1, f) != 1) { fclose(f); RJ("ev.bin short read"); }
      fclose(f); }

    // the verifier rebuilds W1/W2 from ITS OWN registered table copies
    vector<int> W1i, W2i;
    build_weights(W1i, W2i, cosT, sinT, B, C, C_pad, HD);

    // ---- transcript replay ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C);
    absorb_u32(tr, "HD", HD); absorb_u32(tr, "SCALE_R", SCALE_R);
    absorb_g1_tensor(tr, "com_T", com_T);
    absorb_g1_tensor(tr, "com_Y64", com_Y);
    auto u = fs_challenge_vec(tr, logD);
    absorb_fr(tr, "ev", ev);

    const Fr_t inv6 = inv(F_SIX);
    FrTensor E_u = build_eq_tensor(u);

    // ---- hadamard 1 replay ----
    absorb_fr(tr, "c1", hp1.claim_H);
    Fr_t cur = hp1.claim_H;
    vector<Fr_t> ws1;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp1.ev[4*k], hp1.ev[4*k+1], hp1.ev[4*k+2], hp1.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("h1 round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws1.push_back(w);
        cur = lagrange4(e, w, inv6);
    }
    if (!fr_eq(hp1.U_f2, F_ONE)) RJ("h1 U_f2 != 1");
    {
        FrTensor W1_t(D, W1i.data());
        FrTensor E1(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(E_u.gpu_data, W1_t.gpu_data, E1.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t W_f = fold_public(E1, ws1);
        if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp1.S_f2, hp1.U_f2, 2), 2)))
            RJ("h1 weight terminal");
    }
    absorb_fr(tr, "S_f2", hp1.S_f2); absorb_fr(tr, "U_f2", hp1.U_f2);
    vector<Fr_t> pt1(ws1.rbegin(), ws1.rend());
    if (!claim_mode) {
        if (!open_verify(com_T, gen, C_pad, Q, pt1, hp1.S_f2, obdir + "/ipa_T1.bin", tr))
            RJ("IPA opening of h1 terminal vs com_T");
    }
    // (claim mode: the T@pt1 claim is recomputed at the end of verify, after
    // every local check has passed; the IPA absorbs vanish symmetrically)

    // ---- hadamard 2 replay ----
    absorb_fr(tr, "c2", hp2.claim_H);
    cur = hp2.claim_H;
    vector<Fr_t> ws2;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp2.ev[4*k], hp2.ev[4*k+1], hp2.ev[4*k+2], hp2.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("h2 round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws2.push_back(w);
        cur = lagrange4(e, w, inv6);
    }
    if (!fr_eq(hp2.U_f2, F_ONE)) RJ("h2 U_f2 != 1");
    {
        FrTensor W2_t(D, W2i.data());
        FrTensor E2(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(E_u.gpu_data, W2_t.gpu_data, E2.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t W_f = fold_public(E2, ws2);
        if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp2.S_f2, hp2.U_f2, 2), 2)))
            RJ("h2 weight terminal");
    }
    absorb_fr(tr, "S_f2", hp2.S_f2); absorb_fr(tr, "U_f2", hp2.U_f2);
    // the VERIFIER computes pt2' itself: pt2'[fb] = 1 - pt2[fb]
    vector<Fr_t> pt2(ws2.rbegin(), ws2.rend());
    vector<Fr_t> pt2p(pt2);
    pt2p[fb] = h_scalar(F_ONE, pt2[fb], 1);
    if (!claim_mode) {
        if (!open_verify(com_T, gen, C_pad, Q, pt2p, hp2.S_f2, obdir + "/ipa_T2.bin", tr))
            RJ("IPA opening of h2 terminal vs com_T at the flipped point");
        // ---- Y64 opening ----
        if (!open_verify(com_Y, gen, C_pad, Q, u, ev, obdir + "/ipa_Y.bin", tr))
            RJ("IPA opening of ev vs com_Y64");
    }
    // ---- the sum identity (both modes) ----
    if (!fr_eq(h_scalar(hp1.claim_H, hp2.claim_H, 0), ev))
        RJ("c1 + c2 != ev");
    if (claim_mode) {
        // ---- claim recomputation (canonical order T1, T2, Y — the prover's
        // emission order); verdict becomes ACCEPT-conditional ----
        auto mk_claim = [&](const string& tensor, const string& comref,
                            const vector<Fr_t>& point, const Fr_t& eval) {
            BoClaim c;
            c.id = obid + ":" + tensor;
            c.comref = comref;
            c.domain = C_pad; c.n_rows = B;
            c.point = point;
            c.eval = eval;
            claim_emit(vaccdir, c);
        };
        mk_claim("T1", obdir + "/com_T.bin", pt1, hp1.S_f2);
        mk_claim("T2", obdir + "/com_T.bin", pt2p, hp2.S_f2);
        mk_claim("Y", obdir + "/com_Y64.bin", u, ev);
        drvstate_emit(vaccdir, obid, tr);
        prof.lap("claim_emit");
        cout << "ACCEPT-conditional (3 claims emitted; final verdict gated on opening_batch)" << endl;
        return true;
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
    "dims.bin", "com_T.bin", "com_Y64.bin", "ev.bin",
    "hp1.bin", "hp2.bin", "ipa_T1.bin", "ipa_T2.bin", "ipa_Y.bin"};
static long tamper_offset(const string& f) {
    if (f == "dims.bin") return 0;
    if (f == "ev.bin") return 4;                    // raw Fr sequence
    if (f.substr(0, 4) == "ipa_") return -32;       // a_final
    if (f.substr(0, 4) == "com_") return 24;        // first point, x limbs
    return 36;                                      // hp: claim_H 0-31, count 32-35
}

static bool selftest_case(uint B, uint C, uint HD) {
    const uint C_pad = 1u << ceilLog2(C);
    cout << "==== selftest case B=" << B << " C=" << C << " HD=" << HD
         << " (C_pad=" << C_pad << ", flip bit " << ceilLog2(HD) - 1 << ") ====" << endl;
    // toy tables: random int32 in [-2^7, 2^7] (driver is table-agnostic)
    srand(2026 + B * 100 + C * 10 + HD);
    vector<int> cosT((size_t)B * HD), sinT((size_t)B * HD);
    for (auto& v : cosT) v = rand() % 257 - 128;
    for (auto& v : sinT) v = rand() % 257 - 128;
    vector<int> Th((size_t)B * C);
    for (auto& v : Th) v = rand() % 257 - 128;
    Commitment gen = Commitment::random(C_pad);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_rope_ob";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:rope";
    bool all = true;

    prove(obdir, seed, Th, B, C, HD, cosT, sinT, gen, Q, "/tmp/zkob_rope_Y64.i64.bin");
    bool honest = verify(obdir, seed, B, C, HD, cosT, sinT, gen, Q);
    cout << (honest ? "PASS" : "FAIL") << ": honest case "
         << (honest ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && honest;

    // chain-file sanity: Y64[t,e] = T*W1 + Tflip*W2 exactly (spec recompute)
    {
        vector<long> yo((size_t)B * C);
        FILE* f = open_or_die("/tmp/zkob_rope_Y64.i64.bin", "rb");
        if (fread(yo.data(), sizeof(long), yo.size(), f) != yo.size())
            throw runtime_error("Y64 chain file short read");
        fclose(f);
        const uint flipv = HD / 2;
        bool ok = true;
        for (uint t = 0; t < B && ok; t++)
            for (uint e = 0; e < C && ok; e++) {
                long w1 = cosT[(size_t)t * HD + e % HD];
                long w2 = ((e & flipv) ? 1 : -1) * (long)sinT[(size_t)t * HD + e % HD];
                long want = (long)Th[(size_t)t * C + e] * w1
                          + (long)Th[(size_t)t * C + (e ^ flipv)] * w2;
                ok = (yo[(size_t)t * C + e] == want);
            }
        cout << (ok ? "PASS" : "FAIL") << ": Y64 chain file matches the R-ROPE spec" << endl;
        all = all && ok;
    }

    // semantic evil modes: each rejected by EXACTLY the named check
    struct Evil { int mode; size_t idx; const char* expect; const char* what; };
    size_t real_idx = (size_t)1 * C_pad + 1;       // an unpadded grid index
    vector<Evil> evils = {
        {1, real_idx, "c1 + c2 != ev",
            "Y64[idx]+=1, ev honest from evil Y64, hadamards honest"},
        {2, 0, "IPA opening of h2 terminal vs com_T at the flipped point",
            "h2 run with UNPERMUTED T (Tx := T), Y64 consistent with it"},
        {3, real_idx, "h1 weight terminal",
            "bumped cos table W1[idx]+=1 used consistently throughout"},
        {4, real_idx, "h1 U_f2 != 1",
            "h1 ones-buffer U[idx]+=1, c1 computed from it"},
        {5, real_idx, "h2 weight terminal",
            "sigma sign flipped on one W2 entry (sin path) used consistently throughout"},
    };
    string evdir = "/tmp/zkob_rope_evil";
    mkdir(evdir.c_str(), 0755);
    for (auto& ev : evils) {
        prove(evdir, seed, Th, B, C, HD, cosT, sinT, gen, Q, "", ev.mode, ev.idx);
        string reason;
        bool rejected = !verify(evdir, seed, B, C, HD, cosT, sinT, gen, Q, &reason);
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
        bool rejected = !verify(obdir, seed, B, C, HD, cosT, sinT, gen, Q);
        tamper_byte(obdir + "/" + fn, off, -1);
        cout << (rejected ? "PASS" : "FAIL") << ": byte tamper " << fn << "@" << off
             << " rejected: " << (rejected ? "YES" : "NO(!!)") << endl;
        all = all && rejected;
    }
    bool restored = verify(obdir, seed, B, C, HD, cosT, sinT, gen, Q);
    cout << (restored ? "PASS" : "FAIL") << ": restored verify "
         << (restored ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && restored;
    cout << (all ? "CASE PASS" : "CASE FAIL") << endl;
    return all;
}

// claim-mode selftest (Stage C): honest ACCEPT goes conditional-ACCEPT +
// batch ACCEPT; every forgery still rejects at a NAMED locus. Evil 2 is the
// RELOCATED one: its old locus was the h2 flipped-point IPA; driver-local
// checks now pass and the false T@pt2' claim dies in the batch at round0
// (honest-procedure batch over the false list, ZKOB_EVIL=1 class).
static bool selftest_case_claims(uint B, uint C, uint HD) {
    const uint C_pad = 1u << ceilLog2(C);
    cout << "==== selftest (claim mode) B=" << B << " C=" << C << " HD=" << HD
         << " ====" << endl;
    srand(3026 + B * 100 + C * 10 + HD);
    vector<int> cosT((size_t)B * HD), sinT((size_t)B * HD);
    for (auto& v : cosT) v = rand() % 257 - 128;
    for (auto& v : sinT) v = rand() % 257 - 128;
    vector<int> Th((size_t)B * C);
    for (auto& v : Th) v = rand() % 257 - 128;

    string dir = "/tmp/zkob_rope_cm";
    { string c = "rm -rf " + dir; system(c.c_str()); }
    mkdir(dir.c_str(), 0755);
    string acc = dir + "/acc";
    mkdir(acc.c_str(), 0755);
    string run_seed = "selftest";
    string seed = "selftest:rope";
    string obid = "selftest.rope.q";

    map<uint32_t, string> genpaths;
    Commitment gen = Commitment::random(C_pad);
    genpaths[C_pad] = dir + "/gen" + to_string(C_pad) + ".bin";
    gen.save(genpaths[C_pad]);
    string qpath = dir + "/q.bin";
    Commitment::random(2).save(qpath);
    Commitment qg(qpath);
    G1Jacobian_t Q = qg(0);

    prove(dir, seed, Th, B, C, HD, cosT, sinT, gen, Q, "", 0, 0, acc, obid);
    batch_prove(acc, run_seed, genpaths, qpath);

    int vacc_n = 0;
    auto pipeline = [&](string& locus) -> bool {
        string vacc = dir + "/vacc" + to_string(vacc_n++);
        mkdir(vacc.c_str(), 0755);
        string reason;
        if (!verify(dir, seed, B, C, HD, cosT, sinT, gen, Q, &reason, vacc, obid)) {
            locus = "driver:" + reason; return false;
        }
        return batch_verify(acc, vacc, run_seed, genpaths, qpath, &locus);
    };
    int total = 0, fail = 0;
    auto expect = [&](const string& what, const string& want) {
        string locus;
        bool acc_ok = pipeline(locus);
        bool ok = (want == "accept") ? acc_ok
                                     : (!acc_ok && locus.find(want) != string::npos);
        total++; if (!ok) fail++;
        cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << what
             << " -> expected " << want << ", got " << (acc_ok ? "accept" : locus) << endl;
    };

    expect("honest (conditional ACCEPT + batch ACCEPT)", "accept");

    // forgeries the DRIVER still catches (unchanged loci)
    tamper_byte(dir + "/ev.bin", 4, +1);
    expect("ev.bin tamper (driver weight terminal via transcript divergence)", "driver:");
    tamper_byte(dir + "/ev.bin", 4, -1);
    tamper_byte(dir + "/hp1.bin", 36, +1);
    expect("hp1 round-0 eval tamper (driver round check)", "driver:");
    tamper_byte(dir + "/hp1.bin", 36, -1);
    tamper_byte(dir + "/com_T.bin", 24, +1);
    expect("com_T tamper (driver transcript divergence)", "driver:");
    tamper_byte(dir + "/com_T.bin", 24, -1);

    // forgeries that now die in the BATCH (the relocated opening layer)
    tamper_byte(acc + "/claims.bin", -1, +1);
    expect("prover claims.bin eval tamper", "claims_match");
    tamper_byte(acc + "/claims.bin", -1, -1);
    {
        auto cs = claims_load(acc + "/claims.bin");
        auto omitted = vector<BoClaim>(cs.begin(), cs.end() - 1);
        claims_save(acc + "/claims.bin", omitted);
        expect("claim dropped from prover accumulator", "claims_match");
        claims_save(acc + "/claims.bin", cs);
    }
    tamper_byte(acc + "/ipa_batch_" + to_string(C_pad) + ".bin", -32, +1);
    expect("batched IPA a_final tamper", "ipa" + to_string(C_pad));
    tamper_byte(acc + "/ipa_batch_" + to_string(C_pad) + ".bin", -32, -1);

    expect("restored", "accept");

    // semantic evil battery in claim mode
    size_t real_idx = (size_t)1 * C_pad + 1;
    struct EvilCM { int mode; size_t idx; const char* want; const char* what; };
    vector<EvilCM> evils = {
        {1, real_idx, "driver:c1 + c2 != ev", "Y64[idx]+=1 (driver sum identity)"},
        {3, real_idx, "driver:h1 weight terminal", "bumped cos table used consistently"},
        {4, real_idx, "driver:h1 U_f2 != 1", "corrupted h1 ones-buffer"},
        {5, real_idx, "driver:h2 weight terminal", "sigma sign flip used consistently"},
    };
    for (auto& ev : evils) {
        string edir = dir + "/evil", eacc = dir + "/eacc", evacc = dir + "/evacc";
        { string c = "rm -rf " + edir + " " + eacc + " " + evacc; system(c.c_str()); }
        mkdir(edir.c_str(), 0755); mkdir(eacc.c_str(), 0755); mkdir(evacc.c_str(), 0755);
        prove(edir, seed, Th, B, C, HD, cosT, sinT, gen, Q, "", ev.mode, ev.idx, eacc, obid);
        string reason;
        bool rej = !verify(edir, seed, B, C, HD, cosT, sinT, gen, Q, &reason, evacc, obid);
        string locus = "driver:" + reason;
        bool ok = rej && locus.find(ev.want) != string::npos;
        total++; if (!ok) fail++;
        cout << "  [" << (ok ? "PASS" : "FAIL") << "] evil=" << ev.mode << " (" << ev.what
             << ") -> expected " << ev.want << ", got "
             << (rej ? locus : string("accept")) << endl;
    }
    // evil 2 — the RELOCATED locus: driver-local checks pass (conditional
    // ACCEPT), the false T@pt2' claim dies in the batch at round0
    {
        string edir = dir + "/evil2", eacc = dir + "/eacc2", evacc = dir + "/evacc2";
        mkdir(edir.c_str(), 0755); mkdir(eacc.c_str(), 0755); mkdir(evacc.c_str(), 0755);
        prove(edir, seed, Th, B, C, HD, cosT, sinT, gen, Q, "", 2, 0, eacc, obid);
        string reason;
        bool drv_ok = verify(edir, seed, B, C, HD, cosT, sinT, gen, Q, &reason, evacc, obid);
        bool batch_rej = false; string locus = "(driver rejected)";
        if (drv_ok) {
            batch_prove(eacc, run_seed, genpaths, qpath, /*evil=*/1);
            batch_rej = !batch_verify(eacc, evacc, run_seed, genpaths, qpath, &locus);
        }
        bool ok = drv_ok && batch_rej && locus.substr(0, 5) == "round";
        total++; if (!ok) fail++;
        cout << "  [" << (ok ? "PASS" : "FAIL")
             << "] evil=2 (unpermuted rotate_half; RELOCATED locus) -> expected "
                "driver conditional-ACCEPT + batch round0, got driver "
             << (drv_ok ? "accept" : "reject") << " + batch "
             << (batch_rej ? locus : string("accept")) << endl;
    }

    bool ok = fail == 0;
    cout << (ok ? "CASE PASS" : "CASE FAIL") << " (claim mode, " << (total - fail)
         << "/" << total << ")" << endl;
    return ok;
}

static bool selftest_real() {
    const uint B = 1024, C = 768, HD = 64;
    cout << "==== selftest real-scale case B=1024 C=768 HD=64 ====" << endl;
    // real tables: registered files if present, else generated in-driver via
    // host double cos/sin -- NON-AUTHORITATIVE, selftest only (the registered
    // sha256 files from gen_rope_tables.py are the source of truth)
    vector<int> cosT((size_t)B * HD), sinT((size_t)B * HD);
    if (file_size("rope-cos-table.bin") == (long)B * HD * 4 &&
        file_size("rope-sin-table.bin") == (long)B * HD * 4) {
        cout << "loading rope-cos-table.bin / rope-sin-table.bin" << endl;
        cosT = load_table("rope-cos-table.bin", B, HD);
        sinT = load_table("rope-sin-table.bin", B, HD);
    } else {
        cout << "rope tables not found: generating in-driver via host double "
                "cos/sin (NON-AUTHORITATIVE, selftest only)" << endl;
        const uint half = HD / 2;
        for (uint t = 0; t < B; t++)
            for (uint d = 0; d < HD; d++) {
                double ang = (double)t * pow(10000.0, -(double)(d % half) / half);
                cosT[(size_t)t * HD + d] = (int)llround(65536.0 * cos(ang));
                sinT[(size_t)t * HD + d] = (int)llround(65536.0 * sin(ang));
            }
    }
    // T ~ round(N(0, 2^18)): the realistic |q| envelope (design section 8.4)
    mt19937_64 rng(20260610);
    normal_distribution<double> nd(0.0, 262144.0);
    vector<int> Th((size_t)B * C);
    for (auto& v : Th) v = (int)llround(nd(rng));
    const string genpath = "/tmp/gen1024.bin";
    if (file_size(genpath) != 1024L * (long)sizeof(G1Jacobian_t)) {
        cout << "generating " << genpath << " via ppgen..." << endl;
        if (system(("./ppgen 1024 " + genpath).c_str()))
            throw runtime_error("ppgen failed");
    }
    Commitment gen(genpath);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_rope_real";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:rope:real";
    bool all = true;

    auto t0 = chrono::steady_clock::now();
    prove(obdir, seed, Th, B, C, HD, cosT, sinT, gen, Q, "/tmp/zkob_rope_real_Y64.i64.bin");
    auto t1 = chrono::steady_clock::now();
    bool honest = verify(obdir, seed, B, C, HD, cosT, sinT, gen, Q);
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

    tamper_byte(obdir + "/hp1.bin", 36, +1);
    bool rejected = !verify(obdir, seed, B, C, HD, cosT, sinT, gen, Q);
    tamper_byte(obdir + "/hp1.bin", 36, -1);
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

#include "zkob_serve.cuh"
static int zkw_run1(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    // strip the optional claim-mode flag block: --claims <accdir> <obid>
    int base_argc = argc;
    string cm_a, cm_b;
    for (int i = 2; i < argc; i++)
        if (string(argv[i]) == "--claims") {
            base_argc = i;
            if (i + 1 < argc) cm_a = argv[i + 1];
            if (i + 2 < argc) cm_b = argv[i + 2];
            break;
        }
    if (mode == "selftest") {
        bo_probe_kernels();
        cout << "kernel -dlto probes: PASS" << endl;
        setenv("ZKOB_FOLD_CROSSCHECK", "1", 1);
        bool a = selftest_case(8, 6, 2);     // padding cols + flip bit 0
        bool b = selftest_case(4, 8, 4);     // C == C_pad (no padding)
        bool c = selftest_case(16, 12, 4);   // padded 16x16 grid
        bool d = selftest_real();
        bool e = selftest_case_claims(8, 6, 2);
        bool f = selftest_case_claims(4, 8, 4);
        bool g = selftest_case_claims(16, 12, 4);
        bool ok = a && b && c && d && e && f && g;
        cout << (ok ? "ZKOB-ROPE SELFTEST: ALL PASS"
                    : "ZKOB-ROPE SELFTEST: FAIL") << endl;
        return ok ? 0 : 1;
    }
    if (mode == "prove" && (base_argc == 12 || base_argc == 13)) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[5]), C = stoi(argv[6]), HD = stoi(argv[7]);
        vector<int> Th = load_i32(argv[4], B * C);
        vector<int> cosT = load_table(argv[8], B, HD);
        vector<int> sinT = load_table(argv[9], B, HD);
        Commitment gen(argv[10]), qg(argv[11]);
        if (!cm_a.empty() && cm_b.empty())
            throw runtime_error("prove --claims needs <accdir> <obid>");
        prove(obdir, seed, Th, B, C, HD, cosT, sinT, gen, qg(0),
              base_argc == 13 ? argv[12] : "", 0, 0, cm_a, cm_b);
        return 0;
    }
    if (mode == "verify" && base_argc == 11) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[4]), C = stoi(argv[5]), HD = stoi(argv[6]);
        vector<int> cosT = load_table(argv[7], B, HD);
        vector<int> sinT = load_table(argv[8], B, HD);
        Commitment gen(argv[9]), qg(argv[10]);
        if (!cm_a.empty() && cm_b.empty())
            throw runtime_error("verify --claims needs <vaccdir> <obid>");
        return verify(obdir, seed, B, C, HD, cosT, sinT, gen, qg(0),
                      nullptr, cm_a, cm_b) ? 0 : 1;
    }
    cerr << "usage: zkob_rope selftest\n"
         << "       zkob_rope prove  <obdir> <seed> <T-int32> <B> <C> <HD> <cos-int32> <sin-int32> <gen> <q> [Y64-i64-out] [--claims <accdir> <obid>]\n"
         << "       zkob_rope verify <obdir> <seed> <B> <C> <HD> <cos-int32> <sin-int32> <gen> <q> [--claims <vaccdir> <obid>]" << endl;
    return 2;
}

// Stage C2 single-process transport: `serve` keeps this driver resident (one
// CUDA init for the whole walk); every request runs the same zkw_run1 entry.
int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "serve")
        return zkw_serve(argv[0], zkw_run1);
    return zkw_run1(argc, argv);
}
