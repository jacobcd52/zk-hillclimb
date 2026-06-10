// COORDINATOR-BUILT (do not let submission agents edit).
// Real driver for ONE nonlinearity_lookup obligation (mlp.swiglu):
//   (1) MAPPING LOOKUP: every pair (G[i], S[i]) is a row of the PUBLIC table
//       {(x, silu(x)) : x in [low, low+len)} — proven with the logUp lookup on
//       the combined witness  comb = G + r*S  against the combined table
//       T_comb = table + r*mapped, where r is an FS challenge derived AFTER
//       all witness commitments. The verifier forms the combined commitment
//       HOMOMORPHICALLY:  com_comb[j] = com_G[j] + r*com_S[j]  (1-thread
//       helpers; same generators for G and S).  This also range-binds G.
//   (2) HADAMARD SUMCHECK: H = S .* U (elementwise), proven by the eq-weighted
//       degree-3 sumcheck  H~(u_h) = sum_b eq(u_h,b) S(b) U(b), binding com_S,
//       com_U, com_H via IPA openings.
//   The subsequent rescale of H (sf 2^16) is a SEPARATE zkob_rescale run; the
//   orchestrator composes both under the mlp.swiglu obligation id and checks
//   com_H here == com_X there byte-identically.
//
// Layout: all tensors committed row-wise with the SAME generator set gen
// (size C_pad): G, S, U, H, A: B_pad x C_pad; m: (N/C_pad) x C_pad.
// Zero-padding works because the table MUST contain the pair (0, 0)
// (mapped[-low] == 0; true for silu since silu(0)=0 — checked at load).
// Constraints: len = N a power of two, C_pad <= N <= D = B_pad*C_pad, N | D.
//
// FS schedule (one transcript, seed = run_seed:obligation_id):
//   absorb B, C, LOW, LEN, com_G, com_S, com_U, com_H, com_m
//   -> r (pair combiner) -> beta
//   compute A = 1/(comb+beta); absorb com_A -> alpha -> u (logD)
//   lookup rounds: absorb p(0..3) -> w_r;  absorb A_f, S_f, m_f
//   IPA(A), IPA(comb) at u_pt = reverse(w);  IPA(m) at u_m = reverse(w[n1..])
//   -> u_h (logD); absorb claim_H = H~(u_h)
//   hadamard rounds: absorb p(0..3) -> wh_r;  absorb S_f2, U_f2
//   IPA(H) at u_h;  IPA(S), IPA(U) at u_pt2 = reverse(wh)
//
// Files in <obdir>: dims.bin, com_G/com_S/com_U/com_H/com_m/com_A.bin,
//   lookup.bin, hp.bin, ipa_A/ipa_comb/ipa_m/ipa_H/ipa_S/ipa_U.bin
//
// Usage:
//   zkob_glu prove  <obdir> <seed> <G-int32.bin> <U-int32.bin> <B> <C>
//                   <low> <len> <mapped-int32.bin> <gen.bin> <q.bin>
//                   [H-int64-out.bin]
//   zkob_glu verify <obdir> <seed> <B> <C> <low> <len> <mapped-int32.bin>
//                   <gen.bin> <q.bin>
//   zkob_glu selftest
#include "zkob_lookup.cuh"
#include <iostream>
#include <sys/stat.h>
using namespace std;

// load mapped values and enforce the zero-pair layout requirement
static FrTensor load_mapped(const string& path, int low, uint len) {
    FILE* f = open_or_die(path, "rb");
    vector<int> buf(len);
    if (fread(buf.data(), sizeof(int), len, f) != len)
        throw runtime_error("mapped table: short read");
    fclose(f);
    if (len != (1u << ceilLog2(len))) throw runtime_error("table len must be pow2");
    if (low > 0 || low + (int)len <= 0) throw runtime_error("table must contain x=0");
    if (buf[-low] != 0) throw runtime_error("layout needs mapped(0) == 0");
    return FrTensor(len, buf.data());
}

// ---------------- prove ----------------
// evil 1 (selftest): bump S[i] += 1 (wrong mapped value), H recomputed from the
//   evil S so the hadamard part is consistent — the LOOKUP must reject.
// evil 2 (selftest): honest S, bump H[i] += 1 — the HADAMARD must reject.
static void prove(const string& obdir, const string& seed,
                  const FrTensor& G, const FrTensor& U,
                  uint B, uint C, int low, uint len, const FrTensor& mapped,
                  const Commitment& gen, const G1Jacobian_t& Q,
                  const string& h_out_path, int evil = 0, uint evil_idx = 0) {
    const uint B_pad = 1u << ceilLog2(B), C_pad = 1u << ceilLog2(C);
    const uint D = B_pad * C_pad, N = len;
    const uint logD = ceilLog2(D), n1 = ceilLog2(D / N);
    if (gen.size != C_pad) throw runtime_error("generator size != C_pad");
    if (C_pad > N || N > D || D % N)
        throw runtime_error("layout needs C_pad <= N <= D, N | D");
    if (G.size != B * C || U.size != B * C) throw runtime_error("input dims");

    tLookupRangeMapping tl(low, len, mapped);
    FrTensor G_pad = G.pad({B, C});
    FrTensor U_pad = U.pad({B, C});
    auto p = tl(G_pad);                       // (S_pad, m) over all D entries
    FrTensor& S_pad = p.first;
    FrTensor& m = p.second;
    if (evil == 1) {
        k_bump<<<1,1>>>(S_pad.gpu_data, evil_idx, F_ONE, 0);
        cudaDeviceSynchronize();
    }
    FrTensor H_pad = S_pad * U_pad;
    if (evil == 2) {
        k_bump<<<1,1>>>(H_pad.gpu_data, evil_idx, F_ONE, 0);
        cudaDeviceSynchronize();
    }
    if (!h_out_path.empty()) {
        // chain file is the UNPADDED B x C tensor in int64 (values to ~2^46);
        // zkob_rescale re-pads it itself. Strip the C_pad/B_pad zero padding.
        H_pad.save_long(h_out_path);
        if (B != B_pad || C != C_pad) {
            vector<long> padded(D);
            FILE* f = open_or_die(h_out_path, "rb");
            if (fread(padded.data(), sizeof(long), D, f) != D)
                throw runtime_error("chain strip: short read");
            fclose(f);
            f = open_or_die(h_out_path, "wb");
            for (uint b = 0; b < B; b++)
                fwrite(padded.data() + (size_t)b * C_pad, sizeof(long), C, f);
            fclose(f);
        }
    }

    G1TensorJacobian com_G = gen.commit(G_pad);
    G1TensorJacobian com_S = gen.commit(S_pad);
    G1TensorJacobian com_U = gen.commit(U_pad);
    G1TensorJacobian com_H = gen.commit(H_pad);
    G1TensorJacobian com_m = gen.commit(m);   // (N/C_pad) rows
    com_G.save(obdir + "/com_G.bin");
    com_S.save(obdir + "/com_S.bin");
    com_U.save(obdir + "/com_U.bin");
    com_H.save(obdir + "/com_H.bin");
    com_m.save(obdir + "/com_m.bin");
    { FILE* f = open_or_die(obdir + "/dims.bin", "wb");
      int32_t d[4] = {(int32_t)B, (int32_t)C, low, (int32_t)len};
      fwrite(d, sizeof(int32_t), 4, f); fclose(f); }

    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C);
    absorb_u32(tr, "LOW", (uint32_t)low); absorb_u32(tr, "LEN", len);
    absorb_g1_tensor(tr, "com_G", com_G);
    absorb_g1_tensor(tr, "com_S", com_S);
    absorb_g1_tensor(tr, "com_U", com_U);
    absorb_g1_tensor(tr, "com_H", com_H);
    absorb_g1_tensor(tr, "com_m", com_m);
    Fr_t r = fs_challenge_fr(tr);
    Fr_t beta = fs_challenge_fr(tr);

    FrTensor comb = G_pad + S_pad * r;        // combined lookup witness
    FrTensor T_comb = tl.table + tl.mapped_vals * r;

    FrTensor A(D);
    tlookup_inv_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        comb.gpu_data, beta, A.gpu_data, D);
    cudaDeviceSynchronize();
    G1TensorJacobian com_A = gen.commit(A);
    com_A.save(obdir + "/com_A.bin");
    absorb_g1_tensor(tr, "com_A", com_A);
    Fr_t alpha = fs_challenge_fr(tr);
    auto u = fs_challenge_vec(tr, logD);

    FrTensor Bv(N);
    tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        T_comb.gpu_data, beta, Bv.gpu_data, N);
    cudaDeviceSynchronize();
    Fr_t alpha_sq = alpha * alpha;
    Fr_t Cc = alpha_sq - (Bv * m).sum();
    Fr_t claim = alpha + alpha_sq;
    Fr_t inv_ratio = Fr_t{N,0,0,0,0,0,0,0} / Fr_t{D,0,0,0,0,0,0,0};

    LookupProof pf;
    vector<Fr_t> ws;
    fs_phase1(claim, A, comb, Bv, T_comb, m, alpha, beta, Cc,
              inv_ratio, alpha_sq, u, tr, ws, pf, evil != 1);
    write_lookup(obdir + "/lookup.bin", pf);
    absorb_fr(tr, "A_f", pf.A_f); absorb_fr(tr, "S_f", pf.S_f); absorb_fr(tr, "m_f", pf.m_f);

    vector<Fr_t> u_pt(ws.rbegin(), ws.rend());
    vector<Fr_t> u_m(ws.rbegin(), ws.rend() - n1);
    if (evil != 1) {  // convention sanity: fold terminals == ME evaluations
        const uint logG = ceilLog2(C_pad);
        vector<Fr_t> u_col(u_pt.begin(), u_pt.begin() + logG);
        vector<Fr_t> u_row(u_pt.begin() + logG, u_pt.end());
        if (!fr_eq(pf.A_f, A.multi_dim_me({u_row, u_col}, {B_pad, C_pad})) ||
            !fr_eq(pf.S_f, comb.multi_dim_me({u_row, u_col}, {B_pad, C_pad})))
            throw runtime_error("lookup terminal != multi_dim_me (convention bug)");
    }
    open_prove(A, C_pad, gen, Q, u_pt, obdir + "/ipa_A.bin", tr);
    open_prove(comb, C_pad, gen, Q, u_pt, obdir + "/ipa_comb.bin", tr);
    open_prove(m, C_pad, gen, Q, u_m, obdir + "/ipa_m.bin", tr);

    // ---- hadamard part: H = S .* U ----
    auto u_h = fs_challenge_vec(tr, logD);
    HadamardProof hp;
    {
        const uint logG = ceilLog2(C_pad);
        vector<Fr_t> uh_col(u_h.begin(), u_h.begin() + logG);
        vector<Fr_t> uh_row(u_h.begin() + logG, u_h.end());
        hp.claim_H = H_pad.multi_dim_me({uh_row, uh_col}, {B_pad, C_pad});
    }
    absorb_fr(tr, "claim_H", hp.claim_H);
    FrTensor E = build_eq_tensor(u_h);
    FrTensor S2(S_pad), U2(U_pad);            // fold buffers
    vector<Fr_t> wsh;
    fs_hadamard(hp.claim_H, E, S2, U2, tr, wsh, hp, evil != 2);
    absorb_fr(tr, "S_f2", hp.S_f2); absorb_fr(tr, "U_f2", hp.U_f2);
    write_hp(obdir + "/hp.bin", hp);

    vector<Fr_t> u_pt2(wsh.rbegin(), wsh.rend());
    if (evil != 2) {
        const uint logG = ceilLog2(C_pad);
        vector<Fr_t> u_col(u_pt2.begin(), u_pt2.begin() + logG);
        vector<Fr_t> u_row(u_pt2.begin() + logG, u_pt2.end());
        if (!fr_eq(hp.S_f2, S_pad.multi_dim_me({u_row, u_col}, {B_pad, C_pad})) ||
            !fr_eq(hp.U_f2, U_pad.multi_dim_me({u_row, u_col}, {B_pad, C_pad})))
            throw runtime_error("hadamard terminal != multi_dim_me (convention bug)");
    }
    open_prove(H_pad, C_pad, gen, Q, u_h, obdir + "/ipa_H.bin", tr);
    open_prove(S_pad, C_pad, gen, Q, u_pt2, obdir + "/ipa_S.bin", tr);
    open_prove(U_pad, C_pad, gen, Q, u_pt2, obdir + "/ipa_U.bin", tr);
    cout << "PROVED swiglu obligation -> " << obdir << endl;
}

// ---------------- verify (witness-free) ----------------
static bool verify(const string& obdir, const string& seed, uint B, uint C,
                   int low, uint len, const FrTensor& mapped,
                   const Commitment& gen, const G1Jacobian_t& Q) {
    const uint B_pad = 1u << ceilLog2(B), C_pad = 1u << ceilLog2(C);
    const uint D = B_pad * C_pad, N = len;
    const uint logD = ceilLog2(D), n1 = ceilLog2(D / N), n2 = ceilLog2(N);
    if (gen.size != C_pad || C_pad > N || N > D || D % N) {
        cout << "REJECT: bad layout params" << endl; return false;
    }
    { FILE* f = open_or_die(obdir + "/dims.bin", "rb");
      int32_t d[4]; if (fread(d, sizeof(int32_t), 4, f) != 4) { fclose(f); return false; }
      fclose(f);
      if (d[0] != (int32_t)B || d[1] != (int32_t)C || d[2] != low || d[3] != (int32_t)len) {
          cout << "REJECT: dims.bin mismatch" << endl; return false; } }

    G1TensorJacobian com_G(obdir + "/com_G.bin");
    G1TensorJacobian com_S(obdir + "/com_S.bin");
    G1TensorJacobian com_U(obdir + "/com_U.bin");
    G1TensorJacobian com_H(obdir + "/com_H.bin");
    G1TensorJacobian com_m(obdir + "/com_m.bin");
    G1TensorJacobian com_A(obdir + "/com_A.bin");
    if (com_G.size != B_pad || com_S.size != B_pad || com_U.size != B_pad ||
        com_H.size != B_pad || com_A.size != B_pad || com_m.size != N / C_pad) {
        cout << "REJECT: commitment row counts" << endl; return false;
    }
    LookupProof pf = read_lookup(obdir + "/lookup.bin");
    HadamardProof hp = read_hp(obdir + "/hp.bin");
    if (pf.ev.size() != 4 * logD || hp.ev.size() != 4 * logD) {
        cout << "REJECT: round count" << endl; return false;
    }

    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C);
    absorb_u32(tr, "LOW", (uint32_t)low); absorb_u32(tr, "LEN", len);
    absorb_g1_tensor(tr, "com_G", com_G);
    absorb_g1_tensor(tr, "com_S", com_S);
    absorb_g1_tensor(tr, "com_U", com_U);
    absorb_g1_tensor(tr, "com_H", com_H);
    absorb_g1_tensor(tr, "com_m", com_m);
    Fr_t r = fs_challenge_fr(tr);
    Fr_t beta = fs_challenge_fr(tr);
    absorb_g1_tensor(tr, "com_A", com_A);
    Fr_t alpha = fs_challenge_fr(tr);
    auto u = fs_challenge_vec(tr, logD);

    // combined commitment, formed homomorphically (1-thread helpers; the
    // batched-G1-kernel route is -dlto miscompile bait, see GOTCHAS)
    vector<G1Jacobian_t> hcomb(B_pad);
    {
        vector<G1Jacobian_t> hg(B_pad), hs(B_pad);
        cudaMemcpy(hg.data(), com_G.gpu_data, B_pad * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hs.data(), com_S.gpu_data, B_pad * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        for (uint j = 0; j < B_pad; j++) hcomb[j] = h_add(hg[j], h_mul(hs[j], r));
    }
    G1TensorJacobian com_comb(B_pad, hcomb.data());

    // lookup rounds: anchor recomputed, Lagrange-4 chain
    const Fr_t inv6 = inv(F_SIX);
    Fr_t cur = h_scalar(alpha, h_scalar(alpha, alpha, 2), 0);   // alpha + alpha^2
    Fr_t alpha_acc = alpha, alphasq_acc = h_scalar(alpha, alpha, 2);
    vector<Fr_t> ws;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {pf.ev[4*k], pf.ev[4*k+1], pf.ev[4*k+2], pf.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0))) {
            cout << "REJECT: lookup round " << k << " p(0)+p(1) != claim" << endl;
            return false;
        }
        absorb_fr(tr, "p0", e[0]); absorb_fr(tr, "p1", e[1]);
        absorb_fr(tr, "p2", e[2]); absorb_fr(tr, "p3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws.push_back(w);
        cur = lagrange4(e, w, inv6);
        Fr_t eqv = my_eq(u[logD - 1 - k], w);
        alpha_acc = h_scalar(alpha_acc, eqv, 2);
        if (k >= n1) alphasq_acc = h_scalar(alphasq_acc, eqv, 2);
    }
    // B_f, T_f recomputed from the PUBLIC combined table
    Fr_t B_f, T_f;
    {
        tLookupRangeMapping tl(low, len, mapped);
        FrTensor T_comb = tl.table + tl.mapped_vals * r;
        FrTensor B_pub(N);
        tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            T_comb.gpu_data, beta, B_pub.gpu_data, N);
        cudaDeviceSynchronize();
        Fr_t *dB, *dT, *dtmp;
        cudaMalloc(&dB, N * sizeof(Fr_t)); cudaMalloc(&dT, N * sizeof(Fr_t));
        cudaMalloc(&dtmp, N * sizeof(Fr_t));
        cudaMemcpy(dB, B_pub.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dT, T_comb.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
        uint sz = N;
        for (uint k = 0; k < n2; k++) {
            uint nsz = sz >> 1;
            k_fr_fold<<<(nsz + 31) / 32, 32>>>(dB, ws[n1 + k], dtmp, nsz);
            cudaDeviceSynchronize(); std::swap(dB, dtmp);
            k_fr_fold<<<(nsz + 31) / 32, 32>>>(dT, ws[n1 + k], dtmp, nsz);
            cudaDeviceSynchronize(); std::swap(dT, dtmp);
            sz = nsz;
        }
        cudaMemcpy(&B_f, dB, sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&T_f, dT, sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaFree(dB); cudaFree(dT); cudaFree(dtmp);
    }
    // terminal identity
    Fr_t inv_ratio = h_scalar({N,0,0,0,0,0,0,0}, inv({D,0,0,0,0,0,0,0}), 2);
    Fr_t t1 = h_scalar(alpha_acc, h_scalar(pf.A_f, h_scalar(pf.S_f, beta, 0), 2), 2);
    Fr_t t2 = h_scalar(inv_ratio, h_scalar(alphasq_acc, h_scalar(B_f, h_scalar(T_f, beta, 0), 2), 2), 2);
    Fr_t t4 = h_scalar(inv_ratio, h_scalar(pf.m_f, B_f, 2), 2);
    Fr_t rhs = h_scalar(h_scalar(h_scalar(t1, t2, 0), pf.A_f, 0), t4, 1);
    if (!fr_eq(cur, rhs)) {
        cout << "REJECT: lookup terminal identity" << endl; return false;
    }
    absorb_fr(tr, "A_f", pf.A_f); absorb_fr(tr, "S_f", pf.S_f); absorb_fr(tr, "m_f", pf.m_f);

    vector<Fr_t> u_pt(ws.rbegin(), ws.rend());
    vector<Fr_t> u_m(ws.rbegin(), ws.rend() - n1);
    if (!open_verify(com_A, gen, C_pad, Q, u_pt, pf.A_f, obdir + "/ipa_A.bin", tr)) {
        cout << "REJECT: IPA opening of A_f vs com_A" << endl; return false;
    }
    if (!open_verify(com_comb, gen, C_pad, Q, u_pt, pf.S_f, obdir + "/ipa_comb.bin", tr)) {
        cout << "REJECT: IPA opening of S_f vs com_G + r*com_S" << endl; return false;
    }
    if (!open_verify(com_m, gen, C_pad, Q, u_m, pf.m_f, obdir + "/ipa_m.bin", tr)) {
        cout << "REJECT: IPA opening of m_f vs com_m" << endl; return false;
    }

    // ---- hadamard part ----
    auto u_h = fs_challenge_vec(tr, logD);
    absorb_fr(tr, "claim_H", hp.claim_H);
    Fr_t curh = hp.claim_H, eq_acc = F_ONE;
    vector<Fr_t> wsh;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp.ev[4*k], hp.ev[4*k+1], hp.ev[4*k+2], hp.ev[4*k+3]};
        if (!fr_eq(curh, h_scalar(e[0], e[1], 0))) {
            cout << "REJECT: hadamard round " << k << " p(0)+p(1) != claim" << endl;
            return false;
        }
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); wsh.push_back(w);
        curh = lagrange4(e, w, inv6);
        eq_acc = h_scalar(eq_acc, my_eq(u_h[logD - 1 - k], w), 2);
    }
    Fr_t rhsh = h_scalar(eq_acc, h_scalar(hp.S_f2, hp.U_f2, 2), 2);
    if (!fr_eq(curh, rhsh)) {
        cout << "REJECT: hadamard terminal identity" << endl; return false;
    }
    absorb_fr(tr, "S_f2", hp.S_f2); absorb_fr(tr, "U_f2", hp.U_f2);

    vector<Fr_t> u_pt2(wsh.rbegin(), wsh.rend());
    if (!open_verify(com_H, gen, C_pad, Q, u_h, hp.claim_H, obdir + "/ipa_H.bin", tr)) {
        cout << "REJECT: IPA opening of claim_H vs com_H" << endl; return false;
    }
    if (!open_verify(com_S, gen, C_pad, Q, u_pt2, hp.S_f2, obdir + "/ipa_S.bin", tr)) {
        cout << "REJECT: IPA opening of S_f2 vs com_S" << endl; return false;
    }
    if (!open_verify(com_U, gen, C_pad, Q, u_pt2, hp.U_f2, obdir + "/ipa_U.bin", tr)) {
        cout << "REJECT: IPA opening of U_f2 vs com_U" << endl; return false;
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
static bool selftest_case(uint B, uint C, int low, uint len) {
    cout << "---- selftest case B=" << B << " C=" << C
         << " low=" << low << " len=" << len << " ----" << endl;
    const uint C_pad = 1u << ceilLog2(C);
    // toy mapped table: f(x) = x^2 (f(0) = 0 as the layout requires)
    vector<int> map_host(len);
    for (uint i = 0; i < len; i++) { int x = (int)i + low; map_host[i] = x * x; }
    FrTensor mapped(len, map_host.data());
    // toy witness in range
    srand(42 + B + C);
    vector<int> g_host(B * C), u_host(B * C);
    for (uint i = 0; i < B * C; i++) {
        g_host[i] = (rand() % (int)len) + low;
        u_host[i] = (rand() % 13) - 6;
    }
    FrTensor G(B * C, g_host.data()), U(B * C, u_host.data());
    Commitment gen = Commitment::random(C_pad);
    Commitment qg = Commitment::random(1);
    gen.save("/tmp/zkob_glu_gen.bin"); qg.save("/tmp/zkob_glu_q.bin");
    string obdir = "/tmp/zkob_glu_ob";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:glu";

    prove(obdir, seed, G, U, B, C, low, len, mapped, gen, qg(0), "/tmp/zkob_glu_H.i64.bin");
    bool honest = verify(obdir, seed, B, C, low, len, mapped, gen, qg(0));
    cout << "honest: " << (honest ? "ACCEPT" : "REJECT(!!)") << endl;
    if (!honest) return false;

    // byte-tamper forgeries (re-prove honestly before each)
    struct Tamper { const char* file; long off; };
    vector<Tamper> tampers = {
        {"lookup.bin", 4 + 32}, {"lookup.bin", -32}, {"hp.bin", 4},
        {"hp.bin", -32}, {"com_A.bin", 24}, {"com_m.bin", 24},
        {"com_H.bin", 24}, {"ipa_comb.bin", -32}, {"ipa_U.bin", -32},
    };
    for (auto& t : tampers) {
        prove(obdir, seed, G, U, B, C, low, len, mapped, gen, qg(0), "");
        tamper_byte(obdir + "/" + t.file, t.off, 1);
        bool rejected = !verify(obdir, seed, B, C, low, len, mapped, gen, qg(0));
        cout << "tamper " << t.file << "@" << t.off << " rejected: "
             << (rejected ? "YES" : "NO(!!)") << endl;
        if (!rejected) return false;
    }
    // semantic forgery 1: wrong mapped value (S[i] += 1), hadamard consistent
    {
        string evdir = "/tmp/zkob_glu_evil";
        mkdir(evdir.c_str(), 0755);
        prove(evdir, seed, G, U, B, C, low, len, mapped, gen, qg(0), "", 1, 3);
        bool rejected = !verify(evdir, seed, B, C, low, len, mapped, gen, qg(0));
        cout << "semantic wrong-mapping (S[3]+=1) rejected: "
             << (rejected ? "YES" : "NO(!!)") << endl;
        if (!rejected) return false;
    }
    // semantic forgery 2: wrong product (H[i] += 1), lookup consistent
    {
        string evdir = "/tmp/zkob_glu_evil";
        prove(evdir, seed, G, U, B, C, low, len, mapped, gen, qg(0), "", 2, 5);
        bool rejected = !verify(evdir, seed, B, C, low, len, mapped, gen, qg(0));
        cout << "semantic wrong-product (H[5]+=1) rejected: "
             << (rejected ? "YES" : "NO(!!)") << endl;
        if (!rejected) return false;
    }
    cout << "CASE PASS" << endl;
    return true;
}

int main(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    if (mode == "selftest") {
        bool a = selftest_case(8, 4, -8, 16);    // n1 = 1
        bool b = selftest_case(4, 4, -8, 16);    // n1 = 0 (pure phase2)
        bool c = selftest_case(8, 6, -16, 32);   // padded C, m multi-row
        cout << ((a && b && c) ? "ZKOB-GLU SELFTEST: ALL PASS"
                               : "ZKOB-GLU SELFTEST: FAIL") << endl;
        return (a && b && c) ? 0 : 1;
    }
    if (mode == "prove" && (argc == 13 || argc == 14)) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[6]), C = stoi(argv[7]);
        int low = stoi(argv[8]); uint len = stoi(argv[9]);
        FrTensor G = load_int32_tensor(argv[4], B * C);
        FrTensor U = load_int32_tensor(argv[5], B * C);
        FrTensor mapped = load_mapped(argv[10], low, len);
        Commitment gen(argv[11]), qg(argv[12]);
        prove(obdir, seed, G, U, B, C, low, len, mapped, gen, qg(0),
              argc == 14 ? argv[13] : "");
        return 0;
    }
    if (mode == "verify" && argc == 11) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[4]), C = stoi(argv[5]);
        int low = stoi(argv[6]); uint len = stoi(argv[7]);
        FrTensor mapped = load_mapped(argv[8], low, len);
        Commitment gen(argv[9]), qg(argv[10]);
        return verify(obdir, seed, B, C, low, len, mapped, gen, qg(0)) ? 0 : 1;
    }
    cerr << "usage: zkob_glu selftest\n"
         << "       zkob_glu prove  <obdir> <seed> <G-int32> <U-int32> <B> <C> <low> <len> <mapped-int32> <gen> <q> [H-int64-out]\n"
         << "       zkob_glu verify <obdir> <seed> <B> <C> <low> <len> <mapped-int32> <gen> <q>" << endl;
    return 2;
}
