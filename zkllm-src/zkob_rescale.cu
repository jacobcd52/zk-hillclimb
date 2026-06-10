// COORDINATOR-BUILT (do not let submission agents edit).
// Real driver for ONE rescaling_lookup obligation: X = sf * X_ + rem, with
// rem range-checked in [-sf/2, sf/2) via the logUp lookup (algebra pinned in
// vrf_toy_lookup.cu §9; openings via the FS IPA pinned in vrf_toy_ipa.cu §10).
//
// The obligation has two parts:
//  (1) AFFINE LINK, checked HOMOMORPHICALLY on public commitments, per row:
//        com_X[j] == sf * com_X_[j] + com_rem[j]
//      (all three committed with the same generators; zero-padding satisfies
//       the identity trivially: 0 == sf*0 + 0).
//  (2) RANGE PROOF on rem via logUp against the PUBLIC table
//      tLookupRange(-sf/2, sf) (size N = sf, a power of two), with witness
//      S = rem_padded (size D = B_pad * C_pad, padded with zeros; 0 is in the
//      table). Terminals A_f (aux inverse), S_f (rem) and m_f (multiplicities)
//      are IPA opening obligations vs com_A / com_rem / com_m; B_f and T_f are
//      RECOMPUTED by the verifier from the public table.
//
// FS schedule (one transcript per obligation, seed = run_seed:obligation_id):
//   absorb B, C, LOG_SF, com_X, com_X_, com_rem, com_m  -> beta
//   compute A = 1/(rem+beta); absorb com_A              -> alpha
//   -> u (logD challenges; the eq-binding evaluation point)
//   per round r: absorb p(0),p(1),p(2),p(3)             -> w_r
//   absorb A_f, S_f, m_f
//   IPA(A), IPA(rem) at u_pt = reverse(w);  IPA(m) at u_m = reverse(w[n1..])
//   (round r binds the current MSB, so the ME opening point is the reversed
//    round-challenge sequence; pinned in §9/§10.)
//
// Commitment layout (harness registration rules, §8): every tensor is
// committed row-wise with the SAME generator set gen (size C_pad):
//   X, X_, rem, A: B_pad rows x C_pad;  m: (N / C_pad) rows x C_pad.
// Constraints: C_pad <= N <= D (checked).
//
// Files in <obdir>: dims.bin, com_X.bin, com_Xr.bin, com_rem.bin, com_m.bin,
//                   com_A.bin, lookup.bin, ipa_A.bin, ipa_rem.bin, ipa_m.bin
//
// Usage:
//   zkob_rescale prove  <obdir> <seed> <X-int.bin> <B> <C> <LOG_SF>
//                       <gen.bin> <q.bin> [Xr-int-out.bin]
//   zkob_rescale verify <obdir> <seed> <B> <C> <LOG_SF> <gen.bin> <q.bin>
//   zkob_rescale selftest
#include "zkob_lookup.cuh"
#include <iostream>
#include <sys/stat.h>
using namespace std;

// ---------------- prove ----------------
// evil_idx >= 0: selftest-only SEMANTIC forgery — after computing the honest
// multiplicities, shift X_[i] += 1 and rem[i] -= sf. The affine link still
// holds exactly (sf*(X_+1) + (rem-sf) == sf*X_ + rem) but rem leaves the
// range; the verifier's lookup MUST reject. This is precisely the covert
// channel the obligation exists to close.
static void prove(const string& obdir, const string& seed, const FrTensor& X,
                  uint B, uint C, uint log_sf, const Commitment& gen,
                  const G1Jacobian_t& Q, const string& xr_out_path, int evil_idx = -1) {
    const uint sf = 1u << log_sf;
    const uint B_pad = 1u << ceilLog2(B), C_pad = 1u << ceilLog2(C);
    const uint D = B_pad * C_pad, N = sf;
    const uint logD = ceilLog2(D), n1 = ceilLog2(D / N);
    if (gen.size != C_pad) throw runtime_error("generator size != C_pad");
    if (C_pad > N || N > D || D % N)
        throw runtime_error("layout needs C_pad <= N <= D, N | D");

    FrTensor Xr(B * C), rem(B * C);
    rescaling_kernel<<<(B*C + 255) / 256, 256>>>(
        const_cast<Fr_t*>(X.gpu_data), Xr.gpu_data, rem.gpu_data, (long)sf, B * C);
    cudaDeviceSynchronize();
    if (!xr_out_path.empty()) Xr.save_int(xr_out_path);

    // multiplicities from the HONEST rem (also what the evil prover would do:
    // an out-of-range value has no table slot to count into)
    tLookupRange tl(-(int)(sf >> 1), sf);
    FrTensor m = tl.prep(rem.pad({B, C}));
    if (evil_idx >= 0) {
        k_bump<<<1,1>>>(Xr.gpu_data, (uint)evil_idx, F_ONE, 0);
        k_bump<<<1,1>>>(rem.gpu_data, (uint)evil_idx, {sf,0,0,0,0,0,0,0}, 1);
        cudaDeviceSynchronize();
    }
    FrTensor X_pad = X.pad({B, C});
    FrTensor Xr_pad = Xr.pad({B, C});
    FrTensor rem_pad = rem.pad({B, C});      // = lookup witness S, flat D

    G1TensorJacobian com_X = gen.commit(X_pad);
    G1TensorJacobian com_Xr = gen.commit(Xr_pad);
    G1TensorJacobian com_rem = gen.commit(rem_pad);
    G1TensorJacobian com_m = gen.commit(m);  // (N/C_pad) rows
    com_X.save(obdir + "/com_X.bin");
    com_Xr.save(obdir + "/com_Xr.bin");
    com_rem.save(obdir + "/com_rem.bin");
    com_m.save(obdir + "/com_m.bin");
    { FILE* f = open_or_die(obdir + "/dims.bin", "wb");
      uint32_t d[3] = {B, C, log_sf}; fwrite(d, sizeof(uint32_t), 3, f); fclose(f); }

    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C); absorb_u32(tr, "LOG_SF", log_sf);
    absorb_g1_tensor(tr, "com_X", com_X);
    absorb_g1_tensor(tr, "com_Xr", com_Xr);
    absorb_g1_tensor(tr, "com_rem", com_rem);
    absorb_g1_tensor(tr, "com_m", com_m);
    Fr_t beta = fs_challenge_fr(tr);

    FrTensor A(D);
    tlookup_inv_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        rem_pad.gpu_data, beta, A.gpu_data, D);
    cudaDeviceSynchronize();
    G1TensorJacobian com_A = gen.commit(A);
    com_A.save(obdir + "/com_A.bin");
    absorb_g1_tensor(tr, "com_A", com_A);
    Fr_t alpha = fs_challenge_fr(tr);
    auto u = fs_challenge_vec(tr, logD);

    FrTensor Bv(N);
    tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        tl.table.gpu_data, beta, Bv.gpu_data, N);
    cudaDeviceSynchronize();
    Fr_t alpha_sq = alpha * alpha;
    Fr_t Cc = alpha_sq - (Bv * m).sum();
    Fr_t claim = alpha + alpha_sq;
    Fr_t inv_ratio = Fr_t{N,0,0,0,0,0,0,0} / Fr_t{D,0,0,0,0,0,0,0};

    LookupProof pf;
    vector<Fr_t> ws;
    fs_phase1(claim, A, rem_pad, Bv, tl.table, m, alpha, beta, Cc,
              inv_ratio, alpha_sq, u, tr, ws, pf, evil_idx < 0);
    write_lookup(obdir + "/lookup.bin", pf);
    absorb_fr(tr, "A_f", pf.A_f); absorb_fr(tr, "S_f", pf.S_f); absorb_fr(tr, "m_f", pf.m_f);

    // opening points (reversed round-challenge sequences; pinned §9/§10)
    vector<Fr_t> u_pt(ws.rbegin(), ws.rend());
    vector<Fr_t> u_m(ws.rbegin(), ws.rend() - n1);

    if (evil_idx < 0) {  // convention sanity: fold terminals == ME evaluations
        const uint logG = ceilLog2(C_pad);
        vector<Fr_t> u_col(u_pt.begin(), u_pt.begin() + logG);
        vector<Fr_t> u_row(u_pt.begin() + logG, u_pt.end());
        if (!fr_eq(pf.A_f, A.multi_dim_me({u_row, u_col}, {B_pad, C_pad})) ||
            !fr_eq(pf.S_f, rem_pad.multi_dim_me({u_row, u_col}, {B_pad, C_pad})))
            throw runtime_error("terminal != multi_dim_me (convention bug)");
        if (N / C_pad > 1) {
            vector<Fr_t> um_col(u_m.begin(), u_m.begin() + logG);
            vector<Fr_t> um_row(u_m.begin() + logG, u_m.end());
            if (!fr_eq(pf.m_f, m.multi_dim_me({um_row, um_col}, {N / C_pad, C_pad})))
                throw runtime_error("m terminal != multi_dim_me (convention bug)");
        }
    }
    open_prove(A, C_pad, gen, Q, u_pt, obdir + "/ipa_A.bin", tr);
    open_prove(rem_pad, C_pad, gen, Q, u_pt, obdir + "/ipa_rem.bin", tr);
    open_prove(m, C_pad, gen, Q, u_m, obdir + "/ipa_m.bin", tr);
    cout << "PROVED rescaling obligation -> " << obdir << endl;
}
// ---------------- verify (witness-free) ----------------
static bool verify(const string& obdir, const string& seed, uint B, uint C,
                   uint log_sf, const Commitment& gen, const G1Jacobian_t& Q) {
    const uint sf = 1u << log_sf;
    const uint B_pad = 1u << ceilLog2(B), C_pad = 1u << ceilLog2(C);
    const uint D = B_pad * C_pad, N = sf;
    const uint logD = ceilLog2(D), n1 = ceilLog2(D / N), n2 = ceilLog2(N);
    if (gen.size != C_pad || C_pad > N || N > D || D % N) {
        cout << "REJECT: bad layout params" << endl; return false;
    }
    { FILE* f = open_or_die(obdir + "/dims.bin", "rb");
      uint32_t d[3]; if (fread(d, sizeof(uint32_t), 3, f) != 3) { fclose(f); return false; }
      fclose(f);
      if (d[0] != B || d[1] != C || d[2] != log_sf) {
          cout << "REJECT: dims.bin mismatch" << endl; return false; } }

    G1TensorJacobian com_X(obdir + "/com_X.bin");
    G1TensorJacobian com_Xr(obdir + "/com_Xr.bin");
    G1TensorJacobian com_rem(obdir + "/com_rem.bin");
    G1TensorJacobian com_m(obdir + "/com_m.bin");
    G1TensorJacobian com_A(obdir + "/com_A.bin");
    if (com_X.size != B_pad || com_Xr.size != B_pad || com_rem.size != B_pad ||
        com_A.size != B_pad || com_m.size != N / C_pad) {
        cout << "REJECT: commitment row counts" << endl; return false;
    }
    // (1) homomorphic affine link, all rows at once
    {
        uint n_bad = 0;
        vector<G1Jacobian_t> hx(B_pad), hxr(B_pad), hrem(B_pad);
        cudaMemcpy(hx.data(), com_X.gpu_data, B_pad * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hxr.data(), com_Xr.gpu_data, B_pad * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hrem.data(), com_rem.gpu_data, B_pad * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        for (uint j = 0; j < B_pad; j++)
            if (!g1_eq(hx[j], h_add(h_mul(hxr[j], {sf,0,0,0,0,0,0,0}), hrem[j]))) n_bad++;
        if (n_bad) {
            cout << "REJECT: affine link com_X != sf*com_Xr + com_rem ("
                 << n_bad << " rows)" << endl;
            return false;
        }
    }
    LookupProof pf = read_lookup(obdir + "/lookup.bin");
    if (pf.ev.size() != 4 * logD) { cout << "REJECT: round count" << endl; return false; }

    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C); absorb_u32(tr, "LOG_SF", log_sf);
    absorb_g1_tensor(tr, "com_X", com_X);
    absorb_g1_tensor(tr, "com_Xr", com_Xr);
    absorb_g1_tensor(tr, "com_rem", com_rem);
    absorb_g1_tensor(tr, "com_m", com_m);
    Fr_t beta = fs_challenge_fr(tr);
    absorb_g1_tensor(tr, "com_A", com_A);
    Fr_t alpha = fs_challenge_fr(tr);
    auto u = fs_challenge_vec(tr, logD);

    // (2) lookup rounds: anchor RECOMPUTED, then Lagrange-4 chain
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
    // (3) B_f, T_f recomputed from the PUBLIC table, folded with the phase2
    // round challenges in round order (binds current MSB each time; §9)
    Fr_t B_f, T_f;
    {
        tLookupRange tl(-(int)(sf >> 1), sf);
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
    // (4) terminal identity
    Fr_t inv_ratio = h_scalar({N,0,0,0,0,0,0,0}, inv({D,0,0,0,0,0,0,0}), 2);
    Fr_t t1 = h_scalar(alpha_acc, h_scalar(pf.A_f, h_scalar(pf.S_f, beta, 0), 2), 2);
    Fr_t t2 = h_scalar(inv_ratio, h_scalar(alphasq_acc, h_scalar(B_f, h_scalar(T_f, beta, 0), 2), 2), 2);
    Fr_t t4 = h_scalar(inv_ratio, h_scalar(pf.m_f, B_f, 2), 2);
    Fr_t rhs = h_scalar(h_scalar(h_scalar(t1, t2, 0), pf.A_f, 0), t4, 1);
    if (!fr_eq(cur, rhs)) {
        cout << "REJECT: lookup terminal identity" << endl; return false;
    }
    absorb_fr(tr, "A_f", pf.A_f); absorb_fr(tr, "S_f", pf.S_f); absorb_fr(tr, "m_f", pf.m_f);

    // (5) IPA openings bind the terminals to the commitments
    vector<Fr_t> u_pt(ws.rbegin(), ws.rend());
    vector<Fr_t> u_m(ws.rbegin(), ws.rend() - n1);
    if (!open_verify(com_A, gen, C_pad, Q, u_pt, pf.A_f, obdir + "/ipa_A.bin", tr)) {
        cout << "REJECT: IPA opening of A_f vs com_A" << endl; return false;
    }
    if (!open_verify(com_rem, gen, C_pad, Q, u_pt, pf.S_f, obdir + "/ipa_rem.bin", tr)) {
        cout << "REJECT: IPA opening of S_f vs com_rem" << endl; return false;
    }
    if (!open_verify(com_m, gen, C_pad, Q, u_m, pf.m_f, obdir + "/ipa_m.bin", tr)) {
        cout << "REJECT: IPA opening of m_f vs com_m" << endl; return false;
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

static bool selftest_case(uint B, uint C, uint log_sf) {
    cout << "=== selftest B=" << B << " C=" << C << " log_sf=" << log_sf << " ===" << endl;
    const uint C_pad = 1u << ceilLog2(C);
    string dir = "/tmp/zkob_rescale_selftest";
    mkdir(dir.c_str(), 0755);
    string seed = "selftest:rescale";

    FrTensor X = FrTensor::random_int(B * C, 12);
    Commitment gen = Commitment::random(C_pad);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);

    prove(dir, seed, X, B, C, log_sf, gen, Q, "");
    bool honest = verify(dir, seed, B, C, log_sf, gen, Q);
    cout << "honest verify: " << (honest ? "ACCEPT" : "REJECT(!!)") << endl;

    struct { const char* file; long off; const char* what; } cases[] = {
        {"/lookup.bin", 4 + 32, "round-0 p(1)"},        // 4B header + p(0)
        {"/lookup.bin", -32, "m_f terminal"},           // last Fr in file
        {"/ipa_rem.bin", -32, "ipa_rem a_final"},
        {"/com_A.bin", 24, "com_A point"},
        {"/com_rem.bin", 24, "com_rem point"},          // also breaks affine link
        {"/com_m.bin", 24, "com_m point"},
    };
    bool all_rej = true;
    for (auto& c : cases) {
        tamper_byte(dir + c.file, c.off, +1);
        bool rejected = !verify(dir, seed, B, C, log_sf, gen, Q);
        tamper_byte(dir + c.file, c.off, -1);  // restore
        cout << "forgery [" << c.what << "] rejected: " << (rejected ? "YES" : "NO(!!)") << endl;
        all_rej = all_rej && rejected;
    }
    bool restored = verify(dir, seed, B, C, log_sf, gen, Q);
    cout << "restored verify: " << (restored ? "ACCEPT" : "REJECT(!!)") << endl;

    // SEMANTIC forgery: honest procedure, inconsistent witness (covert channel:
    // X_[i] += 1 hidden by rem[i] -= sf). Affine link holds; lookup must catch.
    string edir = "/tmp/zkob_rescale_evil";
    mkdir(edir.c_str(), 0755);
    prove(edir, seed, X, B, C, log_sf, gen, Q, "", (int)(B * C / 2));
    bool evil_rej = !verify(edir, seed, B, C, log_sf, gen, Q);
    cout << "out-of-range rem (covert +1 on X_) rejected: "
         << (evil_rej ? "YES" : "NO(!!)") << endl;

    bool ok = honest && all_rej && restored && evil_rej;
    cout << (ok ? "CASE PASS" : "CASE FAIL") << endl;
    return ok;
}

int main(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    if (mode == "selftest") {
        bool a = selftest_case(8, 4, 4);    // n1=1, n2=4, m rows=4
        bool b = selftest_case(4, 16, 4);   // n1=2, m as a single row
        bool c = selftest_case(8, 6, 6);    // n1=0 (pure phase2), padded C
        cout << ((a && b && c) ? "ZKOB-RESCALE SELFTEST: ALL PASS"
                               : "ZKOB-RESCALE SELFTEST: FAIL") << endl;
        return (a && b && c) ? 0 : 1;
    }
    if (mode == "prove" && (argc == 10 || argc == 11)) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[5]), C = stoi(argv[6]), log_sf = stoi(argv[7]);
        // int64 input: rescale inputs are pre-rescale matmul outputs (> int32)
        FrTensor X = load_long_tensor(argv[4], B * C);
        Commitment gen(argv[8]), qg(argv[9]);
        prove(obdir, seed, X, B, C, log_sf, gen, qg(0), argc == 11 ? argv[10] : "");
        return 0;
    }
    if (mode == "verify" && argc == 9) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[4]), C = stoi(argv[5]), log_sf = stoi(argv[6]);
        Commitment gen(argv[7]), qg(argv[8]);
        return verify(obdir, seed, B, C, log_sf, gen, qg(0)) ? 0 : 1;
    }
    cerr << "usage: zkob_rescale selftest | prove ... | verify ..." << endl;
    return 2;
}
