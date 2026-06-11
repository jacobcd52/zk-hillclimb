// COORDINATOR-BUILT (do not let submission agents edit).
// Real driver for ONE sumcheck_matmul obligation (Y = X @ W), with serialized
// proofs and a fully witness-free verifier. Algebra pinned in vrf_toy_matmul.cu
// (§8) + vrf_toy_ipa.cu (§10); per-round Fiat-Shamir throughout — challenges
// are derived only AFTER absorbing the messages they bind.
//
// Obligation = sumcheck (claim -> claim_X * claim_W) + THREE IPA openings:
//   claim_W vs the REGISTERED weight commitment   (binds the model)
//   claim_X vs the prover-supplied com_X          (chains to previous layer)
//   claim   vs the prover-supplied com_Y          (chains to next layer)
//
// Commitment layout (harness-defined registration; pinned §8):
//   T (R x C) padded per-dim to R_pad x C_pad, committed with C_pad generators
//   -> R_pad row points. Row dim folds with the row challenges (k_com_me
//   orientation); within-row opens via the IPA with b = ME weights of the
//   column challenges. For W: rows=IN (u_input), cols=OUT (u_output).
//   For X: rows=B (u_batch), cols=IN (u_input). For Y: rows=B, cols=OUT.
//
// FS schedule (one transcript per obligation, seed = run_seed:obligation_id):
//   absorb B, IN, OUT, com_X, com_W, com_Y
//   -> u_batch (ceilLog2 B), u_output (ceilLog2 OUT)
//   absorb claim
//   per round r: absorb p(0),p(1),p(2) -> x_r        (u_input[i] = x_{L-1-i})
//   absorb claim_X, claim_W
//   IPA(W), IPA(X), IPA(Y) on the same transcript (L,R absorbed before each x)
//
// Files in <obdir>: dims.bin, com_X.bin, com_Y.bin, sumcheck.bin,
//                   ipa_W.bin, ipa_X.bin, ipa_Y.bin
// com_W comes from the separate REGISTERED file (verifier absorbs that copy;
// a prover whose W doesn't match the registration diverges the transcript).
//
// Usage:
//   zkob_fc commit  <W-int.bin> <IN> <OUT> <gen_out.bin> <com_out.bin>
//   zkob_fc prove   <obdir> <seed> <X-int.bin> <W-int.bin> <B> <IN> <OUT>
//                   <gen_in.bin> <gen_out.bin> <q.bin> [Y-int-out.bin]
//                   [--claims <accdir> <obid> <registered-com_W-path>]
//   zkob_fc verify  <obdir> <seed> <B> <IN> <OUT> <com_W.bin>
//                   <gen_in.bin> <gen_out.bin> <q.bin>
//                   [--claims <vaccdir> <obid>]
//   zkob_fc selftest
//
// CLAIM MODE (Stage A of the transport rebuild, flag-selected; the old
// inline-IPA tail below stays compilable and is the default until Stage C):
// with --claims, prove EMITS its three terminal claims
//   W vs the REGISTERED commitment (point u_output++u_input)   [comref =
//     the registered file path — the F5/F6 pin: the *.commitment_opening
//     manifest id is discharged by this claim inside the batch]
//   X vs com_X (point u_input++u_batch)
//   Y vs com_Y (point u_output++u_batch)
// into <accdir>/claims.bin at the exact program points of the old
// ipa_prove calls, plus its final transcript state (drvstate). Verify in
// claim mode runs every absorb/round/terminal check unchanged, then
// RECOMPUTES the three claims from its own transcript + sumcheck.bin and
// emits them into the VERIFIER's accumulator (<vaccdir>) — the orchestrator
// byte-compares the two lists (opening_batch.claims_match). The driver
// verdict is ACCEPT-conditional; only the batch makes it final.
#include "vrf_common.cuh"
#include "zkob_claims.cuh"
#include "zkfc.cuh"
#include <iostream>
#include <sys/stat.h>
using namespace std;

// upstream zkip kernels (zkfc.cu; extern "C" via KERNEL, not in its header)
KERNEL void zkip_poly_kernel(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0,
                             GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint N_in, uint N_out);
KERNEL void zkip_reduce_kernel(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *new_a,
                               GLOBAL Fr_t *new_b, Fr_t v, uint N_in, uint N_out);

static Fr_t dev_sum_pow2(const Fr_t* d_src, uint n) {  // n pow2; src untouched
    Fr_t *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(Fr_t));
    cudaMalloc(&d_y, (n / 2 + 1) * sizeof(Fr_t));
    cudaMemcpy(d_x, d_src, n * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    uint sz = n;
    while (sz > 1) {
        uint half = sz / 2;
        k_fr_add_pairs<<<(half + 63) / 64, 64>>>(d_x, d_y, half);
        cudaDeviceSynchronize();
        std::swap(d_x, d_y); sz = half;
    }
    Fr_t out; cudaMemcpy(&out, d_x, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y);
    return out;
}

struct SumcheckProof {
    vector<Fr_t> ev;     // 3 per round: p(0), p(1), p(2)
    Fr_t claim, claim_X, claim_W;
};
static void write_sumcheck(const string& path, const SumcheckProof& p) {
    FILE* f = open_or_die(path, "wb");
    write_pod_vec(f, p.ev);
    fwrite(&p.claim, sizeof(Fr_t), 1, f);
    fwrite(&p.claim_X, sizeof(Fr_t), 1, f);
    fwrite(&p.claim_W, sizeof(Fr_t), 1, f);
    fclose(f);
}
static SumcheckProof read_sumcheck(const string& path) {
    FILE* f = open_or_die(path, "rb");
    SumcheckProof p;
    p.ev = read_pod_vec<Fr_t>(f);
    if (fread(&p.claim, sizeof(Fr_t), 1, f) != 1 ||
        fread(&p.claim_X, sizeof(Fr_t), 1, f) != 1 ||
        fread(&p.claim_W, sizeof(Fr_t), 1, f) != 1)
        throw runtime_error("read_sumcheck: terminals");
    fclose(f);
    return p;
}

static FrTensor load_int_tensor(const string& path, uint expect) {
    FILE* f = open_or_die(path, "rb");
    vector<int> buf(expect);
    if (fread(buf.data(), sizeof(int), expect, f) != expect)
        throw runtime_error("short read / wrong dims: " + path);
    fclose(f);
    return FrTensor(expect, buf.data());
}

// ---------------- prove ----------------
static void prove(const string& obdir, const string& seed,
                  const FrTensor& X, const FrTensor& W,
                  uint B, uint IN, uint OUT,
                  const Commitment& gen_in, const Commitment& gen_out,
                  const G1Jacobian_t& Q, const string& y_out_path,
                  const string& accdir = "", const string& obid = "",
                  const string& comW_path = "") {
    const bool claim_mode = !accdir.empty();
    const uint B_pad = 1u << ceilLog2(B), IN_pad = 1u << ceilLog2(IN),
               OUT_pad = 1u << ceilLog2(OUT);
    if (gen_in.size != IN_pad || gen_out.size != OUT_pad)
        throw runtime_error("generator sizes don't match padded dims");

    zkFC fc(IN, OUT, W);
    FrTensor Y = fc(X);
    // int64 on disk: Y values are pre-rescale products (up to ~2^42 at sf=2^16)
    if (!y_out_path.empty()) Y.save_long(y_out_path);

    FrTensor X_padded = X.pad({B, IN});      // B_pad x IN_pad
    FrTensor W_padded = W.pad({IN, OUT});    // IN_pad x OUT_pad
    FrTensor Y_padded = Y.pad({B, OUT});     // B_pad x OUT_pad
    G1TensorJacobian com_X = gen_in.commit(X_padded);
    G1TensorJacobian com_W = gen_out.commit(W_padded);
    G1TensorJacobian com_Y = gen_out.commit(Y_padded);
    com_X.save(obdir + "/com_X.bin");
    com_Y.save(obdir + "/com_Y.bin");

    // dims file (cross-checked by verifier against manifest args)
    { FILE* f = open_or_die(obdir + "/dims.bin", "wb");
      uint32_t d[3] = {B, IN, OUT}; fwrite(d, sizeof(uint32_t), 3, f); fclose(f); }

    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "IN", IN); absorb_u32(tr, "OUT", OUT);
    absorb_g1_tensor(tr, "com_X", com_X);
    absorb_g1_tensor(tr, "com_W", com_W);
    absorb_g1_tensor(tr, "com_Y", com_Y);
    auto u_batch = fs_challenge_vec(tr, ceilLog2(B));
    auto u_output = fs_challenge_vec(tr, ceilLog2(OUT));

    Fr_t claim = Y.multi_dim_me({u_batch, u_output}, {B, OUT});
    absorb_fr(tr, "claim", claim);

    // ---- round-at-a-time sumcheck (per-round FS; see §10) ----
    FrTensor X_red = X.partial_me(u_batch, B, IN);     // size IN
    FrTensor W_red = W.partial_me(u_output, OUT, 1);   // size IN
    const uint L = ceilLog2(IN);
    uint cap = 1u << L;
    Fr_t *d_a, *d_b, *d_a2, *d_b2, *d_o0, *d_o1, *d_o2;
    cudaMalloc(&d_a, cap * sizeof(Fr_t));  cudaMalloc(&d_b, cap * sizeof(Fr_t));
    cudaMalloc(&d_a2, cap * sizeof(Fr_t)); cudaMalloc(&d_b2, cap * sizeof(Fr_t));
    cudaMalloc(&d_o0, cap * sizeof(Fr_t)); cudaMalloc(&d_o1, cap * sizeof(Fr_t));
    cudaMalloc(&d_o2, cap * sizeof(Fr_t));
    cudaMemcpy(d_a, X_red.gpu_data, IN * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_b, W_red.gpu_data, IN * sizeof(Fr_t), cudaMemcpyDeviceToDevice);

    SumcheckProof sc; sc.claim = claim;
    vector<Fr_t> xs;             // round challenges, in round order
    Fr_t cur = claim;
    uint sz = IN;
    for (uint r = 0; r < L; r++) {
        uint N_out = (1u << ceilLog2(sz)) >> 1;
        zkip_poly_kernel<<<(N_out + 63) / 64, 64>>>(d_a, d_b, d_o0, d_o1, d_o2, sz, N_out);
        cudaDeviceSynchronize();
        Fr_t c0 = dev_sum_pow2(d_o0, N_out);
        Fr_t c1 = dev_sum_pow2(d_o1, N_out);
        Fr_t c2 = dev_sum_pow2(d_o2, N_out);
        Fr_t p0 = c0;
        Fr_t p1 = h_scalar(h_scalar(c0, c1, 0), c2, 0);
        Fr_t p2 = h_scalar(h_scalar(c0, h_scalar(F_TWO, c1, 2), 0),
                           h_scalar(h_scalar(F_TWO, F_TWO, 2), c2, 2), 0);
        if (!fr_eq(cur, h_scalar(p0, p1, 0)))
            throw runtime_error("prover sumcheck round inconsistency (witness bug)");
        absorb_fr(tr, "p0", p0); absorb_fr(tr, "p1", p1); absorb_fr(tr, "p2", p2);
        Fr_t x = fs_challenge_fr(tr);
        xs.push_back(x);
        sc.ev.push_back(p0); sc.ev.push_back(p1); sc.ev.push_back(p2);
        cur = lagrange3(p0, p1, p2, x);
        zkip_reduce_kernel<<<(N_out + 63) / 64, 64>>>(d_a, d_b, d_a2, d_b2, x, sz, N_out);
        cudaDeviceSynchronize();
        std::swap(d_a, d_a2); std::swap(d_b, d_b2);
        sz = N_out;
    }
    // u_input as upstream consumes it: round r used u_input[L-1-r]
    vector<Fr_t> u_input(L);
    for (uint i = 0; i < L; i++) u_input[i] = xs[L - 1 - i];

    Fr_t claim_X = X.multi_dim_me({u_batch, u_input}, {B, IN});
    Fr_t claim_W = W.multi_dim_me({u_input, u_output}, {IN, OUT});
    Fr_t a_fin, b_fin;
    cudaMemcpy(&a_fin, d_a, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b_fin, d_b, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a2); cudaFree(d_b2);
    cudaFree(d_o0); cudaFree(d_o1); cudaFree(d_o2);
    if (!fr_eq(a_fin, claim_X) || !fr_eq(b_fin, claim_W))
        throw runtime_error("folded terminals != multi_dim_me claims (convention bug)");
    if (!fr_eq(cur, h_scalar(claim_X, claim_W, 2)))
        throw runtime_error("terminal product mismatch (witness bug)");
    sc.claim_X = claim_X; sc.claim_W = claim_W;
    absorb_fr(tr, "claim_X", claim_X); absorb_fr(tr, "claim_W", claim_W);
    write_sumcheck(obdir + "/sumcheck.bin", sc);

    if (claim_mode) {
        // ---- claim emission (replaces the inline IPAs; same program point,
        // same order W, X, Y; the driver transcript simply ends here) ----
        auto mk = [&](const string& tensor, const string& comref, uint32_t domain,
                      uint32_t n_rows, const vector<Fr_t>& u_col,
                      const vector<Fr_t>& u_row, const Fr_t& eval) {
            BoClaim c;
            c.id = obid + ":" + tensor;
            c.comref = comref;
            c.domain = domain; c.n_rows = n_rows;
            c.point = u_col;
            c.point.insert(c.point.end(), u_row.begin(), u_row.end());
            c.eval = eval;
            claim_emit(accdir, c);
        };
        string witW = accdir + "/wit_" + obid + "_W.fr";
        string witX = accdir + "/wit_" + obid + "_X.fr";
        string witY = accdir + "/wit_" + obid + "_Y.fr";
        W_padded.save(witW); X_padded.save(witX); Y_padded.save(witY);
        mk("W", comW_path, OUT_pad, IN_pad, u_output, u_input, claim_W);
        witref_emit(accdir, comW_path, witW);
        mk("X", obdir + "/com_X.bin", IN_pad, B_pad, u_input, u_batch, claim_X);
        witref_emit(accdir, obdir + "/com_X.bin", witX);
        mk("Y", obdir + "/com_Y.bin", OUT_pad, B_pad, u_output, u_batch, claim);
        witref_emit(accdir, obdir + "/com_Y.bin", witY);
        drvstate_emit(accdir, obid, tr);
        cout << "PROVED matmul obligation (claim mode, 3 claims emitted) -> " << obdir << endl;
        return;
    }

    // ---- OLD TAIL: inline IPA openings (same transcript; order W, X, Y) ----
    FrTensor aW = W_padded.partial_me(u_input, OUT_pad);    // fold IN_pad rows
    auto bW = me_weights(u_output);
    write_ipa(obdir + "/ipa_W.bin", ipa_prove(aW.gpu_data, bW, gen_out.gpu_data, Q, OUT_pad, tr));

    FrTensor aX = X_padded.partial_me(u_batch, IN_pad);     // fold B_pad rows
    auto bX = me_weights(u_input);
    write_ipa(obdir + "/ipa_X.bin", ipa_prove(aX.gpu_data, bX, gen_in.gpu_data, Q, IN_pad, tr));

    FrTensor aY = Y_padded.partial_me(u_batch, OUT_pad);
    write_ipa(obdir + "/ipa_Y.bin", ipa_prove(aY.gpu_data, bW, gen_out.gpu_data, Q, OUT_pad, tr));

    cout << "PROVED matmul obligation -> " << obdir << endl;
}

// ---------------- verify (witness-free) ----------------
static bool verify(const string& obdir, const string& seed,
                   uint B, uint IN, uint OUT, const string& com_W_path,
                   const Commitment& gen_in, const Commitment& gen_out,
                   const G1Jacobian_t& Q,
                   const string& vaccdir = "", const string& obid = "") {
    const bool claim_mode = !vaccdir.empty();
    BoTimer prof("fc_verify");
    const uint B_pad = 1u << ceilLog2(B), IN_pad = 1u << ceilLog2(IN),
               OUT_pad = 1u << ceilLog2(OUT);
    if (gen_in.size != IN_pad || gen_out.size != OUT_pad) {
        cout << "REJECT: generator sizes don't match dims" << endl; return false;
    }
    // dims cross-check
    { FILE* f = open_or_die(obdir + "/dims.bin", "rb");
      uint32_t d[3]; if (fread(d, sizeof(uint32_t), 3, f) != 3) { fclose(f); return false; }
      fclose(f);
      if (d[0] != B || d[1] != IN || d[2] != OUT) {
          cout << "REJECT: dims.bin mismatch" << endl; return false; } }

    G1TensorJacobian com_X(obdir + "/com_X.bin");
    G1TensorJacobian com_W(com_W_path);             // REGISTERED commitment
    G1TensorJacobian com_Y(obdir + "/com_Y.bin");
    if (com_X.size != B_pad || com_W.size != IN_pad || com_Y.size != B_pad) {
        cout << "REJECT: commitment row counts" << endl; return false;
    }
    SumcheckProof sc = read_sumcheck(obdir + "/sumcheck.bin");
    const uint L = ceilLog2(IN);
    if (sc.ev.size() != 3 * L) { cout << "REJECT: round count" << endl; return false; }

    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "IN", IN); absorb_u32(tr, "OUT", OUT);
    absorb_g1_tensor(tr, "com_X", com_X);
    absorb_g1_tensor(tr, "com_W", com_W);
    absorb_g1_tensor(tr, "com_Y", com_Y);
    auto u_batch = fs_challenge_vec(tr, ceilLog2(B));
    auto u_output = fs_challenge_vec(tr, ceilLog2(OUT));
    absorb_fr(tr, "claim", sc.claim);
    prof.lap("absorb");

    Fr_t cur = sc.claim;
    vector<Fr_t> xs;
    for (uint r = 0; r < L; r++) {
        Fr_t p0 = sc.ev[3*r], p1 = sc.ev[3*r+1], p2 = sc.ev[3*r+2];
        if (!fr_eq(cur, h_scalar(p0, p1, 0))) {
            cout << "REJECT: sumcheck round " << r << " p(0)+p(1) != claim" << endl;
            return false;
        }
        absorb_fr(tr, "p0", p0); absorb_fr(tr, "p1", p1); absorb_fr(tr, "p2", p2);
        Fr_t x = fs_challenge_fr(tr);
        xs.push_back(x);
        cur = lagrange3(p0, p1, p2, x);
    }
    vector<Fr_t> u_input(L);
    for (uint i = 0; i < L; i++) u_input[i] = xs[L - 1 - i];
    if (!fr_eq(cur, h_scalar(sc.claim_X, sc.claim_W, 2))) {
        cout << "REJECT: terminal claim != claim_X * claim_W" << endl; return false;
    }
    absorb_fr(tr, "claim_X", sc.claim_X); absorb_fr(tr, "claim_W", sc.claim_W);
    prof.lap("rounds");

    if (claim_mode) {
        // ---- claim recomputation (the verifier-side accumulator entry; the
        // orchestrator byte-compares this against the prover's claims.bin).
        // Points come from THIS verify's own FS replay; evals are the
        // FS-bound terminals it just checked. The W claim carries the
        // REGISTERED com path as comref (F5/F6 discharge pin).
        auto mk = [&](const string& tensor, const string& comref, uint32_t domain,
                      uint32_t n_rows, const vector<Fr_t>& u_col,
                      const vector<Fr_t>& u_row, const Fr_t& eval) {
            BoClaim c;
            c.id = obid + ":" + tensor;
            c.comref = comref;
            c.domain = domain; c.n_rows = n_rows;
            c.point = u_col;
            c.point.insert(c.point.end(), u_row.begin(), u_row.end());
            c.eval = eval;
            claim_emit(vaccdir, c);
        };
        mk("W", com_W_path, OUT_pad, IN_pad, u_output, u_input, sc.claim_W);
        mk("X", obdir + "/com_X.bin", IN_pad, B_pad, u_input, u_batch, sc.claim_X);
        mk("Y", obdir + "/com_Y.bin", OUT_pad, B_pad, u_output, u_batch, sc.claim);
        drvstate_emit(vaccdir, obid, tr);
        prof.lap("claim_emit");
        cout << "ACCEPT-conditional (3 claims emitted; final verdict gated on opening_batch)" << endl;
        return true;
    }

    // ---- OLD TAIL: inline IPA verifies ----
    // IPA W: rows IN_pad folded by u_input, b from u_output, eval claim_W
    {
        G1Jacobian_t C0 = fold_chain(com_W.gpu_data, IN_pad, u_input, 0);
        G1Jacobian_t P0 = h_add(C0, h_mul(Q, sc.claim_W));
        if (!ipa_verify(gen_out.gpu_data, OUT_pad, Q, P0, u_output,
                        read_ipa(obdir + "/ipa_W.bin"), tr)) {
            cout << "REJECT: IPA opening of claim_W vs registered weight commitment" << endl;
            return false;
        }
    }
    // IPA X: rows B_pad folded by u_batch, b from u_input, eval claim_X
    {
        G1Jacobian_t C0 = fold_chain(com_X.gpu_data, B_pad, u_batch, 0);
        G1Jacobian_t P0 = h_add(C0, h_mul(Q, sc.claim_X));
        if (!ipa_verify(gen_in.gpu_data, IN_pad, Q, P0, u_input,
                        read_ipa(obdir + "/ipa_X.bin"), tr)) {
            cout << "REJECT: IPA opening of claim_X vs com_X" << endl;
            return false;
        }
    }
    // IPA Y: rows B_pad folded by u_batch, b from u_output, eval claim
    {
        G1Jacobian_t C0 = fold_chain(com_Y.gpu_data, B_pad, u_batch, 0);
        G1Jacobian_t P0 = h_add(C0, h_mul(Q, sc.claim));
        if (!ipa_verify(gen_out.gpu_data, OUT_pad, Q, P0, u_output,
                        read_ipa(obdir + "/ipa_Y.bin"), tr)) {
            cout << "REJECT: IPA opening of claim vs com_Y" << endl;
            return false;
        }
    }
    prof.lap("ipa_tail");
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

static bool selftest_case(uint B, uint IN, uint OUT) {
    cout << "=== selftest B=" << B << " IN=" << IN << " OUT=" << OUT << " ===" << endl;
    const uint IN_pad = 1u << ceilLog2(IN), OUT_pad = 1u << ceilLog2(OUT);
    string dir = "/tmp/zkob_fc_selftest";
    mkdir(dir.c_str(), 0755);
    string seed = "selftest:matmul";

    FrTensor X = FrTensor::random_int(B * IN, 8);
    FrTensor W = FrTensor::random_int(IN * OUT, 8);
    Commitment gen_in = Commitment::random(IN_pad);
    Commitment gen_out = Commitment::random(OUT_pad);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);

    // registered weight commitment
    G1TensorJacobian com_W = gen_out.commit(W.pad({IN, OUT}));
    com_W.save(dir + "/com_W.bin");

    prove(dir, seed, X, W, B, IN, OUT, gen_in, gen_out, Q, "");
    bool honest = verify(dir, seed, B, IN, OUT, dir + "/com_W.bin", gen_in, gen_out, Q);
    cout << "honest verify: " << (honest ? "ACCEPT" : "REJECT(!!)") << endl;

    // forgeries: tampering any artifact must reject
    struct { const char* file; long off; const char* what; } cases[] = {
        {"/sumcheck.bin", 4 + 32, "round-0 p(1)"},                  // 4B header + p0
        {"/sumcheck.bin", -32, "claim_W terminal"},                 // last Fr in file
        {"/ipa_W.bin", -32, "ipa_W a_final"},
        {"/ipa_X.bin", 8 + 16, "ipa_X round-0 L point"},            // 2 hdrs + into L[0]
        {"/com_Y.bin", 24, "com_Y point"},
        {"/com_W.bin", 24, "registered com_W"},
    };
    bool all_rej = true;
    for (auto& c : cases) {
        tamper_byte(dir + c.file, c.off, +1);
        bool rejected = !verify(dir, seed, B, IN, OUT, dir + "/com_W.bin", gen_in, gen_out, Q);
        tamper_byte(dir + c.file, c.off, -1);  // restore
        cout << "forgery [" << c.what << "] rejected: " << (rejected ? "YES" : "NO(!!)") << endl;
        all_rej = all_rej && rejected;
    }
    // sanity: file restored -> verifies again
    bool restored = verify(dir, seed, B, IN, OUT, dir + "/com_W.bin", gen_in, gen_out, Q);
    cout << "restored verify: " << (restored ? "ACCEPT" : "REJECT(!!)") << endl;

    bool ok = honest && all_rej && restored;
    cout << (ok ? "CASE PASS" : "CASE FAIL") << endl;
    return ok;
}

// claim-mode selftest: honest ACCEPT goes conditional-ACCEPT + batch ACCEPT;
// every forgery still rejects, now at a NAMED locus (driver-side checks where
// the driver still catches it; the batch where the opening layer used to).
static bool selftest_case_claims(uint B, uint IN, uint OUT) {
    cout << "=== selftest (claim mode) B=" << B << " IN=" << IN << " OUT=" << OUT
         << " ===" << endl;
    const uint IN_pad = 1u << ceilLog2(IN), OUT_pad = 1u << ceilLog2(OUT);
    string dir = "/tmp/zkob_fc_selftest_cm";
    { string c = "rm -rf " + dir; system(c.c_str()); }
    mkdir(dir.c_str(), 0755);
    string acc = dir + "/acc";
    mkdir(acc.c_str(), 0755);
    string run_seed = "selftest";              // batch seed = selftest:opening_batch
    string seed = "selftest:matmul";           // obligation transcript seed
    string obid = "selftest.matmul";

    FrTensor X = FrTensor::random_int(B * IN, 8);
    FrTensor W = FrTensor::random_int(IN * OUT, 8);
    // ONE generator vector per domain SIZE (the registration invariant the
    // per-domain RLC grouping relies on; gen_in == gen_out when sizes match)
    map<uint32_t, Commitment*> genobj;
    map<uint32_t, string> genpaths;
    for (uint32_t G : {IN_pad, OUT_pad})
        if (!genobj.count(G)) {
            genobj[G] = new Commitment(Commitment::random(G));
            genpaths[G] = dir + "/gen" + to_string(G) + ".bin";
            genobj[G]->save(genpaths[G]);
        }
    Commitment& gen_in = *genobj[IN_pad];
    Commitment& gen_out = *genobj[OUT_pad];
    string qpath = dir + "/q.bin";
    Commitment::random(2).save(qpath);         // 2-slot q (Q + H slot, §4.4)
    Commitment qg(qpath);
    G1Jacobian_t Q = qg(0);

    string comW_path = dir + "/com_W.bin";
    gen_out.commit(W.pad({IN, OUT})).save(comW_path);

    prove(dir, seed, X, W, B, IN, OUT, gen_in, gen_out, Q, "", acc, obid, comW_path);

    // pipeline: fc verify (fresh verifier accumulator) -> batch verify
    int vacc_n = 0;
    auto pipeline = [&](string& locus) -> bool {
        string vacc = dir + "/vacc" + to_string(vacc_n++);
        mkdir(vacc.c_str(), 0755);
        if (!verify(dir, seed, B, IN, OUT, comW_path, gen_in, gen_out, Q, vacc, obid)) {
            locus = "driver"; return false;
        }
        return batch_verify(acc, vacc, run_seed, genpaths, qpath, &locus);
    };

    int total = 0, fail = 0;
    auto expect = [&](const string& what, const string& want) {
        string locus;
        bool acc_ok = pipeline(locus);
        bool ok = (want == "accept") ? acc_ok : (!acc_ok && locus == want);
        total++; if (!ok) fail++;
        cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << what
             << " -> expected " << want << ", got " << (acc_ok ? "accept" : locus) << endl;
    };

    batch_prove(acc, run_seed, genpaths, qpath);
    expect("honest (conditional ACCEPT + batch ACCEPT)", "accept");

    // forgeries that the DRIVER still catches (unchanged loci)
    tamper_byte(dir + "/sumcheck.bin", 4 + 32, +1);
    expect("sumcheck round-0 p(1) tamper (driver round check)", "driver");
    tamper_byte(dir + "/sumcheck.bin", 4 + 32, -1);
    tamper_byte(dir + "/sumcheck.bin", -32, +1);
    expect("claim_W terminal tamper (driver terminal product)", "driver");
    tamper_byte(dir + "/sumcheck.bin", -32, -1);
    tamper_byte(dir + "/com_Y.bin", 24, +1);
    expect("com_Y tamper (driver transcript divergence)", "driver");
    tamper_byte(dir + "/com_Y.bin", 24, -1);
    tamper_byte(comW_path, 24, +1);
    expect("registered com_W tamper (driver transcript divergence)", "driver");
    tamper_byte(comW_path, 24, -1);

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
    tamper_byte(acc + "/ipa_batch_" + to_string(OUT_pad) + ".bin", -32, +1);
    expect("batched IPA a_final tamper", "ipa" + to_string(OUT_pad));
    tamper_byte(acc + "/ipa_batch_" + to_string(OUT_pad) + ".bin", -32, -1);

    expect("restored", "accept");

    for (auto& g : genobj) delete g.second;
    bool ok = fail == 0;
    cout << (ok ? "CASE PASS" : "CASE FAIL") << " (claim mode, " << (total - fail)
         << "/" << total << ")" << endl;
    return ok;
}

int main(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    // strip the optional claim-mode flag block: --claims <args...>
    int base_argc = argc;
    string cm_a, cm_b, cm_c;
    for (int i = 2; i < argc; i++)
        if (string(argv[i]) == "--claims") {
            base_argc = i;
            if (i + 1 < argc) cm_a = argv[i + 1];
            if (i + 2 < argc) cm_b = argv[i + 2];
            if (i + 3 < argc) cm_c = argv[i + 3];
            break;
        }
    if (mode == "selftest") {
        bo_probe_kernels();
        cout << "kernel -dlto probes: PASS" << endl;
        bool a = selftest_case(4, 6, 3);
        bool b = selftest_case(8, 8, 8);
        bool c = selftest_case(16, 12, 5);
        bool d = selftest_case_claims(4, 6, 3);
        bool e = selftest_case_claims(8, 8, 8);    // IN_pad == OUT_pad: shared gen
        bool f = selftest_case_claims(16, 12, 5);  // two domains in one batch
        bool ok = a && b && c && d && e && f;
        cout << (ok ? "ZKOB-FC SELFTEST: ALL PASS" : "ZKOB-FC SELFTEST: FAIL") << endl;
        return ok ? 0 : 1;
    }
    if (mode == "commit" && base_argc == 7) {
        uint IN = stoi(argv[3]), OUT = stoi(argv[4]);
        FrTensor W = load_int_tensor(argv[2], IN * OUT);
        Commitment gen_out(argv[5]);
        gen_out.commit(W.pad({IN, OUT})).save(argv[6]);
        cout << "registered commitment -> " << argv[6] << endl;
        return 0;
    }
    if (mode == "prove" && (base_argc == 12 || base_argc == 13)) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[6]), IN = stoi(argv[7]), OUT = stoi(argv[8]);
        FrTensor X = load_int_tensor(argv[4], B * IN);
        FrTensor W = load_int_tensor(argv[5], IN * OUT);
        Commitment gen_in(argv[9]), gen_out(argv[10]), qg(argv[11]);
        if (!cm_a.empty() && (cm_b.empty() || cm_c.empty()))
            throw runtime_error("prove --claims needs <accdir> <obid> <registered-com_W>");
        prove(obdir, seed, X, W, B, IN, OUT, gen_in, gen_out, qg(0),
              base_argc == 13 ? argv[12] : "", cm_a, cm_b, cm_c);
        return 0;
    }
    if (mode == "verify" && base_argc == 11) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[4]), IN = stoi(argv[5]), OUT = stoi(argv[6]);
        Commitment gen_in(argv[8]), gen_out(argv[9]), qg(argv[10]);
        if (!cm_a.empty() && cm_b.empty())
            throw runtime_error("verify --claims needs <vaccdir> <obid>");
        return verify(obdir, seed, B, IN, OUT, argv[7], gen_in, gen_out, qg(0),
                      cm_a, cm_b) ? 0 : 1;
    }
    cerr << "usage: zkob_fc selftest | commit ... | prove ... [--claims <accdir> <obid> <com_W>] | verify ... [--claims <vaccdir> <obid>]" << endl;
    return 2;
}
