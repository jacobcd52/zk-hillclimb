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
// WPRIV MODE (Stage D weight privacy, flag-selected on top of claim mode):
// with --wpriv <waccdir> <W-blinds.bin>, the registered com_W is the HIDING
// Pedersen registration (D1: rows + s_r*H; the prover RECOMPUTES it from
// (W, blinds) so a tampered registration file still diverges the transcript),
// the zkip sumcheck switches to COMMITTED ROUND MESSAGES (D2/F7: C_p0/1/2
// absorbed, homomorphic p(0)+p(1)==cur checks, lagrange3_g1 fold — the round
// evals, which are weight functionals per §4.1, never appear in plaintext),
// claim_W is shipped ONLY as C_W = claim_W*Q + t_W*H, the terminal product
// check becomes a Schnorr PoK of the H-component of C_cur - claim_X*C_W, and
// the W claim enters the WEIGHT accumulator with EvalVar=Committed. X and Y
// stay public claims in the public accumulator (activations are the
// statement). Artifacts: wsc.bin replaces sumcheck.bin (no plaintext round
// evals, no claim_W).
#include "vrf_common.cuh"
#include "zkob_claims.cuh"
#include "zkob_wpriv.cuh"
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

// wpriv sumcheck artifact: committed rounds + public claim/claim_X + C_W +
// the terminal Schnorr. NO plaintext round eval and NO claim_W anywhere.
struct WscProof {
    vector<G1Jacobian_t> cps;   // 3 round commitments per round
    Fr_t claim, claim_X;
    G1Jacobian_t C_W;
    SchnorrH sch;
    G1Jacobian_t C_claim;       // apriv (D5): committed initial claim; hides the
                                // Y MLE eval (= ceval of the committed Y claim)
};
// apriv replaces the public `claim` scalar with the committed C_claim (so the
// output activation eval never appears in plaintext). Non-apriv layout is
// byte-identical to before (the existing wpriv selftest offsets rely on this).
static void write_wsc(const string& path, const WscProof& p, bool apriv = false) {
    FILE* f = open_or_die(path, "wb");
    write_pod_vec(f, p.cps);
    if (apriv) fwrite(&p.C_claim, sizeof(G1Jacobian_t), 1, f);
    else       fwrite(&p.claim, sizeof(Fr_t), 1, f);
    fwrite(&p.claim_X, sizeof(Fr_t), 1, f);
    fwrite(&p.C_W, sizeof(G1Jacobian_t), 1, f);
    fwrite(&p.sch.A, sizeof(G1Jacobian_t), 1, f);
    fwrite(&p.sch.z, sizeof(Fr_t), 1, f);
    fclose(f);
}
static WscProof read_wsc(const string& path, bool apriv = false) {
    FILE* f = open_or_die(path, "rb");
    WscProof p;
    p.cps = read_pod_vec<G1Jacobian_t>(f);
    bool head = apriv ? (fread(&p.C_claim, sizeof(G1Jacobian_t), 1, f) == 1)
                      : (fread(&p.claim, sizeof(Fr_t), 1, f) == 1);
    if (!head ||
        fread(&p.claim_X, sizeof(Fr_t), 1, f) != 1 ||
        fread(&p.C_W, sizeof(G1Jacobian_t), 1, f) != 1 ||
        fread(&p.sch.A, sizeof(G1Jacobian_t), 1, f) != 1 ||
        fread(&p.sch.z, sizeof(Fr_t), 1, f) != 1)
        throw runtime_error("read_wsc: body");
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
                  const string& comW_path = "",
                  const string& waccdir = "", const string& wblindpath = "",
                  const G1Jacobian_t* Hp = nullptr, bool apriv = false) {
    const bool claim_mode = !accdir.empty();
    const bool wpriv = !waccdir.empty();
    if (wpriv && !claim_mode)
        throw runtime_error("wpriv requires claim mode");
    if (wpriv && (wblindpath.empty() || !Hp))
        throw runtime_error("wpriv needs the registration blinds path and H");
    if (apriv && !wpriv)
        throw runtime_error("apriv (activation privacy) builds on wpriv");
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
    if (wpriv) {
        // D1: the registered commitment is hiding; the prover RECOMPUTES it
        // from (W, blinds) — a substituted/tampered registration file still
        // diverges the transcript against the verifier's copy
        auto s = wp_blinds_load(wblindpath);
        if (s.size() != IN_pad)
            throw runtime_error("wpriv: blind count != IN_pad");
        wp_hide_rows(com_W, s, *Hp);
    }
    G1TensorJacobian com_Y = gen_out.commit(Y_padded);
    vector<Fr_t> s_Y;
    if (apriv) {
        // D5: hide the OUTPUT activation commitment (fresh per-proof row blinds;
        // com_Y has B_pad row points -> B_pad blinds). Must happen before the
        // commitment is saved/absorbed so the transcript binds the hiding form.
        s_Y.resize(B_pad);
        for (auto& z : s_Y) z = wp_rand();
        wp_hide_rows(com_Y, s_Y, *Hp);
    }
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
    Fr_t t_Y = F_ZERO;
    G1Jacobian_t C_claim;
    if (apriv) {
        // D5: commit the initial claim (= the Y eval). The sumcheck then runs
        // from a hidden running claim; C_claim doubles as the Y claim's ceval.
        t_Y = wp_rand();
        C_claim = ped_qh(claim, t_Y, Q, *Hp);
        absorb_g1(tr, "Cclaim", C_claim);
    } else {
        absorb_fr(tr, "claim", claim);
    }

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
    WscProof wsc; wsc.claim = claim;
    Fr_t tau = apriv ? t_Y : F_ZERO;   // running committed-claim blind (apriv: t_Y)
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
        Fr_t x;
        if (wpriv) {
            // D2/F7: committed round messages; the one blind constraint per
            // round (tau0 + tau1 == tau_cur) is what the verifier's
            // homomorphic C_p0 + C_p1 == C_cur check enforces
            Fr_t tau0 = wp_rand(), tau2 = wp_rand();
            Fr_t tau1 = h_scalar(tau, tau0, 1);
            G1Jacobian_t C0 = ped_qh(p0, tau0, Q, *Hp);
            G1Jacobian_t C1 = ped_qh(p1, tau1, Q, *Hp);
            G1Jacobian_t C2 = ped_qh(p2, tau2, Q, *Hp);
            absorb_g1(tr, "wp0", C0); absorb_g1(tr, "wp1", C1); absorb_g1(tr, "wp2", C2);
            wsc.cps.push_back(C0); wsc.cps.push_back(C1); wsc.cps.push_back(C2);
            x = fs_challenge_fr(tr);
            tau = lagrange3(tau0, tau1, tau2, x);
        } else {
            absorb_fr(tr, "p0", p0); absorb_fr(tr, "p1", p1); absorb_fr(tr, "p2", p2);
            x = fs_challenge_fr(tr);
            sc.ev.push_back(p0); sc.ev.push_back(p1); sc.ev.push_back(p2);
        }
        xs.push_back(x);
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
    if (wpriv) {
        // D2/D3 terminal: claim_W ships ONLY inside C_W; the product check
        // becomes a Schnorr PoK of delta with
        //   C_cur - claim_X*C_W = (tau - claim_X*t_W)*H = delta*H
        Fr_t t_W = wp_rand();
        G1Jacobian_t C_W = ped_qh(claim_W, t_W, Q, *Hp);
        absorb_fr(tr, "claim_X", claim_X);
        absorb_g1(tr, "C_W", C_W);
        wsc.claim_X = claim_X;
        wsc.C_W = C_W;
        wsc.C_claim = C_claim;        // apriv: hidden initial claim (= Y ceval)
        Fr_t delta = h_scalar(tau, h_scalar(claim_X, t_W, 2), 1);
        wsc.sch = schnorr_h_prove(delta, *Hp, tr);
        write_wsc(obdir + "/wsc.bin", wsc, apriv);
        // claim routing: W (Committed) -> weight accumulator with the blind
        // stash; X (plain) -> public accumulator; Y -> public (wpriv) OR
        // Committed into the hiding batch (apriv).
        string witW = waccdir + "/wit_" + obid + "_W.fr";
        string witX = accdir + "/wit_" + obid + "_X.fr";
        string witY = (apriv ? waccdir : accdir) + "/wit_" + obid + "_Y.fr";
        W_padded.save(witW); X_padded.save(witX); Y_padded.save(witY);
        {
            BoClaim c;
            c.id = obid + ":W";
            c.comref = comW_path;
            c.domain = OUT_pad; c.n_rows = IN_pad;
            c.point = u_output;
            c.point.insert(c.point.end(), u_input.begin(), u_input.end());
            c.tag = BO_EVAL_COMMITTED;
            c.ceval = C_W;
            claim_emit(waccdir, c);
            wp_cblind_emit(waccdir, c.id, claim_W, t_W);
            witref_emit(waccdir, comW_path, witW);
            wp_blindref_emit(waccdir, comW_path, wblindpath);
        }
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
        mk("X", obdir + "/com_X.bin", IN_pad, B_pad, u_input, u_batch, claim_X);
        witref_emit(accdir, obdir + "/com_X.bin", witX);
        if (apriv) {
            // D5: Y committed (ceval = C_claim) into the hiding batch; com_Y is
            // hidden, blinds stashed prover-private and blindref'd like W.
            string yblindpath = waccdir + "/comY_" + obid + ".blinds.bin";
            wp_blinds_save(yblindpath, s_Y);
            BoClaim c;
            c.id = obid + ":Y";
            c.comref = obdir + "/com_Y.bin";
            c.domain = OUT_pad; c.n_rows = B_pad;
            c.point = u_output;
            c.point.insert(c.point.end(), u_batch.begin(), u_batch.end());
            c.tag = BO_EVAL_COMMITTED;
            c.ceval = C_claim;
            claim_emit(waccdir, c);
            wp_cblind_emit(waccdir, c.id, claim, t_Y);
            witref_emit(waccdir, obdir + "/com_Y.bin", witY);
            wp_blindref_emit(waccdir, obdir + "/com_Y.bin", yblindpath);
        } else {
            mk("Y", obdir + "/com_Y.bin", OUT_pad, B_pad, u_output, u_batch, claim);
            witref_emit(accdir, obdir + "/com_Y.bin", witY);
        }
        drvstate_emit(accdir, obid, tr);
        drvstate_emit(waccdir, obid, tr);
        cout << "PROVED matmul obligation (" << (apriv ? "apriv: W+Y committed + X public"
                                                       : "wpriv: W committed + 2 public claims")
             << ") -> " << obdir << endl;
        return;
    }
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
                   const string& vaccdir = "", const string& obid = "",
                   const string& wvaccdir = "", const G1Jacobian_t* Hp = nullptr,
                   bool apriv = false) {
    const bool claim_mode = !vaccdir.empty();
    const bool wpriv = !wvaccdir.empty();
    if (wpriv && (!claim_mode || !Hp))
        throw runtime_error("wpriv verify needs claim mode and H");
    if (apriv && !wpriv)
        throw runtime_error("apriv verify builds on wpriv");
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
    SumcheckProof sc;
    WscProof wsc;
    const uint L = ceilLog2(IN);
    if (wpriv) {
        wsc = read_wsc(obdir + "/wsc.bin", apriv);
        if (wsc.cps.size() != 3 * L) { cout << "REJECT: round count" << endl; return false; }
        sc.claim_X = wsc.claim_X;
        if (!apriv) sc.claim = wsc.claim;   // apriv: initial claim is committed
    } else {
        sc = read_sumcheck(obdir + "/sumcheck.bin");
        if (sc.ev.size() != 3 * L) { cout << "REJECT: round count" << endl; return false; }
    }

    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "IN", IN); absorb_u32(tr, "OUT", OUT);
    absorb_g1_tensor(tr, "com_X", com_X);
    absorb_g1_tensor(tr, "com_W", com_W);
    absorb_g1_tensor(tr, "com_Y", com_Y);
    auto u_batch = fs_challenge_vec(tr, ceilLog2(B));
    auto u_output = fs_challenge_vec(tr, ceilLog2(OUT));
    if (apriv) absorb_g1(tr, "Cclaim", wsc.C_claim);   // committed initial claim
    else       absorb_fr(tr, "claim", sc.claim);
    prof.lap("absorb");

    Fr_t cur = sc.claim;
    G1Jacobian_t C_cur;
    if (wpriv) C_cur = apriv ? wsc.C_claim : h_mul(Q, sc.claim);
    vector<Fr_t> xs;
    for (uint r = 0; r < L; r++) {
        Fr_t x;
        if (wpriv) {
            // homomorphic round check over the committed round messages
            const G1Jacobian_t &C0 = wsc.cps[3*r], &C1 = wsc.cps[3*r+1], &C2 = wsc.cps[3*r+2];
            if (!g1_eq(h_add(C0, C1), C_cur)) {
                cout << "REJECT: sumcheck round " << r << " C_p0+C_p1 != C_cur" << endl;
                return false;
            }
            absorb_g1(tr, "wp0", C0); absorb_g1(tr, "wp1", C1); absorb_g1(tr, "wp2", C2);
            x = fs_challenge_fr(tr);
            C_cur = lagrange3_g1(C0, C1, C2, x);
        } else {
            Fr_t p0 = sc.ev[3*r], p1 = sc.ev[3*r+1], p2 = sc.ev[3*r+2];
            if (!fr_eq(cur, h_scalar(p0, p1, 0))) {
                cout << "REJECT: sumcheck round " << r << " p(0)+p(1) != claim" << endl;
                return false;
            }
            absorb_fr(tr, "p0", p0); absorb_fr(tr, "p1", p1); absorb_fr(tr, "p2", p2);
            x = fs_challenge_fr(tr);
            cur = lagrange3(p0, p1, p2, x);
        }
        xs.push_back(x);
    }
    vector<Fr_t> u_input(L);
    for (uint i = 0; i < L; i++) u_input[i] = xs[L - 1 - i];
    if (wpriv) {
        // terminal: PoK of the H-component of C_cur - claim_X*C_W (a prover
        // with cur != claim_X*claim_W would need a Q/H relation -> DLOG)
        absorb_fr(tr, "claim_X", wsc.claim_X);
        absorb_g1(tr, "C_W", wsc.C_W);
        G1Jacobian_t E = h_sub(C_cur, h_mul(wsc.C_W, wsc.claim_X));
        if (!schnorr_h_verify(E, wsc.sch, *Hp, tr)) {
            cout << "REJECT: terminal Schnorr (committed claim_W product)" << endl;
            return false;
        }
    } else {
        if (!fr_eq(cur, h_scalar(sc.claim_X, sc.claim_W, 2))) {
            cout << "REJECT: terminal claim != claim_X * claim_W" << endl; return false;
        }
        absorb_fr(tr, "claim_X", sc.claim_X); absorb_fr(tr, "claim_W", sc.claim_W);
    }
    prof.lap("rounds");

    if (wpriv) {
        // claim recomputation: W (Committed, ceval = the FS-bound C_W) into
        // the verifier's WEIGHT accumulator; X, Y (plain) into the public one
        {
            BoClaim c;
            c.id = obid + ":W";
            c.comref = com_W_path;
            c.domain = OUT_pad; c.n_rows = IN_pad;
            c.point = u_output;
            c.point.insert(c.point.end(), u_input.begin(), u_input.end());
            c.tag = BO_EVAL_COMMITTED;
            c.ceval = wsc.C_W;
            claim_emit(wvaccdir, c);
        }
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
        mk("X", obdir + "/com_X.bin", IN_pad, B_pad, u_input, u_batch, wsc.claim_X);
        if (apriv) {
            // D5: Y recomputed as a Committed claim (ceval = the FS-bound
            // C_claim) into the verifier's WEIGHT/hiding accumulator.
            BoClaim c;
            c.id = obid + ":Y";
            c.comref = obdir + "/com_Y.bin";
            c.domain = OUT_pad; c.n_rows = B_pad;
            c.point = u_output;
            c.point.insert(c.point.end(), u_batch.begin(), u_batch.end());
            c.tag = BO_EVAL_COMMITTED;
            c.ceval = wsc.C_claim;
            claim_emit(wvaccdir, c);
        } else {
            mk("Y", obdir + "/com_Y.bin", OUT_pad, B_pad, u_output, u_batch, wsc.claim);
        }
        drvstate_emit(vaccdir, obid, tr);
        drvstate_emit(wvaccdir, obid, tr);
        prof.lap("claim_emit");
        cout << "ACCEPT-conditional (" << (apriv ? "apriv: 2 committed (W,Y) + X public"
                                                 : "wpriv: 1 committed + 2 public claims")
             << " emitted)" << endl;
        return true;
    }

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

// wpriv selftest: D1-D3 through the REAL driver + both batches; D4 leak
// regression against the actual claim_W (read from the prover-private blind
// stash); every forgery still rejects at a named locus.
static bool selftest_case_wpriv(uint B, uint IN, uint OUT) {
    cout << "=== selftest (wpriv mode) B=" << B << " IN=" << IN << " OUT=" << OUT
         << " ===" << endl;
    const uint IN_pad = 1u << ceilLog2(IN), OUT_pad = 1u << ceilLog2(OUT);
    string dir = "/tmp/zkob_fc_selftest_wp";
    { string c = "rm -rf " + dir; system(c.c_str()); }
    mkdir(dir.c_str(), 0755);
    string acc = dir + "/acc", wacc = dir + "/wacc";
    mkdir(acc.c_str(), 0755); mkdir(wacc.c_str(), 0755);
    string run_seed = "selftest";
    string seed = "selftest:matmul";
    string obid = "selftest.matmul";

    FrTensor X = FrTensor::random_int(B * IN, 8);
    FrTensor W = FrTensor::random_int(IN * OUT, 8);
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
    Commitment::random(2).save(qpath);
    Commitment qg(qpath);
    G1Jacobian_t Q = qg(0), H = qg(1);

    // D1: hiding registration of W (fresh per-row blinds, prover-private)
    string comW_path = dir + "/com_W.bin";
    string blindpath = dir + "/com_W.blinds.bin";
    {
        G1TensorJacobian comW = gen_out.commit(W.pad({IN, OUT}));
        vector<Fr_t> s(IN_pad);
        for (auto& x : s) x = wp_rand();
        wp_hide_rows(comW, s, H);
        comW.save(comW_path);
        wp_blinds_save(blindpath, s);
    }

    prove(dir, seed, X, W, B, IN, OUT, gen_in, gen_out, Q, "", acc, obid,
          comW_path, wacc, blindpath, &H);

    int vacc_n = 0;
    auto pipeline = [&](string& locus) -> bool {
        string vacc = dir + "/vacc" + to_string(vacc_n);
        string wvacc = dir + "/wvacc" + to_string(vacc_n);
        vacc_n++;
        mkdir(vacc.c_str(), 0755); mkdir(wvacc.c_str(), 0755);
        if (!verify(dir, seed, B, IN, OUT, comW_path, gen_in, gen_out, Q,
                    vacc, obid, wvacc, &H)) {
            locus = "driver"; return false;
        }
        if (!batch_verify(acc, vacc, run_seed, genpaths, qpath, &locus))
            return false;
        // wbatch loci are already "w"-prefixed (wclaims_match, wipa<G>, ...)
        if (!wbatch_verify(wacc, wvacc, run_seed, genpaths, qpath, &locus))
            return false;
        return true;
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
    auto check = [&](const string& what, bool ok) {
        total++; if (!ok) fail++;
        cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << what << endl;
    };

    batch_prove(acc, run_seed, genpaths, qpath);
    wbatch_prove(wacc, run_seed, genpaths, qpath);
    expect("honest (driver + public batch + weight batch)", "accept");

    // ---- D4 leak regression: the true claim_W (from the prover-private
    // blind stash) must appear in NO verifier-consumed artifact ----
    {
        Fr_t claim_W = wp_cblinds_load(wacc).at(obid + ":W").first;
        vector<string> artifacts = {
            dir + "/dims.bin", dir + "/com_X.bin", dir + "/com_Y.bin",
            dir + "/wsc.bin", comW_path,
            acc + "/claims.bin", acc + "/drvstates.bin",
            acc + "/batch_sumcheck.bin", acc + "/batch_vfin.bin",
            wacc + "/claims.bin", wacc + "/drvstates.bin",
            wacc + "/wbatch_sumcheck.bin", wacc + "/wbatch_vfin.bin"};
        for (auto& g : genpaths) {
            artifacts.push_back(acc + "/ipa_batch_" + to_string(g.first) + ".bin");
            artifacts.push_back(wacc + "/wipa_batch_" + to_string(g.first) + ".bin");
        }
        auto hits = wp_leak_scan(artifacts, claim_W);
        for (auto& h : hits) cout << "    LEAK: claim_W found in " << h << endl;
        check("D4: claim_W absent from every proof artifact (wpriv)", hits.empty());
        // positive control: the PLAIN claim-mode artifacts (previous selftest
        // case) DO carry their claim_W — validates the scanner cross-artifact
        string cm = "/tmp/zkob_fc_selftest_cm";
        if (bo_file_exists(cm + "/sumcheck.bin")) {
            auto sc_cm = read_sumcheck(cm + "/sumcheck.bin");
            check("D4 positive control: plain-mode claims.bin leaks its claim_W",
                  wp_file_contains(cm + "/acc/claims.bin", sc_cm.claim_W));
        }
    }

    // ---- forgeries: driver-level ----
    long g1sz = (long)sizeof(G1Jacobian_t);
    tamper_byte(dir + "/wsc.bin", 4 + 5, +1);                 // round-0 C_p0
    expect("wsc round-0 C_p0 tamper (homomorphic round check)", "driver");
    tamper_byte(dir + "/wsc.bin", 4 + 5, -1);
    tamper_byte(dir + "/wsc.bin", 4 + g1sz + 5, +1);          // round-0 C_p1
    expect("wsc round-0 C_p1 tamper", "driver");
    tamper_byte(dir + "/wsc.bin", 4 + g1sz + 5, -1);
    tamper_byte(dir + "/wsc.bin", 4 + 2 * g1sz + 5, +1);      // round-0 C_p2
    expect("wsc round-0 C_p2 tamper (fold divergence)", "driver");
    tamper_byte(dir + "/wsc.bin", 4 + 2 * g1sz + 5, -1);
    tamper_byte(dir + "/wsc.bin", -1, +1);                    // Schnorr z
    expect("wsc terminal Schnorr response tamper", "driver");
    tamper_byte(dir + "/wsc.bin", -1, -1);
    {
        // C_W tamper: offset = 4 + 3L*g1 + 2 Fr
        uint Lr = ceilLog2(IN);
        long off = 4 + 3L * Lr * g1sz + 64 + 5;
        tamper_byte(dir + "/wsc.bin", off, +1);
        expect("wsc C_W tamper (terminal Schnorr binds C_W)", "driver");
        tamper_byte(dir + "/wsc.bin", off, -1);
    }
    tamper_byte(comW_path, 24, +1);
    expect("registered (hiding) com_W tamper -> transcript divergence", "driver");
    tamper_byte(comW_path, 24, -1);
    tamper_byte(dir + "/com_Y.bin", 24, +1);
    expect("com_Y tamper (driver transcript divergence)", "driver");
    tamper_byte(dir + "/com_Y.bin", 24, -1);

    // ---- forgeries: batch-level ----
    tamper_byte(acc + "/claims.bin", -1, +1);
    expect("public claims.bin tamper", "claims_match");
    tamper_byte(acc + "/claims.bin", -1, -1);
    tamper_byte(wacc + "/claims.bin", -1, +1);
    expect("weight claims.bin (C_W bytes) tamper", "wclaims_match");
    tamper_byte(wacc + "/claims.bin", -1, -1);
    tamper_byte(wacc + "/wipa_batch_" + to_string(OUT_pad) + ".bin", -1, +1);
    expect("weight ZK-IPA tamper", "wipa" + to_string(OUT_pad));
    tamper_byte(wacc + "/wipa_batch_" + to_string(OUT_pad) + ".bin", -1, -1);
    {
        // registration hash-pin at the WEIGHT batch (comref-hash absorb):
        // tamper the registered com AFTER the driver checks (run wbatch only)
        tamper_byte(comW_path, 24, +1);
        string wl;
        bool acc_w = wbatch_verify(wacc, dir + "/wvacc0", run_seed, genpaths, qpath, &wl);
        // single weight claim: C_cur0 = ceval is rho-INDEPENDENT, so the G0
        // comref-hash divergence first bites at the round-0 challenge ->
        // locus wround1 (multi-claim batches die at wround0; see wselftest)
        check("registered-com hash-pin at the weight batch (post-driver tamper)",
              !acc_w && wl == "wround1");
        tamper_byte(comW_path, 24, -1);
    }
    expect("restored", "accept");

    for (auto& g : genobj) delete g.second;
    bool ok = fail == 0;
    cout << (ok ? "CASE PASS" : "CASE FAIL") << " (wpriv mode, " << (total - fail)
         << "/" << total << ")" << endl;
    return ok;
}

// apriv selftest (D5 activation privacy): the OUTPUT activation Y is hidden on
// top of weight privacy. W and Y are committed claims (hiding batch), X stays
// public. D5 leak regression: the Y MLE eval (claim) AND claim_W must appear in
// NO verifier-consumed artifact. Every forgery still rejects at a named locus.
static bool selftest_case_apriv(uint B, uint IN, uint OUT) {
    cout << "=== selftest (apriv mode) B=" << B << " IN=" << IN << " OUT=" << OUT
         << " ===" << endl;
    const uint IN_pad = 1u << ceilLog2(IN), OUT_pad = 1u << ceilLog2(OUT);
    string dir = "/tmp/zkob_fc_selftest_ap";
    { string c = "rm -rf " + dir; system(c.c_str()); }
    mkdir(dir.c_str(), 0755);
    string acc = dir + "/acc", wacc = dir + "/wacc";
    mkdir(acc.c_str(), 0755); mkdir(wacc.c_str(), 0755);
    string run_seed = "selftest";
    string seed = "selftest:matmul";
    string obid = "selftest.matmul";

    FrTensor X = FrTensor::random_int(B * IN, 8);
    FrTensor W = FrTensor::random_int(IN * OUT, 8);
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
    Commitment::random(2).save(qpath);
    Commitment qg(qpath);
    G1Jacobian_t Q = qg(0), H = qg(1);

    // D1: hiding registration of W (as in the wpriv path)
    string comW_path = dir + "/com_W.bin";
    string blindpath = dir + "/com_W.blinds.bin";
    {
        G1TensorJacobian comW = gen_out.commit(W.pad({IN, OUT}));
        vector<Fr_t> s(IN_pad);
        for (auto& x : s) x = wp_rand();
        wp_hide_rows(comW, s, H);
        comW.save(comW_path);
        wp_blinds_save(blindpath, s);
    }

    prove(dir, seed, X, W, B, IN, OUT, gen_in, gen_out, Q, "", acc, obid,
          comW_path, wacc, blindpath, &H, /*apriv=*/true);

    int vacc_n = 0;
    auto pipeline = [&](string& locus) -> bool {
        string vacc = dir + "/vacc" + to_string(vacc_n);
        string wvacc = dir + "/wvacc" + to_string(vacc_n);
        vacc_n++;
        mkdir(vacc.c_str(), 0755); mkdir(wvacc.c_str(), 0755);
        if (!verify(dir, seed, B, IN, OUT, comW_path, gen_in, gen_out, Q,
                    vacc, obid, wvacc, &H, /*apriv=*/true)) {
            locus = "driver"; return false;
        }
        if (!batch_verify(acc, vacc, run_seed, genpaths, qpath, &locus))
            return false;
        if (!wbatch_verify(wacc, wvacc, run_seed, genpaths, qpath, &locus))
            return false;
        return true;
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
    auto check = [&](const string& what, bool ok) {
        total++; if (!ok) fail++;
        cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << what << endl;
    };

    batch_prove(acc, run_seed, genpaths, qpath);
    wbatch_prove(wacc, run_seed, genpaths, qpath);
    expect("honest (driver + public batch[X] + hiding batch[W,Y])", "accept");

    // ---- D5 leak regression: the Y eval (claim) AND claim_W must be absent ----
    {
        auto cbl = wp_cblinds_load(wacc);
        Fr_t claim_Y = cbl.at(obid + ":Y").first;     // the hidden output eval
        Fr_t claim_W = cbl.at(obid + ":W").first;
        vector<string> artifacts = {
            dir + "/dims.bin", dir + "/com_X.bin", dir + "/com_Y.bin",
            dir + "/wsc.bin", comW_path,
            acc + "/claims.bin", acc + "/drvstates.bin",
            acc + "/batch_sumcheck.bin", acc + "/batch_vfin.bin",
            wacc + "/claims.bin", wacc + "/drvstates.bin",
            wacc + "/wbatch_sumcheck.bin", wacc + "/wbatch_vfin.bin"};
        for (auto& g : genpaths) {
            artifacts.push_back(acc + "/ipa_batch_" + to_string(g.first) + ".bin");
            artifacts.push_back(wacc + "/wipa_batch_" + to_string(g.first) + ".bin");
        }
        auto hitsY = wp_leak_scan(artifacts, claim_Y);
        for (auto& h : hitsY) cout << "    LEAK: claim_Y found in " << h << endl;
        check("D5: Y eval (claim) absent from every proof artifact", hitsY.empty());
        auto hitsW = wp_leak_scan(artifacts, claim_W);
        check("D5: claim_W still absent (weight privacy intact)", hitsW.empty());
    }

    // ---- forgeries ----
    tamper_byte(dir + "/com_Y.bin", 24, +1);
    expect("hidden com_Y tamper (driver transcript divergence)", "driver");
    tamper_byte(dir + "/com_Y.bin", 24, -1);
    tamper_byte(dir + "/wsc.bin", -1, +1);            // terminal Schnorr z (last byte)
    expect("wsc terminal Schnorr response tamper", "driver");
    tamper_byte(dir + "/wsc.bin", -1, -1);
    tamper_byte(wacc + "/claims.bin", -1, +1);        // a committed claim's bytes
    expect("hiding-batch claims.bin tamper (W or Y)", "wclaims_match");
    tamper_byte(wacc + "/claims.bin", -1, -1);
    expect("restored", "accept");

    for (auto& g : genobj) delete g.second;
    bool ok = fail == 0;
    cout << (ok ? "CASE PASS" : "CASE FAIL") << " (apriv mode, " << (total - fail)
         << "/" << total << ")" << endl;
    return ok;
}

#include "zkob_serve.cuh"
static int zkw_run1(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    // strip the optional flag blocks: --claims <a> <b> <c>, --wpriv <a> [<b>],
    // --hiding <q.bin> <blinds_out>
    int base_argc = argc;
    string cm_a, cm_b, cm_c, wp_a, wp_b, hd_a, hd_b;
    bool ap_flag = false;
    for (int i = 2; i < argc; i++) {
        string s = argv[i];
        if (s == "--apriv") {
            if (base_argc == argc) base_argc = i;
            ap_flag = true;
        } else if (s == "--claims") {
            if (base_argc == argc) base_argc = i;
            if (i + 1 < argc) cm_a = argv[i + 1];
            if (i + 2 < argc) cm_b = argv[i + 2];
            if (i + 3 < argc) cm_c = argv[i + 3];
        } else if (s == "--wpriv") {
            if (base_argc == argc) base_argc = i;
            if (i + 1 < argc) wp_a = argv[i + 1];
            if (i + 2 < argc && string(argv[i + 2]).rfind("--", 0) != 0) wp_b = argv[i + 2];
        } else if (s == "--hiding") {
            if (base_argc == argc) base_argc = i;
            if (i + 1 < argc) hd_a = argv[i + 1];
            if (i + 2 < argc) hd_b = argv[i + 2];
        }
    }
    if (mode == "selftest") {
        bo_probe_kernels();
        wpriv_probe();
        cout << "kernel -dlto probes (incl. wpriv shapes): PASS" << endl;
        // Stage B: batched-fold convention cross-check live in every batch verify
        setenv("ZKOB_FOLD_CROSSCHECK", "1", 1);
        setenv("ZKOB_BATCH_SELFCHECK", "1", 1);
        bool a = selftest_case(4, 6, 3);
        bool b = selftest_case(8, 8, 8);
        bool c = selftest_case(16, 12, 5);
        bool d = selftest_case_claims(4, 6, 3);
        bool e = selftest_case_claims(8, 8, 8);    // IN_pad == OUT_pad: shared gen
        bool f = selftest_case_claims(16, 12, 5);  // two domains in one batch
        bool g = selftest_case_wpriv(4, 6, 3);     // Stage D: weight privacy
        bool h = selftest_case_wpriv(16, 12, 5);   // two domains, wpriv
        bool i = selftest_case_apriv(4, 6, 3);     // Stage D5: + activation (Y) privacy
        bool j = selftest_case_apriv(16, 12, 5);   // two domains, apriv
        bool ok = a && b && c && d && e && f && g && h && i && j;
        cout << (ok ? "ZKOB-FC SELFTEST: ALL PASS" : "ZKOB-FC SELFTEST: FAIL") << endl;
        return ok ? 0 : 1;
    }
    if (mode == "commit" && base_argc == 7) {
        uint IN = stoi(argv[3]), OUT = stoi(argv[4]);
        FrTensor W = load_int_tensor(argv[2], IN * OUT);
        Commitment gen_out(argv[5]);
        G1TensorJacobian com = gen_out.commit(W.pad({IN, OUT}));
        if (!hd_a.empty()) {
            // D1 hiding registration: com[r] += s_r*H, blinds saved
            // prover-private (hd_b); H = q.bin slot 1
            if (hd_b.empty()) throw runtime_error("--hiding needs <q.bin> <blinds_out>");
            Commitment qg(hd_a);
            if (qg.size < 2) throw runtime_error("--hiding: q.bin has no H slot");
            const uint IN_pad = 1u << ceilLog2(IN);
            vector<Fr_t> s(IN_pad);
            for (auto& x : s) x = wp_rand();
            wp_hide_rows(com, s, qg(1));
            wp_blinds_save(hd_b, s);
        }
        com.save(argv[6]);
        cout << "registered commitment" << (hd_a.empty() ? "" : " (HIDING)")
             << " -> " << argv[6] << endl;
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
        if (!wp_a.empty() && wp_b.empty())
            throw runtime_error("prove --wpriv needs <waccdir> <W-blinds.bin>");
        G1Jacobian_t Hslot;
        const G1Jacobian_t* Hp = nullptr;
        if (!wp_a.empty()) {
            if (qg.size < 2) throw runtime_error("--wpriv: q.bin has no H slot");
            Hslot = qg(1); Hp = &Hslot;
        }
        prove(obdir, seed, X, W, B, IN, OUT, gen_in, gen_out, qg(0),
              base_argc == 13 ? argv[12] : "", cm_a, cm_b, cm_c, wp_a, wp_b, Hp, ap_flag);
        return 0;
    }
    if (mode == "verify" && base_argc == 11) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[4]), IN = stoi(argv[5]), OUT = stoi(argv[6]);
        Commitment gen_in(argv[8]), gen_out(argv[9]), qg(argv[10]);
        if (!cm_a.empty() && cm_b.empty())
            throw runtime_error("verify --claims needs <vaccdir> <obid>");
        G1Jacobian_t Hslot;
        const G1Jacobian_t* Hp = nullptr;
        if (!wp_a.empty()) {
            if (qg.size < 2) throw runtime_error("--wpriv: q.bin has no H slot");
            Hslot = qg(1); Hp = &Hslot;
        }
        return verify(obdir, seed, B, IN, OUT, argv[7], gen_in, gen_out, qg(0),
                      cm_a, cm_b, wp_a, Hp, ap_flag) ? 0 : 1;
    }
    cerr << "usage: zkob_fc selftest | commit ... [--hiding <q.bin> <blinds_out>] | "
            "prove ... [--claims <accdir> <obid> <com_W>] [--wpriv <waccdir> <W-blinds>] | "
            "verify ... [--claims <vaccdir> <obid>] [--wpriv <wvaccdir>]" << endl;
    return 2;
}

// Stage C2 single-process transport: `serve` keeps this driver resident (one
// CUDA init for the whole walk); every request runs the same zkw_run1 entry.
int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "serve")
        return zkw_serve(argv[0], zkw_run1);
    return zkw_run1(argc, argv);
}
