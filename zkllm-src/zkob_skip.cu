// COORDINATOR-BUILT (do not let submission agents edit).
// Driver for ONE skip_connection obligation: Z = A + B, checked PURELY
// homomorphically on public row commitments (same generator set for all
// three): com_Z[j] == com_A[j] + com_B[j]. No witness, no proof artifact —
// the commitment files ARE the proof; chaining to neighbor obligations is
// byte-equality of those files (orchestrator's job).
//
// Uses the 1-thread host helpers ONLY (h_add/g1_eq) — batched G1 kernels of
// new shapes are -dlto miscompile bait (see GOTCHAS; the zkob_rescale affine
// kernel was silently wrong on all rows).
//
// Usage:
//   zkob_skip add    <com_A.bin> <com_B.bin> <com_Z_out.bin>   (prover side)
//                    [--claims <accdir> <obid>]
//   zkob_skip verify <com_A.bin> <com_B.bin> <com_Z.bin>
//                    [--claims <vaccdir> <obid>]
//   zkob_skip selftest
//
// CLAIM MODE (Stage C of the transport rebuild): zkob_skip is the one driver
// with NO opening tail (TRANSPORT_REBUILD_DESIGN §3: "0 — no change at all").
// --claims is accepted for orchestrator uniformity and emits ZERO claims and
// no drvstate (there is no per-obligation transcript here; the ⊕ point check
// is the whole obligation). The verdict is therefore FINAL even in claim
// mode — nothing is deferred to opening_batch. The claim-mode selftest pins
// that routing a skip run through an accumulator leaves it byte-untouched
// and the batch over the other drivers' claims still ACCEPTs.
#include "zkob_claims.cuh"
#include <iostream>
#include <sys/stat.h>
using namespace std;

static vector<G1Jacobian_t> to_host(const G1TensorJacobian& t) {
    vector<G1Jacobian_t> h(t.size);
    cudaMemcpy(h.data(), t.gpu_data, t.size * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    return h;
}

static void hom_add(const string& pa, const string& pb, const string& pz) {
    G1TensorJacobian A(pa), B(pb);
    if (A.size != B.size) throw runtime_error("commitment row counts differ");
    auto ha = to_host(A), hb = to_host(B);
    vector<G1Jacobian_t> hz(A.size);
    for (uint j = 0; j < A.size; j++) hz[j] = h_add(ha[j], hb[j]);
    G1TensorJacobian Z(A.size, hz.data());
    Z.save(pz);
    cout << "WROTE homomorphic sum -> " << pz << endl;
}

static bool verify(const string& pa, const string& pb, const string& pz) {
    G1TensorJacobian A(pa), B(pb), Z(pz);
    if (A.size != B.size || A.size != Z.size) {
        cout << "REJECT: commitment row counts" << endl; return false;
    }
    auto ha = to_host(A), hb = to_host(B), hz = to_host(Z);
    for (uint j = 0; j < A.size; j++)
        if (!g1_eq(hz[j], h_add(ha[j], hb[j]))) {
            cout << "REJECT: skip add fails at row " << j << endl; return false;
        }
    cout << "ACCEPT" << endl;
    return true;
}

// claim-mode selftest: skip emits ZERO claims; an accumulator pre-seeded with
// one honest synthetic claim (standing in for a neighbor driver's emission)
// must pass through a skip add+verify byte-unchanged, and the batch over it
// must still ACCEPT. Also re-pins the driver-side forgery in claim mode.
static bool selftest_claims() {
    cout << "=== selftest (claim mode: zero-claims routing) ===" << endl;
    string dir = "/tmp/zkob_skip_cm";
    { string c = "rm -rf " + dir; system(c.c_str()); }
    mkdir(dir.c_str(), 0755);
    string acc = dir + "/acc", vacc = dir + "/vacc";
    mkdir(acc.c_str(), 0755); mkdir(vacc.c_str(), 0755);
    string run_seed = "selftest";

    // the skip obligation itself
    uint B = 8, C = 5, C_pad = 8;
    FrTensor X = FrTensor::random_int(B * C, 10);
    FrTensor Y = FrTensor::random_int(B * C, 10);
    FrTensor Z = X + Y;
    Commitment gen = Commitment::random(C_pad);
    gen.commit(X.pad({B, C})).save(dir + "/A.bin");
    gen.commit(Y.pad({B, C})).save(dir + "/B.bin");
    gen.commit(Z.pad({B, C})).save(dir + "/Z.bin");

    // synthetic neighbor claim: a 1-row x 4-col tensor opened at a random
    // column point (eval = <T, me_weights(u_col)>, the pinned MLE convention)
    const uint G = 4, logG = 2;
    FrTensor T = FrTensor::random_int(G, 8);
    Commitment ngen = Commitment::random(G);
    map<uint32_t, string> genpaths;
    genpaths[G] = dir + "/gen4.bin";
    ngen.save(genpaths[G]);
    string qpath = dir + "/q.bin";
    Commitment::random(2).save(qpath);
    G1TensorJacobian com_T = ngen.commit(T);
    com_T.save(dir + "/com_T.bin");
    fs::Transcript ntr("selftest:neighbor");
    auto u_col = fs_challenge_vec(ntr, logG);
    Fr_t ev = F_ZERO;
    {
        vector<Fr_t> hT(G);
        cudaMemcpy(hT.data(), T.gpu_data, G * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        auto w = me_weights(u_col);
        for (uint i = 0; i < G; i++) ev = h_scalar(ev, h_scalar(hT[i], w[i], 2), 0);
    }
    BoClaim c;
    c.id = "selftest.neighbor:T"; c.comref = dir + "/com_T.bin";
    c.domain = G; c.n_rows = 1; c.point = u_col; c.eval = ev;
    claim_emit(acc, c);
    string witT = acc + "/wit_T.fr";
    T.save(witT);
    witref_emit(acc, dir + "/com_T.bin", witT);
    drvstate_emit(acc, "selftest.neighbor", ntr);
    claim_emit(vacc, c);
    drvstate_emit(vacc, "selftest.neighbor", ntr);

    auto sha = [&](const string& p) { uint8_t h[32]; bo_file_sha256(p, h);
        return string((const char*)h, 32); };
    string acc_before = sha(acc + "/claims.bin");

    // skip add + verify with --claims routing (must emit nothing)
    hom_add(dir + "/A.bin", dir + "/B.bin", dir + "/Z2.bin");
    bool honest = verify(dir + "/A.bin", dir + "/B.bin", dir + "/Z.bin");
    bool hom = verify(dir + "/A.bin", dir + "/B.bin", dir + "/Z2.bin");
    cout << "claim mode: 0 claims emitted (no opening tail; verdict final)" << endl;

    bool acc_unchanged = (sha(acc + "/claims.bin") == acc_before) &&
                         !bo_file_exists(acc + "/wit_selftest.skip.fr");
    cout << "accumulator byte-unchanged by skip: " << (acc_unchanged ? "YES" : "NO(!!)") << endl;

    batch_prove(acc, run_seed, genpaths, qpath);
    string locus;
    bool batch_ok = batch_verify(acc, vacc, run_seed, genpaths, qpath, &locus);
    cout << "batch over neighbor claim still ACCEPTs: " << (batch_ok ? "YES" : "NO(!!)") << endl;

    // driver-side forgery unchanged in claim mode
    FrTensor Zb = X + Y;
    { Fr_t h; cudaMemcpy(&h, Zb.gpu_data + 7, sizeof(Fr_t), cudaMemcpyDeviceToHost);
      h = h_scalar(h, F_ONE, 0);
      cudaMemcpy(Zb.gpu_data + 7, &h, sizeof(Fr_t), cudaMemcpyHostToDevice); }
    gen.commit(Zb.pad({B, C})).save(dir + "/Zbad.bin");
    bool forged = !verify(dir + "/A.bin", dir + "/B.bin", dir + "/Zbad.bin");
    cout << "forged Z rejected (claim mode): " << (forged ? "YES" : "NO(!!)") << endl;

    bool ok = honest && hom && acc_unchanged && batch_ok && forged;
    cout << (ok ? "CASE PASS" : "CASE FAIL") << " (claim mode)" << endl;
    return ok;
}

int main(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    // strip the optional claim-mode flag block: --claims <accdir> <obid>
    // (accepted for orchestrator uniformity; zkob_skip emits ZERO claims)
    int base_argc = argc;
    bool claim_mode = false;
    for (int i = 2; i < argc; i++)
        if (string(argv[i]) == "--claims") { base_argc = i; claim_mode = true; break; }
    if (mode == "selftest") {
        // commit two random int tensors and their sum; verify; then tamper
        uint B = 8, C = 5, C_pad = 8;
        FrTensor X = FrTensor::random_int(B * C, 10);
        FrTensor Y = FrTensor::random_int(B * C, 10);
        FrTensor Z = X + Y;
        Commitment gen = Commitment::random(C_pad);
        gen.commit(X.pad({B, C})).save("/tmp/zkob_skip_A.bin");
        gen.commit(Y.pad({B, C})).save("/tmp/zkob_skip_B.bin");
        gen.commit(Z.pad({B, C})).save("/tmp/zkob_skip_Z.bin");
        bool honest = verify("/tmp/zkob_skip_A.bin", "/tmp/zkob_skip_B.bin", "/tmp/zkob_skip_Z.bin");
        // hom_add path must equal the witness commitment
        hom_add("/tmp/zkob_skip_A.bin", "/tmp/zkob_skip_B.bin", "/tmp/zkob_skip_Z2.bin");
        bool hom = verify("/tmp/zkob_skip_A.bin", "/tmp/zkob_skip_B.bin", "/tmp/zkob_skip_Z2.bin");
        // forgery: commit X+Y with one element bumped
        FrTensor Zb = X + Y;
        { Fr_t one = F_ONE; Fr_t* p; cudaMalloc(&p, sizeof(Fr_t));
          cudaMemcpy(p, Zb.gpu_data + 7, sizeof(Fr_t), cudaMemcpyDeviceToDevice);
          Fr_t h; cudaMemcpy(&h, p, sizeof(Fr_t), cudaMemcpyDeviceToHost);
          h = h_scalar(h, one, 0);
          cudaMemcpy(Zb.gpu_data + 7, &h, sizeof(Fr_t), cudaMemcpyHostToDevice);
          cudaFree(p); }
        gen.commit(Zb.pad({B, C})).save("/tmp/zkob_skip_Zbad.bin");
        bool forged = !verify("/tmp/zkob_skip_A.bin", "/tmp/zkob_skip_B.bin", "/tmp/zkob_skip_Zbad.bin");
        cout << "honest: " << (honest ? "ACCEPT" : "REJECT(!!)")
             << "  hom_add: " << (hom ? "ACCEPT" : "REJECT(!!)")
             << "  forged Z rejected: " << (forged ? "YES" : "NO(!!)") << endl;
        bool old_ok = honest && hom && forged;
        cout << (old_ok ? "old-mode case PASS" : "old-mode case FAIL") << endl;
        bo_probe_kernels();
        cout << "kernel -dlto probes: PASS" << endl;
        setenv("ZKOB_FOLD_CROSSCHECK", "1", 1);
        bool cm_ok = selftest_claims();
        bool ok = old_ok && cm_ok;
        cout << (ok ? "ZKOB-SKIP SELFTEST: ALL PASS" : "ZKOB-SKIP SELFTEST: FAIL") << endl;
        return ok ? 0 : 1;
    }
    if (mode == "add" && base_argc == 5) {
        hom_add(argv[2], argv[3], argv[4]);
        if (claim_mode) cout << "claim mode: 0 claims emitted (skip has no opening tail)" << endl;
        return 0;
    }
    if (mode == "verify" && base_argc == 5) {
        bool ok = verify(argv[2], argv[3], argv[4]);
        if (ok && claim_mode)
            cout << "claim mode: 0 claims emitted; verdict FINAL (not gated on opening_batch)" << endl;
        return ok ? 0 : 1;
    }
    cerr << "usage: zkob_skip selftest | add a b z [--claims <accdir> <obid>] | verify a b z [--claims <vaccdir> <obid>]" << endl;
    return 2;
}
