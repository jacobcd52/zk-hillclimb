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
//   zkob_skip verify <com_A.bin> <com_B.bin> <com_Z.bin>
//   zkob_skip selftest
#include "vrf_common.cuh"
#include <iostream>
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

int main(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
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
        bool ok = honest && hom && forged;
        cout << (ok ? "ZKOB-SKIP SELFTEST: ALL PASS" : "ZKOB-SKIP SELFTEST: FAIL") << endl;
        return ok ? 0 : 1;
    }
    if (mode == "add" && argc == 5) { hom_add(argv[2], argv[3], argv[4]); return 0; }
    if (mode == "verify" && argc == 5) return verify(argv[2], argv[3], argv[4]) ? 0 : 1;
    cerr << "usage: zkob_skip selftest | add a b z | verify a b z" << endl;
    return 2;
}
