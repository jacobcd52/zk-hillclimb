// COORDINATOR-BUILT (do not let submission agents edit).
// Toy-scale ground truth for witness-free SUMCHECK (zkip) verification of one
// matmul claim, plus the glue to the pinned commitment-opening verify.
//
// Replicates zkFC::prove with injected challenges:
//   X (B x IN), W (IN x OUT), Y = X @ W (matrixMultiplyOptimized)
//   u_batch (logB), u_input (logIN), u_output (logOUT)
//   claim   = Y.multi_dim_me({u_batch, u_output}, {B, OUT})
//   X_red   = X.partial_me(u_batch, B, IN)      (size IN)
//   W_red   = W.partial_me(u_output, OUT, 1)    (size IN)
//   final   = zkip(claim, X_red, W_red, u_input, proof)   <- round polys
//   claim_X = X.multi_dim_me({u_batch, u_input}, {B, IN})
//   claim_W = W.multi_dim_me({u_input, u_output}, {IN, OUT})
//   upstream asserts final == claim_X * claim_W (witness); we serialize claim_X,
//   claim_W and DISCHARGE claim_W via the commitment opening (vrf_toy_open.cu).
//
// Serialization: Polynomial's coefficients are private, so each degree-2 round
// poly is shipped as THREE EVALUATIONS p(0), p(1), p(2) (extracted with
// operator(), which uses the same integer-view arithmetic as the kernels).
//
// Witness-free verifier, given {claim, round evals, claim_X, claim_W,
// u_batch/u_input/u_output, com(W_padded), pp generators, opening proof, eval}:
//   rounds consume u_input from the END: round r uses u_input[L-1-r]
//   per round: check claim == p(0) + p(1); claim' = Lagrange3(p0,p1,p2)(u_r)
//     Lagrange3: p(u) = p0*(u-1)(u-2)/2 + p1*u(2-u) + p2*u(u-1)/2
//     (INV2 = (r+1)/2 hardcoded; self-checked at startup: 2*INV2 == 1)
//   terminal: claim_final == claim_X * claim_W   (integer-view product)
//   claim_W discharge (verifyWeightClaim mapping, proof.cu):
//     u_cat = u_output ++ u_input; W_padded = W.pad({IN, OUT}) per-dim pow2
//     com rows = IN_pad (folded by u_input), generators = OUT_pad (by u_output)
//     pinned opening verify (see vrf_toy_open.cu) + CHECK opening eval == claim_W
//   claim_X is the caller's obligation (chains to the previous layer / input).
//
// Same -dlto miscompilation workaround as vrf_toy_open.cu: every G1 fold
// kernel has exactly ONE G1Jacobian_mul branch; odd levels identity-padded.
#include "commitment.cuh"
#include "zkfc.cuh"
#include "polynomial.cuh"
#include <iostream>
using namespace std;

// ---- 1-thread device helpers (bit-identical host-side field/point ops) ----
KERNEL void k_scalar_op(Fr_t a, Fr_t b, int op, GLOBAL Fr_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    if (op == 0) *out = blstrs__scalar__Scalar_add(a, b);
    if (op == 1) *out = blstrs__scalar__Scalar_sub(a, b);
    if (op == 2) *out = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a, b)); // plain a*b (int view)
}
KERNEL void k_g1_mul(G1Jacobian_t g, Fr_t c, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = G1Jacobian_mul(g, c);
}
KERNEL void k_g1_addsub(G1Jacobian_t a, G1Jacobian_t b, int sub, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = blstrs__g1__G1Affine_add(a, sub ? G1Jacobian_minus(b) : b);
}
// pair-fold, me_open generator orientation: g' = g1 + u*(g0 - g1). SINGLE branch.
KERNEL void k_gen_fold(GLOBAL G1Jacobian_t* g, Fr_t u, GLOBAL G1Jacobian_t* gout, uint new_size) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    gout[gid] = blstrs__g1__G1Affine_add(g[2*gid+1],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(g[2*gid], G1Jacobian_minus(g[2*gid+1])), u));
}
// pair-fold, scalar/ME orientation: c' = c0 + u*(c1 - c0). SINGLE branch.
KERNEL void k_com_me(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* cout, uint new_size) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    cout[gid] = blstrs__g1__G1Affine_add(c[2*gid],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(c[2*gid+1], G1Jacobian_minus(c[2*gid])), u));
}

static Fr_t h_scalar(Fr_t a, Fr_t b, int op) {
    Fr_t *d, h; cudaMalloc(&d, sizeof(Fr_t));
    k_scalar_op<<<1,1>>>(a, b, op, d);
    cudaMemcpy(&h, d, sizeof(Fr_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
static G1Jacobian_t h_mul(G1Jacobian_t g, Fr_t c) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_mul<<<1,1>>>(g, c, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
static G1Jacobian_t h_add(G1Jacobian_t a, G1Jacobian_t b) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_addsub<<<1,1>>>(a, b, 0, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
static bool g1_eq(G1Jacobian_t a, G1Jacobian_t b) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_addsub<<<1,1>>>(a, b, 1, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d);
    for (int i = 0; i < 12; i++) if (h.z.val[i] != 0) return false;
    return true;
}
static bool fr_eq(const Fr_t& a, const Fr_t& b) {
    for (int i = 0; i < 8; i++) if (a.val[i] != b.val[i]) return false;
    return true;
}

// generic G1 fold chain with identity-padding for odd levels (vrf_toy_open.cu).
static G1Jacobian_t fold_chain(const G1Jacobian_t* dev_src, uint size,
                               const vector<Fr_t>& us, int orientation) {
    uint cap = size + 1;
    G1Jacobian_t* d_a; cudaMalloc(&d_a, cap * sizeof(G1Jacobian_t));
    G1Jacobian_t* d_b; cudaMalloc(&d_b, cap * sizeof(G1Jacobian_t));
    cudaMemcpy(d_a, dev_src, size * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    uint sz = size;
    for (auto& u : us) {
        if (sz % 2) { cudaMemset(d_a + sz, 0, sizeof(G1Jacobian_t)); sz += 1; }
        uint nsz = sz / 2;
        if (orientation == 0) k_com_me<<<(nsz + 31) / 32, 32>>>(d_a, u, d_b, nsz);
        else                  k_gen_fold<<<(nsz + 31) / 32, 32>>>(d_a, u, d_b, nsz);
        cudaDeviceSynchronize();
        std::swap(d_a, d_b); sz = nsz;
    }
    G1Jacobian_t out;
    cudaMemcpy(&out, d_a, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b);
    return out;
}

// ---- integer-view constants ----
static const Fr_t F_ZERO = {0, 0, 0, 0, 0, 0, 0, 0};
static const Fr_t F_ONE  = {1, 0, 0, 0, 0, 0, 0, 0};
static const Fr_t F_TWO  = {2, 0, 0, 0, 0, 0, 0, 0};
// INV2 = (r+1)/2 for r = blstrs__scalar__Scalar_P (LE 32-bit limbs); self-checked in main.
static const Fr_t F_INV2 = {2147483649u, 2147483647u, 2147429887u, 2849952257u,
                            80800770u, 429714436u, 2496577188u, 972477353u};

// Lagrange-3 evaluation from {p(0), p(1), p(2)} at u, all integer-view:
// p(u) = p0*(u-1)(u-2)/2 + p1*u(2-u) + p2*u(u-1)/2
static Fr_t lagrange3(const Fr_t& p0, const Fr_t& p1, const Fr_t& p2, const Fr_t& u) {
    Fr_t um1 = h_scalar(u, F_ONE, 1);
    Fr_t um2 = h_scalar(u, F_TWO, 1);
    Fr_t l0 = h_scalar(h_scalar(um1, um2, 2), F_INV2, 2);
    Fr_t l1 = h_scalar(u, h_scalar(F_TWO, u, 1), 2);
    Fr_t l2 = h_scalar(h_scalar(u, um1, 2), F_INV2, 2);
    Fr_t acc = h_scalar(p0, l0, 2);
    acc = h_scalar(acc, h_scalar(p1, l1, 2), 0);
    acc = h_scalar(acc, h_scalar(p2, l2, 2), 0);
    return acc;
}

static bool run_case(uint B, uint IN, uint OUT) {
    cout << "=== case B=" << B << " IN=" << IN << " OUT=" << OUT << " ===" << endl;
    const uint IN_pad = 1u << ceilLog2(IN), OUT_pad = 1u << ceilLog2(OUT);

    // ---- setup: weights, commitment (public pp + public com), input ----
    FrTensor W = FrTensor::random_int(IN * OUT, 8);
    FrTensor X = FrTensor::random_int(B * IN, 8);
    Commitment gen = Commitment::random(OUT_pad);
    FrTensor W_padded = W.pad({IN, OUT});                  // IN_pad x OUT_pad
    G1TensorJacobian com = gen.commit(W_padded);           // IN_pad row points
    zkFC fc(IN, OUT, W);
    FrTensor Y = fc(X);                                    // B x OUT

    // ---- prover side: zkFC::prove with injected challenges ----
    auto u_batch  = random_vec(ceilLog2(B));
    auto u_input  = random_vec(ceilLog2(IN));
    auto u_output = random_vec(ceilLog2(OUT));

    Fr_t claim = Y.multi_dim_me({u_batch, u_output}, {B, OUT});
    FrTensor X_red = X.partial_me(u_batch, B, IN);
    FrTensor W_red = W.partial_me(u_output, OUT, 1);
    vector<Polynomial> proof;
    Fr_t final_claim = zkip(claim, X_red, W_red, u_input, proof);  // throws internally on mismatch
    Fr_t claim_X = X.multi_dim_me({u_batch, u_input}, {B, IN});
    Fr_t claim_W = W.multi_dim_me({u_input, u_output}, {IN, OUT});
    cout << "rounds: " << proof.size() << " (expect " << u_input.size() << ")" << endl;

    // serialize round polys as evaluations p(0), p(1), p(2)
    vector<Fr_t> ev0, ev1, ev2;
    for (auto& p : proof) {
        ev0.push_back(p(F_ZERO)); ev1.push_back(p(F_ONE)); ev2.push_back(p(F_TWO));
    }

    // sanity: padded ME == unpadded ME (partial_me odd branch == implicit zero pad)
    Fr_t claim_W_padded = W_padded.multi_dim_me({u_input, u_output}, {IN_pad, OUT_pad});
    bool pad_ok = fr_eq(claim_W, claim_W_padded);
    cout << "padded/unpadded claim_W agree: " << (pad_ok ? "YES" : "NO") << endl;

    // opening proof for claim_W (verifyWeightClaim mapping: u_cat = u_output ++ u_input)
    FrTensor t_row = W_padded.partial_me(u_input, OUT_pad);   // fold IN_pad rows
    vector<G1Jacobian_t> open_proof;
    Fr_t open_eval = Commitment::me_open(t_row, gen, u_output.begin(), u_output.end(), open_proof);

    // ---- verifier side (NO access to X / W / Y) ----
    bool ok = true;

    // (1) sumcheck rounds from serialized evals; u consumed from the END
    Fr_t cur = claim;
    const uint L = u_input.size();
    for (uint r = 0; r < L; r++) {
        Fr_t u_r = u_input[L - 1 - r];
        if (!fr_eq(cur, h_scalar(ev0[r], ev1[r], 0))) {
            cout << "round " << r << ": claim != p(0)+p(1)  FAIL" << endl; ok = false;
        }
        Fr_t nxt = lagrange3(ev0[r], ev1[r], ev2[r], u_r);
        // cross-check Lagrange3 against the actual Polynomial object (prover-side only)
        if (!fr_eq(nxt, proof[r](u_r))) {
            cout << "round " << r << ": Lagrange3 != Polynomial(u)  FAIL" << endl; ok = false;
        }
        cur = nxt;
    }
    // (2) terminal: claim_final == claim_X * claim_W (integer-view product)
    bool term_ok = fr_eq(cur, h_scalar(claim_X, claim_W, 2));
    cout << "terminal claim == claim_X*claim_W: " << (term_ok ? "PASS" : "FAIL") << endl;
    bool fin_ok = fr_eq(cur, final_claim);  // matches what upstream zkip returned
    cout << "verifier terminal == upstream final_claim: " << (fin_ok ? "YES" : "NO") << endl;

    // (3) discharge claim_W via the pinned commitment-opening verify
    //     com rows (IN_pad) folded by u_input; generators (OUT_pad) by u_output.
    bool glue_ok = fr_eq(open_eval, claim_W);
    cout << "opening eval == claim_W (glue): " << (glue_ok ? "YES" : "NO") << endl;
    G1Jacobian_t C = fold_chain(com.gpu_data, IN_pad, u_input, 0);
    bool open_ok = true;
    for (uint i = 0; i < u_output.size(); i++) {
        G1Jacobian_t T = open_proof[3*i], T0 = open_proof[3*i+1], T1 = open_proof[3*i+2];
        if (!g1_eq(T, C)) { cout << "open round " << i << ": T != C  FAIL" << endl; open_ok = false; }
        Fr_t uu = u_output[i];
        Fr_t omu  = h_scalar(F_ONE, uu, 1);
        Fr_t c_t  = h_scalar(uu, omu, 2);
        Fr_t c_t0 = h_scalar(omu, omu, 2);
        Fr_t c_t1 = h_scalar(uu, uu, 2);
        C = h_add(h_add(h_mul(T0, c_t0), h_mul(T, c_t)), h_mul(T1, c_t1));
    }
    G1Jacobian_t G_final = fold_chain(gen.gpu_data, OUT_pad, u_output, 1);
    bool g_ok = g1_eq(G_final, open_proof.back());
    bool open_fin = g1_eq(C, h_mul(G_final, claim_W));   // verify against claim_W directly
    cout << "opening verify (G_final, C_L == G*claim_W): "
         << (g_ok && open_fin && open_ok ? "PASS" : "FAIL") << endl;

    // ---- forgeries: each must be rejected ----
    // F1: tamper round-0 p(1) -> p(0)+p(1) check breaks
    bool f1 = !fr_eq(claim, h_scalar(ev0[0], h_scalar(ev1[0], F_ONE, 0), 0));
    // F2: claim_W + 1 -> terminal product check breaks (claim_X != 0 w.h.p.)
    bool f2 = !fr_eq(cur, h_scalar(claim_X, h_scalar(claim_W, F_ONE, 0), 2));
    // F3: claim_W + 1 against the honest opening -> C_L == G*claim_W' breaks
    bool f3 = !g1_eq(C, h_mul(G_final, h_scalar(claim_W, F_ONE, 0)));
    cout << "forged round poly rejected: " << (f1 ? "YES" : "NO(!!)")
         << "  forged claim_W rejected (terminal): " << (f2 ? "YES" : "NO(!!)")
         << "  (opening): " << (f3 ? "YES" : "NO(!!)") << endl;

    bool all = ok && pad_ok && term_ok && fin_ok && glue_ok && open_ok && g_ok && open_fin
               && f1 && f2 && f3;
    cout << (all ? "CASE PASS" : "CASE FAIL") << endl;
    return all;
}

int main() {
    // INV2 self-check: 2 * INV2 == 1 in the integer view
    if (!fr_eq(h_scalar(F_INV2, F_TWO, 2), F_ONE)) {
        cout << "INV2 SELF-CHECK FAILED -- hardcoded constant is wrong" << endl; return 2;
    }
    cout << "INV2 self-check passed" << endl;
    bool a = run_case(4, 8, 4);    // powers of two everywhere
    bool b = run_case(4, 6, 3);    // both matmul dims non-pow2 (pad + zkip zero-fill)
    bool c = run_case(2, 8, 3);    // OUT non-pow2 only
    bool d = run_case(8, 16, 16);  // a bit bigger
    cout << ((a && b && c && d) ? "TOY-MATMUL-VERIFY: ALL PASS" : "TOY-MATMUL-VERIFY: FAIL") << endl;
    return (a && b && c && d) ? 0 : 1;
}
