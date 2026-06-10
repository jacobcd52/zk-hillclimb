// COORDINATOR-BUILT (do not let submission agents edit).
// Toy-scale ground truth for the witness-free commitment-opening verifier.
//
// Pins the verify-side algebra of Commitment::me_open at sizes where every
// intermediate can be checked:
//   scalars t (N), generators (GEN), com = commit(t) (COM = N/GEN row points)
//   u = u_in(ceilLog2 GEN) ++ u_out(ceilLog2 COM)  [open() takes u_out from the END]
//   prover: t_row = t.partial_me(u_out, N/COM); me_open(t_row, gen, u_in) -> proof
//   verifier (witness-free, only {com, pp generators, u, proof, eval}):
//     C_0 = raw-limb ME of com at u_out      (NOT com(u_out): G1_me_step unmonts
//           the challenge; me_open/partial_me use raw limbs. upstream's
//           com(u_out) result -- open()'s g_temp -- is dead code, so the
//           inconsistency was invisible upstream.)
//     per round i (challenge u_i, proof triple T,T0,T1):
//        T == C_i                                   (binding consistency)
//        C_{i+1} = (1-u)^2*T0 + u(1-u)*T + u^2*T1   (fold identity)
//     G_final = pairwise generator fold of pp with u_in: g' = g1 + u*(g0-g1)
//               (recomputed from PUBLIC pp -- never read from the proof)
//     accept iff C_L == G_final * eval
// All Fr coefficient arithmetic matches the kernels' "mont limbs as integers"
// view: plain modmul(a,b) = mont(montmul(a,b)).
//
// !!! MISCOMPILATION WORKAROUND (found empirically, see vrf_toy_debug2.cu):
// with this project's -dlto build, a kernel containing TWO branches that each
// call G1Jacobian_mul produces wrong results (even when one branch is dead).
// Therefore every fold kernel here has exactly ONE branch (pairs only), and
// odd-size levels are handled by PADDING with the identity point (all-zero
// bytes, z=0), which reproduces upstream's odd-branch semantics exactly:
//   scalar orientation: c_last + u*(INF - c_last) = (1-u)*c_last   (matches)
//   me_open generators: INF + u*(g_last - INF)    = u*g_last       (matches)
#include "commitment.cuh"
#include "proof.cuh"
#include <iostream>
using namespace std;

// ---- 1-thread device helpers so host can do field/point ops bit-identically
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
    // Jacobian coords aren't unique: equal iff a - b == point at infinity (z==0).
    // The add formula handles P + (-P): u1==u2, s1!=s2 -> h=0 -> z3=0.
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_addsub<<<1,1>>>(a, b, 1, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d);
    for (int i = 0; i < 12; i++) if (h.z.val[i] != 0) return false;
    return true;
}

// generic G1 fold chain with identity-padding for odd levels.
// orientation 0 = scalar/ME (k_com_me), 1 = me_open generators (k_gen_fold).
static G1Jacobian_t fold_chain(const G1Jacobian_t* dev_src, uint size,
                               const vector<Fr_t>& us, int orientation) {
    uint cap = size + 1;  // room for one pad slot at any level
    G1Jacobian_t* d_a; cudaMalloc(&d_a, cap * sizeof(G1Jacobian_t));
    G1Jacobian_t* d_b; cudaMalloc(&d_b, cap * sizeof(G1Jacobian_t));
    cudaMemcpy(d_a, dev_src, size * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    uint sz = size;
    for (auto& u : us) {
        if (sz % 2) {  // pad with identity (all-zero bytes, z=0)
            cudaMemset(d_a + sz, 0, sizeof(G1Jacobian_t));
            sz += 1;
        }
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

static bool run_case(uint GEN, uint N) {
    const uint COM = N / GEN;
    cout << "=== case GEN=" << GEN << " N=" << N << " COM=" << COM << " ===" << endl;
    Commitment gen = Commitment::random(GEN);
    FrTensor t = FrTensor::random_int(N, 16);
    G1TensorJacobian com = gen.commit(t);

    auto u = random_vec(ceilLog2(GEN) + ceilLog2(COM));
    vector<Fr_t> u_out(u.end() - ceilLog2(COM), u.end());
    vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(COM));

    // ---- prover side (witness): exactly what Commitment::open does
    FrTensor t_row = t.partial_me(u_out, N / COM);
    vector<G1Jacobian_t> proof;
    Fr_t eval = Commitment::me_open(t_row, gen, u_in.begin(), u_in.end(), proof);
    cout << "proof points: " << proof.size() << " (expect " << 3 * u_in.size() + 1 << ")" << endl;

    // ---- verifier side (NO access to t / t_row) ----
    G1Jacobian_t C = fold_chain(com.gpu_data, COM, u_out, 0);  // C_0
    Fr_t ONE = {1, 0, 0, 0, 0, 0, 0, 0};  // integer-view 1
    bool ok = true;
    for (uint i = 0; i < u_in.size(); i++) {
        G1Jacobian_t T = proof[3*i], T0 = proof[3*i+1], T1 = proof[3*i+2];
        if (!g1_eq(T, C)) { cout << "round " << i << ": T != C  FAIL" << endl; ok = false; }
        Fr_t uu = u_in[i];
        Fr_t omu = h_scalar(ONE, uu, 1);                 // 1 - u
        Fr_t c_t  = h_scalar(uu, omu, 2);                // u(1-u)
        Fr_t c_t0 = h_scalar(omu, omu, 2);               // (1-u)^2
        Fr_t c_t1 = h_scalar(uu, uu, 2);                 // u^2
        C = h_add(h_add(h_mul(T0, c_t0), h_mul(T, c_t)), h_mul(T1, c_t1));
    }
    // recompute folded generator from PUBLIC pp only
    G1Jacobian_t G_final = fold_chain(gen.gpu_data, GEN, u_in, 1);
    bool g_ok = g1_eq(G_final, proof.back());
    cout << "recomputed G_final == proof's pushed generator: " << (g_ok ? "YES" : "NO") << endl;
    bool final_ok = g1_eq(C, h_mul(G_final, eval));
    cout << "final check C_L == G_final * eval: " << (final_ok ? "PASS" : "FAIL") << endl;

    // ---- forgery sanity at toy scale: perturb eval -> must FAIL
    Fr_t bad_eval = h_scalar(eval, ONE, 0); // eval + 1
    bool forg_ok = !g1_eq(C, h_mul(G_final, bad_eval));
    cout << "forged eval rejected: " << (forg_ok ? "YES" : "NO(!!)") << endl;

    bool all = ok && g_ok && final_ok && forg_ok;
    cout << (all ? "CASE PASS" : "CASE FAIL") << endl;
    return all;
}

int main() {
    bool a = run_case(8, 32);   // powers of two everywhere
    bool b = run_case(6, 24);   // odd level inside me_open fold (6->3->2->1)
    bool c = run_case(8, 24);   // odd number of commitment rows (COM=3)
    cout << ((a && b && c) ? "TOY-OPEN-VERIFY: ALL PASS" : "TOY-OPEN-VERIFY: FAIL") << endl;
    return (a && b && c) ? 0 : 1;
}
