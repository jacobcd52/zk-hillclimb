// Debug bisection for vrf_toy_open: replicate the FULL me_open recursion
// (scalars + generators + temps) with standalone kernels and compare against
// the real proof at every level. Localizes exactly where my model of
// me_open_step diverges from reality.
#include "commitment.cuh"
#include "proof.cuh"
#include <iostream>
using namespace std;

// my replication of me_open_step, verbatim semantics
KERNEL void k_step(GLOBAL Fr_t* s, GLOBAL G1Jacobian_t* g, Fr_t u,
                   GLOBAL Fr_t* ns, GLOBAL G1Jacobian_t* ng,
                   GLOBAL G1Jacobian_t* T, GLOBAL G1Jacobian_t* T0, GLOBAL G1Jacobian_t* T1,
                   uint old_size, uint new_size) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    uint i0 = 2 * gid, i1 = 2 * gid + 1;
    if (i1 >= old_size) {
        ns[gid] = blstrs__scalar__Scalar_sub(s[i0],
            blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, s[i0])));
        ng[gid] = G1Jacobian_mul(g[i0], u);
        T[gid] = G1Jacobian_mul(g[i0], s[i0]);
        T0[gid] = blstrs__g1__G1Affine_ZERO;
        T1[gid] = blstrs__g1__G1Affine_ZERO;
        return;
    }
    ns[gid] = blstrs__scalar__Scalar_add(s[i0], blstrs__scalar__Scalar_mont(
        blstrs__scalar__Scalar_mul(u, blstrs__scalar__Scalar_sub(s[i1], s[i0]))));
    ng[gid] = blstrs__g1__G1Affine_add(g[i1],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(g[i0], G1Jacobian_minus(g[i1])), u));
    T[gid] = blstrs__g1__G1Affine_add(G1Jacobian_mul(g[i0], s[i0]), G1Jacobian_mul(g[i1], s[i1]));
    T0[gid] = G1Jacobian_mul(g[i1], s[i0]);
    T1[gid] = G1Jacobian_mul(g[i0], s[i1]);
}

// raw-limb ME fold over G1 points (orientation of Fr_partial_me_step)
KERNEL void k_com_me(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* cout_,
                     uint old_size, uint new_size) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    uint c0 = 2 * gid, c1 = 2 * gid + 1;
    if (c1 >= old_size) {
        cout_[gid] = blstrs__g1__G1Affine_add(c[c0], G1Jacobian_minus(G1Jacobian_mul(c[c0], u)));
        return;
    }
    cout_[gid] = blstrs__g1__G1Affine_add(c[c0],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(c[c1], G1Jacobian_minus(c[c0])), u));
}
KERNEL void k_scalar_op(Fr_t a, Fr_t b, int op, GLOBAL Fr_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    if (op == 0) *out = blstrs__scalar__Scalar_add(a, b);
    if (op == 1) *out = blstrs__scalar__Scalar_sub(a, b);
    if (op == 2) *out = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a, b));
}
KERNEL void k_g1_mul(G1Jacobian_t g, Fr_t c, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = G1Jacobian_mul(g, c);
}
KERNEL void k_g1_add2(G1Jacobian_t a, G1Jacobian_t b, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = blstrs__g1__G1Affine_add(a, b);
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
    k_g1_add2<<<1,1>>>(a, b, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}

// staged version of the k_com_me even-case expression, dumping intermediates
KERNEL void k_com_me_staged(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* st) {
    if (GET_GLOBAL_ID() > 0) return;
    st[0] = G1Jacobian_minus(c[0]);                       // -c0
    st[1] = blstrs__g1__G1Affine_add(c[1], st[0]);        // c1 - c0
    st[2] = G1Jacobian_mul(st[1], u);                     // u*(c1-c0)
    st[3] = blstrs__g1__G1Affine_add(c[0], st[2]);        // c0 + u*(c1-c0)
}

KERNEL void k_g1_sub1(G1Jacobian_t a, G1Jacobian_t b, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = blstrs__g1__G1Affine_add(a, G1Jacobian_minus(b));
}
static bool g1_eq(G1Jacobian_t a, G1Jacobian_t b) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_sub1<<<1,1>>>(a, b, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d);
    for (int i = 0; i < 12; i++) if (h.z.val[i] != 0) return false;
    return true;
}
// direct inner product <s, g> with raw-limb scalar mul
KERNEL void k_ip(GLOBAL Fr_t* s, GLOBAL G1Jacobian_t* g, GLOBAL G1Jacobian_t* out, uint n) {
    if (GET_GLOBAL_ID() > 0) return;
    G1Jacobian_t acc = blstrs__g1__G1Affine_ZERO;
    for (uint i = 0; i < n; i++)
        acc = blstrs__g1__G1Affine_add(acc, G1Jacobian_mul(g[i], s[i]));
    *out = acc;
}
static G1Jacobian_t h_ip(Fr_t* d_s, G1Jacobian_t* d_g, uint n) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_ip<<<1,1>>>(d_s, d_g, d, n);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}

int main() {
    const uint GEN = 8, N = 32, COM = N / GEN;
    Commitment gen = Commitment::random(GEN);
    FrTensor t = FrTensor::random_int(N, 16);
    G1TensorJacobian com = gen.commit(t);

    auto u = random_vec(5);
    vector<Fr_t> u_out(u.end() - ceilLog2(COM), u.end());
    vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(COM));

    FrTensor t_row = t.partial_me(u_out, N / COM);
    vector<G1Jacobian_t> proof;
    Fr_t eval = Commitment::me_open(t_row, gen, u_in.begin(), u_in.end(), proof);

    // sanity 0: g1_eq self-test
    G1Jacobian_t P = proof[0];
    cout << "sanity g1_eq(P,P): " << g1_eq(P, P)
         << "   g1_eq(P,proof[1]): " << g1_eq(P, proof[1]) << endl;

    // sanity 1: proof[0] (T at round 0) == <t_row, gen> directly?
    Fr_t* d_s; cudaMalloc(&d_s, GEN * sizeof(Fr_t));
    cudaMemcpy(d_s, t_row.gpu_data, GEN * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    G1Jacobian_t* d_g; cudaMalloc(&d_g, GEN * sizeof(G1Jacobian_t));
    cudaMemcpy(d_g, gen.gpu_data, GEN * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    G1Jacobian_t ip0 = h_ip(d_s, d_g, GEN);
    cout << "proof[0] == <t_row, gen> (raw-limb ip): " << g1_eq(proof[0], ip0) << endl;

    // sanity 2: <t_row, gen> == my com-ME of com at u_out?  (the round-0 verifier eq)
    // com rows: com_j = sum_k mul(g_k, t[j*GEN+k])  -- check row 0 directly too
    Fr_t* d_t; cudaMalloc(&d_t, N * sizeof(Fr_t));
    cudaMemcpy(d_t, t.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    G1Jacobian_t row0 = h_ip(d_t, d_g, GEN);  // <t[0:8], gen>
    G1Jacobian_t com0; cudaMemcpy(&com0, com.gpu_data, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cout << "com[0] == <t[0:8], gen>: " << g1_eq(row0, com0) << endl;

    // full replication of me_open, comparing every pushed point
    uint sz = GEN;
    for (uint r = 0; r < u_in.size(); r++) {
        uint nsz = (sz + 1) / 2;
        Fr_t* d_ns; cudaMalloc(&d_ns, nsz * sizeof(Fr_t));
        G1Jacobian_t *d_ng, *d_T, *d_T0, *d_T1;
        cudaMalloc(&d_ng, nsz * sizeof(G1Jacobian_t));
        cudaMalloc(&d_T, nsz * sizeof(G1Jacobian_t));
        cudaMalloc(&d_T0, nsz * sizeof(G1Jacobian_t));
        cudaMalloc(&d_T1, nsz * sizeof(G1Jacobian_t));
        k_step<<<(nsz + 31) / 32, 32>>>(d_s, d_g, u_in[r], d_ns, d_ng, d_T, d_T0, d_T1, sz, nsz);
        cudaDeviceSynchronize();
        // sum the temps the same way me_open does: G1TensorJacobian::sum()
        G1TensorJacobian tT(nsz), tT0(nsz), tT1(nsz);
        cudaMemcpy(tT.gpu_data, d_T, nsz * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(tT0.gpu_data, d_T0, nsz * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(tT1.gpu_data, d_T1, nsz * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
        cout << "round " << r
             << "  T==proof: "  << g1_eq(tT.sum(),  proof[3*r])
             << "  T0==proof: " << g1_eq(tT0.sum(), proof[3*r+1])
             << "  T1==proof: " << g1_eq(tT1.sum(), proof[3*r+2]) << endl;
        cudaFree(d_s); cudaFree(d_g); cudaFree(d_T); cudaFree(d_T0); cudaFree(d_T1);
        d_s = d_ns; d_g = d_ng; sz = nsz;
    }
    // final: my folded generator vs proof.back(); my folded scalar vs eval
    G1Jacobian_t myG; cudaMemcpy(&myG, d_g, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    Fr_t myS; cudaMemcpy(&myS, d_s, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cout << "my G_final == proof.back(): " << g1_eq(myG, proof.back()) << endl;
    bool s_eq = true;
    for (int i = 0; i < 8; i++) if (myS.val[i] != eval.val[i]) s_eq = false;
    cout << "my folded scalar == eval: " << s_eq << endl;

    // ===== toy-verifier ops, tested piecewise against the proof =====
    // (a) C_0 = raw-limb ME of com at u_out  -- must equal proof[0]
    G1Jacobian_t* d_c; cudaMalloc(&d_c, COM * sizeof(G1Jacobian_t));
    cudaMemcpy(d_c, com.gpu_data, COM * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    uint csz = COM;
    for (auto& uu : u_out) {
        uint ncsz = (csz + 1) / 2;
        G1Jacobian_t* d_cn; cudaMalloc(&d_cn, ncsz * sizeof(G1Jacobian_t));
        k_com_me<<<(ncsz + 31) / 32, 32>>>(d_c, uu, d_cn, csz, ncsz);
        cudaDeviceSynchronize();
        cudaFree(d_c); d_c = d_cn; csz = ncsz;
    }
    G1Jacobian_t C0; cudaMemcpy(&C0, d_c, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cout << "(a) com-ME(u_out) == proof[0]: " << g1_eq(C0, proof[0]) << endl;

    // (b) verifier C-chain: C_{i+1} = (1-u)^2 T0 + u(1-u) T + u^2 T1, check T_i == C_i
    Fr_t ONE = {1, 0, 0, 0, 0, 0, 0, 0};
    G1Jacobian_t C = C0;
    for (uint i = 0; i < u_in.size(); i++) {
        cout << "(b) round " << i << " T==C: " << g1_eq(proof[3*i], C) << endl;
        Fr_t uu = u_in[i];
        Fr_t omu = h_scalar(ONE, uu, 1);
        Fr_t c_t  = h_scalar(uu, omu, 2);
        Fr_t c_t0 = h_scalar(omu, omu, 2);
        Fr_t c_t1 = h_scalar(uu, uu, 2);
        C = h_add(h_add(h_mul(proof[3*i+1], c_t0), h_mul(proof[3*i], c_t)),
                  h_mul(proof[3*i+2], c_t1));
    }
    cout << "(b) final C_L == G_final * eval: " << g1_eq(C, h_mul(proof.back(), eval)) << endl;

    // (c) explicit weights: which w assignment satisfies sum_j w_j com_j == proof[0]?
    vector<G1Jacobian_t> h_com(COM);
    cudaMemcpy(h_com.data(), com.gpu_data, COM * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    auto wsum = [&](Fr_t a, Fr_t b) {   // a = challenge on bit0 (rows 0/1), b on bit1
        Fr_t oma = h_scalar(ONE, a, 1), omb = h_scalar(ONE, b, 1);
        Fr_t w0 = h_scalar(oma, omb, 2), w1 = h_scalar(a, omb, 2);
        Fr_t w2 = h_scalar(oma, b, 2),   w3 = h_scalar(a, b, 2);
        return h_add(h_add(h_mul(h_com[0], w0), h_mul(h_com[1], w1)),
                     h_add(h_mul(h_com[2], w2), h_mul(h_com[3], w3)));
    };
    cout << "(c) w(bit0=u_out[0], bit1=u_out[1]): " << g1_eq(wsum(u_out[0], u_out[1]), proof[0]) << endl;
    cout << "(c) w(bit0=u_out[1], bit1=u_out[0]): " << g1_eq(wsum(u_out[1], u_out[0]), proof[0]) << endl;
    // (d) and t_row directly: my manual fold of t with each order, vs t_row
    vector<Fr_t> h_t(N), h_tr(GEN);
    cudaMemcpy(h_t.data(), t.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tr.data(), t_row.gpu_data, GEN * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    for (int order = 0; order < 2; order++) {
        Fr_t a = u_out[order], b = u_out[1 - order];
        bool all = true;
        for (uint k = 0; k < GEN; k++) {
            // fold rows with a on (0,1),(2,3) then b
            Fr_t r01 = h_scalar(h_t[k], h_scalar(a, h_scalar(h_t[GEN+k], h_t[k], 1), 2), 0);
            Fr_t r23 = h_scalar(h_t[2*GEN+k], h_scalar(a, h_scalar(h_t[3*GEN+k], h_t[2*GEN+k], 1), 2), 0);
            Fr_t r = h_scalar(r01, h_scalar(b, h_scalar(r23, r01, 1), 2), 0);
            for (int i = 0; i < 8; i++) if (r.val[i] != h_tr[k].val[i]) all = false;
        }
        cout << "(d) manual fold order " << order << " == t_row: " << all << endl;
    }

    // (e) one k_com_me round vs the same fold done with h_* helpers
    G1Jacobian_t* d_in; cudaMalloc(&d_in, COM * sizeof(G1Jacobian_t));
    cudaMemcpy(d_in, com.gpu_data, COM * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    G1Jacobian_t* d_out1; cudaMalloc(&d_out1, 2 * sizeof(G1Jacobian_t));
    k_com_me<<<1, 32>>>(d_in, u_out[0], d_out1, COM, 2);
    cudaDeviceSynchronize();
    vector<G1Jacobian_t> h_o1(2);
    cudaMemcpy(h_o1.data(), d_out1, 2 * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    auto h_sub = [&](G1Jacobian_t x, G1Jacobian_t y) {  // x - y via k_g1_sub1
        G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
        k_g1_sub1<<<1,1>>>(x, y, d);
        cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
    };
    G1Jacobian_t ref01 = h_add(h_com[0], h_mul(h_sub(h_com[1], h_com[0]), u_out[0]));
    G1Jacobian_t ref23 = h_add(h_com[2], h_mul(h_sub(h_com[3], h_com[2]), u_out[0]));
    cout << "(e) launch err: " << cudaGetLastError() << endl;
    cout << "(e) k_com_me[0] == c0+u(c1-c0): " << g1_eq(h_o1[0], ref01)
         << "   k_com_me[1] == c2+u(c3-c2): " << g1_eq(h_o1[1], ref23) << endl;
    // piecewise: diff = c1 - c0, scaled = u*(c1-c0), done host-side vs in-kernel value
    G1Jacobian_t hd = h_sub(h_com[1], h_com[0]);
    G1Jacobian_t hs = h_mul(hd, u_out[0]);
    // raw limb dump of the kernel result vs reference (first x limb)
    printf("(e) kern[0].x[0]=%08x  ref01.x[0]=%08x\n", h_o1[0].x.val[0], ref01.x.val[0]);
    printf("(e) hd.x[0]=%08x hs.x[0]=%08x  u_out0[0]=%08x\n",
           hd.x.val[0], hs.x.val[0], u_out[0].val[0]);
    // (f) staged kernel: where does the in-kernel computation diverge from h_*?
    G1Jacobian_t* d_st; cudaMalloc(&d_st, 4 * sizeof(G1Jacobian_t));
    k_com_me_staged<<<1,1>>>(d_in, u_out[0], d_st);
    cudaDeviceSynchronize();
    vector<G1Jacobian_t> h_st(4);
    cudaMemcpy(h_st.data(), d_st, 4 * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cout << "(f) staged c1-c0 == h_sub: " << g1_eq(h_st[1], hd)
         << "  staged u*(c1-c0) == h_mul: " << g1_eq(h_st[2], hs)
         << "  staged full == ref01: " << g1_eq(h_st[3], ref01) << endl;
    // and the second round on the reference points
    G1Jacobian_t refC = h_add(ref01, h_mul(h_sub(ref23, ref01), u_out[1]));
    cout << "(e) two-step h_* fold == proof[0]: " << g1_eq(refC, proof[0]) << endl;
    cout << "(e) two-step h_* fold == wsum:     " << g1_eq(refC, wsum(u_out[0], u_out[1])) << endl;
    return 0;
}
