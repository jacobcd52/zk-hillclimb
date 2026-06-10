// Minimal matrix: why does k_com_me differ from the staged version?
// Axes: thread count, guard style, indices variable vs literal, nesting.
#include "commitment.cuh"
#include <iostream>
using namespace std;

KERNEL void k_v1(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* out,
                 uint old_size, uint new_size) {       // = k_com_me verbatim
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    uint c0 = 2 * gid, c1 = 2 * gid + 1;
    if (c1 >= old_size) {
        out[gid] = blstrs__g1__G1Affine_add(c[c0], G1Jacobian_minus(G1Jacobian_mul(c[c0], u)));
        return;
    }
    out[gid] = blstrs__g1__G1Affine_add(c[c0],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(c[c1], G1Jacobian_minus(c[c0])), u));
}
KERNEL void k_v2(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* out) { // literals, 1 thread
    if (GET_GLOBAL_ID() > 0) return;
    out[0] = blstrs__g1__G1Affine_add(c[0],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(c[1], G1Jacobian_minus(c[0])), u));
}
KERNEL void k_v3(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* out) { // staged locals, 1 thread
    if (GET_GLOBAL_ID() > 0) return;
    G1Jacobian_t d = blstrs__g1__G1Affine_add(c[1], G1Jacobian_minus(c[0]));
    G1Jacobian_t s = G1Jacobian_mul(d, u);
    out[0] = blstrs__g1__G1Affine_add(c[0], s);
}
KERNEL void k_v4(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* out) { // variable indices
    if (GET_GLOBAL_ID() > 0) return;
    uint i0 = 0, i1 = 1;
    out[0] = blstrs__g1__G1Affine_add(c[i0],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(c[i1], G1Jacobian_minus(c[i0])), u));
}
KERNEL void k_v5(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* out,
                 uint old_size, uint new_size) {       // extra uint params, literals
    if (GET_GLOBAL_ID() > 0) return;
    out[0] = blstrs__g1__G1Affine_add(c[0],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(c[1], G1Jacobian_minus(c[0])), u));
}
KERNEL void k_v6(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* out,
                 uint old_size, uint new_size) {       // v1 guard, literals, no odd branch
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    out[gid] = blstrs__g1__G1Affine_add(c[0],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(c[1], G1Jacobian_minus(c[0])), u));
}
KERNEL void k_v7(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* out,
                 uint old_size, uint new_size) {       // v1 minus the odd branch
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    uint c0 = 2 * gid, c1 = 2 * gid + 1;
    out[gid] = blstrs__g1__G1Affine_add(c[c0],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(c[c1], G1Jacobian_minus(c[c0])), u));
}
KERNEL void k_v8(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* out,
                 uint old_size, uint new_size) {       // v1 with if/else, no early return
    const uint gid = GET_GLOBAL_ID();
    if (gid < new_size) {
        uint c0 = 2 * gid, c1 = 2 * gid + 1;
        if (c1 >= old_size) {
            out[gid] = blstrs__g1__G1Affine_add(c[c0], G1Jacobian_minus(G1Jacobian_mul(c[c0], u)));
        } else {
            out[gid] = blstrs__g1__G1Affine_add(c[c0],
                G1Jacobian_mul(blstrs__g1__G1Affine_add(c[c1], G1Jacobian_minus(c[c0])), u));
        }
    }
}
KERNEL void k_sub(G1Jacobian_t a, G1Jacobian_t b, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = blstrs__g1__G1Affine_add(a, G1Jacobian_minus(b));
}
KERNEL void k_mu(G1Jacobian_t a, Fr_t x, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = G1Jacobian_mul(a, x);
}
KERNEL void k_ad(G1Jacobian_t a, G1Jacobian_t b, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = blstrs__g1__G1Affine_add(a, b);
}
static bool g1_eq(G1Jacobian_t a, G1Jacobian_t b) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_sub<<<1,1>>>(a, b, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d);
    for (int i = 0; i < 12; i++) if (h.z.val[i] != 0) return false;
    return true;
}
template <typename K, typename... A> G1Jacobian_t h1(K k, A... a) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k<<<1,1>>>(a..., d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}

int main() {
    Commitment pts = Commitment::random(2);
    auto u = random_vec(1);
    G1Jacobian_t hc[2];
    cudaMemcpy(hc, pts.gpu_data, 2 * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);

    // reference via 1-op kernels
    G1Jacobian_t ref = h1(k_ad, hc[0], h1(k_mu, h1(k_sub, hc[1], hc[0]), u[0]));

    G1Jacobian_t* d_out; cudaMalloc(&d_out, sizeof(G1Jacobian_t));
    G1Jacobian_t r;
    auto get = [&]() { cudaMemcpy(&r, d_out, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); return r; };

    k_v1<<<1,32>>>(pts.gpu_data, u[0], d_out, 2, 1); cudaDeviceSynchronize();
    cout << "v1 <<<1,32>>> == ref: " << g1_eq(get(), ref) << endl;
    k_v1<<<1,1>>>(pts.gpu_data, u[0], d_out, 2, 1); cudaDeviceSynchronize();
    cout << "v1 <<<1,1>>>  == ref: " << g1_eq(get(), ref) << endl;
    k_v2<<<1,1>>>(pts.gpu_data, u[0], d_out); cudaDeviceSynchronize();
    cout << "v2 (literals) == ref: " << g1_eq(get(), ref) << endl;
    k_v3<<<1,1>>>(pts.gpu_data, u[0], d_out); cudaDeviceSynchronize();
    cout << "v3 (locals)   == ref: " << g1_eq(get(), ref) << endl;
    k_v4<<<1,1>>>(pts.gpu_data, u[0], d_out); cudaDeviceSynchronize();
    cout << "v4 (var idx)  == ref: " << g1_eq(get(), ref) << endl;
    k_v5<<<1,1>>>(pts.gpu_data, u[0], d_out, 2, 1); cudaDeviceSynchronize();
    cout << "v5 (+uints)   == ref: " << g1_eq(get(), ref) << endl;
    k_v6<<<1,1>>>(pts.gpu_data, u[0], d_out, 2, 1); cudaDeviceSynchronize();
    cout << "v6 (guard)    == ref: " << g1_eq(get(), ref) << endl;
    k_v7<<<1,1>>>(pts.gpu_data, u[0], d_out, 2, 1); cudaDeviceSynchronize();
    cout << "v7 (v1-odd)   == ref: " << g1_eq(get(), ref) << endl;
    k_v8<<<1,1>>>(pts.gpu_data, u[0], d_out, 2, 1); cudaDeviceSynchronize();
    cout << "v8 (if/else)  == ref: " << g1_eq(get(), ref) << endl;
    k_v8<<<1,32>>>(pts.gpu_data, u[0], d_out, 2, 1); cudaDeviceSynchronize();
    cout << "v8 <<<1,32>>> == ref: " << g1_eq(get(), ref) << endl;
    // odd-branch correctness too: 3 points, fold -> 2 (pair + leftover*(1-u))
    Commitment pts3 = Commitment::random(3);
    G1Jacobian_t hc3[3];
    cudaMemcpy(hc3, pts3.gpu_data, 3 * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    G1Jacobian_t* d_out2; cudaMalloc(&d_out2, 2 * sizeof(G1Jacobian_t));
    k_v8<<<1,32>>>(pts3.gpu_data, u[0], d_out2, 3, 2); cudaDeviceSynchronize();
    G1Jacobian_t r2[2];
    cudaMemcpy(r2, d_out2, 2 * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    G1Jacobian_t ref0 = h1(k_ad, hc3[0], h1(k_mu, h1(k_sub, hc3[1], hc3[0]), u[0]));
    G1Jacobian_t ref1 = h1(k_sub, hc3[2], h1(k_mu, hc3[2], u[0]));   // (1-u)*c2
    cout << "v8 odd pair   == ref: " << g1_eq(r2[0], ref0)
         << "  v8 odd leftover == (1-u)c2: " << g1_eq(r2[1], ref1) << endl;
    return 0;
}
