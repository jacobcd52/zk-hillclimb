// Public wrappers + helpers shared by zkprove_dump and zkverify, reusing zkfc
// internals. Kept separate so we don't modify upstream .cu files.
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "polynomial.cuh"
#include "zkfs.cuh"
#include <vector>
#include <stdexcept>
using namespace std;

// defined (non-static) in zkfc.cu
Polynomial zkip_step_poly(const FrTensor& a, const FrTensor& b, const Fr_t& u);
KERNEL void zkip_reduce_kernel(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *new_a, GLOBAL Fr_t *new_b, Fr_t v, uint N_in, uint N_out);

static const Fr_t Z0{0,0,0,0,0,0,0,0};
static const Fr_t Z1{1,0,0,0,0,0,0,0};
static const Fr_t Z2{2,0,0,0,0,0,0,0};

Polynomial zkip_step_poly_pub(const FrTensor& a, const FrTensor& b, const Fr_t& u) {
    return zkip_step_poly(a, b, u);
}

void zkip_reduce_pub(const FrTensor& a, const FrTensor& b, FrTensor& new_a, FrTensor& new_b,
                     const Fr_t& v, uint N_in, uint N_out) {
    zkip_reduce_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        a.gpu_data, b.gpu_data, new_a.gpu_data, new_b.gpu_data, v, N_in, N_out);
    cudaDeviceSynchronize();
}

// Extract degree-2 polynomial coefficients [c0,c1,c2] by evaluation+interpolation
// over points {0,1,2}. Robust against Polynomial's private members.
vector<Fr_t> poly_coeffs(Polynomial& p, uint width) {
    Fr_t p0 = p(Z0), p1 = p(Z1), p2 = p(Z2);
    Fr_t two{2,0,0,0,0,0,0,0};
    Fr_t inv2 = Z1 / two;
    Fr_t c0 = p0;
    Fr_t c2 = (p2 - (two * p1) + p0) * inv2;
    Fr_t c1 = p1 - p0 - c2;
    vector<Fr_t> out{c0, c1, c2};
    while (out.size() < width) out.push_back(Z0);
    return out;
}

// Evaluate a serialized round poly (coeff vector) at x — host side (Horner).
Fr_t eval_coeffs(const vector<Fr_t>& c, const Fr_t& x) {
    Fr_t acc = Z0;
    for (int i = (int)c.size() - 1; i >= 0; --i) acc = acc * x + c[i];
    return acc;
}

// Generator-only fold matching me_open_step's new_generators update:
//   gid1 in range: ng = g1 + u*(g0 - g1)
//   gid1 oob:      ng = u*g0
#include "g1-tensor.cuh"
G1Jacobian_t G1Jacobian_mul_host(G1Jacobian_t a, Fr_t x); // not needed; use device
KERNEL void me_gen_fold(const G1Jacobian_t* g, G1Jacobian_t* ng, Fr_t u, uint old_size, uint new_size) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    uint g0 = 2*gid, g1 = 2*gid+1;
    if (g1 >= old_size) {
        ng[gid] = G1Jacobian_mul(g[g0], u);
        return;
    }
    ng[gid] = blstrs__g1__G1Affine_add(g[g1], G1Jacobian_mul(blstrs__g1__G1Affine_add(g[g0], G1Jacobian_minus(g[g1])), u));
}
