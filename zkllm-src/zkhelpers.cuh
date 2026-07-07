#ifndef ZKHELPERS_CUH
#define ZKHELPERS_CUH
#include "fr-tensor.cuh"
#include "polynomial.cuh"
#include <vector>
using std::vector;

Polynomial zkip_step_poly_pub(const FrTensor& a, const FrTensor& b, const Fr_t& u);
void zkip_reduce_pub(const FrTensor& a, const FrTensor& b, FrTensor& new_a, FrTensor& new_b,
                     const Fr_t& v, uint N_in, uint N_out);
vector<Fr_t> poly_coeffs(Polynomial& p, uint width);
Fr_t eval_coeffs(const vector<Fr_t>& c, const Fr_t& x);

#include "g1-tensor.cuh"
KERNEL void me_gen_fold(const G1Jacobian_t* g, G1Jacobian_t* ng, Fr_t u, uint old_size, uint new_size);
#endif
