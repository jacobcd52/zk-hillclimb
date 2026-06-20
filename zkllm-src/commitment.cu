#include "commitment.cuh"

Commitment Commitment::random(uint size)
{
    Commitment out(size, G1Jacobian_generator);
    out *= FrTensor::random(size);
    return out; 
}

// KERNEL void com_sum_row_kernel(const G1Jacobian_t* arr, G1Jacobian_t* arr_out, uint m, uint n) {
//     auto row = GET_GLOBAL_ID();
//     if (row < m) {
//         G1Jacobian_t rowSum = arr[row * n];
//         for (uint i = 1; i < n; ++ i) {
//             rowSum = blstrs__g1__G1Affine_add(rowSum, arr[row * n + i]);
//         }
//         arr_out[row] = rowSum;
//     }
    
// }

// reference per-element MSM (full 256-bit mul per (row,col) then rowwise sum)
G1TensorJacobian Commitment::commit_naive(const FrTensor& t) const
{
    if (t.size % size != 0) throw std::runtime_error("Commitment::commit - Incompatible dimensions");

    uint m = t.size / size;
    G1TensorJacobian temp = (*this) * t;
    return temp.rowwise_sum(m, size);
}

// ---------------- Pippenger (bucket) MSM (P2 speed lever) ----------------
// Each row commitment out[r] = sum_i scalars[r,i] * gen[i] is computed by the
// bucket method instead of `size` full scalar muls: PIP_C-bit windows, one
// thread per (row, window) accumulating into 2^PIP_C local buckets, reduced via
// the running-sum trick, then windows combined by Horner. Produces the SAME
// group element as commit_naive (a different but projectively-equal Jacobian
// rep); used as the single consistent commit path so all file-level
// byte-compares (comref / chaining / hash-pin) stay consistent and the
// in-circuit g1_eq checks (projective) are unaffected.
#define PIP_C 4u
#define PIP_NBUCKET (1u << PIP_C)
#define PIP_W ((256u + PIP_C - 1u) / PIP_C)   // 64 windows of 4 bits

// Extract the w-th PIP_C-bit window of the scalar's RAW limbs. G1Jacobian_mul
// multiplies by x.val[] directly (no unmont), so we must too -> identical point.
DEVICE uint pip_digit(Fr_t s, uint w) {
    uint bitpos = w * PIP_C;
    uint limb = bitpos >> 5, off = bitpos & 31;   // PIP_C | 32 -> no cross-limb
    if (limb >= blstrs__scalar__Scalar_LIMBS) return 0;
    return (s.val[limb] >> off) & (PIP_NBUCKET - 1u);
}

KERNEL void pippenger_bucket_kernel(const G1Jacobian_t* gen, const Fr_t* scalars,
                                    G1Jacobian_t* partials, uint m, uint n) {
    const uint tid = GET_GLOBAL_ID();
    if (tid >= m * PIP_W) return;
    const uint row = tid / PIP_W, w = tid % PIP_W;
    G1Jacobian_t bucket[PIP_NBUCKET];
    for (uint b = 0; b < PIP_NBUCKET; b++) bucket[b] = blstrs__g1__G1Affine_ZERO;
    const Fr_t* srow = scalars + (size_t)row * n;
    for (uint i = 0; i < n; i++) {
        const uint d = pip_digit(srow[i], w);
        if (d) bucket[d] = blstrs__g1__G1Affine_add(bucket[d], gen[i]);
    }
    G1Jacobian_t acc = blstrs__g1__G1Affine_ZERO, run = blstrs__g1__G1Affine_ZERO;
    for (int b = (int)PIP_NBUCKET - 1; b >= 1; b--) {     // sum_b b*bucket[b]
        acc = blstrs__g1__G1Affine_add(acc, bucket[b]);
        run = blstrs__g1__G1Affine_add(run, acc);
    }
    partials[(size_t)row * PIP_W + w] = run;
}

KERNEL void pippenger_combine_kernel(const G1Jacobian_t* partials, G1Jacobian_t* out, uint m) {
    const uint row = GET_GLOBAL_ID();
    if (row >= m) return;
    G1Jacobian_t acc = blstrs__g1__G1Affine_ZERO;
    for (int w = (int)PIP_W - 1; w >= 0; w--) {           // Horner: *2^PIP_C then +partial
        for (uint k = 0; k < PIP_C; k++) acc = blstrs__g1__G1Affine_double(acc);
        acc = blstrs__g1__G1Affine_add(acc, partials[(size_t)row * PIP_W + w]);
    }
    out[row] = acc;
}

G1TensorJacobian Commitment::commit_pippenger(const FrTensor& t) const
{
    if (t.size % size != 0) throw std::runtime_error("Commitment::commit_pippenger - Incompatible dimensions");
    uint m = t.size / size;
    G1TensorJacobian partials(m * PIP_W);
    pippenger_bucket_kernel<<<(m * PIP_W + G1NumThread - 1) / G1NumThread, G1NumThread>>>(
        gpu_data, t.gpu_data, partials.gpu_data, m, size);
    cudaDeviceSynchronize();
    G1TensorJacobian out(m);
    pippenger_combine_kernel<<<(m + G1NumThread - 1) / G1NumThread, G1NumThread>>>(
        partials.gpu_data, out.gpu_data, m);
    cudaDeviceSynchronize();
    return out;
}

G1TensorJacobian Commitment::commit(const FrTensor& t) const
{
    return commit_pippenger(t);
}

DEVICE G1Jacobian_t commit_int_dev_func(G1Jacobian_t a, Fr_t s) {
    const int x = scalar_to_int(s);
    G1Jacobian_t out = blstrs__g1__G1Affine_ZERO;
    #pragma unroll
    for (uint i = 0; i < 31; ++ i) {
        if ((x >> i) & 1) out = blstrs__g1__G1Affine_add(out, a);
        a = blstrs__g1__G1Affine_double(a);
    }
    
    if (x < 0) out = blstrs__g1__G1Affine_add(out, G1Jacobian_minus(a));
    return out;
}

KERNEL void commit_int_kernel(const G1Jacobian_t* generators, const Fr_t* scalars, G1Jacobian_t* out, uint n, uint m) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= m * n) return;
    out[gid] = commit_int_dev_func(generators[gid % n], scalars[gid]);
}

G1TensorJacobian Commitment::commit_int (const FrTensor& t) const{
    if (t.size % size != 0) throw std::runtime_error("Commitment::commit_int - Incompatible dimensions");

    uint m = t.size / size;
    G1TensorJacobian temp(t.size);
    commit_int_kernel<<<(m*size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, temp.gpu_data, size, m);
    cudaDeviceSynchronize();
    return temp.rowwise_sum(m, size);
}

G1TensorJacobian Commitment::commit_int_multi(const vector<FrTensor>& ts) const{
    uint num_row = 0;
    for (auto& t : ts) {
        if (t.size % size != 0) throw std::runtime_error("Commitment::commit_int_multi - Incompatible dimensions");
        num_row += t.size / size;
    }

    G1TensorJacobian temp(num_row * size);
    auto temp_start = temp.gpu_data;
    for (auto& t: ts)
    {
        uint m = t.size / size;
        commit_int_kernel<<<(m*size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, temp_start, size, m);
        cudaDeviceSynchronize();
        temp_start += m * size;
    }
    return temp.rowwise_sum(temp.size / size, size);
}

KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators, Fr_t u, // always assume that scalars and u is in mont form
    GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
    GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0, GLOBAL G1Jacobian_t* temp_out1, 
    uint old_size, uint new_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;

    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    if (gid1 >= old_size) {
        new_scalars[gid] = blstrs__scalar__Scalar_sub(scalars[gid0], 
            blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, scalars[gid0]))
        );
        new_generators[gid] = G1Jacobian_mul(generators[gid0], u);
        temp_out[gid] = G1Jacobian_mul(generators[gid0], scalars[gid0]);
        temp_out0[gid] = blstrs__g1__G1Affine_ZERO;
        temp_out1[gid] = blstrs__g1__G1Affine_ZERO;
        return;
    }


    new_scalars[gid] = blstrs__scalar__Scalar_add(scalars[gid0], blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, blstrs__scalar__Scalar_sub(scalars[gid1], scalars[gid0]))));
    new_generators[gid] = blstrs__g1__G1Affine_add(generators[gid1], G1Jacobian_mul(blstrs__g1__G1Affine_add(generators[gid0], G1Jacobian_minus(generators[gid1])), u));
    temp_out[gid] = blstrs__g1__G1Affine_add(G1Jacobian_mul(generators[gid0], scalars[gid0]), G1Jacobian_mul(generators[gid1], scalars[gid1]));
    temp_out0[gid] = G1Jacobian_mul(generators[gid1], scalars[gid0]);
    temp_out1[gid] = G1Jacobian_mul(generators[gid0], scalars[gid1]);
}

Fr_t Commitment::me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof)
{
    if (t.size != generators.size) throw std::runtime_error("Commitment::me_open - Incompatible dimensions "+ std::to_string(t.size) + " " + std::to_string(generators.size));
    if (begin >= end)
    {
        proof.push_back(generators(0));
        return t(0);
    }
    uint new_size = (t.size + 1) / 2;
    FrTensor new_scalars(new_size);
    Commitment new_generators(new_size);
    G1TensorJacobian temp(new_size), temp0(new_size), temp1(new_size);
    me_open_step<<<(new_size+G1NumThread-1)/G1NumThread,G1NumThread>>>(t.gpu_data, generators.gpu_data, *begin, 
    new_scalars.gpu_data, new_generators.gpu_data, temp.gpu_data, temp0.gpu_data, temp1.gpu_data, 
    t.size, new_size);
    cudaDeviceSynchronize();
    proof.push_back(temp.sum());
    proof.push_back(temp0.sum());
    proof.push_back(temp1.sum());
    return me_open(new_scalars, new_generators, begin + 1, end, proof);
}



Fr_t Commitment::open(const FrTensor& t, const G1TensorJacobian& com, const vector<Fr_t>& u) const
{
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));
    auto g_temp = (com.size == 1)? com(0) : com(u_out);
    // if (size != (1 << u_in.size())) throw std::runtime_error("Incompatible dimensions");
    vector<G1Jacobian_t> proof;
    return me_open(t.partial_me(u_out, t.size / com.size), *this, u_in.begin(), u_in.end(), proof);
}

Weight create_weight(string generator_filename, string weight_filename, string com_filename, uint in_dim, uint out_dim) {
    Commitment generator(generator_filename);
    FrTensor weight = FrTensor::from_int_bin(weight_filename);
    G1TensorJacobian com(com_filename);
    return {generator, weight, com, in_dim, out_dim};
}