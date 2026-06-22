// P3.2 GPU Merkle tree over a Goldilocks codeword, using a device SHA-256.
// This is the hash-commitment cost (Basefold/FRI commit = NTT encode + Merkle).
// SHA-256 is the conservative choice; an algebraic hash (Poseidon2) would be
// cheaper still, so this UNDER-states the eventual hash-PCS advantage.
#pragma once
#include <cstdint>
#include "p3_goldilocks.cuh"

#define P3_MERKLE_THREADS 256

__host__ __device__ __forceinline__ uint32_t p3_ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

// SHA-256 over `len` bytes (len <= 64 here), output 32 bytes. Self-contained.
__device__ void p3_sha256(const uint8_t* data, uint32_t len, uint8_t out[32]) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};

    uint8_t msg[128];                 // len<=64 -> padded <=128 (2 blocks)
    uint32_t total = len;
    for (uint32_t i = 0; i < len; i++) msg[i] = data[i];
    msg[len] = 0x80;
    uint32_t padded = ((len + 8) / 64 + 1) * 64;
    for (uint32_t i = len + 1; i < padded - 8; i++) msg[i] = 0;
    uint64_t bitlen = (uint64_t)total * 8;
    for (int i = 0; i < 8; i++) msg[padded - 1 - i] = (uint8_t)(bitlen >> (8 * i));

    for (uint32_t blk = 0; blk < padded; blk += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)msg[blk+4*i] << 24) | ((uint32_t)msg[blk+4*i+1] << 16) |
                   ((uint32_t)msg[blk+4*i+2] << 8) | (uint32_t)msg[blk+4*i+3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = p3_ror(w[i-15],7) ^ p3_ror(w[i-15],18) ^ (w[i-15] >> 3);
            uint32_t s1 = p3_ror(w[i-2],17) ^ p3_ror(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = p3_ror(e,6) ^ p3_ror(e,11) ^ p3_ror(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = p3_ror(a,2) ^ p3_ror(a,13) ^ p3_ror(a,22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    for (int i = 0; i < 8; i++) {
        out[4*i]   = (uint8_t)(h[i] >> 24); out[4*i+1] = (uint8_t)(h[i] >> 16);
        out[4*i+2] = (uint8_t)(h[i] >> 8);  out[4*i+3] = (uint8_t)(h[i]);
    }
}

// Fixed-input SHA-256 compression of exactly 64 bytes: one block from the standard
// IV, NO length-padding block.  Collision-resistant (the SHA-256 compression IS the
// CR primitive) and HALVES internal-node cost vs full SHA-256, which pads 64 bytes
// into a second, all-padding block.  Used for Merkle internal nodes (leaves keep
// full SHA-256 over the 8-byte value -> distinct domains).  __host__ __device__ so
// the GPU prover and host verifier compute byte-identical node hashes.
__host__ __device__ __forceinline__ void p3_sha256_compress64(const uint8_t in[64], uint8_t out[32]) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)in[4*i] << 24) | ((uint32_t)in[4*i+1] << 16) |
               ((uint32_t)in[4*i+2] << 8) | (uint32_t)in[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = p3_ror(w[i-15],7) ^ p3_ror(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = p3_ror(w[i-2],17) ^ p3_ror(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = p3_ror(e,6) ^ p3_ror(e,11) ^ p3_ror(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + K[i] + w[i];
        uint32_t S0 = p3_ror(a,2) ^ p3_ror(a,13) ^ p3_ror(a,22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    for (int i = 0; i < 8; i++) {
        out[4*i]   = (uint8_t)(h[i] >> 24); out[4*i+1] = (uint8_t)(h[i] >> 16);
        out[4*i+2] = (uint8_t)(h[i] >> 8);  out[4*i+3] = (uint8_t)(h[i]);
    }
}

__global__ void p3_merkle_leaf_kernel(const gl_t* codeword, uint8_t* out, uint32_t M) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    uint8_t buf[8];
    gl_t v = codeword[i];
    #pragma unroll
    for (int k = 0; k < 8; k++) buf[k] = (uint8_t)(v >> (8 * k));
    p3_sha256(buf, 8, out + (size_t)i * 32);
}

__global__ void p3_merkle_internal_kernel(const uint8_t* in, uint8_t* out, uint32_t cnt) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cnt) return;
    uint8_t buf[64];
    #pragma unroll
    for (int k = 0; k < 32; k++) { buf[k] = in[(size_t)(2*i)*32 + k]; buf[32+k] = in[(size_t)(2*i+1)*32 + k]; }
    p3_sha256_compress64(buf, out + (size_t)i * 32);
}

// Build a Merkle tree over M (power of 2) codeword elements; write 32-byte root.
// d_scratchA/B are each >= M*32 bytes of device scratch.
static void p3_merkle_build(const gl_t* d_codeword, uint32_t M,
                            uint8_t* d_scratchA, uint8_t* d_scratchB, uint8_t root_out[32]) {
    uint32_t blocks = (M + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS;
    p3_merkle_leaf_kernel<<<blocks, P3_MERKLE_THREADS>>>(d_codeword, d_scratchA, M);
    uint8_t* cur = d_scratchA; uint8_t* nxt = d_scratchB;
    for (uint32_t cnt = M >> 1; cnt >= 1; cnt >>= 1) {
        uint32_t b = (cnt + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS;
        p3_merkle_internal_kernel<<<b, P3_MERKLE_THREADS>>>(cur, nxt, cnt);
        uint8_t* t = cur; cur = nxt; nxt = t;
        if (cnt == 1) break;
    }
    cudaMemcpy(root_out, cur, 32, cudaMemcpyDeviceToHost);
}
