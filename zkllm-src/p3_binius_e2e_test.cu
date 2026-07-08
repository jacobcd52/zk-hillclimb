// Binius substrate step 5 (design doc section 21.5): ONE real relation proven
// end-to-end over the binary tower, with the committed-data win measured
// against the Goldilocks path.
//
// Relation: N private 8-bit integer additions s = a + b (9-bit result).  This
// is the exact shape of the carry problem section 17 item 5 flagged: over
// GF(2^k) field addition is XOR (no carries), so INTEGER arithmetic is
// rebuilt as a ripple-carry adder over committed F_2 bit-slices:
//     s_0 = a_0 + b_0                    c_1 = a_0 b_0
//     s_i = a_i + b_i + c_i              c_{i+1} = maj(a_i, b_i, c_i)  i=1..6
//     s_7 = a_7 + b_7 + c_7              s_8 = maj(a_7, b_7, c_7)
// (all +/maj over F_2; maj = ab+ac+bc).  32 bit-columns of N rows -- a(8),
// b(8), s(9), carries(7) -- are stacked (column id = top 5 index bits) into
// ONE Binius PCS commitment at TRUE bit width; booleanity is structural (the
// packed commitment can only contain bits).  The 16 constraints are gamma-
// batched into a degree-3 zerocheck (eq * quadratic) over T_128; the 32
// column evals the verifier needs at the sumcheck endpoint are bound to the
// commitment by ONE stacked opening at (zeta, rho_sel) worth
// sum_j eq(rho_sel, j) * finals[j].
//
// Teeth: honest accept; violated sum bit / violated carry / wrong carry-out
// reject; tampered finals, root and PCS binding reject.
// Measurement: committed bytes + commit/prove/verify time vs the SAME 32N
// bit-witness committed the way the Goldilocks prover does today (one gl_t
// per bit, rate-1/4 GPU NTT, 8-byte-leaf SHA-256 Merkle tree -- the actual
// p3_ntt/p3_merkle kernels of the production prover).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <omp.h>
#include "p3_binius_sumcheck.cuh"
#include "p3_ntt.cuh"

static int np_ = 0, nf_ = 0;
static void ck(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) np_++; else nf_++;
}
static uint64_t rs_ = 0x5eed5eed5eed5eedULL;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- the batched adder constraint (columns: 0=eq, 1..8=a, 9..16=b,
//      17..25=s, 26..32=c_1..c_7) ----
struct AdderCtx { bf128_t g[16]; };
static bf128_t C_adder(const bf128_t* w, const void* vctx) {
    const AdderCtx* ctx = (const AdderCtx*)vctx;
    const bf128_t *A = w + 1, *B = w + 9, *S = w + 17, *Cin = w + 26;  // Cin[i] = c_{i+1}
    bf128_t acc = bf128_zero();
    for (int i = 0; i < 8; i++) {                     // sum bits
        bf128_t k = bf128_add(S[i], bf128_add(A[i], B[i]));
        if (i >= 1) k = bf128_add(k, Cin[i - 1]);
        acc = bf128_add(acc, bf128_mul(ctx->g[i], k));
    }
    for (int i = 0; i < 8; i++) {                     // carry chain
        bf128_t ci = (i == 0) ? bf128_zero() : Cin[i - 1];
        bf128_t maj = bf128_mul(A[i], B[i]);
        maj = bf128_add(maj, bf128_mul(bf128_add(A[i], B[i]), ci));   // ab + (a+b)c
        bf128_t out = (i < 7) ? Cin[i] : S[8];
        acc = bf128_add(acc, bf128_mul(ctx->g[8 + i], bf128_add(out, maj)));
    }
    return bf128_mul(w[0], acc);
}

// stacked witness: bits[col * N + x], cols in the order above (a,b,s,c)
static void gen_witness(int lN, std::vector<uint8_t>& bits, bool valid) {
    size_t N = (size_t)1 << lN;
    bits.assign(32 * N, 0);
    for (size_t x = 0; x < N; x++) {
        uint32_t a = (uint32_t)(rnd() & 0xff), b = (uint32_t)(rnd() & 0xff);
        uint32_t s = a + b, c = 0, carry = 0;
        for (int i = 0; i < 8; i++) {
            bits[(0 + i) * N + x] = (a >> i) & 1;
            bits[(8 + i) * N + x] = (b >> i) & 1;
            uint32_t ai = (a >> i) & 1, bi = (b >> i) & 1;
            carry = (ai & bi) | (ai & carry) | (bi & carry);
            if (i < 7) { c |= carry << (i + 1); bits[(25 + i) * N + x] = carry; }
        }
        (void)c;
        for (int i = 0; i < 9; i++) bits[(16 + i) * N + x] = (s >> i) & 1;
    }
    (void)valid;
}

struct E2EProof {
    uint8_t root[32];
    BfScProof sc;
    BfPcsProof pcs;
    size_t bytes() const {
        return 32 + sc.rounds.size() * 16 + sc.finals.size() * 16 + pcs.bytes();
    }
};
struct E2EStats { double commit_ms, sc_ms, open_ms, verify_ms; size_t committed; };

static const int NCOL = 32, K = 33, DEG = 3;

static void e2e_prove(int lN, const std::vector<uint8_t>& bits, E2EProof& pf, E2EStats& st) {
    size_t N = (size_t)1 << lN;
    BfPcsParams p;
    p.l = lN + 5; p.lcol = p.l / 2; p.lrow = p.l - p.lcol; p.Q = 100;
    fs::Transcript tr("binius-e2e");
    tr.absorb("stmt-adder8", &lN, sizeof lN);
    BfPcsCommit C;
    double t0 = now_ms();
    bfpcs_commit(p, bits.data(), tr, C);
    st.commit_ms = now_ms() - t0; st.committed = C.committed_bytes;
    memcpy(pf.root, C.root, 32);
    // zerocheck challenges + columns
    std::vector<bf128_t> rz(lN);
    for (auto& x : rz) x = bf_chal128(tr);
    AdderCtx ctx; ctx.g[0] = bf128_one();
    bf128_t gamma = bf_chal128(tr);
    for (int j = 1; j < 16; j++) ctx.g[j] = bf128_mul(ctx.g[j - 1], gamma);
    t0 = now_ms();
    std::vector<std::vector<bf128_t>> cols(K);
    bf_eq_table(rz.data(), lN, cols[0]);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < NCOL; j++) {
        cols[1 + j].resize(N);
        for (size_t x = 0; x < N; x++)
            cols[1 + j][x] = bits[(size_t)j * N + x] ? bf128_one() : bf128_zero();
    }
    std::vector<bf128_t> zeta;
    bf_sumcheck_prove(lN, K, cols, DEG, C_adder, &ctx, tr, pf.sc, zeta);
    st.sc_ms = now_ms() - t0;
    // stacked opening at (zeta, rho_sel)
    std::vector<bf128_t> rho(5), eqsel, R(p.l);
    for (auto& x : rho) x = bf_chal128(tr);
    bf_eq_table(rho.data(), 5, eqsel);
    bf128_t V = bf128_zero();
    for (int j = 0; j < NCOL; j++)
        V = bf128_add(V, bf128_mul(eqsel[j], pf.sc.finals[1 + j]));
    for (int t = 0; t < lN; t++) R[t] = zeta[t];
    for (int t = 0; t < 5; t++) R[lN + t] = rho[t];
    t0 = now_ms();
    bfpcs_open(C, R.data(), V, tr, pf.pcs);
    st.open_ms = now_ms() - t0;
}

static bool e2e_verify(int lN, const E2EProof& pf, E2EStats* st = nullptr) {
    double t0 = now_ms();
    BfPcsParams p;
    p.l = lN + 5; p.lcol = p.l / 2; p.lrow = p.l - p.lcol; p.Q = 100;
    fs::Transcript tr("binius-e2e");
    tr.absorb("stmt-adder8", &lN, sizeof lN);
    tr.absorb("bfpcs-root", pf.root, 32);
    std::vector<bf128_t> rz(lN);
    for (auto& x : rz) x = bf_chal128(tr);
    AdderCtx ctx; ctx.g[0] = bf128_one();
    bf128_t gamma = bf_chal128(tr);
    for (int j = 1; j < 16; j++) ctx.g[j] = bf128_mul(ctx.g[j - 1], gamma);
    if ((int)pf.sc.finals.size() != K) return false;
    std::vector<bf128_t> zeta; bf128_t E;
    if (!bf_sumcheck_verify(pf.sc, bf128_zero(), tr, zeta, &E)) return false;
    if (!bf128_eq(E, C_adder(pf.sc.finals.data(), &ctx))) return false;
    // eq column is public: recompute eq(rz, zeta) = prod (1 + rz_t + zeta_t)
    bf128_t eqv = bf128_one();
    for (int t = 0; t < lN; t++)
        eqv = bf128_mul(eqv, bf128_add(bf128_one(), bf128_add(rz[t], zeta[t])));
    if (!bf128_eq(pf.sc.finals[0], eqv)) return false;
    // bind the 32 column evals to the commitment via one stacked opening
    std::vector<bf128_t> rho(5), eqsel, R(p.l);
    for (auto& x : rho) x = bf_chal128(tr);
    bf_eq_table(rho.data(), 5, eqsel);
    bf128_t V = bf128_zero();
    for (int j = 0; j < NCOL; j++)
        V = bf128_add(V, bf128_mul(eqsel[j], pf.sc.finals[1 + j]));
    for (int t = 0; t < lN; t++) R[t] = zeta[t];
    for (int t = 0; t < 5; t++) R[lN + t] = rho[t];
    bool ok = bfpcs_verify(p, pf.root, R.data(), V, tr, pf.pcs);
    if (st) st->verify_ms = now_ms() - t0;
    return ok;
}

// ---- Goldilocks baseline: the SAME 32N bit-witness committed the way the
// production prover commits it today (one gl_t per bit value, zero-padded
// rate-1/4 GPU NTT, 8-byte-leaf GPU SHA-256 Merkle) ----
static void gl_baseline(int lN, const std::vector<uint8_t>& bits,
                        double& ms, size_t& bytes, uint8_t root[32]) {
    int logM = lN + 5 + 2;
    size_t M = (size_t)1 << logM, nmsg = (size_t)32 << lN;
    std::vector<gl_t> msg(M, 0);
    for (size_t i = 0; i < nmsg; i++) msg[i] = bits[i];
    gl_t *d_msg, *d_cw; uint8_t *d_a, *d_b;
    cudaMalloc(&d_msg, M * sizeof(gl_t)); cudaMalloc(&d_cw, M * sizeof(gl_t));
    cudaMalloc(&d_a, M * 32); cudaMalloc(&d_b, M * 32);
    P3Ntt ntt((uint32_t)logM);
    cudaDeviceSynchronize();
    double t0 = now_ms();
    cudaMemcpy(d_msg, msg.data(), M * sizeof(gl_t), cudaMemcpyHostToDevice);
    ntt.run(d_msg, d_cw, true);
    p3_merkle_build(d_cw, (uint32_t)M, d_a, d_b, root);
    cudaDeviceSynchronize();
    ms = now_ms() - t0;
    bytes = M * sizeof(gl_t) + (2 * M - 1) * 32;      // codeword + tree nodes
    cudaFree(d_msg); cudaFree(d_cw); cudaFree(d_a); cudaFree(d_b);
}

int main(int argc, char** argv) {
    int lN_teeth = 14, lN_meas = (argc > 1) ? atoi(argv[1]) : 18;
#ifndef __CUDA_ARCH__
    bf16_tab();                                   // one-time log/exp build
#endif
    #pragma omp parallel
    { volatile int warm = omp_get_thread_num(); (void)warm; }
    // ---- teeth at lN=14 (16k additions) ----
    {
        int lN = lN_teeth;
        size_t N = (size_t)1 << lN;
        std::vector<uint8_t> bits;
        gen_witness(lN, bits, true);
        E2EProof pf; E2EStats st;
        e2e_prove(lN, bits, pf, st);
        E2EStats sv;
        ck("honest 8-bit adder witness accepts (16k rows)", e2e_verify(lN, pf, &sv));
        { auto b2 = bits; b2[(size_t)19 * N + 777] ^= 1;      // s_3 of row 777
          E2EProof q; E2EStats s2; e2e_prove(lN, b2, q, s2);
          ck("ONE violated sum bit (s_3, 1 row of 16k) rejects", !e2e_verify(lN, q)); }
        { auto b2 = bits; b2[(size_t)28 * N + 1234] ^= 1;     // c_4 of row 1234
          E2EProof q; E2EStats s2; e2e_prove(lN, b2, q, s2);
          ck("ONE violated carry (c_4, 1 row) rejects", !e2e_verify(lN, q)); }
        { auto b2 = bits; b2[(size_t)24 * N + 4095] ^= 1;     // s_8 carry-out
          E2EProof q; E2EStats s2; e2e_prove(lN, b2, q, s2);
          ck("wrong carry-out (s_8, 1 row) rejects", !e2e_verify(lN, q)); }
        { auto q = pf; q.sc.finals[5].lo ^= 1;
          ck("tampered column eval (finals) rejects", !e2e_verify(lN, q)); }
        { auto q = pf; q.root[7] ^= 1;
          ck("tampered commitment root rejects", !e2e_verify(lN, q)); }
        { auto q = pf; q.pcs.t[3].hi ^= 2;
          ck("tampered PCS opening rejects", !e2e_verify(lN, q)); }
    }
    // ---- measured A/B vs Goldilocks at lN_meas ----
    for (int lN : {16, lN_meas}) {
        size_t N = (size_t)1 << lN;
        std::vector<uint8_t> bits;
        gen_witness(lN, bits, true);
        E2EProof pf; E2EStats st;
        e2e_prove(lN, bits, pf, st);
        E2EStats sv;
        bool ok = e2e_verify(lN, pf, &sv);
        char msg[128];
        snprintf(msg, sizeof msg, "measured run lN=%d (%zu additions, 2^%d witness bits) accepts",
                 lN, N, lN + 5);
        ck(msg, ok);
        double glms; size_t glbytes; uint8_t glroot[32];
        gl_baseline(lN, bits, glms, glbytes, glroot);
        double bin_prove = st.commit_ms + st.sc_ms + st.open_ms;
        printf("  [meas] lN=%d  BINIUS committed %.2f MB (commit %.0f ms) | "
               "prove total %.0f ms (sc %.0f, open %.0f) | proof %.1f KB | verify %.0f ms\n",
               lN, st.committed / 1048576.0, st.commit_ms, bin_prove, st.sc_ms,
               st.open_ms, pf.bytes() / 1024.0, sv.verify_ms);
        printf("  [meas] lN=%d  GOLDILOCKS committed %.2f MB (commit %.0f ms; "
               "same witness, gl_t-per-bit rate-1/4 NTT + 8B-leaf Merkle)\n",
               lN, glbytes / 1048576.0, glms);
        printf("  [meas] lN=%d  committed-data ratio %.1fx  commit-time ratio %.1fx\n",
               lN, (double)glbytes / st.committed, glms / st.commit_ms * 1.0);
    }

    printf("\nBINIUS-E2E: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
