// Selftest for p3_binius_pcs.cuh (design doc section 21.3).
// Teeth: honest commit/open/verify accepts at two sizes; every tampering
// vector (claimed value, eval row, proximity row, column data, Merkle path,
// wrong point) rejects; a CHEATING PROVER that lies about a single witness
// bit relative to the committed codeword is caught by the spot-check
// consistency test; a codeword corrupted after encoding (tree rebuilt
// honestly, so Merkle passes) is caught by the same test.  Also checks the
// T_128 limb-wise re-encode used by the verifier against the T_16 encode.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include "p3_binius_pcs.cuh"

static int np_ = 0, nf_ = 0;
static void ck(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) np_++; else nf_++;
}
static uint64_t rs_ = 0x0123456789abcdefULL;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }
static bf128_t rnd128() { return {rnd(), rnd()}; }

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    // ---- T_128 limb-wise encode == T_16 encode (embedded + linearity) ----
    {
        int m = 8; uint32_t n = 1u << m;
        BfNtt nt; bfntt_init(nt, m);
        std::vector<bf16_t> f(n);
        std::vector<bf128_t> F(n);
        for (uint32_t i = 0; i < n; i++) { f[i] = (bf16_t)(rnd() & 0xffff); F[i] = bf128_from16(f[i]); }
        bfntt_fwd_host(nt, f.data());
        bfntt_fwd_host128(nt, F.data());
        uint32_t bad = 0;
        for (uint32_t i = 0; i < n; i++)
            if (!bf128_eq(F[i], bf128_from16(f[i]))) bad++;
        ck("Enc128(embedded T_16 row) == embed(Enc16(row))", bad == 0);
        // T_128-combination commutes with encoding
        const int R = 5;
        std::vector<std::vector<bf16_t>> rows(R, std::vector<bf16_t>(n));
        std::vector<bf128_t> comb(n, bf128_zero()), c(R);
        for (int i = 0; i < R; i++) {
            c[i] = rnd128();
            for (uint32_t j = 0; j < n; j++) rows[i][j] = (bf16_t)(rnd() & 0xffff);
            for (uint32_t j = 0; j < n; j++)
                comb[j] = bf128_add(comb[j], bf128_smul16(c[i], rows[i][j]));
        }
        bfntt_fwd_host128(nt, comb.data());
        for (int i = 0; i < R; i++) bfntt_fwd_host(nt, rows[i].data());
        bad = 0;
        for (uint32_t j = 0; j < n; j++) {
            bf128_t s = bf128_zero();
            for (int i = 0; i < R; i++) s = bf128_add(s, bf128_smul16(c[i], rows[i][j]));
            if (!bf128_eq(s, comb[j])) bad++;
        }
        ck("Enc128(sum c_i * row_i) == sum c_i * Enc16(row_i)", bad == 0);
    }
    // ---- honest accept at two shapes + full tamper battery at l=20 ----
    for (int cfg = 0; cfg < 2; cfg++) {
        BfPcsParams p;
        p.l = cfg ? 20 : 16;
        p.lrow = cfg ? 10 : 8;
        p.lcol = p.l - p.lrow;
        p.Q = 100;
        size_t n = (size_t)1 << p.l;
        std::vector<uint8_t> bits(n);
        for (auto& b : bits) b = (uint8_t)(rnd() & 1);
        std::vector<bf128_t> r(p.l);
        for (auto& x : r) x = rnd128();
        bf128_t v = bf_ml_eval_bits(bits.data(), p.l, r.data());

        fs::Transcript tp("bfpcs-test");
        BfPcsCommit C;
        bfpcs_commit(p, bits.data(), tp, C);
        BfPcsProof pf;
        double t0 = now_ms();
        bfpcs_open(C, r.data(), v, tp, pf);
        double t_open = now_ms() - t0;
        auto vfy = [&](const BfPcsProof& q, bf128_t val, const bf128_t* pt) {
            fs::Transcript tv("bfpcs-test");
            tv.absorb("bfpcs-root", C.root, 32);
            return bfpcs_verify(p, C.root, pt, val, tv, q);
        };
        t0 = now_ms();
        bool ok = vfy(pf, v, r.data());
        double t_vfy = now_ms() - t0;
        char msg[128];
        snprintf(msg, sizeof msg, "honest accept l=%d (2^%d bits; commit %.1f ms, "
                 "committed %.2f MB, proof %.1f KB, open %.1f ms, verify %.1f ms)",
                 p.l, p.l, C.commit_ms, C.committed_bytes / 1048576.0,
                 pf.bytes() / 1024.0, t_open, t_vfy);
        ck(msg, ok);
        if (cfg == 0) continue;

        { auto q = pf; q.t[rnd() % q.t.size()].lo ^= 1;
          ck("tampered eval row t rejects", !vfy(q, v, r.data())); }
        { auto q = pf; q.u[rnd() % q.u.size()].hi ^= 1;
          ck("tampered proximity row u rejects", !vfy(q, v, r.data())); }
        { auto q = pf; q.cols[rnd() % q.cols.size()] ^= 1;
          ck("tampered column data rejects (Merkle)", !vfy(q, v, r.data())); }
        { auto q = pf; q.paths[rnd() % q.paths.size()] ^= 1;
          ck("tampered Merkle path rejects", !vfy(q, v, r.data())); }
        { bf128_t v2 = v; v2.lo ^= 1;
          ck("wrong claimed value rejects", !vfy(pf, v2, r.data())); }
        { auto r2 = r; r2[3].lo ^= 1;
          ck("wrong evaluation point rejects", !vfy(pf, v, r2.data())); }
        // cheating prover: witness bit flipped AFTER commitment; t/u/value are
        // internally consistent for the modified witness, columns are not
        {
            auto bits2 = bits;
            bits2[rnd() % n] ^= 1;
            bf128_t v2 = bf_ml_eval_bits(bits2.data(), p.l, r.data());
            BfPcsCommit C2 = C;              // committed codeword/tree unchanged
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < (int64_t)C.n_rows; i++)
                bf_pack_bits(bits2.data() + ((size_t)i << p.lcol), (size_t)1 << p.lcol,
                             C2.msg.data() + (size_t)i * C2.pc);
            fs::Transcript t2("bfpcs-test");
            t2.absorb("bfpcs-root", C.root, 32);
            BfPcsProof q;
            bfpcs_open(C2, r.data(), v2, t2, q);
            ck("cheating prover (1 witness bit flipped vs commitment) rejects",
               !vfy(q, v2, r.data()));
        }
        // corrupted codeword with an honestly rebuilt tree: Merkle passes,
        // spot-check consistency must catch it
        {
            BfPcsCommit C3 = C;
            for (uint32_t j = 0; j < C.nc; j += 2)
                C3.cw[j] = (bf16_t)(rnd() & 0xffff);     // wreck half of row 0
            bfpcs_tree(C3);
            fs::Transcript t3("bfpcs-test");
            t3.absorb("bfpcs-root", C3.root, 32);
            BfPcsProof q;
            bfpcs_open(C3, r.data(), v, t3, q);
            fs::Transcript tv("bfpcs-test");
            tv.absorb("bfpcs-root", C3.root, 32);
            ck("corrupted codeword (honest tree rebuild) rejects",
               !bfpcs_verify(p, C3.root, r.data(), v, tv, q));
        }
    }

    printf("\nBINIUS-PCS: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
