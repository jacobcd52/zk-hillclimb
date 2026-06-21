// P3.3 selftest: FRI low-degree test -- honest accept + every tamper rejects.
//   ./p3_fri_selftest
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
using namespace p3fri;

static uint64_t rng_s = 0xC0FFEE;
static uint64_t rng() { rng_s = rng_s*6364136223846793005ULL+1442695040888963407ULL; uint64_t z=rng_s; z^=z>>31; return z; }

// RS-encode N0 coeffs onto the size-2^logM0 subgroup (Horner per point).
static std::vector<gl_t> rs_encode(const std::vector<gl_t>& coeff, uint32_t logM0) {
    uint32_t M0 = 1u << logM0; gl_t w = gl_root_of_unity(logM0);
    std::vector<gl_t> cw(M0);
    for (uint32_t j = 0; j < M0; j++) {
        gl_t xj = gl_pow(w, j), p = 0;
        for (int i = (int)coeff.size() - 1; i >= 0; i--) p = gl_add(gl_mul(p, xj), coeff[i]);
        cw[j] = p;
    }
    return cw;
}

static int npass = 0, nfail = 0;
static void check(const char* name, bool cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
    if (cond) npass++; else nfail++;
}

int main() {
    printf("=== P3.3  FRI low-degree test selftest ===\n");

    // honest accept across a couple of sizes
    for (auto cfg : std::vector<std::pair<uint32_t,uint32_t>>{{6,1},{8,2},{10,1}}) {
        uint32_t logN = cfg.first, R = cfg.second, N0 = 1u << logN;
        std::vector<gl_t> coeff(N0); for (auto& c : coeff) c = rng() % GL_P;
        auto cw = rs_encode(coeff, logN + R);
        auto pf = prove(cw, logN, R, 32);
        const char* why = nullptr;
        char nm[64]; snprintf(nm, sizeof nm, "honest accept logN=%u R=%u", logN, R);
        check(nm, verify(pf, "p3-fri", &why));
        if (why && std::string(why) != "ok") printf("      (why=%s)\n", why);
    }

    // tamper battery on a fixed config
    uint32_t logN = 8, R = 2, N0 = 1u << logN;
    std::vector<gl_t> coeff(N0); for (auto& c : coeff) c = rng() % GL_P;
    auto cw = rs_encode(coeff, logN + R);
    const char* why = nullptr;

    { auto pf = prove(cw, logN, R, 32); check("baseline honest", verify(pf, "p3-fri", &why)); }

    { auto pf = prove(cw, logN, R, 32); pf.queries[0].rounds[0].a = gl_add(pf.queries[0].rounds[0].a, 1);
      check("tamper revealed value -> reject", !verify(pf, "p3-fri", &why)); printf("      why=%s\n", why); }

    { auto pf = prove(cw, logN, R, 32); pf.queries[0].rounds[0].pa[0][0] ^= 1;
      check("tamper merkle path -> reject", !verify(pf, "p3-fri", &why)); printf("      why=%s\n", why); }

    { auto pf = prove(cw, logN, R, 32); pf.final_word.back() = gl_add(pf.final_word.back(), 1);
      check("tamper final (non-constant) -> reject", !verify(pf, "p3-fri", &why)); printf("      why=%s\n", why); }

    { auto pf = prove(cw, logN, R, 32); pf.queries[0].rounds[2].a = gl_add(pf.queries[0].rounds[2].a, 7);
      check("tamper mid-round value -> reject", !verify(pf, "p3-fri", &why)); printf("      why=%s\n", why); }

    { auto pf = prove(cw, logN, R, 32); pf.roots[1][0] ^= 1;
      check("tamper a round root -> reject", !verify(pf, "p3-fri", &why)); printf("      why=%s\n", why); }

    { auto pf = prove(cw, logN, R, 32); check("wrong verifier seed -> reject", !verify(pf, "WRONG", &why)); printf("      why=%s\n", why); }

    // soundness: a genuinely high-degree word, folded honestly, must be rejected
    { std::vector<gl_t> hi(1u << (logN + R)); for (auto& v : hi) v = rng() % GL_P;
      auto pf = prove(hi, logN, R, 64);
      check("high-degree word -> reject", !verify(pf, "p3-fri", &why)); printf("      why=%s\n", why); }

    printf("\nP3.3 FRI: %d passed, %d failed -> %s\n", npass, nfail, nfail==0 ? "ALL PASS" : "FAIL");
    return nfail == 0 ? 0 : 1;
}
