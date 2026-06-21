// P3.4 selftest: Basefold multilinear evaluation opening.
//   ./p3_basefold_selftest
// honest accept (eval proven) + reject on wrong value / tampered sumcheck /
// tampered codeword / tampered final / wrong opening point.
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
#include "p3_basefold.cuh"
using namespace p3bf;

static uint64_t rng_s = 0xBEEF1234;
static uint64_t rng() { rng_s = rng_s*6364136223846793005ULL+1442695040888963407ULL; uint64_t z=rng_s; z^=z>>31; return z; }

static std::vector<gl_t> rs_encode_ref(const std::vector<gl_t>& coeff, uint32_t logM0) {
    uint32_t M0 = 1u << logM0; gl_t w = gl_root_of_unity(logM0);
    std::vector<gl_t> cw(M0);
    for (uint32_t j = 0; j < M0; j++) {
        gl_t xj = gl_pow(w, j), p = 0;
        for (int i = (int)coeff.size() - 1; i >= 0; i--) p = gl_add(gl_mul(p, xj), coeff[i]);
        cw[j] = p;
    }
    return cw;
}

static int npass=0, nfail=0;
static void check(const char* n, bool c){ printf("  [%s] %s\n", c?"PASS":"FAIL", n); if(c)npass++; else nfail++; }

int main() {
    printf("=== P3.4  Basefold evaluation opening selftest ===\n");

    for (auto cfg : std::vector<std::pair<uint32_t,uint32_t>>{{4,1},{6,2},{8,1}}) {
        uint32_t v = cfg.first, R = cfg.second, N = 1u << v;
        std::vector<gl_t> c(N); for (auto& x : c) x = rng() % GL_P;
        std::vector<gl_t> z(v); for (auto& x : z) x = rng() % GL_P;
        gl_t y = eval_h(c, build_eq(z));
        auto cw = rs_encode_ref(c, v + R);
        auto pf = prove_eval(c, z, y, R, 32, cw);
        const char* why=nullptr; char nm[64]; snprintf(nm,sizeof nm,"honest open v=%u R=%u",v,R);
        check(nm, verify_eval(pf,"p3-bf",&why)); if(why && std::string(why)!="ok") printf("      why=%s\n",why);
    }

    // tamper battery
    uint32_t v=8, R=2, N=1u<<v;
    std::vector<gl_t> c(N); for (auto& x : c) x = rng()%GL_P;
    std::vector<gl_t> z(v); for (auto& x : z) x = rng()%GL_P;
    gl_t y = eval_h(c, build_eq(z));
    auto cw = rs_encode_ref(c, v+R);
    const char* why=nullptr;

    { auto pf=prove_eval(c,z,y,R,32,cw); check("baseline honest", verify_eval(pf,"p3-bf",&why)); }

    { auto pf=prove_eval(c,z,gl_add(y,1),R,32,cw);   // prove a wrong value
      check("wrong claimed value -> reject", !verify_eval(pf,"p3-bf",&why)); printf("      why=%s\n",why); }

    { auto pf=prove_eval(c,z,y,R,32,cw); pf.msgs[0].s0 = gl_add(pf.msgs[0].s0,1);
      check("tamper sumcheck msg -> reject", !verify_eval(pf,"p3-bf",&why)); printf("      why=%s\n",why); }

    { auto pf=prove_eval(c,z,y,R,32,cw); pf.queries[0].rounds[0].a = gl_add(pf.queries[0].rounds[0].a,1);
      check("tamper codeword value -> reject", !verify_eval(pf,"p3-bf",&why)); printf("      why=%s\n",why); }

    { auto pf=prove_eval(c,z,y,R,32,cw); pf.queries[0].rounds[1].pa[0][0] ^= 1;
      check("tamper merkle path -> reject", !verify_eval(pf,"p3-bf",&why)); printf("      why=%s\n",why); }

    { auto pf=prove_eval(c,z,y,R,32,cw); pf.final_word[0] = gl_add(pf.final_word[0],1);
      check("tamper final constant -> reject", !verify_eval(pf,"p3-bf",&why)); printf("      why=%s\n",why); }

    { auto pf=prove_eval(c,z,y,R,32,cw); pf.z[0] = gl_add(pf.z[0],1);   // verifier-visible z changed
      check("inconsistent opening point -> reject", !verify_eval(pf,"p3-bf",&why)); printf("      why=%s\n",why); }

    printf("\nP3.4 BASEFOLD: %d passed, %d failed -> %s\n", npass, nfail, nfail==0?"ALL PASS":"FAIL");
    return nfail==0?0:1;
}
