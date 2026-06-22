#include <cstdio>
#include <vector>
#include "p3_basefold.cuh"
#include "fs_transcript.hpp"

static gl_t rnd(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    uint64_t z = s ^ (s >> 31);
    return z % GL_P;
}

int main() {
    const uint32_t base_log = 9;        // real slice size 512
    const uint32_t v = base_log + 1;    // + ex variable
    const uint32_t R = 1;
    const uint32_t Q = 20;
    const uint32_t n = 1u << base_log;
    const uint32_t N = 1u << v;

    std::vector<gl_t> real(n), mask(n), c(N);
    uint64_t s = 12345;
    for (uint32_t i = 0; i < n; i++) {
        real[i] = (17ULL * i + 3ULL) % GL_P;
        mask[i] = rnd(s);
        c[i] = real[i];
        c[n + i] = mask[i];
    }

    std::vector<gl_t> z(v);
    for (uint32_t i = 0; i < v; i++) {
        z[i] = rnd(s);
        if (z[i] == 0) z[i] = 7 + i;
    }

    std::vector<gl_t> cw = p3bf::rs_encode(c, R);
    gl_t y = p3bf::eval_h(c, p3bf::build_eq(z));

    const char* seed = "leak-test";
    p3bf::EvalProof pf = p3bf::prove_eval(c, z, y, R, Q, cw, seed);

    // Replay Basefold opening transcript to recover internal alphas.
    fs::Transcript tr(seed);
    tr.absorb("z", pf.z.data(), pf.z.size() * sizeof(gl_t));
    tr.absorb("y", &pf.y, sizeof(gl_t));

    std::vector<gl_t> alpha(v);
    for (uint32_t r = 0; r < v; r++) {
        tr.absorb("root", pf.roots[r].data(), 32);
        tr.absorb("sc", &pf.msgs[r], sizeof(p3bf::SumMsg));
        alpha[r] = p3bf::alpha_from(tr);
    }

    std::vector<gl_t> alpha_prefix(alpha.begin(), alpha.end() - 1);
    std::vector<gl_t> z_prefix(pf.z.begin(), pf.z.end() - 1);

    gl_t E = p3bf::eq_point(alpha_prefix, z_prefix);
    gl_t rex = pf.z.back();
    const p3bf::SumMsg& last = pf.msgs.back();

    gl_t leaked;
    gl_t denom0 = gl_mul(E, gl_sub(1ULL, rex));

    if (denom0 != 0) {
        leaked = gl_mul(last.s0, gl_inv(denom0));
    } else {
        // Fallback, e.g. rex == 1.
        gl_t denom1 = gl_mul(E, rex);
        gl_t three_rex_minus_one = gl_sub(gl_add(rex, gl_add(rex, rex)), 1ULL);
        gl_t denom2 = gl_mul(E, three_rex_minus_one);
        gl_t C1 = gl_mul(last.s1, gl_inv(denom1));
        gl_t twoC1_minus_C0 = gl_mul(last.s2, gl_inv(denom2));
        leaked = gl_sub(gl_add(C1, C1), twoC1_minus_C0);
    }

    gl_t truth = p3bf::eval_h(real, p3bf::build_eq(alpha_prefix));

    printf("leaked=%llu\ntruth =%llu\nmatch=%d\n",
           (unsigned long long)leaked,
           (unsigned long long)truth,
           leaked == truth);

    return leaked == truth ? 0 : 1;
}