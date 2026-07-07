#include "p3_private_fc.cuh"
#include <iostream>
#include <vector>

static std::vector<gl_t> rand_vec(size_t n, uint64_t& s) {
    std::vector<gl_t> v(n);
    for (auto& x : v) x = p3pfc::prng(s);
    return v;
}

static std::vector<gl_t> augment(const std::vector<gl_t>& real,
                                 const std::vector<gl_t>& mask) {
    std::vector<gl_t> h(2 * real.size());
    for (size_t i = 0; i < real.size(); ++i) {
        h[i] = real[i];
        h[real.size() + i] = mask[i];
    }
    return h;
}

int main() {
    const uint32_t bb = 5, ii = 5, oo = 5;
    const uint32_t R = 1, Q = 20;
    const uint32_t B = 1u << bb, IN = 1u << ii, OUT = 1u << oo;

    // False statement: X = 0, W = 0, but Y[0] = 1.
    std::vector<gl_t> X(B * IN, 0), W(IN * OUT, 0), Y(B * OUT, 0);
    Y[0] = 1;

    uint64_t s = 1234567;
    auto RX = rand_vec(X.size(), s);
    auto RW = rand_vec(W.size(), s);
    auto RY = rand_vec(Y.size(), s);

    auto Xh = augment(X, RX);
    auto Wh = augment(W, RW);
    auto Yh = augment(Y, RY);

    p3pfc::Proof pf{};
    pf.bb = bb; pf.ii = ii; pf.oo = oo; pf.R = R; pf.Q = Q;

    std::vector<gl_t> cwX, cwW, cwY;
    pf.rootX = p3bf::commit(Xh, R, cwX);
    pf.rootW = p3bf::commit(Wh, R, cwW);
    pf.rootY = p3bf::commit(Yh, R, cwY);

    fs::Transcript tr("p3-pfc");
    tr.absorb("rX", pf.rootX.data(), 32);
    tr.absorb("rW", pf.rootW.data(), 32);
    tr.absorb("rY", pf.rootY.data(), 32);

    auto r_i = p3pfc::chal_vec(tr, bb);
    auto r_k = p3pfc::chal_vec(tr, oo);

    // Forge the masked sumcheck transcript.
    pf.Sq = 0;
    tr.absorb("Sq", &pf.Sq, sizeof pf.Sq);
    gl_t rho = p3pfc::chal(tr);
    if (rho == 0) {
        std::cerr << "rho=0; rerun with different roots/masks\n";
        return 3;
    }

    gl_t claim = gl_mul(rho, pf.Sq); // zero
    std::vector<gl_t> r;
    const uint32_t V = ii + 3;

    for (uint32_t rd = 0; rd < V; ++rd) {
        // Any quadratic with s0+s1 = claim is accepted by the chain.
        // With claim=0, all-zero messages keep claim zero forever.
        p3pfc::SumMsg m{0, claim, 0};
        pf.msgs.push_back(m);
        tr.absorb("sc", &m, sizeof m);
        gl_t a = p3pfc::chal(tr);
        r.push_back(a);
        claim = p3pfc::quad_eval(m.s0, m.s1, m.s2, a);
    }

    std::vector<gl_t> r_j(r.begin(), r.begin() + ii);
    gl_t rexX = r[ii], rexW = r[ii + 1], rey = r[ii + 2];

    auto zX = p3pfc::cat(p3pfc::cat(r_j, r_i), std::vector<gl_t>{rexX});
    auto zW = p3pfc::cat(p3pfc::cat(r_k, r_j), std::vector<gl_t>{rexW});
    auto zY = p3pfc::cat(p3pfc::cat(r_k, r_i), std::vector<gl_t>{rey});

    gl_t yX = p3bf::eval_h(Xh, p3bf::build_eq(zX));
    gl_t yW = p3bf::eval_h(Wh, p3bf::build_eq(zW));
    gl_t yY = p3bf::eval_h(Yh, p3bf::build_eq(zY));

    pf.openX = p3bf::prove_eval(Xh, zX, yX, R, Q, cwX, "pfc-X");
    pf.openW = p3bf::prove_eval(Wh, zW, yW, R, Q, cwW, "pfc-W");
    pf.openY = p3bf::prove_eval(Yh, zY, yY, R, Q, cwY, "pfc-Y");

    gl_t mX = pf.openX.y, mW = pf.openW.y, mY = pf.openY.y;
    gl_t c2 = (gl_t)((2ull * IN) % GL_P);

    gl_t term1 = gl_mul(gl_sub(1ULL, rey), mY);
    gl_t term2 = gl_mul(
        gl_mul(gl_sub(1ULL, rexX), gl_sub(1ULL, rexW)),
        gl_mul(mX, mW)
    );
    gl_t summand_final = gl_sub(term1, gl_mul(c2, term2));

    // Free variable: choose qr to force final tie.
    pf.qr = gl_mul(gl_sub(claim, summand_final), gl_inv(rho));

    const char* why = nullptr;
    bool ok = p3pfc::verify(pf, Q, R, &why);

    std::cout << "false statement: X=0, W=0, Y[0]=1\n";
    std::cout << "verify returned " << ok << ", why=" << (why ? why : "(null)") << "\n";
    return ok ? 0 : 2;
}