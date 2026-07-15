// p3_int_selftest.cu -- gadget battery for the INTEGER layer gadget set.
//
//   nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp -I. p3_int_selftest.cu -o /root/p3_int_selftest
//   cd /root/zkllm && /root/p3_int_selftest
//
// Every gadget: honest accept (bitwise vs the canonical replay; cross-checked
// against int_layer_ref.py at the layer level) + adversarial rejects with
// per-gadget teeth, in BOTH non-zk and zk modes.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_int_gadgets.cuh"
using std::vector;
using namespace p3ig;

static int np_ = 0, nf_ = 0;
static void ck(const char* n, bool c) {
    printf("  [%s] %s\n", c ? "PASS" : "FAIL", n);
    if (c) np_++; else nf_++;
}

static uint64_t rng_s = 42;
static inline uint64_t rnd64() {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s;
}
static inline int64_t rnds(int64_t bound) {     // uniform in (-bound, bound)
    return (int64_t)(rnd64() % (2 * (uint64_t)bound - 1)) - (bound - 1);
}

static const uint32_t R = 2, Q = 24;

// ---------------- rescale ----------------
static void test_rescale(const Tables& T, bool zk) {
    const char* zs = zk ? " [zk]" : "";
    struct Case { uint32_t le, sf; int rng; int64_t abound; const char* nm; };
    Case cases[] = {
        {8, 16, p3irs::RNG_S20, 1LL << 34, "sf16/S20"},
        {7, 27, p3irs::RNG_S16, 1LL << 41, "sf27/S16"},
        {7, 14, p3irs::RNG_S20, 1LL << 32, "sf14/S20"},
    };
    for (auto& c : cases) {
        vector<gl_t> acc((size_t)1 << c.le);
        for (auto& a : acc) a = gsig(rnds(c.abound));
        p3irs::Wit w = p3irs::gen_witness(c.le, c.sf, c.rng, acc);
        Col CX = commit_col_nc(w.acc, R), CY = commit_col_nc(w.y, R);
        p3irs::Operands o{&CX, &CY};
        fs::Transcript tp("irs-t");
        p3irs::Proof pf = p3irs::prove(tp, w, T, o, R, Q);
        fs::Transcript tv("irs-t");
        const char* why = nullptr;
        bool ok = p3irs::verify(tv, T, pf, CX.root, CY.root, c.le, c.sf, c.rng,
                                Q, R, &why);
        char nm[96]; snprintf(nm, sizeof nm, "rescale %s honest accepts%s (%s)", c.nm, zs, why);
        ck(nm, ok);
    }
    // adversarial: each tamper must be rejected (prover throw counts as reject)
    struct TCase { int mode; const char* nm; };
    TCase tc[] = {{p3irs::IT_SHIFT, "y-shift (rem out of table)"},
                  {p3irs::IT_LIMB, "limb flip (zero-check)"},
                  {p3irs::IT_RANGE, "y out of window (range lookup)"}};
    for (auto& t : tc) {
        vector<gl_t> acc(1 << 8);
        for (auto& a : acc) a = gsig(rnds(1LL << 34));
        p3irs::Tamper tm{t.mode, 5};
        bool rejected = false;
        const char* why = nullptr;
        try {
            p3irs::Wit w = p3irs::gen_witness(8, 16, p3irs::RNG_S20, acc, &tm);
            Col CX = commit_col_nc(w.acc, R), CY = commit_col_nc(w.y, R);
            p3irs::Operands o{&CX, &CY};
            fs::Transcript tp("irs-t");
            p3irs::Proof pf = p3irs::prove(tp, w, T, o, R, Q, /*strict=*/false);
            fs::Transcript tv("irs-t");
            rejected = !p3irs::verify(tv, T, pf, CX.root, CY.root, 8, 16,
                                      p3irs::RNG_S20, Q, R, &why);
        } catch (const std::exception&) { rejected = true; }
        char nm[96]; snprintf(nm, sizeof nm, "rescale rejects %s%s", t.nm, zs);
        ck(nm, rejected);
    }
}

// ---------------- matmul ----------------
// commit Y for a set of (view, operands) instances with the zk mask linkage
static Col commit_y_linked(const vector<gl_t>& yvals, uint32_t vY,
                           const vector<std::tuple<p3imm::OpView, p3imm::OpView,
                                                   p3imm::OpView, uint32_t, uint32_t,
                                                   uint32_t>>& insts) {
    if (!p3zkc::G.on) return commit_col_nc(yvals, R);
    vector<gl_t> m1((size_t)1 << vY, 0);
    for (auto& t : insts)
        p3imm::accum_mask1(m1, std::get<0>(t), std::get<1>(t), std::get<2>(t),
                           std::get<3>(t), std::get<4>(t), std::get<5>(t));
    vector<gl_t> mask = p3zkc::mk_linked(vY, m1);
    return commit_col_nc(yvals, R, &mask);
}

static void test_matmul(const Tables& T, bool zk) {
    (void)T;
    const char* zs = zk ? " [zk]" : "";
    // ---- direct layouts ----
    {
        const uint32_t lj = 4, lk = 3, li = 5;
        vector<gl_t> X((size_t)1 << (lj + li)), W((size_t)1 << (lj + lk));
        for (auto& x : X) x = gsig(rnds(1LL << 19));
        for (auto& x : W) x = gsig(rnds(1LL << 19));
        Col CX = commit_col_nc(X, R), CW = commit_col_nc(W, R);
        p3imm::OpView xv = p3imm::direct_x(&CX, lj, li);
        p3imm::OpView wv = p3imm::direct_w(&CW, lj, lk);
        vector<gl_t> Y = p3imm::compute_y(X, xv, W, wv, lj, lk, li);
        p3imm::OpView yv = p3imm::direct_y(nullptr, lk, li);
        Col CY = commit_y_linked(Y, lk + li, {{xv, wv, yv, lj, lk, li}});
        yv.c = &CY; yv.root = CY.root;
        fs::Transcript tp("imm-t");
        p3imm::Proof pf = p3imm::prove(tp, xv, wv, yv, lj, lk, li, R, Q);
        fs::Transcript tv("imm-t");
        const char* why = nullptr;
        bool ok = p3imm::verify(tv, pf, xv, wv, yv, lj, lk, li, Q, R, &why);
        char nm[96]; snprintf(nm, sizeof nm, "matmul direct honest accepts%s (%s)", zs, why);
        ck(nm, ok);

        // tamper: Y grid value flipped (committed accumulator forged)
        vector<gl_t> Yb = Y; Yb[3] = gl_add(Yb[3], 1ULL);
        Col CYb = commit_y_linked(Yb, lk + li, {{xv, wv, yv, lj, lk, li}});
        p3imm::OpView yvb = p3imm::direct_y(&CYb, lk, li);
        fs::Transcript tp2("imm-t");
        p3imm::Proof pf2 = p3imm::prove(tp2, xv, wv, yvb, lj, lk, li, R, Q, false);
        fs::Transcript tv2("imm-t");
        bool bad = p3imm::verify(tv2, pf2, xv, wv, yvb, lj, lk, li, Q, R, &why);
        snprintf(nm, sizeof nm, "matmul rejects forged Y value%s", zs);
        ck(nm, !bad);

        // tamper: operand swapped (proof over different X than committed... the
        // committed X root is the statement; forge X inside the grid)
        vector<gl_t> Xb = X; Xb[7] = gl_add(Xb[7], 1ULL);
        Col CXb = commit_col_nc(Xb, R);
        p3imm::OpView xvb = p3imm::direct_x(&CXb, lj, li);
        // honest Y from the ORIGINAL X: sub-proof about (Xb, W, Y) must fail
        Col CY2 = commit_y_linked(Y, lk + li, {{xvb, wv, yv, lj, lk, li}});
        p3imm::OpView yv2 = p3imm::direct_y(&CY2, lk, li);
        fs::Transcript tp3("imm-t");
        p3imm::Proof pf3 = p3imm::prove(tp3, xvb, wv, yv2, lj, lk, li, R, Q, false);
        fs::Transcript tv3("imm-t");
        bad = p3imm::verify(tv3, pf3, xvb, wv, yv2, lj, lk, li, Q, R, &why);
        snprintf(nm, sizeof nm, "matmul rejects X/Y mismatch%s", zs);
        ck(nm, !bad);
    }
    // ---- sliced / transposed views into shared grids (the attention shapes) ----
    {
        // Q grid: T=2^5 tokens x d=2^4 cols; head slice h=1 (top d-bit fixed),
        // K as W via transpose view; Y = instance a=1 block of a 2-instance
        // shared scores grid  [k(seq) | i(seq) | a]
        const uint32_t lseq = 5, ldh = 3, lda = 4;
        vector<gl_t> Qg((size_t)1 << (lda + lseq)), Kg((size_t)1 << (lda + lseq));
        for (auto& x : Qg) x = gsig(rnds(1LL << 15));
        for (auto& x : Kg) x = gsig(rnds(1LL << 15));
        Col CQ = commit_col_nc(Qg, R), CK = commit_col_nc(Kg, R);
        // X view: X(i,j) = Qg[(j | h<<ldh) | i<<lda], h = 1
        p3imm::OpView xv; xv.c = &CQ; xv.root = CQ.root; xv.v = lda + lseq;
        for (uint32_t b = 0; b < ldh; b++) xv.sel.push_back({p3imm::S_J, (uint8_t)b});
        xv.sel.push_back({p3imm::S_C, 1});
        for (uint32_t b = 0; b < lseq; b++) xv.sel.push_back({p3imm::S_A, (uint8_t)b});
        // W view: W(j,k) = Kg[(j | h<<ldh) | k<<lda]  (transpose comes free)
        p3imm::OpView wv; wv.c = &CK; wv.root = CK.root; wv.v = lda + lseq;
        for (uint32_t b = 0; b < ldh; b++) wv.sel.push_back({p3imm::S_J, (uint8_t)b});
        wv.sel.push_back({p3imm::S_C, 1});
        for (uint32_t b = 0; b < lseq; b++) wv.sel.push_back({p3imm::S_A, (uint8_t)b});
        vector<gl_t> Yi = p3imm::compute_y(Qg, xv, Kg, wv, ldh, lseq, lseq);
        // shared scores grid, instance a=1 holds Yi, instance 0 random junk
        vector<gl_t> Sg((size_t)1 << (2 * lseq + 1));
        for (auto& x : Sg) x = gsig(rnds(1LL << 30));
        p3imm::OpView yv; yv.v = 2 * lseq + 1;
        for (uint32_t b = 0; b < lseq; b++) yv.sel.push_back({p3imm::S_A, (uint8_t)b});
        for (uint32_t b = 0; b < lseq; b++) yv.sel.push_back({p3imm::S_I, (uint8_t)b});
        yv.sel.push_back({p3imm::S_C, 1});
        for (uint32_t k = 0; k < (1u << lseq); k++)
            for (uint32_t i = 0; i < (1u << lseq); i++)
                Sg[p3imm::y_off(yv, k, i)] = Yi[k | ((size_t)i << lseq)];
        Col CS = commit_y_linked(Sg, 2 * lseq + 1, {{xv, wv, yv, ldh, lseq, lseq}});
        yv.c = &CS; yv.root = CS.root;
        fs::Transcript tp("imm-t2");
        p3imm::Proof pf = p3imm::prove(tp, xv, wv, yv, ldh, lseq, lseq, R, Q);
        fs::Transcript tv("imm-t2");
        const char* why = nullptr;
        bool ok = p3imm::verify(tv, pf, xv, wv, yv, ldh, lseq, lseq, Q, R, &why);
        char nm[110];
        snprintf(nm, sizeof nm, "matmul slice+transpose+shared-Y honest accepts%s (%s)", zs, why);
        ck(nm, ok);

        // tamper inside the claimed instance block
        vector<gl_t> Sb = Sg;
        Sb[p3imm::y_off(yv, 2, 3)] = gl_add(Sb[p3imm::y_off(yv, 2, 3)], 1ULL);
        Col CSb = commit_y_linked(Sb, 2 * lseq + 1, {{xv, wv, yv, ldh, lseq, lseq}});
        p3imm::OpView yvb = yv; yvb.c = &CSb; yvb.root = CSb.root;
        fs::Transcript tp2("imm-t2");
        p3imm::Proof pf2 = p3imm::prove(tp2, xv, wv, yvb, ldh, lseq, lseq, R, Q, false);
        fs::Transcript tv2("imm-t2");
        bool bad = p3imm::verify(tv2, pf2, xv, wv, yvb, ldh, lseq, lseq, Q, R, &why);
        snprintf(nm, sizeof nm, "matmul rejects tamper in instance block%s", zs);
        ck(nm, !bad);
    }
}

// ---------------- residual add ----------------
static void test_add(const Tables& T, bool zk) {
    const char* zs = zk ? " [zk]" : "";
    const uint32_t le = 8;
    vector<gl_t> a(1 << le), b(1 << le);
    for (auto& x : a) x = gsig(rnds(1LL << 17));
    for (auto& x : b) x = gsig(rnds(1LL << 17));
    {
        p3iadd::Wit w = p3iadd::gen_witness(le, a, b);
        Col CA = commit_col_nc(a, R), CB2 = commit_col_nc(b, R), CO = commit_col_nc(w.out, R);
        p3iadd::Operands o{&CA, &CB2, &CO};
        fs::Transcript tp("iad-t");
        p3iadd::Proof pf = p3iadd::prove(tp, w, T, o, R, Q);
        fs::Transcript tv("iad-t");
        const char* why = nullptr;
        bool ok = p3iadd::verify(tv, T, pf, CA.root, CB2.root, CO.root, le, Q, R, &why);
        char nm[96]; snprintf(nm, sizeof nm, "resadd honest accepts%s (%s)", zs, why);
        ck(nm, ok);
    }
    for (int mode : {p3iadd::AT_OUT, p3iadd::AT_RANGE}) {
        vector<gl_t> aa = a, bb = b;
        if (mode == p3iadd::AT_RANGE) { aa[5] = gsig(1LL << 18); bb[5] = gsig((1LL << 18) + 77); }
        p3iadd::Tamper tm{mode, 5};
        bool rejected = false;
        try {
            p3iadd::Wit w = p3iadd::gen_witness(le, aa, bb, &tm);
            Col CA = commit_col_nc(aa, R), CB2 = commit_col_nc(bb, R), CO = commit_col_nc(w.out, R);
            p3iadd::Operands o{&CA, &CB2, &CO};
            fs::Transcript tp("iad-t");
            p3iadd::Proof pf = p3iadd::prove(tp, w, T, o, R, Q, false);
            fs::Transcript tv("iad-t");
            const char* why = nullptr;
            rejected = !p3iadd::verify(tv, T, pf, CA.root, CB2.root, CO.root, le, Q, R, &why);
        } catch (const std::exception&) { rejected = true; }
        char nm[96];
        snprintf(nm, sizeof nm, "resadd rejects %s%s",
                 mode == p3iadd::AT_OUT ? "forged out (zero-check)" : "out-of-range (lookup)", zs);
        ck(nm, rejected);
    }
}

// ---------------- rmsnorm ----------------
static void test_rms(const Tables& T, bool zk) {
    const char* zs = zk ? " [zk]" : "";
    const uint32_t lT = 3, ld = 6;
    vector<gl_t> x((size_t)1 << (lT + ld)), g((size_t)1 << ld);
    for (auto& v : x) v = gsig(rnds(1LL << 17));
    for (auto& v : g) v = gsig(rnds(1LL << 17));
    auto run = [&](const p3irms::Tamper* tm, const char* nm, bool expect_ok) {
        bool ok = false, threw = false;
        const char* why = "?";
        try {
            p3irms::Wit w = p3irms::gen_witness(lT, ld, x, g, T, tm);
            Col CX = commit_col_nc(x, R), CG = commit_col_nc(g, R), CY = commit_col_nc(w.y, R);
            p3irms::Operands o{&CX, &CG, &CY};
            fs::Transcript tp("irm-t");
            p3irms::Proof pf = p3irms::prove(tp, w, T, o, R, Q, expect_ok);
            fs::Transcript tv("irm-t");
            ok = p3irms::verify(tv, T, pf, CX.root, CG.root, CY.root, lT, ld, Q, R, &why);
        } catch (const std::exception& e) { threw = true; why = e.what(); }
        char b[128];
        if (expect_ok) snprintf(b, sizeof b, "rmsnorm honest accepts%s (%s)", zs, why);
        else snprintf(b, sizeof b, "rmsnorm rejects %s%s", nm, zs);
        ck(b, expect_ok ? (ok && !threw) : (!ok || threw));
    };
    run(nullptr, "", true);
    p3irms::Tamper t1{p3irms::RT_M, 2, 0};  run(&t1, "inflated M (row-sum)", false);
    p3irms::Tamper t2{p3irms::RT_R, 2, 0};  run(&t2, "R+1 (limb lookup)", false);
    p3irms::Tamper t3{p3irms::RT_W, 2, 3};  run(&t3, "forged W (zero-check)", false);
    p3irms::Tamper t4{p3irms::RT_Y, 2, 3};  run(&t4, "forged Y (zero-check)", false);
}

// ---------------- rope ----------------
static void test_rope(const Tables& T, bool zk) {
    const char* zs = zk ? " [zk]" : "";
    const uint32_t lT = 4, ld = 5, lseq = 4, ldh = 4;
    p3irope::Pub P; P.lseq = lseq; P.ldh = ldh;
    const uint32_t seq = 1u << lseq, dh2 = 1u << (ldh - 1);
    P.ct.resize((size_t)seq * dh2); P.st.resize((size_t)seq * dh2);
    for (uint32_t s = 0; s < seq; s++)
        for (uint32_t j = 0; j < dh2; j++) {
            double ang = (double)s * pow(10000.0, -2.0 * j / (double)(1u << ldh));
            P.ct[(size_t)s * dh2 + j] = (int32_t)llround(cos(ang) * 16384.0);
            P.st[(size_t)s * dh2 + j] = (int32_t)llround(sin(ang) * 16384.0);
        }
    vector<gl_t> q((size_t)1 << (lT + ld));
    for (auto& v : q) v = gsig(rnds(1LL << 17));
    auto run = [&](const p3irope::Tamper* tm, const char* nm, bool expect_ok) {
        bool ok = false, threw = false;
        const char* why = "?";
        try {
            p3irope::Wit w = p3irope::gen_witness(lT, ld, P, q, tm);
            Col CQ = commit_col_nc(q, R), CY = commit_col_nc(w.y, R);
            p3irope::Operands o{&CQ, &CY};
            fs::Transcript tp("irp-t");
            p3irope::Proof pf = p3irope::prove(tp, w, T, P, o, R, Q, expect_ok);
            fs::Transcript tv("irp-t");
            ok = p3irope::verify(tv, T, P, pf, CQ.root, CY.root, lT, ld, Q, R, &why);
        } catch (const std::exception& e) { threw = true; why = e.what(); }
        char b[128];
        if (expect_ok) snprintf(b, sizeof b, "rope honest accepts%s (%s)", zs, why);
        else snprintf(b, sizeof b, "rope rejects %s%s", nm, zs);
        ck(b, expect_ok ? (ok && !threw) : (!ok || threw));
    };
    run(nullptr, "", true);
    // tamper at position t=3 (t=0 has sin=0, where the rotation is a no-op)
    p3irope::Tamper t1{p3irope::RPT_UNROT, (size_t)(3u << ld) | 7};
    run(&t1, "unrotated operand (flipped-point claim)", false);
    p3irope::Tamper t2{p3irope::RPT_Y, (size_t)(3u << ld) | 7};
    run(&t2, "y-shift (rem lookup)", false);
}

// ---------------- softmax ----------------
static void test_smx(const Tables& T, bool zk) {
    const char* zs = zk ? " [zk]" : "";
    const uint32_t la = 1, lseq = 4;
    vector<gl_t> z((size_t)1 << (la + 2 * lseq));
    for (auto& v : z) v = gsig(rnds(1LL << 13));
    auto run = [&](const p3ismx::Tamper* tm, const char* nm, bool expect_ok) {
        bool ok = false, threw = false;
        const char* why = "?";
        try {
            p3ismx::Wit w = p3ismx::gen_witness(la, lseq, z, T, tm);
            Col CZ = commit_col_nc(z, R), CP = commit_col_nc(w.p, R);
            p3ismx::Operands o{&CZ, &CP};
            fs::Transcript tp("ism-t");
            p3ismx::Proof pf = p3ismx::prove(tp, w, T, o, R, Q, expect_ok);
            fs::Transcript tv("ism-t");
            ok = p3ismx::verify(tv, T, pf, CZ.root, CP.root, la, lseq, Q, R, &why);
        } catch (const std::exception& e) { threw = true; why = e.what(); }
        char b[128];
        if (expect_ok) snprintf(b, sizeof b, "softmax honest accepts%s (%s)", zs, why);
        else snprintf(b, sizeof b, "softmax rejects %s%s", nm, zs);
        ck(b, expect_ok ? (ok && !threw) : (!ok || threw));
    };
    run(nullptr, "", true);
    p3ismx::Tamper t1{p3ismx::ST_MX, 5, 0};  run(&t1, "mx+1 (attainment)", false);
    p3ismx::Tamper t2{p3ismx::ST_E, 5, 2};   run(&t2, "forged exp (EXPT lookup)", false);
    p3ismx::Tamper t3{p3ismx::ST_P, 5, 2};   run(&t3, "P+1 (bracket limbs)", false);
    p3ismx::Tamper t4{p3ismx::ST_S, 5, 0};   run(&t4, "inflated S (row-sum)", false);
}

// ---------------- swiglu ----------------
static void test_swg(const Tables& T, bool zk) {
    const char* zs = zk ? " [zk]" : "";
    const uint32_t le = 8;
    vector<gl_t> g(1 << le), u(1 << le);
    for (auto& v : g) v = gsig(rnds(1LL << 17));
    for (auto& v : u) v = gsig(rnds(1LL << 17));
    auto run = [&](const p3iswg::Tamper* tm, const char* nm, bool expect_ok) {
        bool ok = false, threw = false;
        const char* why = "?";
        try {
            p3iswg::Wit w = p3iswg::gen_witness(le, g, u, T, tm);
            Col CG = commit_col_nc(g, R), CU = commit_col_nc(u, R), CM = commit_col_nc(w.mo, R);
            p3iswg::Operands o{&CG, &CU, &CM};
            fs::Transcript tp("isw-t");
            p3iswg::Proof pf = p3iswg::prove(tp, w, T, o, R, Q, expect_ok);
            fs::Transcript tv("isw-t");
            ok = p3iswg::verify(tv, T, pf, CG.root, CU.root, CM.root, le, Q, R, &why);
        } catch (const std::exception& e) { threw = true; why = e.what(); }
        char b[128];
        if (expect_ok) snprintf(b, sizeof b, "swiglu honest accepts%s (%s)", zs, why);
        else snprintf(b, sizeof b, "swiglu rejects %s%s", nm, zs);
        ck(b, expect_ok ? (ok && !threw) : (!ok || threw));
    };
    run(nullptr, "", true);
    p3iswg::Tamper t1{p3iswg::GT_SIL, 9};  run(&t1, "forged silu (SILU lookup)", false);
    p3iswg::Tamper t2{p3iswg::GT_M, 9};    run(&t2, "forged M (zero-check)", false);
}

int main() {
    printf("=== INT gadget selftest (rescale + matmul) ===\n");
    p3fri::g_gpu_merkle = true;
    p3bf::p3_enable_mempool();
    Tables T = build_tables(64, 11);            // d=64, btop=11 (sf=27)

    for (int zk = 0; zk <= 1; zk++) {
        p3zkc::G.on = zk != 0;
        printf("-- mode %s --\n", zk ? "zk" : "plain");
        test_rescale(T, zk != 0);
        test_matmul(T, zk != 0);
        test_add(T, zk != 0);
        test_rms(T, zk != 0);
        test_rope(T, zk != 0);
        test_smx(T, zk != 0);
        test_swg(T, zk != 0);
    }
    p3zkc::G.on = false;
    printf("\nINT-GADGETS: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
