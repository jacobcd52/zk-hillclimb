// COORDINATOR-BUILT (do not let submission agents edit).
// 1) Demonstrates the me_open STEERING ATTACK: upstream's fold-opening protocol
//    is forgeable because the fold coefficients (the evaluation point) are known
//    to the prover BEFORE it constructs the otherwise-unconstrained T0/T1 points.
//    Attack: at the last round set T1' = T1 + u^{-2} * Delta with
//    Delta = G_final * (eval' - eval). Then C_L shifts by exactly Delta and the
//    §7 verify chain (vrf_toy_open) ACCEPTS the forged eval'. Unsound even
//    interactively, since open() takes the whole challenge vector as input.
// 2) Pins the sound replacement: a Bulletproofs-style inner-product argument
//    (IPA) over the GEN-sized within-row vector, with per-round Fiat-Shamir
//    challenges derived AFTER absorbing that round's (L, R).
//
//    Statement: public (com rows, u_out, u_in, eval, pp = {g, Q}).
//      C_0  = raw-limb ME fold of com rows at u_out  (pinned §7 step 1; equals
//             <g, t_row> for honest commitments — asserted here at toy scale)
//      b    = ME weight vector of u_in: b_k = prod_i (k>>i&1 ? u_in[i] : 1-u_in[i])
//             (bit i of k pairs with u_in[i]; pinned here via <t_row,b> == me_open eval)
//      P_0  = C_0 + eval*Q
//    Round (n -> n/2), lo = front half, hi = back half:
//      L = <a_lo, g_hi> + <a_lo, b_hi>*Q ;  R = <a_hi, g_lo> + <a_hi, b_lo>*Q
//      absorb L, R  ->  x (nonzero, < r via top-limb % 1944954707, same dist as random_vec)
//      a' = x*a_lo + x^{-1}*a_hi ; b' = x^{-1}*b_lo + x*b_hi ; g' = x^{-1}*g_lo + x*g_hi
//      P' = x^2*L + P + x^{-2}*R          (invariant: P = <a,g> + <a,b>*Q)
//    Final: prover sends a_f. Verifier recomputes g_f, b_f from public data and
//    the re-derived challenges, checks  P_L == a_f*g_f + (a_f*b_f)*Q.
//    g_f is computed BOTH by explicit fold and by the s-vector MSM
//    (s_i = prod_r (bit_{MSB-r}(i) ? x_r : x_r^{-1})) and cross-checked, pinning
//    the formula the real driver will use.
//
// NOTE (documented, accepted for phase 0): pp generators are known-dlog
// (Commitment::random = G*r). Fine: the auditing side runs setup. The IPA also
// leaks a_f and eval (no blinding) — weight privacy of the opened row is NOT
// provided yet; deferred, documented in PHASE0_NOTES §10.
//
// GEN must be a power of two for the IPA (pp is; weights are padded).
//
// Also probes the -dlto miscompilation surface: a straight-line kernel with TWO
// G1Jacobian_mul calls (no branches) is cross-checked against a two-pass
// single-mul reference. (The known-bad shape is two BRANCHES each calling
// G1Jacobian_mul — see vrf_toy_debug2.cu.)
#include "commitment.cuh"
#include "proof.cuh"
#include "polynomial.cuh"
#include "fs_transcript.hpp"
#include <iostream>
using namespace std;

const Fr_t F_ONE = {1, 0, 0, 0, 0, 0, 0, 0};

// ---- 1-thread device helpers (same idioms as vrf_toy_open.cu) ----
KERNEL void k_g1_mul(G1Jacobian_t g, Fr_t c, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = G1Jacobian_mul(g, c);
}
KERNEL void k_g1_addsub(G1Jacobian_t a, G1Jacobian_t b, int sub, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = blstrs__g1__G1Affine_add(a, sub ? G1Jacobian_minus(b) : b);
}
// pair-fold, scalar/ME orientation: c' = c0 + u*(c1 - c0). SINGLE branch.
KERNEL void k_com_me(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* cout, uint new_size) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    cout[gid] = blstrs__g1__G1Affine_add(c[2*gid],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(c[2*gid+1], G1Jacobian_minus(c[2*gid])), u));
}
// pair-fold, me_open generator orientation: g' = g1 + u*(g0 - g1). SINGLE branch.
KERNEL void k_gen_fold(GLOBAL G1Jacobian_t* g, Fr_t u, GLOBAL G1Jacobian_t* gout, uint new_size) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    gout[gid] = blstrs__g1__G1Affine_add(g[2*gid+1],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(g[2*gid], G1Jacobian_minus(g[2*gid+1])), u));
}
// miscompile probe: TWO G1Jacobian_mul in STRAIGHT LINE (no branches).
KERNEL void k_two_mul(GLOBAL G1Jacobian_t* a, GLOBAL G1Jacobian_t* b, Fr_t x, Fr_t y,
                      GLOBAL G1Jacobian_t* out, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    out[gid] = blstrs__g1__G1Affine_add(G1Jacobian_mul(a[gid], x), G1Jacobian_mul(b[gid], y));
}

static G1Jacobian_t h_mul(G1Jacobian_t g, Fr_t c) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_mul<<<1,1>>>(g, c, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
static G1Jacobian_t h_add(G1Jacobian_t a, G1Jacobian_t b) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_addsub<<<1,1>>>(a, b, 0, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
static bool g1_eq(G1Jacobian_t a, G1Jacobian_t b) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_addsub<<<1,1>>>(a, b, 1, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d);
    for (int i = 0; i < 12; i++) if (h.z.val[i] != 0) return false;
    return true;
}
static bool fr_eq(const Fr_t& a, const Fr_t& b) {
    for (int i = 0; i < 8; i++) if (a.val[i] != b.val[i]) return false;
    return true;
}

// generic G1 fold chain with identity-padding for odd levels (pinned §7).
static G1Jacobian_t fold_chain(const G1Jacobian_t* dev_src, uint size,
                               const vector<Fr_t>& us, int orientation) {
    uint cap = size + 1;
    G1Jacobian_t* d_a; cudaMalloc(&d_a, cap * sizeof(G1Jacobian_t));
    G1Jacobian_t* d_b; cudaMalloc(&d_b, cap * sizeof(G1Jacobian_t));
    cudaMemcpy(d_a, dev_src, size * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    uint sz = size;
    for (auto& u : us) {
        if (sz % 2) { cudaMemset(d_a + sz, 0, sizeof(G1Jacobian_t)); sz += 1; }
        uint nsz = sz / 2;
        if (orientation == 0) k_com_me<<<(nsz + 31) / 32, 32>>>(d_a, u, d_b, nsz);
        else                  k_gen_fold<<<(nsz + 31) / 32, 32>>>(d_a, u, d_b, nsz);
        cudaDeviceSynchronize();
        std::swap(d_a, d_b); sz = nsz;
    }
    G1Jacobian_t out;
    cudaMemcpy(&out, d_a, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b);
    return out;
}

// ---- host-side small-vector algebra (toy scale; integer-view Fr ops) ----
static Fr_t ip(const vector<Fr_t>& a, uint oa, const vector<Fr_t>& b, uint ob, uint n) {
    Fr_t acc = a[oa] * b[ob];
    for (uint k = 1; k < n; k++) acc = acc + a[oa + k] * b[ob + k];
    return acc;
}
static G1Jacobian_t msm(const vector<G1Jacobian_t>& g, uint og,
                        const vector<Fr_t>& a, uint oa, uint n) {
    G1Jacobian_t acc = h_mul(g[og], a[oa]);
    for (uint k = 1; k < n; k++) acc = h_add(acc, h_mul(g[og + k], a[oa + k]));
    return acc;
}
static vector<G1Jacobian_t> g1_to_host(const G1Jacobian_t* dev, uint n) {
    vector<G1Jacobian_t> out(n);
    cudaMemcpy(out.data(), dev, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    return out;
}

// ME weight vector of u: b_k = prod_i (k>>i & 1 ? u[i] : 1 - u[i]), size 2^|u|.
static vector<Fr_t> me_weights(const vector<Fr_t>& u) {
    uint L = u.size(), n = 1u << L;
    vector<Fr_t> b(n);
    for (uint k = 0; k < n; k++) {
        Fr_t w = F_ONE;
        for (uint i = 0; i < L; i++) w = w * (((k >> i) & 1) ? u[i] : (F_ONE - u[i]));
        b[k] = w;
    }
    return b;
}

// ---- Fiat-Shamir glue ----
static void absorb_g1(fs::Transcript& tr, const string& label, const G1Jacobian_t& p) {
    tr.absorb(label, &p, sizeof(G1Jacobian_t));
}
static void absorb_fr(fs::Transcript& tr, const string& label, const Fr_t& x) {
    tr.absorb(label, &x, sizeof(Fr_t));
}
// challenge -> Fr: 8 LE uint32 limbs, top limb % 1944954707 (same dist as
// random_vec, guarantees < r); re-derive on the (negligible) zero outcome.
static Fr_t fs_challenge_fr(fs::Transcript& tr) {
    while (true) {
        uint8_t buf[32]; tr.challenge_bytes(buf);
        Fr_t x;
        for (int i = 0; i < 8; i++)
            x.val[i] = uint32_t(buf[4*i]) | (uint32_t(buf[4*i+1]) << 8)
                     | (uint32_t(buf[4*i+2]) << 16) | (uint32_t(buf[4*i+3]) << 24);
        x.val[7] %= 1944954707;
        for (int i = 0; i < 8; i++) if (x.val[i]) return x;
    }
}
// transcript binding the full statement before any round challenge
static fs::Transcript statement_transcript(const vector<G1Jacobian_t>& com_host,
                                           const vector<Fr_t>& u_out,
                                           const vector<Fr_t>& u_in, const Fr_t& eval) {
    fs::Transcript tr("vrf-toy-ipa");
    tr.absorb("com", com_host.data(), com_host.size() * sizeof(G1Jacobian_t));
    tr.absorb("u_out", u_out.data(), u_out.size() * sizeof(Fr_t));
    tr.absorb("u_in", u_in.data(), u_in.size() * sizeof(Fr_t));
    absorb_fr(tr, "eval", eval);
    return tr;
}

// ---- the OLD (§7 / upstream me_open) verify chain, for the attack demo ----
static bool old_verify(const G1TensorJacobian& com, const Commitment& gen,
                       const vector<Fr_t>& u_in, const vector<Fr_t>& u_out,
                       const vector<G1Jacobian_t>& proof, const Fr_t& eval) {
    G1Jacobian_t C = fold_chain(com.gpu_data, com.size, u_out, 0);
    for (uint i = 0; i < u_in.size(); i++) {
        G1Jacobian_t T = proof[3*i], T0 = proof[3*i+1], T1 = proof[3*i+2];
        if (!g1_eq(T, C)) return false;
        Fr_t uu = u_in[i], omu = F_ONE - uu;
        C = h_add(h_add(h_mul(T0, omu * omu), h_mul(T, uu * omu)), h_mul(T1, uu * uu));
    }
    G1Jacobian_t G_final = fold_chain(gen.gpu_data, gen.size, u_in, 1);
    if (!g1_eq(G_final, proof.back())) return false;
    return g1_eq(C, h_mul(G_final, eval));
}

// ---- IPA prover ----
struct IpaProof {
    vector<G1Jacobian_t> L, R;
    Fr_t a_final;
};

static IpaProof ipa_prove(vector<Fr_t> a, vector<Fr_t> b, vector<G1Jacobian_t> g,
                          const G1Jacobian_t& Q, G1Jacobian_t P, fs::Transcript tr,
                          bool check_invariant) {
    IpaProof pf;
    uint n = a.size();
    while (n > 1) {
        uint h = n / 2;
        G1Jacobian_t Lp = h_add(msm(g, h, a, 0, h), h_mul(Q, ip(a, 0, b, h, h)));
        G1Jacobian_t Rp = h_add(msm(g, 0, a, h, h), h_mul(Q, ip(a, h, b, 0, h)));
        absorb_g1(tr, "L", Lp); absorb_g1(tr, "R", Rp);
        Fr_t x = fs_challenge_fr(tr), xi = inv(x);
        for (uint k = 0; k < h; k++) {
            a[k] = x * a[k] + xi * a[k + h];
            b[k] = xi * b[k] + x * b[k + h];
            g[k] = h_add(h_mul(g[k], xi), h_mul(g[k + h], x));
        }
        a.resize(h); b.resize(h); g.resize(h);
        P = h_add(h_add(h_mul(Lp, x * x), P), h_mul(Rp, xi * xi));
        if (check_invariant) {
            G1Jacobian_t want = h_add(msm(g, 0, a, 0, h), h_mul(Q, ip(a, 0, b, 0, h)));
            if (!g1_eq(P, want)) cout << "  prover invariant BROKEN at n=" << n << endl;
        }
        pf.L.push_back(Lp); pf.R.push_back(Rp);
        n = h;
    }
    pf.a_final = a[0];
    return pf;
}

// ---- IPA verifier (witness-free: public g, Q, com-fold, u_in, eval, proof) ----
static bool ipa_verify(const vector<G1Jacobian_t>& g_pub, const G1Jacobian_t& Q,
                       const G1Jacobian_t& P0, const vector<Fr_t>& u_in,
                       const IpaProof& pf, fs::Transcript tr, bool report) {
    uint rounds = pf.L.size();
    if ((1u << rounds) != g_pub.size()) return false;
    vector<Fr_t> xs(rounds), xis(rounds);
    G1Jacobian_t P = P0;
    vector<Fr_t> b = me_weights(u_in);
    for (uint r = 0; r < rounds; r++) {
        absorb_g1(tr, "L", pf.L[r]); absorb_g1(tr, "R", pf.R[r]);
        xs[r] = fs_challenge_fr(tr); xis[r] = inv(xs[r]);
        P = h_add(h_add(h_mul(pf.L[r], xs[r] * xs[r]), P), h_mul(pf.R[r], xis[r] * xis[r]));
        uint h = b.size() / 2;
        for (uint k = 0; k < h; k++) b[k] = xis[r] * b[k] + xs[r] * b[k + h];
        b.resize(h);
    }
    // g_f two ways: explicit fold, and the s-vector MSM the real driver will use
    vector<G1Jacobian_t> g = g_pub;
    for (uint r = 0; r < rounds; r++) {
        uint h = g.size() / 2;
        for (uint k = 0; k < h; k++) g[k] = h_add(h_mul(g[k], xis[r]), h_mul(g[k + h], xs[r]));
        g.resize(h);
    }
    vector<Fr_t> s(g_pub.size());
    for (uint i = 0; i < g_pub.size(); i++) {
        Fr_t w = F_ONE;
        for (uint r = 0; r < rounds; r++)
            w = w * (((i >> (rounds - 1 - r)) & 1) ? xs[r] : xis[r]);
        s[i] = w;
    }
    G1Jacobian_t g_f2 = msm(g_pub, 0, s, 0, g_pub.size());
    if (report)
        cout << "  s-vector MSM == explicit g fold: " << (g1_eq(g[0], g_f2) ? "YES" : "NO(!!)") << endl;
    if (!g1_eq(g[0], g_f2)) return false;
    return g1_eq(P, h_add(h_mul(g[0], pf.a_final), h_mul(Q, pf.a_final * b[0])));
}

static bool run_case(uint GEN, uint N) {
    const uint COM = N / GEN;
    cout << "=== case GEN=" << GEN << " N=" << N << " COM=" << COM << " ===" << endl;
    Commitment gen = Commitment::random(GEN);
    Commitment qgen = Commitment::random(1);
    G1Jacobian_t Q = qgen(0);
    FrTensor t = FrTensor::random_int(N, 16);
    G1TensorJacobian com = gen.commit(t);

    auto u = random_vec(ceilLog2(GEN) + ceilLog2(COM));
    vector<Fr_t> u_out(u.end() - ceilLog2(COM), u.end());
    vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(COM));

    // ---- reference: upstream me_open (prover-side witness path) ----
    FrTensor t_row = t.partial_me(u_out, N / COM);
    vector<G1Jacobian_t> me_proof;
    Fr_t eval = Commitment::me_open(t_row, gen, u_in.begin(), u_in.end(), me_proof);
    bool old_ok = old_verify(com, gen, u_in, u_out, me_proof, eval);
    cout << "old verify (honest me_open): " << (old_ok ? "PASS" : "FAIL") << endl;

    // ---- STEERING ATTACK on the old protocol: forge eval+1 ----
    // Delta = G_final * 1; T1_last' = T1_last + u_last^{-2} * Delta
    // => C_L' = C_L + u_last^2 * (u_last^{-2} * Delta) = C_L + Delta.
    G1Jacobian_t G_final = fold_chain(gen.gpu_data, GEN, u_in, 1);
    Fr_t u_last = u_in.back();
    vector<G1Jacobian_t> forged = me_proof;
    uint last = u_in.size() - 1;
    forged[3*last + 2] = h_add(me_proof[3*last + 2], h_mul(G_final, inv(u_last * u_last)));
    bool attack = old_verify(com, gen, u_in, u_out, forged, eval + F_ONE);
    cout << "STEERING ATTACK: old verify accepts forged eval+1: "
         << (attack ? "YES (old protocol UNSOUND, as predicted)" : "NO (?!)") << endl;

    // ---- pin the IPA statement ----
    // a = t_row (integer view), b = ME weights of u_in, claim <a,b> == eval
    vector<Fr_t> a(GEN);
    for (uint k = 0; k < GEN; k++) a[k] = t_row(k);
    vector<Fr_t> b = me_weights(u_in);
    bool b_ok = fr_eq(ip(a, 0, b, 0, GEN), eval);
    cout << "<t_row, b> == me_open eval (b bit-order pinned): " << (b_ok ? "YES" : "NO(!!)") << endl;

    // C_0 (verifier-computable com fold) must equal <g, t_row>
    G1Jacobian_t C0 = fold_chain(com.gpu_data, COM, u_out, 0);
    vector<G1Jacobian_t> g_host = g1_to_host(gen.gpu_data, GEN);
    bool c0_ok = g1_eq(C0, msm(g_host, 0, a, 0, GEN));
    cout << "C_0 == <g, t_row>: " << (c0_ok ? "YES" : "NO(!!)") << endl;

    // ---- IPA prove + verify ----
    vector<G1Jacobian_t> com_host = g1_to_host(com.gpu_data, COM);
    G1Jacobian_t P0 = h_add(C0, h_mul(Q, eval));
    fs::Transcript tr0 = statement_transcript(com_host, u_out, u_in, eval);
    IpaProof pf = ipa_prove(a, b, g_host, Q, P0, tr0, true);
    cout << "IPA rounds: " << pf.L.size() << " (expect " << u_in.size() << ")" << endl;
    bool ipa_ok = ipa_verify(g_host, Q, P0, u_in, pf, tr0, true);
    cout << "IPA verify (honest): " << (ipa_ok ? "PASS" : "FAIL") << endl;

    // ---- forgeries against the NEW protocol ----
    // f1: forged eval' = eval+1 (statement change -> new P0 AND new transcript)
    Fr_t eval1 = eval + F_ONE;
    G1Jacobian_t P0f = h_add(C0, h_mul(Q, eval1));
    fs::Transcript trf = statement_transcript(com_host, u_out, u_in, eval1);
    bool f1 = !ipa_verify(g_host, Q, P0f, u_in, pf, trf, false);
    // f1b: same forged statement, but the prover gets to RE-RUN honestly on the
    // wrong claim (the steering-attack analog: full adaptive freedom over L/R)
    IpaProof pff = ipa_prove(a, b, g_host, Q, P0f, trf, false);
    bool f1b = !ipa_verify(g_host, Q, P0f, u_in, pff, trf, false);
    // f2: tamper round-0 L (challenges re-derive differently -> reject)
    IpaProof pf2 = pf; pf2.L[0] = h_add(pf2.L[0], G1Jacobian_generator);
    bool f2 = !ipa_verify(g_host, Q, P0, u_in, pf2, tr0, false);
    // f3: tamper final scalar
    IpaProof pf3 = pf; pf3.a_final = pf3.a_final + F_ONE;
    bool f3 = !ipa_verify(g_host, Q, P0, u_in, pf3, tr0, false);
    cout << "forgeries rejected: eval+1: " << (f1 ? "YES" : "NO(!!)")
         << ", adaptive eval+1: " << (f1b ? "YES" : "NO(!!)")
         << ", L tamper: " << (f2 ? "YES" : "NO(!!)")
         << ", a_f tamper: " << (f3 ? "YES" : "NO(!!)") << endl;

    bool all = old_ok && attack && b_ok && c0_ok && ipa_ok && f1 && f1b && f2 && f3;
    cout << (all ? "CASE PASS" : "CASE FAIL") << endl;
    return all;
}

// straight-line two-mul kernel vs two-pass single-mul reference
static bool two_mul_probe() {
    const uint n = 8;
    Commitment ga = Commitment::random(n), gb = Commitment::random(n);
    auto xy = random_vec(2);
    G1Jacobian_t* d_out; cudaMalloc(&d_out, n * sizeof(G1Jacobian_t));
    k_two_mul<<<1, n>>>(ga.gpu_data, gb.gpu_data, xy[0], xy[1], d_out, n);
    cudaDeviceSynchronize();
    vector<G1Jacobian_t> out = g1_to_host(d_out, n); cudaFree(d_out);
    vector<G1Jacobian_t> ha = g1_to_host(ga.gpu_data, n), hb = g1_to_host(gb.gpu_data, n);
    bool ok = true;
    for (uint i = 0; i < n; i++)
        if (!g1_eq(out[i], h_add(h_mul(ha[i], xy[0]), h_mul(hb[i], xy[1])))) ok = false;
    cout << "straight-line two-G1Jacobian_mul kernel correct: "
         << (ok ? "YES" : "NO — MISCOMPILED, real driver must use two passes") << endl;
    return ok;  // informational: harness uses 1-thread helpers either way
}

int main() {
    bool probe = two_mul_probe();
    bool a = run_case(8, 32);    // powers of two everywhere
    bool b = run_case(16, 64);   // deeper IPA (4 rounds)
    bool c = run_case(8, 24);    // odd number of commitment rows (COM=3 fold pad)
    cout << "two-mul probe: " << (probe ? "clean" : "MISCOMPILED (workaround in place)") << endl;
    cout << ((a && b && c) ? "TOY-IPA-VERIFY: ALL PASS" : "TOY-IPA-VERIFY: FAIL") << endl;
    return (a && b && c) ? 0 : 1;
}
