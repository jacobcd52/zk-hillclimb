// Real driver for ONE row-max obligation (zkob_rowmax), per
// STAGE3_FAITHFUL_DESIGN.md §2 (DESIGN FINAL 2026-06-11). Proves, with zero
// advice freedom on the value, that a committed per-row scalar mx[i] equals
// max_{j in allowed(i)} z[i,j] over a committed B x NCOL int32 grid z:
//   (X1) BIN  - S binary on the whole padded grid:
//        0 = sum_b eq(u_bin,b) S(b) (S(b)-1); verifier REQUIRES U_f2 == S_f2 - 1.
//   (X2) SUM  - 1 = sum_b [bcast(eq(u_s)) . AL](b) S(b) 1  (one-hot over allowed)
//        MASK - 0 = sum_b [bcast(eq(u_m)) . (1-AL)](b) S(b) 1 (nothing outside)
//   (X3) ATT  - ev_mx = m~x(u_a) = sum_b bcast(eq(u_a))(b) S(b) z(b) (attainment;
//        pure-broadcast weight -> verifier eq_acc shortcut, rmsnorm-validated)
//   (X4) DOM  - Df = AL.(mx_bcast - z) bound by c1 (eq(u_r).AL weight on z),
//        c2 (same weight on the never-committed mx_bcast, terminal opened vs
//        com_mx at the row-bit suffix), NPL plane openings of the limb tensor
//        L, the plain-field identity c2 - c1 == v0 [+ LEN_R*v1], and the limb
//        range lookup L vs tLookupRange(0, LEN_R)  =>  mx >= z on allowed.
//   (X5) T-BIND (vpad + t*) - 1 = sum_b W_t(b) S(b) 1 with
//        W_t[i*NCOL+j] = eq(u_t)[i]*[j == t*[i]]  =>  S[i, t*[i]] = 1.
// Constant claims (BIN 0, SUM 1, MASK 0, T-BIND 1) are PROTOCOL CONSTANTS:
// imposed by the verifier at round 0 AND required equal to the serialized
// claim_H; never absorbed. Data-dependent claims (ev_mx, c1, c2) absorbed.
//
// Masking regimes (AL public, never committed; derived from B/NCOL/MODE/V):
//   causal: AL[i,j] = (j <= i), requires B == NCOL, V = 0
//   vpad:   AL[i,j] = (j < V),  requires 0 < V <= NCOL (column pads excluded
//           by the mask weights and forced out of S by MASK+BIN, design §2.2)
//
// Layout: flat grid index = i*NCOL + j (column bits low); L flat index =
// plane*B*NCOL + i*NCOL + j (plane bits top). gen_grid (size NCOL) commits all
// grid tensors row-wise; gen_mx (size B) commits mx as ONE row of B values.
// Limb lookup: D_L = NPL*B*NCOL, N = LEN_R, NCOL <= LEN_R <= D_L, LEN_R | D_L.
//
// ONE new Fr-only kernel (design §2.8): k_pp_expand, generalizing
// k_eq_expand's (1-c, c) doubling to arbitrary (a, b) pairs; powers the
// driver-local fast_me_weights / fast_s_vector so the gen-32768 IPAs do not
// inherit the me_weights host-loop hot spot. Both fast helpers are
// cross-checked element-exact against the slow header versions in the
// evil==0 convention checks (toy AND gen-1024 scale); a mismatch THROWS
// (STOP-and-report). The IPA protocol itself is untouched. No new G1 kernels.
//
// Usage:
//   zkob_rowmax prove  <obdir> <seed> <z-int32.bin> <B> <NCOL> <MODE> <V>
//                      <LEN_R> <NPL> <gen_grid.bin> <gen_mx.bin> <q.bin>
//                      [mx-int32-out.bin] [tstar-int32.bin]
//   zkob_rowmax verify <obdir> <seed> <B> <NCOL> <MODE> <V> <LEN_R> <NPL>
//                      <gen_grid.bin> <gen_mx.bin> <q.bin> [tstar-int32.bin]
//   zkob_rowmax selftest
// MODE is the literal string causal|vpad. mx-out may be "-" to skip writing.
// The driver does NOT mkdir the obdir. Input z file holds the UNPADDED
// B x V (vpad) or B x NCOL (causal) int32 values; vpad zero-pads V -> NCOL.
#include "zkob_lookup.cuh"
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <climits>
#include <cstring>
#include <thread>
#include <atomic>
#include <functional>
#include <algorithm>
#include <memory>
#include <sys/stat.h>
using namespace std;

static_assert(sizeof(long) == 8, "host math needs 64-bit long");

// ---------------------------------------------------------------------------
// The ONE new kernel (design §2.8). Fr-only (glu precedent; Fr kernels are not
// in the -dlto miscompile family). Montgomery convention: mont-ify all factors
// except one -> mul(mont(a), in_plain) is the plain product.
KERNEL void k_pp_expand(GLOBAL Fr_t* in, Fr_t a, Fr_t b, GLOBAL Fr_t* out, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    Fr_t am = blstrs__scalar__Scalar_mont(a);
    Fr_t bm = blstrs__scalar__Scalar_mont(b);
    out[gid]     = blstrs__scalar__Scalar_mul(am, in[gid]);
    out[gid + n] = blstrs__scalar__Scalar_mul(bm, in[gid]);
}

// device doubling: after step i, index bit i pairs (as[i] if bit=0, bs[i] if
// bit=1) -- the build_eq_tensor recurrence with arbitrary pair factors.
// Returns a device buffer of size 2^L; caller cudaFrees.
static Fr_t* pp_build_dev(const vector<Fr_t>& as, const vector<Fr_t>& bs) {
    const uint L = as.size();
    Fr_t *cur, *nxt;
    cudaMalloc(&cur, sizeof(Fr_t) << L);
    cudaMalloc(&nxt, sizeof(Fr_t) << L);
    cudaMemcpy(cur, &F_ONE, sizeof(Fr_t), cudaMemcpyHostToDevice);
    uint n = 1;
    for (uint i = 0; i < L; i++) {
        k_pp_expand<<<(n + 255) / 256, 256>>>(cur, as[i], bs[i], nxt, n);
        cudaDeviceSynchronize();
        std::swap(cur, nxt);
        n <<= 1;
    }
    cudaFree(nxt);
    return cur;
}
// fast ME weights: b_k = prod_i (k>>i & 1 ? u[i] : 1-u[i]) -- pairs (1-u_i, u_i)
static Fr_t* fast_me_weights_dev(const vector<Fr_t>& u) {
    vector<Fr_t> as(u.size()), bs(u.size());
    for (uint i = 0; i < u.size(); i++) {
        as[i] = h_scalar(F_ONE, u[i], 1);
        bs[i] = u[i];
    }
    return pp_build_dev(as, bs);
}
static vector<Fr_t> fast_me_weights(const vector<Fr_t>& u) {
    Fr_t* d = fast_me_weights_dev(u);
    vector<Fr_t> h(1u << u.size());
    cudaMemcpy(h.data(), d, sizeof(Fr_t) * h.size(), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return h;
}
// fast IPA s-vector: s_i = prod_r (bit_{R-1-r}(i) ? xs[r] : xis[r]) -- bit b
// pairs (xis[R-1-b], xs[R-1-b]), MSB-first order matching ipa_verify's pinned
// s_i product.
static Fr_t* fast_s_vector_dev(const vector<Fr_t>& xs, const vector<Fr_t>& xis) {
    const uint R = xs.size();
    vector<Fr_t> as(R), bs(R);
    for (uint t = 0; t < R; t++) {
        as[t] = xis[R - 1 - t];
        bs[t] = xs[R - 1 - t];
    }
    return pp_build_dev(as, bs);
}

// fast variant of the header's open_prove (only me_weights moves to device)
static void fast_open_prove(const FrTensor& t, uint G, const Commitment& gen,
                            const G1Jacobian_t& Q, const vector<Fr_t>& u_pt,
                            const string& path, fs::Transcript& tr) {
    const uint logG = ceilLog2(G);
    vector<Fr_t> u_col(u_pt.begin(), u_pt.begin() + logG);
    vector<Fr_t> u_row(u_pt.begin() + logG, u_pt.end());
    if (u_row.empty()) {
        write_ipa(path, ipa_prove(t.gpu_data, fast_me_weights(u_col), gen.gpu_data, Q, G, tr));
    } else {
        FrTensor a = t.partial_me(u_row, G);
        write_ipa(path, ipa_prove(a.gpu_data, fast_me_weights(u_col), gen.gpu_data, Q, G, tr));
    }
}
// fast variant of the header's ipa_verify: identical protocol and algebra;
// the b fold's final value equals <b, s> with the SAME s-vector (front/back
// fold algebra), so b_f and g_f both come from one device s build. All field
// ops are exact canonical mod-p arithmetic, so the result is bit-identical to
// the header's incremental fold.
static bool fast_ipa_verify(const G1Jacobian_t* d_g, uint n, const G1Jacobian_t& Q,
                            const G1Jacobian_t& P0, const vector<Fr_t>& u_b,
                            const IpaProof& pf, fs::Transcript& tr) {
    const uint rounds = pf.L.size();
    if (rounds != pf.R.size()) return false;
    if (n != (1u << rounds)) return false;
    if ((1u << u_b.size()) != n) return false;
    vector<Fr_t> xs(rounds), xis(rounds);
    G1Jacobian_t P = P0;
    for (uint r = 0; r < rounds; r++) {
        absorb_g1(tr, "L", pf.L[r]);
        absorb_g1(tr, "R", pf.R[r]);
        xs[r] = fs_challenge_fr(tr);
        xis[r] = inv(xs[r]);
        P = h_add(h_add(h_mul(pf.L[r], h_scalar(xs[r], xs[r], 2)), P),
                  h_mul(pf.R[r], h_scalar(xis[r], xis[r], 2)));
    }
    Fr_t* d_s = fast_s_vector_dev(xs, xis);
    Fr_t* d_b = fast_me_weights_dev(u_b);
    Fr_t b_f = dev_ip(d_b, d_s, n);
    G1Jacobian_t g_f = dev_msm(d_g, d_s, n);
    cudaFree(d_s);
    cudaFree(d_b);
    return g1_eq(P, h_add(h_mul(g_f, pf.a_final), h_mul(Q, h_scalar(pf.a_final, b_f, 2))));
}
static bool fast_open_verify(const G1TensorJacobian& com, const Commitment& gen, uint G,
                             const G1Jacobian_t& Q, const vector<Fr_t>& u_pt, const Fr_t& eval,
                             const string& path, fs::Transcript& tr) {
    const uint logG = ceilLog2(G);
    if (u_pt.size() < logG) return false;
    if (com.size != (1u << (u_pt.size() - logG))) return false;
    vector<Fr_t> u_col(u_pt.begin(), u_pt.begin() + logG);
    vector<Fr_t> u_row(u_pt.begin() + logG, u_pt.end());
    G1Jacobian_t C0 = fold_chain(com.gpu_data, com.size, u_row, 0);
    G1Jacobian_t P0 = h_add(C0, h_mul(Q, eval));
    return fast_ipa_verify(gen.gpu_data, G, Q, P0, u_col, read_ipa(path), tr);
}

// fast-vs-slow cross-checks (design §2.8/§2.9 pinned; mismatch = STOP)
static void crosscheck_fast_helpers(const vector<Fr_t>& u_me, const vector<Fr_t>& xs_src) {
    {   // fast_me_weights vs the slow header me_weights, element-exact
        vector<Fr_t> slow = me_weights(u_me);
        vector<Fr_t> fastv = fast_me_weights(u_me);
        for (size_t k = 0; k < slow.size(); k++)
            if (!fr_eq(slow[k], fastv[k]))
                throw runtime_error("STOP: fast_me_weights != me_weights at element "
                                    + to_string(k) + " (design §2.8 cross-check failed)");
    }
    {   // fast_s_vector vs the slow ipa_verify s_i product, element-exact
        const uint R = xs_src.size();
        vector<Fr_t> xs(xs_src), xis(R);
        for (uint r = 0; r < R; r++) xis[r] = inv(xs[r]);
        const uint n = 1u << R;
        vector<Fr_t> s_slow(n);
        for (uint i = 0; i < n; i++) {
            Fr_t w = F_ONE;
            for (uint r = 0; r < R; r++)
                w = h_scalar(w, ((i >> (R - 1 - r)) & 1) ? xs[r] : xis[r], 2);
            s_slow[i] = w;
        }
        Fr_t* d = fast_s_vector_dev(xs, xis);
        vector<Fr_t> s_fast(n);
        cudaMemcpy(s_fast.data(), d, sizeof(Fr_t) * n, cudaMemcpyDeviceToHost);
        cudaFree(d);
        for (uint i = 0; i < n; i++)
            if (!fr_eq(s_slow[i], s_fast[i]))
                throw runtime_error("STOP: fast_s_vector != slow s-vector at element "
                                    + to_string(i) + " (design §2.8 cross-check failed)");
    }
}

// ---------------------------------------------------------------------------
// Iterative, allocation-disciplined clone of the header's fs_hadamard
// (zkob_lookup.cuh). Same kernels (k_hp3_step, k_fr_fold), same FrTensor::sum
// reduction, same absorb labels and challenge schedule, same terminal reads --
// every emitted value is the identical field element, so the transcript and
// all proof bytes are unchanged. The ONLY difference is buffer lifetime: the
// header recursion keeps each frame's o0..o3 (4h) AND fold halves (3h) alive
// through the whole descent (~7x the grid on top of three head COPIES); here
// each round's buffers are freed before the next round allocates and the
// round-1 inputs are read in place (no head copies), capping the call's
// working set at ~2.75x(n/2) Fr above the inputs (§2.5 freeing discipline,
// load-bearing at the vpad 2^25 grid).
static void lean_hadamard(Fr_t claim, const Fr_t* E0, const Fr_t* S0, const Fr_t* U0,
                          uint n, fs::Transcript& tr, vector<Fr_t>& ws,
                          HadamardProof& out, bool strict) {
    Fr_t *Ec = nullptr, *Sc = nullptr, *Uc = nullptr;   // owned fold buffers
    uint sz = n;
    while (sz > 1) {
        const uint h = sz >> 1;
        const Fr_t* pe = Ec ? Ec : E0;
        const Fr_t* ps = Sc ? Sc : S0;
        const Fr_t* pu = Uc ? Uc : U0;
        array<Fr_t,4> e;
        {
            FrTensor o0(h), o1(h), o2(h), o3(h);
            k_hp3_step<<<(h + 255) / 256, 256>>>(
                const_cast<Fr_t*>(pe), const_cast<Fr_t*>(ps), const_cast<Fr_t*>(pu),
                o0.gpu_data, o1.gpu_data, o2.gpu_data, o3.gpu_data, h);
            cudaDeviceSynchronize();
            e = {o0.sum(), o1.sum(), o2.sum(), o3.sum()};
        }                                               // o0..o3 freed here
        if (strict && !fr_eq(claim, h_scalar(e[0], e[1], 0)))
            throw runtime_error("hadamard round inconsistency (witness bug)");
        for (int i = 0; i < 4; i++) out.ev.push_back(e[i]);
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws.push_back(w);
        Fr_t *nE, *nS, *nU;
        cudaMalloc(&nE, sizeof(Fr_t) * h);
        cudaMalloc(&nS, sizeof(Fr_t) * h);
        cudaMalloc(&nU, sizeof(Fr_t) * h);
        k_fr_fold<<<(h + 255) / 256, 256>>>(const_cast<Fr_t*>(pe), w, nE, h);
        k_fr_fold<<<(h + 255) / 256, 256>>>(const_cast<Fr_t*>(ps), w, nS, h);
        k_fr_fold<<<(h + 255) / 256, 256>>>(const_cast<Fr_t*>(pu), w, nU, h);
        cudaDeviceSynchronize();
        if (Ec) cudaFree(Ec);
        if (Sc) cudaFree(Sc);
        if (Uc) cudaFree(Uc);
        Ec = nE; Sc = nS; Uc = nU;
        claim = lagrange4(e, w, inv(F_SIX));
        sz = h;
    }
    cudaMemcpy(&out.S_f2, Sc ? Sc : S0, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out.U_f2, Uc ? Uc : U0, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    if (Ec) cudaFree(Ec);
    if (Sc) cudaFree(Sc);
    if (Uc) cudaFree(Uc);
}

// ---------------------------------------------------------------------------
// chunked row-wise commit. Upstream Commitment::commit materializes TWO full
// G1Jacobian buffers of t.size points (144 B/element: the elementwise-mul
// output plus rowwise_sum's working copy) -- ~18 GiB transient at the vpad
// 2^26 shape, which OOMs the 24 GiB card with the witness tensors resident
// (unchecked cudaMalloc -> sticky illegal access). Committing in row blocks
// through the SAME upstream kernels is bit-identical per row (each output row
// depends only on that row's scalars) and caps the transient at ~1.2 GiB.
// No new kernels.
static G1TensorJacobian commit_chunked(const Commitment& gen, const FrTensor& t) {
    const uint G = gen.size;
    if (t.size % G) throw runtime_error("commit_chunked: size not a row multiple");
    const uint rows = t.size / G;
    const uint CHUNK_ROWS = max(1u, (1u << 22) / G);
    if (rows <= CHUNK_ROWS) return gen.commit(t);
    G1TensorJacobian out(rows);
    for (uint r0 = 0; r0 < rows; r0 += CHUNK_ROWS) {
        const uint nr = min(CHUNK_ROWS, rows - r0);
        FrTensor chunk(nr * G);
        cudaMemcpy(chunk.gpu_data, t.gpu_data + (size_t)r0 * G,
                   sizeof(Fr_t) * nr * G, cudaMemcpyDeviceToDevice);
        G1TensorJacobian part = gen.commit(chunk);
        cudaMemcpy(out.gpu_data + r0, part.gpu_data,
                   sizeof(G1Jacobian_t) * nr, cudaMemcpyDeviceToDevice);
    }
    return out;
}

// ---------------------------------------------------------------------------
// shared layout guards (honest-prover throws, design §2.1)
static void layout_guards(uint B, uint NCOL, uint MODE, uint V, uint LEN_R, uint NPL,
                          uint gen_size, uint genmx_size) {
    if (B != (1u << ceilLog2(B))) throw runtime_error("B not a power of two");
    if (NCOL != (1u << ceilLog2(NCOL))) throw runtime_error("NCOL not a power of two");
    if (LEN_R != (1u << ceilLog2(LEN_R))) throw runtime_error("LEN_R not a power of two");
    if (NPL != 1 && NPL != 2) throw runtime_error("NPL must be 1 or 2");
    if (MODE == 0) {
        if (B != NCOL) throw runtime_error("causal requires B == NCOL");
        if (V != 0) throw runtime_error("causal requires V == 0");
    } else if (MODE == 1) {
        if (V == 0 || V > NCOL) throw runtime_error("vpad requires 0 < V <= NCOL");
    } else throw runtime_error("MODE must be 0 (causal) or 1 (vpad)");
    if (gen_size != NCOL) throw runtime_error("gen_grid size != NCOL");
    if (genmx_size != B) throw runtime_error("gen_mx size != B");
    const unsigned long long DL = (unsigned long long)NPL * B * NCOL;
    if (NCOL > LEN_R || LEN_R > DL || DL % LEN_R)
        throw runtime_error("limb lookup layout needs NCOL <= LEN_R <= NPL*D, LEN_R | NPL*D");
}

// allowed-set mask AL (public, never committed)
static vector<int> build_al(uint B, uint NCOL, uint MODE, uint V) {
    vector<int> al((size_t)B * NCOL);
    for (uint i = 0; i < B; i++)
        for (uint j = 0; j < NCOL; j++)
            al[(size_t)i * NCOL + j] = (MODE == 0) ? (j <= i ? 1 : 0) : (j < V ? 1 : 0);
    return al;
}

static Fr_t fr_of_ll(long long v) {
    long l = (long)v;
    FrTensor t(1, &l);
    return t(0);
}

// low 64 bits of (p - d) for small d >= 1 (p = BLS12-381 scalar modulus;
// p's low 64 bits are 0xffffffff00000001, no borrow for any d we use)
static unsigned long long p_minus_low64(unsigned long long d) {
    return 0xffffffff00000001ULL - d;
}

// ---------------------------------------------------------------------------
// memory diagnostics (env ZKOB_MEMLOG=1; selftest-only, zero protocol effect).
// mon_phase_peak is also fed by the selftest monitor thread so each mark
// reports the true sampled peak of the phase it closes, not just the
// instantaneous usage at the boundary.
static atomic<long long> mon_phase_peak{0};
static void mem_mark(const char* label) {
    if (!getenv("ZKOB_MEMLOG")) return;
    size_t fr = 0, to = 0;
    cudaMemGetInfo(&fr, &to);
    long long used = (long long)(to - fr);
    long long pk = mon_phase_peak.load();
    if (used > pk) pk = used;
    mon_phase_peak.store(used);
    cout << "[mem] " << label << ": now " << (double)used / (1u << 30)
         << " GiB, phase peak " << (double)pk / (1u << 30) << " GiB" << endl;
}

// ---------------------------------------------------------------------------
// prove
// evil (selftest only; honest PROCEDURE on an inconsistent witness):
//  1: mx[row] += 1; S honest at the true argmax; Df/limbs recomputed from the
//     evil mx -> only ATT round 0 can reject (absorbed ev_mx vs honest <S,z>).
//  2: mx[row] -= 1 at a row whose second-distinct value is max-1; S moved to
//     that j; the true argmax position's residual is -1, limbs stored as the
//     low limbs of its field representative -> only "DOM bracket identity".
//  3: fractional selector: at a scanned row with two allowed j1 != j2 whose
//     values are distinct and both != mx, the honest one-hot is zeroed and
//     S[j1]=c, S[j2]=1-c with c=(mx-z2)/(z1-z2) (field; c outside {0,1}),
//     mx honest -> only "BIN round 0" (THE certifying evil).
//  4: two-hot: S[i,j2] += 1 at an allowed j2 with z=0 -> only "SUM round 0".
//  5: selector on a pad column S[i,V] += 1 (vpad, V < NCOL) -> "MASK round 0".
//  6: out-of-range limb with compensating carry (lo += LEN_R, hi -= 1; NPL=2;
//     m_L from the HONEST limbs) -> only "limb lookup round 0".
//  7: corrupted broadcast buffer in c2 (mx_bcast[idx] += 1, c2 absorbed from
//     that run, all else honest) -> "IPA opening of c2 terminal vs com_mx".
//  8: wrong served token t*[row] (caller passes the evil t*; S honest)
//     -> only "T-BIND round 0".
static void prove(const string& obdir, const string& seed,
                  const vector<int>& zh_in, uint B, uint NCOL, uint MODE, uint V,
                  uint LEN_R, uint NPL,
                  const Commitment& gen, const Commitment& genmx, const G1Jacobian_t& Q,
                  const string& mx_out, const vector<int>* tstar,
                  int evil = 0, uint evil_i = 0, uint evil_j = 0) {
    const uint D = B * NCOL, DL = NPL * D;
    const uint logB = ceilLog2(B), logC = ceilLog2(NCOL);
    const uint logD = logB + logC, logDL = logD + ceilLog2(NPL);
    const uint n1 = logDL - ceilLog2(LEN_R);
    layout_guards(B, NCOL, MODE, V, LEN_R, NPL, gen.size, genmx.size);
    if (tstar && MODE == 0) throw runtime_error("t* supplied in causal mode");
    if (tstar) {
        if (tstar->size() != (size_t)B) throw runtime_error("t* size != B");
        for (uint i = 0; i < B; i++)
            if ((*tstar)[i] < 0 || (uint)(*tstar)[i] >= V)
                throw runtime_error("t* token out of [0, V)");
    }
    const long long LIMIT = (NPL == 1) ? (long long)LEN_R
                                       : (long long)LEN_R * (long long)LEN_R;

    // ---- host integer chain (int64; all bounds < 2^45, design §2.5) ----
    vector<int> zh((size_t)D, 0);                  // padded grid
    if (MODE == 1) {
        if (zh_in.size() != (size_t)B * V) throw runtime_error("input dims");
        for (uint i = 0; i < B; i++)
            for (uint j = 0; j < V; j++) {
                int v = zh_in[(size_t)i * V + j];
                if ((long long)llabs((long long)v) >= (1LL << 25))
                    throw runtime_error("vpad |z| >= 2^25 (envelope guard)");
                zh[(size_t)i * NCOL + j] = v;
            }
    } else {
        if (zh_in.size() != (size_t)D) throw runtime_error("input dims");
        zh = zh_in;
    }
    vector<int> alh = build_al(B, NCOL, MODE, V);

    // mx + canonical lowest-index argmax over the allowed set
    vector<long> mxh(B);
    vector<uint> amax(B);
    for (uint i = 0; i < B; i++) {
        long long best = LLONG_MIN;
        uint bj = 0;
        for (uint j = 0; j < NCOL; j++)
            if (alh[(size_t)i * NCOL + j] && (long long)zh[(size_t)i * NCOL + j] > best) {
                best = zh[(size_t)i * NCOL + j];
                bj = j;
            }
        mxh[i] = (long)best;
        amax[i] = bj;
    }

    uint e2_oldj = 0;                              // evil 2 bookkeeping
    if (evil == 1) mxh[evil_i] += 1;               // S stays at the true argmax
    if (evil == 2) {
        const uint r = evil_i;
        const long long m = mxh[r];
        e2_oldj = amax[r];
        uint j2 = NCOL;                            // lowest allowed j with z == m-1
        for (uint j = 0; j < NCOL && j2 == NCOL; j++)
            if (alh[(size_t)r * NCOL + j] && (long long)zh[(size_t)r * NCOL + j] == m - 1)
                j2 = j;
        if (j2 == NCOL) throw runtime_error("evil 2 setup: no second-distinct = max-1");
        mxh[r] = (long)(m - 1);
        amax[r] = j2;                              // attainment stays consistent
    }

    vector<int> Sh((size_t)D, 0);
    for (uint i = 0; i < B; i++) Sh[(size_t)i * NCOL + amax[i]] = 1;
    if (evil == 4) {
        const uint r = evil_i;
        uint j2 = NCOL;                            // allowed, z == 0, not the argmax
        for (uint j = 0; j < NCOL && j2 == NCOL; j++)
            if (alh[(size_t)r * NCOL + j] && zh[(size_t)r * NCOL + j] == 0 && j != amax[r])
                j2 = j;
        if (j2 == NCOL) throw runtime_error("evil 4 setup: no allowed zero entry");
        Sh[(size_t)r * NCOL + j2] += 1;
    }
    if (evil == 5) {
        if (MODE != 1 || V >= NCOL) throw runtime_error("evil 5 setup: needs vpad pads");
        Sh[(size_t)evil_i * NCOL + V] += 1;        // pad column: z = 0, AL = 0
    }

    // dominance residual limbs
    vector<long> Lh((size_t)DL, 0);
    for (uint i = 0; i < B; i++)
        for (uint j = 0; j < NCOL; j++) {
            const size_t idx = (size_t)i * NCOL + j;
            if (!alh[idx]) continue;               // Df identically 0 at pads/masked
            long long df = mxh[i] - (long long)zh[idx];
            if (evil == 2 && i == evil_i && j == e2_oldj) {
                // residual -1: store the low limbs of the field representative
                unsigned long long lo64 = p_minus_low64((unsigned long long)(-df));
                Lh[idx] = (long)(lo64 % LEN_R);
                if (NPL == 2) Lh[(size_t)D + idx] = (long)((lo64 / LEN_R) % LEN_R);
                continue;
            }
            if (df < 0) throw runtime_error("Df < 0 (witness bug)");
            if (df >= LIMIT) throw runtime_error("allowed Df >= LEN_R^NPL");
            if (NPL == 1) {
                Lh[idx] = (long)df;
            } else {
                Lh[idx] = (long)(df % LEN_R);
                Lh[(size_t)D + idx] = (long)(df / LEN_R);
            }
        }
    vector<long> Lh_honest;
    if (evil == 6) {
        if (NPL != 2) throw runtime_error("evil 6 setup: needs NPL = 2");
        size_t idx = 0;
        while (idx < (size_t)D && Lh[(size_t)D + idx] < 1) idx++;
        if (idx >= (size_t)D) throw runtime_error("evil 6 setup: no entry with hi limb >= 1");
        Lh_honest = Lh;
        Lh[idx] += (long)LEN_R;                    // lo leaves [0, LEN_R)
        Lh[(size_t)D + idx] -= 1;                  // compensating carry: value unchanged
    }

    if (!mx_out.empty() && mx_out != "-") {        // chain file: unpadded B int32
        vector<int> mi(B);
        for (uint i = 0; i < B; i++) mi[i] = (int)mxh[i];
        FILE* f = open_or_die(mx_out, "wb");
        fwrite(mi.data(), sizeof(int), B, f);
        fclose(f);
    }

    // ---- tensors / commitments (all PLAIN Fr values; §2.5 lifetime
    // discipline: z and S are committed then freed until after the limb
    // lookup -- S's exact post-edit device bytes are stashed host-side -- and
    // L is freed after the lookup and re-uploaded for the DOM plane
    // openings. Commitment and proof bytes are unchanged; only buffer
    // lifetimes differ.) ----
    mem_mark("prove start");
    FrTensor mx_t(B, mxh.data());
    uint e3_r = 0, e3_j1 = 0, e3_j2 = 0;
    Fr_t e3_c = F_ZERO;
    if (evil == 3) {
        // a GENUINELY fractional selector needs two allowed positions whose
        // values are distinct and BOTH != mx: c = (mx - z2)/(z1 - z2) is then
        // outside {0,1} (c = 1 iff z1 = mx, c = 0 iff z2 = mx). The honest
        // one-hot at the argmax is zeroed; row mass moves entirely to (j1, j2).
        // Attainment stays exact: c*z1 + (1-c)*z2 = mx by construction.
        uint r = B, j1 = 0, j2 = 0;
        for (uint i = 0; i < B && r == B; i++) {
            uint a = NCOL, b = NCOL;
            for (uint j = 0; j < NCOL; j++) {
                if (!alh[(size_t)i * NCOL + j]) continue;
                if ((long long)zh[(size_t)i * NCOL + j] == mxh[i]) continue;
                if (a == NCOL) { a = j; continue; }
                if (zh[(size_t)i * NCOL + j] != zh[(size_t)i * NCOL + a]) { b = j; break; }
            }
            if (b != NCOL) { r = i; j1 = a; j2 = b; }
        }
        if (r == B) throw runtime_error("evil 3 setup: no row with two distinct non-max allowed values");
        e3_r = r; e3_j1 = j1; e3_j2 = j2;
        long long z1 = zh[(size_t)r * NCOL + j1], z2 = zh[(size_t)r * NCOL + j2];
        e3_c = h_scalar(fr_of_ll(mxh[r] - z2), inv(fr_of_ll(z1 - z2)), 2);
    }
    G1TensorJacobian com_z = [&] {
        FrTensor z_t(D, zh.data());
        return commit_chunked(gen, z_t);
    }();                                           // z freed until after LIMB
    vector<Fr_t> S_bytes((size_t)D);
    G1TensorJacobian com_S = [&] {
        FrTensor S_t(D, Sh.data());
        if (evil == 3) {                           // device edits, then stash
            Fr_t omc = h_scalar(F_ONE, e3_c, 1);
            cudaMemcpy(S_t.gpu_data + (size_t)e3_r * NCOL + amax[e3_r], &F_ZERO,
                       sizeof(Fr_t), cudaMemcpyHostToDevice);
            cudaMemcpy(S_t.gpu_data + (size_t)e3_r * NCOL + e3_j1, &e3_c,
                       sizeof(Fr_t), cudaMemcpyHostToDevice);
            cudaMemcpy(S_t.gpu_data + (size_t)e3_r * NCOL + e3_j2, &omc,
                       sizeof(Fr_t), cudaMemcpyHostToDevice);
        }
        G1TensorJacobian c = commit_chunked(gen, S_t);
        cudaMemcpy(S_bytes.data(), S_t.gpu_data, sizeof(Fr_t) * D,
                   cudaMemcpyDeviceToHost);
        return c;
    }();                                           // S freed until after LIMB
    G1TensorJacobian com_mx = genmx.commit(mx_t);
    auto L_t = make_unique<FrTensor>(DL, Lh.data());
    tLookupRange tlR(0, LEN_R);
    FrTensor m_L = (evil == 6) ? tlR.prep(FrTensor(DL, Lh_honest.data()))
                               : tlR.prep(*L_t);
    G1TensorJacobian com_L  = commit_chunked(gen, *L_t);
    G1TensorJacobian com_mL = commit_chunked(gen, m_L);
    com_z.save(obdir + "/com_z.bin");
    com_S.save(obdir + "/com_S.bin");
    com_mx.save(obdir + "/com_mx.bin");
    com_L.save(obdir + "/com_L.bin");
    com_mL.save(obdir + "/com_m_L.bin");
    { FILE* f = open_or_die(obdir + "/dims.bin", "wb");
      uint32_t d[6] = {B, NCOL, MODE, V, LEN_R, NPL};
      fwrite(d, sizeof(uint32_t), 6, f); fclose(f); }

    mem_mark("witness tensors + commits");

    // ---- transcript ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "NCOL", NCOL);
    absorb_u32(tr, "MODE", MODE); absorb_u32(tr, "V", V);
    absorb_u32(tr, "LEN_R", LEN_R); absorb_u32(tr, "NPL", NPL);
    if (MODE == 1 && tstar)
        tr.absorb("TSTAR", tstar->data(), (size_t)B * sizeof(int));
    absorb_g1_tensor(tr, "com_z", com_z);
    absorb_g1_tensor(tr, "com_S", com_S);
    absorb_g1_tensor(tr, "com_mx", com_mx);
    absorb_g1_tensor(tr, "com_L", com_L);
    absorb_g1_tensor(tr, "com_m_L", com_mL);

    // ---- LIMB: limb range lookup ----
    Fr_t beta_L = fs_challenge_fr(tr);
    LookupProof pfL;
    vector<Fr_t> ws_L;
    {
        FrTensor A_L(DL);
        tlookup_inv_kernel<<<(DL+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            L_t->gpu_data, beta_L, A_L.gpu_data, DL);
        cudaDeviceSynchronize();
        G1TensorJacobian com_AL = commit_chunked(gen, A_L);
        com_AL.save(obdir + "/com_A_L.bin");
        absorb_g1_tensor(tr, "com_A_L", com_AL);
        Fr_t alpha_L = fs_challenge_fr(tr);
        auto u_L = fs_challenge_vec(tr, logDL);
        {
            FrTensor BvL(LEN_R);
            tlookup_inv_kernel<<<(LEN_R+FrNumThread-1)/FrNumThread,FrNumThread>>>(
                tlR.table.gpu_data, beta_L, BvL.gpu_data, LEN_R);
            cudaDeviceSynchronize();
            Fr_t alpha_sq = alpha_L * alpha_L;
            Fr_t Cc = alpha_sq - (BvL * m_L).sum();
            Fr_t claim = alpha_L + alpha_sq;
            Fr_t inv_ratio = Fr_t{LEN_R,0,0,0,0,0,0,0} / Fr_t{DL,0,0,0,0,0,0,0};
            fs_phase1(claim, A_L, *L_t, BvL, tlR.table, m_L, alpha_L, beta_L, Cc,
                      inv_ratio, alpha_sq, u_L, tr, ws_L, pfL, evil != 6);
        }
        write_lookup(obdir + "/lookup_L.bin", pfL);
        absorb_fr(tr, "A_f", pfL.A_f); absorb_fr(tr, "S_f", pfL.S_f);
        absorb_fr(tr, "m_f", pfL.m_f);
        vector<Fr_t> u_ptL(ws_L.rbegin(), ws_L.rend());
        vector<Fr_t> u_mL(ws_L.rbegin(), ws_L.rend() - n1);
        if (evil == 0) {        // convention: fold terminals == ME evaluations
            vector<Fr_t> u_col(u_ptL.begin(), u_ptL.begin() + logC);
            vector<Fr_t> u_row(u_ptL.begin() + logC, u_ptL.end());
            if (!fr_eq(pfL.A_f, A_L.multi_dim_me({u_row, u_col}, {NPL * B, NCOL})) ||
                !fr_eq(pfL.S_f, L_t->multi_dim_me({u_row, u_col}, {NPL * B, NCOL})))
                throw runtime_error("limb lookup terminal != multi_dim_me (convention bug)");
        }
        fast_open_prove(A_L, NCOL, gen, Q, u_ptL, obdir + "/ipa_A_L.bin", tr);
        fast_open_prove(*L_t, NCOL, gen, Q, u_ptL, obdir + "/ipa_L_lk.bin", tr);
        fast_open_prove(m_L, NCOL, gen, Q, u_mL,  obdir + "/ipa_m_L.bin", tr);
    }   // A_L freed here (§2.5 memory requirement)
    L_t.reset();                                   // re-uploaded for DOM planes
    mem_mark("LIMB");

    // re-materialize z (same int buffer) and S (its exact committed device
    // bytes): identical field elements, so all downstream bytes match
    auto z_t = make_unique<FrTensor>(D, zh.data());    // freed after c1
    FrTensor S_t(D);
    cudaMemcpy(S_t.gpu_data, S_bytes.data(), sizeof(Fr_t) * D,
               cudaMemcpyHostToDevice);
    vector<Fr_t>().swap(S_bytes);
    auto AL_t = make_unique<FrTensor>(D, alh.data());  // public mask; freed after c2
    vector<int> onesh((size_t)D, 1);
    FrTensor ones_t(D, onesh.data());
    onesh.clear(); onesh.shrink_to_fit();

    // ---- BIN: binarity sumcheck (claim 0, protocol constant) ----
    auto u_bin = fs_challenge_vec(tr, logD);
    HadamardProof hp_bin;
    hp_bin.claim_H = F_ZERO;
    vector<Fr_t> ws_bin;
    {
        // U = S - 1 materialized as the int buffer S-1 (values -1/0; FrTensor
        // int ctor handles the mod-p); evil 3 overwrites the two field entries
        vector<int> Uh((size_t)D);
        for (size_t k = 0; k < (size_t)D; k++) Uh[k] = Sh[k] - 1;
        FrTensor U_t(D, Uh.data());
        if (evil == 3) {
            Fr_t m1  = h_scalar(F_ZERO, F_ONE, 1);           // -1 = 0 - 1 (zeroed argmax)
            Fr_t cm1 = h_scalar(e3_c, F_ONE, 1);             // c - 1
            Fr_t mc  = h_scalar(F_ZERO, e3_c, 1);            // -c = (1-c) - 1
            cudaMemcpy(U_t.gpu_data + (size_t)e3_r * NCOL + amax[e3_r], &m1,
                       sizeof(Fr_t), cudaMemcpyHostToDevice);
            cudaMemcpy(U_t.gpu_data + (size_t)e3_r * NCOL + e3_j1, &cm1,
                       sizeof(Fr_t), cudaMemcpyHostToDevice);
            cudaMemcpy(U_t.gpu_data + (size_t)e3_r * NCOL + e3_j2, &mc,
                       sizeof(Fr_t), cudaMemcpyHostToDevice);
        }
        Fr_t* E_raw = fast_me_weights_dev(u_bin);  // == build_eq_tensor values
        lean_hadamard(F_ZERO, E_raw, S_t.gpu_data, U_t.gpu_data, D,
                      tr, ws_bin, hp_bin, evil != 3);
        cudaFree(E_raw);
    }
    absorb_fr(tr, "S_f2", hp_bin.S_f2); absorb_fr(tr, "U_f2", hp_bin.U_f2);
    write_hp(obdir + "/hp_bin.bin", hp_bin);
    vector<Fr_t> pt_bin(ws_bin.rbegin(), ws_bin.rend());
    if (evil == 0) {
        vector<Fr_t> u_col(pt_bin.begin(), pt_bin.begin() + logC);
        vector<Fr_t> u_row(pt_bin.begin() + logC, pt_bin.end());
        if (!fr_eq(hp_bin.S_f2, S_t.multi_dim_me({u_row, u_col}, {B, NCOL})) ||
            !fr_eq(hp_bin.U_f2, h_scalar(hp_bin.S_f2, F_ONE, 1)))
            throw runtime_error("BIN terminal != multi_dim_me / S_f2-1 (convention bug)");
    }
    fast_open_prove(S_t, NCOL, gen, Q, pt_bin, obdir + "/ipa_S_bin.bin", tr);
    mem_mark("BIN");

    // ---- SUM: one-hot-over-allowed (claim 1, protocol constant) ----
    auto u_s = fs_challenge_vec(tr, logB);
    HadamardProof hp_sum;
    hp_sum.claim_H = F_ONE;
    vector<Fr_t> ws_sum;
    {
        FrTensor W(D);
        {
            FrTensor eq_b = build_eq_tensor(u_s);  // size B (tiny)
            FrTensor Wtmp(D);
            k_bcast_rows<<<(D + 255) / 256, 256>>>(eq_b.gpu_data, Wtmp.gpu_data, NCOL, D);
            cudaDeviceSynchronize();
            k_fr_emul<<<(D + 63) / 64, 64>>>(Wtmp.gpu_data, AL_t->gpu_data, W.gpu_data, D);
            cudaDeviceSynchronize();
        }                                          // Wtmp freed pre-sumcheck
        lean_hadamard(F_ONE, W.gpu_data, S_t.gpu_data, ones_t.gpu_data, D,
                      tr, ws_sum, hp_sum, evil != 4);
    }
    absorb_fr(tr, "S_f2", hp_sum.S_f2); absorb_fr(tr, "U_f2", hp_sum.U_f2);
    write_hp(obdir + "/hp_sum.bin", hp_sum);
    vector<Fr_t> pt_s(ws_sum.rbegin(), ws_sum.rend());
    if (evil == 0) {
        vector<Fr_t> u_col(pt_s.begin(), pt_s.begin() + logC);
        vector<Fr_t> u_row(pt_s.begin() + logC, pt_s.end());
        if (!fr_eq(hp_sum.S_f2, S_t.multi_dim_me({u_row, u_col}, {B, NCOL})) ||
            !fr_eq(hp_sum.U_f2, F_ONE))
            throw runtime_error("SUM terminal != multi_dim_me / 1 (convention bug)");
    }
    fast_open_prove(S_t, NCOL, gen, Q, pt_s, obdir + "/ipa_S_sum.bin", tr);
    mem_mark("SUM");

    // ---- MASK: nothing-selected-outside (claim 0, protocol constant) ----
    auto u_m = fs_challenge_vec(tr, logB);
    HadamardProof hp_mask;
    hp_mask.claim_H = F_ZERO;
    vector<Fr_t> ws_mask;
    {
        FrTensor W(D);
        {
            FrTensor eq_b = build_eq_tensor(u_m);  // size B (tiny)
            FrTensor Wtmp(D);
            k_bcast_rows<<<(D + 255) / 256, 256>>>(eq_b.gpu_data, Wtmp.gpu_data, NCOL, D);
            cudaDeviceSynchronize();
            FrTensor NAL = ones_t - *AL_t;
            k_fr_emul<<<(D + 63) / 64, 64>>>(Wtmp.gpu_data, NAL.gpu_data, W.gpu_data, D);
            cudaDeviceSynchronize();
        }                                          // Wtmp, NAL freed pre-sumcheck
        lean_hadamard(F_ZERO, W.gpu_data, S_t.gpu_data, ones_t.gpu_data, D,
                      tr, ws_mask, hp_mask, evil != 5);
    }
    absorb_fr(tr, "S_f2", hp_mask.S_f2); absorb_fr(tr, "U_f2", hp_mask.U_f2);
    write_hp(obdir + "/hp_mask.bin", hp_mask);
    vector<Fr_t> pt_m(ws_mask.rbegin(), ws_mask.rend());
    if (evil == 0) {
        vector<Fr_t> u_col(pt_m.begin(), pt_m.begin() + logC);
        vector<Fr_t> u_row(pt_m.begin() + logC, pt_m.end());
        if (!fr_eq(hp_mask.S_f2, S_t.multi_dim_me({u_row, u_col}, {B, NCOL})) ||
            !fr_eq(hp_mask.U_f2, F_ONE))
            throw runtime_error("MASK terminal != multi_dim_me / 1 (convention bug)");
    }
    fast_open_prove(S_t, NCOL, gen, Q, pt_m, obdir + "/ipa_S_mask.bin", tr);
    mem_mark("MASK");

    // ---- ATT: attainment (claim ev_mx, absorbed) ----
    auto u_a = fs_challenge_vec(tr, logB);
    if (evil == 0)      // pinned fast-vs-slow cross-check (toy AND gen-1024 scale)
        crosscheck_fast_helpers(u_a, vector<Fr_t>(u_bin.begin(),
                                u_bin.begin() + min<uint>(logD, 10)));
    HadamardProof hp_att;
    hp_att.claim_H = mx_t.multi_dim_me({u_a}, {B});           // ev_mx
    absorb_fr(tr, "ev_mx", hp_att.claim_H);
    vector<Fr_t> ws_att;
    {
        FrTensor eq_a = build_eq_tensor(u_a);      // size B (tiny)
        FrTensor W(D);
        k_bcast_rows<<<(D + 255) / 256, 256>>>(eq_a.gpu_data, W.gpu_data, NCOL, D);
        cudaDeviceSynchronize();
        lean_hadamard(hp_att.claim_H, W.gpu_data, S_t.gpu_data, z_t->gpu_data, D,
                      tr, ws_att, hp_att, evil != 1);
    }
    absorb_fr(tr, "S_f2", hp_att.S_f2); absorb_fr(tr, "U_f2", hp_att.U_f2);
    write_hp(obdir + "/hp_att.bin", hp_att);
    vector<Fr_t> pt_a(ws_att.rbegin(), ws_att.rend());
    if (evil == 0) {
        vector<Fr_t> u_col(pt_a.begin(), pt_a.begin() + logC);
        vector<Fr_t> u_row(pt_a.begin() + logC, pt_a.end());
        if (!fr_eq(hp_att.S_f2, S_t.multi_dim_me({u_row, u_col}, {B, NCOL})) ||
            !fr_eq(hp_att.U_f2, z_t->multi_dim_me({u_row, u_col}, {B, NCOL})))
            throw runtime_error("ATT terminal != multi_dim_me (convention bug)");
    }
    fast_open_prove(S_t,  NCOL, gen,   Q, pt_a, obdir + "/ipa_S_att.bin", tr);
    fast_open_prove(*z_t, NCOL, gen,   Q, pt_a, obdir + "/ipa_z_att.bin", tr);
    fast_open_prove(mx_t, B,    genmx, Q, u_a,  obdir + "/ipa_mx_att.bin", tr);
    mem_mark("ATT");

    // ---- DOM: dominance binding ----
    auto u_r = fs_challenge_vec(tr, logD);
    vector<Fr_t> ur_col(u_r.begin(), u_r.begin() + logC);
    vector<Fr_t> ur_row(u_r.begin() + logC, u_r.end());
    // c1: weight eq(u_r).AL on z
    HadamardProof hp_c1;
    {
        FrTensor ALz = *AL_t * *z_t;
        hp_c1.claim_H = ALz.multi_dim_me({ur_row, ur_col}, {B, NCOL});
    }
    absorb_fr(tr, "c1", hp_c1.claim_H);
    vector<Fr_t> ws_c1;
    {
        FrTensor W(D);
        {
            Fr_t* eq_r = fast_me_weights_dev(u_r); // == build_eq_tensor values
            k_fr_emul<<<(D + 63) / 64, 64>>>(eq_r, AL_t->gpu_data, W.gpu_data, D);
            cudaDeviceSynchronize();
            cudaFree(eq_r);
        }
        lean_hadamard(hp_c1.claim_H, W.gpu_data, z_t->gpu_data, ones_t.gpu_data, D,
                      tr, ws_c1, hp_c1, true);
    }
    absorb_fr(tr, "S_f2", hp_c1.S_f2); absorb_fr(tr, "U_f2", hp_c1.U_f2);
    write_hp(obdir + "/hp_c1.bin", hp_c1);
    vector<Fr_t> pt_c1(ws_c1.rbegin(), ws_c1.rend());
    if (evil == 0) {
        vector<Fr_t> u_col(pt_c1.begin(), pt_c1.begin() + logC);
        vector<Fr_t> u_row(pt_c1.begin() + logC, pt_c1.end());
        if (!fr_eq(hp_c1.S_f2, z_t->multi_dim_me({u_row, u_col}, {B, NCOL})) ||
            !fr_eq(hp_c1.U_f2, F_ONE))
            throw runtime_error("c1 terminal != multi_dim_me / 1 (convention bug)");
    }
    fast_open_prove(*z_t, NCOL, gen, Q, pt_c1, obdir + "/ipa_z_c1.bin", tr);
    z_t.reset();                                   // last z use was ipa_z_c1
    mem_mark("c1");

    // c2: same weight on the never-committed broadcast mx_bcast
    HadamardProof hp_c2;
    vector<Fr_t> ws_c2;
    {
        FrTensor mxb(D);
        k_bcast_rows<<<(D + 255) / 256, 256>>>(mx_t.gpu_data, mxb.gpu_data, NCOL, D);
        cudaDeviceSynchronize();
        if (evil == 7) {
            k_bump<<<1,1>>>(mxb.gpu_data, evil_i * NCOL + evil_j, F_ONE, 0);
            cudaDeviceSynchronize();
        }
        {
            FrTensor ALm = *AL_t * mxb;
            hp_c2.claim_H = ALm.multi_dim_me({ur_row, ur_col}, {B, NCOL});
        }
        absorb_fr(tr, "c2", hp_c2.claim_H);
        {
            FrTensor W(D);
            {
                Fr_t* eq_r = fast_me_weights_dev(u_r);  // == build_eq_tensor values
                k_fr_emul<<<(D + 63) / 64, 64>>>(eq_r, AL_t->gpu_data, W.gpu_data, D);
                cudaDeviceSynchronize();
                cudaFree(eq_r);
            }
            lean_hadamard(hp_c2.claim_H, W.gpu_data, mxb.gpu_data, ones_t.gpu_data, D,
                          tr, ws_c2, hp_c2, true);
        }
        if (evil == 0) {
            vector<Fr_t> pt(ws_c2.rbegin(), ws_c2.rend());
            vector<Fr_t> u_col(pt.begin(), pt.begin() + logC);
            vector<Fr_t> u_row(pt.begin() + logC, pt.end());
            if (!fr_eq(hp_c2.S_f2, mxb.multi_dim_me({u_row, u_col}, {B, NCOL})) ||
                !fr_eq(hp_c2.S_f2, mx_t.multi_dim_me({u_row}, {B})) ||
                !fr_eq(hp_c2.U_f2, F_ONE))
                throw runtime_error("c2 terminal != multi_dim_me / bcast suffix (convention bug)");
        }
    }
    absorb_fr(tr, "S_f2", hp_c2.S_f2); absorb_fr(tr, "U_f2", hp_c2.U_f2);
    write_hp(obdir + "/hp_c2.bin", hp_c2);
    vector<Fr_t> pt_c2(ws_c2.rbegin(), ws_c2.rend());
    vector<Fr_t> pt_c2_rows(pt_c2.begin() + logC, pt_c2.end());
    fast_open_prove(mx_t, B, genmx, Q, pt_c2_rows, obdir + "/ipa_mx_c2.bin", tr);
    AL_t.reset();                                  // last AL use was c2's weight
    mem_mark("c2");

    // limb-plane openings + reconstruction values (L re-uploaded from the
    // unchanged host limbs: identical field elements as the committed tensor)
    L_t = make_unique<FrTensor>(DL, Lh.data());
    Fr_t lv[2] = {F_ZERO, F_ZERO};
    if (NPL == 1) {
        lv[0] = L_t->multi_dim_me({ur_row, ur_col}, {B, NCOL});
        absorb_fr(tr, "v0", lv[0]);
        { FILE* f = open_or_die(obdir + "/lvals.bin", "wb");
          fwrite(lv, sizeof(Fr_t), 1, f); fclose(f); }
        if (evil == 0) {        // c2 - c1 == v0 (plain field, design §2.3 DOM)
            Fr_t lhs = h_scalar(hp_c2.claim_H, hp_c1.claim_H, 1);
            if (!fr_eq(lhs, lv[0]))
                throw runtime_error("DOM identity fails on honest witness (convention bug)");
        }
        fast_open_prove(*L_t, NCOL, gen, Q, u_r, obdir + "/ipa_L_p0.bin", tr);
    } else {
        const char* labels[2] = {"v0", "v1"};
        for (uint p = 0; p < 2; p++) {
            vector<Fr_t> u_pt(u_r);
            u_pt.push_back(p ? F_ONE : F_ZERO);    // plane bit = flat bit logD
            vector<Fr_t> u_row2(u_pt.begin() + logC, u_pt.end());
            lv[p] = L_t->multi_dim_me({u_row2, ur_col}, {2 * B, NCOL});
            absorb_fr(tr, labels[p], lv[p]);
        }
        { FILE* f = open_or_die(obdir + "/lvals.bin", "wb");
          fwrite(lv, sizeof(Fr_t), 2, f); fclose(f); }
        if (evil == 0) {
            for (uint p = 0; p < 2; p++) {         // plane opening == plane slice ME
                FrTensor plane(D, Lh.data() + (size_t)p * D);
                if (!fr_eq(lv[p], plane.multi_dim_me({ur_row, ur_col}, {B, NCOL})))
                    throw runtime_error("L plane opening != plane-slice multi_dim_me (convention bug)");
            }
            Fr_t rec = h_scalar(lv[0], h_scalar(Fr_t{LEN_R,0,0,0,0,0,0,0}, lv[1], 2), 0);
            Fr_t lhs = h_scalar(hp_c2.claim_H, hp_c1.claim_H, 1);
            if (!fr_eq(lhs, rec))
                throw runtime_error("DOM identity fails on honest witness (convention bug)");
        }
        for (uint p = 0; p < 2; p++) {
            vector<Fr_t> u_pt(u_r);
            u_pt.push_back(p ? F_ONE : F_ZERO);
            fast_open_prove(*L_t, NCOL, gen, Q, u_pt,
                            obdir + (p ? "/ipa_L_p1.bin" : "/ipa_L_p0.bin"), tr);
        }
    }
    L_t.reset();

    mem_mark("DOM planes");

    // ---- T-BIND: served-token binding (vpad + t*; claim 1, protocol constant) ----
    if (MODE == 1 && tstar) {
        auto u_t = fs_challenge_vec(tr, logB);
        HadamardProof hp_tb;
        hp_tb.claim_H = F_ONE;
        vector<Fr_t> ws_tb;
        {
            FrTensor eq_t = build_eq_tensor(u_t);
            vector<Fr_t> eqh(B);
            cudaMemcpy(eqh.data(), eq_t.gpu_data, sizeof(Fr_t) * B, cudaMemcpyDeviceToHost);
            vector<Fr_t> Wth((size_t)D, F_ZERO);
            for (uint i = 0; i < B; i++)
                Wth[(size_t)i * NCOL + (uint)(*tstar)[i]] = eqh[i];
            FrTensor W(D, Wth.data());
            lean_hadamard(F_ONE, W.gpu_data, S_t.gpu_data, ones_t.gpu_data, D,
                          tr, ws_tb, hp_tb, evil != 8);
        }
        absorb_fr(tr, "S_f2", hp_tb.S_f2); absorb_fr(tr, "U_f2", hp_tb.U_f2);
        write_hp(obdir + "/hp_tbind.bin", hp_tb);
        vector<Fr_t> pt_t(ws_tb.rbegin(), ws_tb.rend());
        if (evil == 0) {
            vector<Fr_t> u_col(pt_t.begin(), pt_t.begin() + logC);
            vector<Fr_t> u_row(pt_t.begin() + logC, pt_t.end());
            if (!fr_eq(hp_tb.S_f2, S_t.multi_dim_me({u_row, u_col}, {B, NCOL})) ||
                !fr_eq(hp_tb.U_f2, F_ONE))
                throw runtime_error("T-BIND terminal != multi_dim_me / 1 (convention bug)");
        }
        fast_open_prove(S_t, NCOL, gen, Q, pt_t, obdir + "/ipa_S_tbind.bin", tr);
    }
    mem_mark("T-BIND/end");
    cout << "PROVED rowmax obligation -> " << obdir << endl;
}

// ---------------------------------------------------------------------------
// verify (witness-free)
#define RJ(msg) do { ostringstream oss_; oss_ << msg; \
    cout << "REJECT: " << oss_.str() << endl; \
    if (reason) *reason = oss_.str(); return false; } while (0)

static Fr_t fold_public(const FrTensor& W, const vector<Fr_t>& ws) {
    uint sz = W.size;
    Fr_t *da, *db;
    cudaMalloc(&da, sz * sizeof(Fr_t));
    cudaMalloc(&db, (sz / 2 + 1) * sizeof(Fr_t));
    cudaMemcpy(da, W.gpu_data, sz * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    for (auto& w : ws) {
        uint nsz = sz >> 1;
        k_fr_fold<<<(nsz + 31) / 32, 32>>>(da, w, db, nsz);
        cudaDeviceSynchronize();
        std::swap(da, db); sz = nsz;
    }
    Fr_t out;
    cudaMemcpy(&out, da, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db);
    return out;
}

static bool verify(const string& obdir, const string& seed,
                   uint B, uint NCOL, uint MODE, uint V, uint LEN_R, uint NPL,
                   const Commitment& gen, const Commitment& genmx, const G1Jacobian_t& Q,
                   const vector<int>* tstar, string* reason = nullptr) {
    const uint D = B * NCOL, DL = NPL * D;
    const uint logB = ceilLog2(B), logC = ceilLog2(NCOL);
    const uint logD = logB + logC, logDL = logD + ceilLog2(NPL);
    const uint n1 = logDL - ceilLog2(LEN_R);
    // layout guards (RJ form)
    if (B != (1u << logB) || NCOL != (1u << logC) ||
        LEN_R != (1u << ceilLog2(LEN_R)) || (NPL != 1 && NPL != 2) ||
        (MODE == 0 && (B != NCOL || V != 0)) ||
        (MODE == 1 && (V == 0 || V > NCOL)) || MODE > 1 ||
        gen.size != NCOL || genmx.size != B ||
        NCOL > LEN_R || LEN_R > DL || DL % LEN_R)
        RJ("bad layout params");
    if (tstar && MODE == 0) RJ("t* supplied in causal mode");
    if (tstar) {
        if (tstar->size() != (size_t)B) RJ("t* size != B");
        for (uint i = 0; i < B; i++)
            if ((*tstar)[i] < 0 || (uint)(*tstar)[i] >= V)
                RJ("t* token out of [0, V)");
    }
    { FILE* f = open_or_die(obdir + "/dims.bin", "rb");
      uint32_t d[6];
      if (fread(d, sizeof(uint32_t), 6, f) != 6) { fclose(f); RJ("dims.bin short read"); }
      fclose(f);
      if (d[0] != B || d[1] != NCOL || d[2] != MODE || d[3] != V ||
          d[4] != LEN_R || d[5] != NPL)
          RJ("dims.bin mismatch"); }

    G1TensorJacobian com_z (obdir + "/com_z.bin");
    G1TensorJacobian com_S (obdir + "/com_S.bin");
    G1TensorJacobian com_mx(obdir + "/com_mx.bin");
    G1TensorJacobian com_L (obdir + "/com_L.bin");
    G1TensorJacobian com_mL(obdir + "/com_m_L.bin");
    G1TensorJacobian com_AL(obdir + "/com_A_L.bin");
    if (com_z.size != B || com_S.size != B || com_mx.size != 1 ||
        com_L.size != NPL * B || com_AL.size != NPL * B ||
        com_mL.size != LEN_R / NCOL)
        RJ("commitment row counts");

    LookupProof pfL = read_lookup(obdir + "/lookup_L.bin");
    HadamardProof hp_bin  = read_hp(obdir + "/hp_bin.bin");
    HadamardProof hp_sum  = read_hp(obdir + "/hp_sum.bin");
    HadamardProof hp_mask = read_hp(obdir + "/hp_mask.bin");
    HadamardProof hp_att  = read_hp(obdir + "/hp_att.bin");
    HadamardProof hp_c1   = read_hp(obdir + "/hp_c1.bin");
    HadamardProof hp_c2   = read_hp(obdir + "/hp_c2.bin");
    HadamardProof hp_tb;
    const bool tb = (MODE == 1 && tstar);
    if (tb) hp_tb = read_hp(obdir + "/hp_tbind.bin");
    Fr_t lv[2] = {F_ZERO, F_ZERO};
    { FILE* f = open_or_die(obdir + "/lvals.bin", "rb");
      if (fread(lv, sizeof(Fr_t), NPL, f) != NPL) { fclose(f); RJ("lvals.bin short read"); }
      fclose(f); }
    if (pfL.ev.size() != 4 * logDL) RJ("limb lookup round count");
    if (hp_bin.ev.size() != 4 * logD || hp_sum.ev.size() != 4 * logD ||
        hp_mask.ev.size() != 4 * logD || hp_att.ev.size() != 4 * logD ||
        hp_c1.ev.size() != 4 * logD || hp_c2.ev.size() != 4 * logD ||
        (tb && hp_tb.ev.size() != 4 * logD))
        RJ("hadamard round count");
    // constant claims: protocol constants, never absorbed; the serialized
    // claim_H must equal the constant the verifier imposes at round 0
    if (!fr_eq(hp_bin.claim_H, F_ZERO))  RJ("BIN claim_H != protocol constant 0");
    if (!fr_eq(hp_sum.claim_H, F_ONE))   RJ("SUM claim_H != protocol constant 1");
    if (!fr_eq(hp_mask.claim_H, F_ZERO)) RJ("MASK claim_H != protocol constant 0");
    if (tb && !fr_eq(hp_tb.claim_H, F_ONE)) RJ("T-BIND claim_H != protocol constant 1");

    vector<int> alh = build_al(B, NCOL, MODE, V);
    FrTensor AL_t(D, alh.data());
    const Fr_t inv6 = inv(F_SIX);

    // ---- transcript replay ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "NCOL", NCOL);
    absorb_u32(tr, "MODE", MODE); absorb_u32(tr, "V", V);
    absorb_u32(tr, "LEN_R", LEN_R); absorb_u32(tr, "NPL", NPL);
    if (tb) tr.absorb("TSTAR", tstar->data(), (size_t)B * sizeof(int));
    absorb_g1_tensor(tr, "com_z", com_z);
    absorb_g1_tensor(tr, "com_S", com_S);
    absorb_g1_tensor(tr, "com_mx", com_mx);
    absorb_g1_tensor(tr, "com_L", com_L);
    absorb_g1_tensor(tr, "com_m_L", com_mL);

    // ---- LIMB ----
    Fr_t beta_L = fs_challenge_fr(tr);
    absorb_g1_tensor(tr, "com_A_L", com_AL);
    Fr_t alpha_L = fs_challenge_fr(tr);
    auto u_L = fs_challenge_vec(tr, logDL);
    Fr_t cur = h_scalar(alpha_L, h_scalar(alpha_L, alpha_L, 2), 0);
    Fr_t alpha_acc = alpha_L, alphasq_acc = h_scalar(alpha_L, alpha_L, 2);
    vector<Fr_t> ws_L;
    for (uint k = 0; k < logDL; k++) {
        array<Fr_t,4> e = {pfL.ev[4*k], pfL.ev[4*k+1], pfL.ev[4*k+2], pfL.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("limb lookup round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "p0", e[0]); absorb_fr(tr, "p1", e[1]);
        absorb_fr(tr, "p2", e[2]); absorb_fr(tr, "p3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_L.push_back(w);
        cur = lagrange4(e, w, inv6);
        Fr_t eqv = my_eq(u_L[logDL - 1 - k], w);
        alpha_acc = h_scalar(alpha_acc, eqv, 2);
        if (k >= n1) alphasq_acc = h_scalar(alphasq_acc, eqv, 2);
    }
    Fr_t B_f, T_f;
    {
        tLookupRange tlR(0, LEN_R);
        FrTensor B_pub(LEN_R);
        tlookup_inv_kernel<<<(LEN_R+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            tlR.table.gpu_data, beta_L, B_pub.gpu_data, LEN_R);
        cudaDeviceSynchronize();
        vector<Fr_t> ws2(ws_L.begin() + n1, ws_L.end());
        B_f = fold_public(B_pub, ws2);
        T_f = fold_public(tlR.table, ws2);
    }
    {
        Fr_t inv_ratio = h_scalar({LEN_R,0,0,0,0,0,0,0}, inv({DL,0,0,0,0,0,0,0}), 2);
        Fr_t t1 = h_scalar(alpha_acc, h_scalar(pfL.A_f, h_scalar(pfL.S_f, beta_L, 0), 2), 2);
        Fr_t t2 = h_scalar(inv_ratio, h_scalar(alphasq_acc, h_scalar(B_f, h_scalar(T_f, beta_L, 0), 2), 2), 2);
        Fr_t t4 = h_scalar(inv_ratio, h_scalar(pfL.m_f, B_f, 2), 2);
        Fr_t rhs = h_scalar(h_scalar(h_scalar(t1, t2, 0), pfL.A_f, 0), t4, 1);
        if (!fr_eq(cur, rhs)) RJ("limb lookup terminal identity");
    }
    absorb_fr(tr, "A_f", pfL.A_f); absorb_fr(tr, "S_f", pfL.S_f); absorb_fr(tr, "m_f", pfL.m_f);
    vector<Fr_t> u_ptL(ws_L.rbegin(), ws_L.rend());
    vector<Fr_t> u_mL(ws_L.rbegin(), ws_L.rend() - n1);
    if (!fast_open_verify(com_AL, gen, NCOL, Q, u_ptL, pfL.A_f, obdir + "/ipa_A_L.bin", tr))
        RJ("IPA opening of A_f vs com_A_L");
    if (!fast_open_verify(com_L, gen, NCOL, Q, u_ptL, pfL.S_f, obdir + "/ipa_L_lk.bin", tr))
        RJ("IPA opening of S_f vs com_L");
    if (!fast_open_verify(com_mL, gen, NCOL, Q, u_mL, pfL.m_f, obdir + "/ipa_m_L.bin", tr))
        RJ("IPA opening of m_f vs com_m_L");

    // ---- BIN (pure eq weight: my_eq accumulator over all rounds) ----
    auto u_bin = fs_challenge_vec(tr, logD);
    cur = F_ZERO;                                  // imposed protocol constant
    Fr_t eq_acc = F_ONE;
    vector<Fr_t> ws_bin;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_bin.ev[4*k], hp_bin.ev[4*k+1], hp_bin.ev[4*k+2], hp_bin.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("BIN round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_bin.push_back(w);
        cur = lagrange4(e, w, inv6);
        eq_acc = h_scalar(eq_acc, my_eq(u_bin[logD - 1 - k], w), 2);
    }
    // load-bearing: U is bound to com_S only through U_f2 == S_f2 - 1
    if (!fr_eq(hp_bin.U_f2, h_scalar(hp_bin.S_f2, F_ONE, 1)))
        RJ("BIN U_f2 != S_f2 - 1");
    if (!fr_eq(cur, h_scalar(eq_acc, h_scalar(hp_bin.S_f2, hp_bin.U_f2, 2), 2)))
        RJ("BIN terminal identity");
    absorb_fr(tr, "S_f2", hp_bin.S_f2); absorb_fr(tr, "U_f2", hp_bin.U_f2);
    vector<Fr_t> pt_bin(ws_bin.rbegin(), ws_bin.rend());
    if (!fast_open_verify(com_S, gen, NCOL, Q, pt_bin, hp_bin.S_f2, obdir + "/ipa_S_bin.bin", tr))
        RJ("IPA opening of S (BIN) vs com_S");

    // ---- SUM (rebuild+fold weight bcast(eq(u_s)) . AL) ----
    auto u_s = fs_challenge_vec(tr, logB);
    cur = F_ONE;                                   // imposed protocol constant
    vector<Fr_t> ws_sum;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_sum.ev[4*k], hp_sum.ev[4*k+1], hp_sum.ev[4*k+2], hp_sum.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("SUM round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_sum.push_back(w);
        cur = lagrange4(e, w, inv6);
    }
    if (!fr_eq(hp_sum.U_f2, F_ONE)) RJ("SUM U_f2 != 1");
    {
        FrTensor eq_b = build_eq_tensor(u_s);
        FrTensor Wtmp(D), W(D);
        k_bcast_rows<<<(D + 255) / 256, 256>>>(eq_b.gpu_data, Wtmp.gpu_data, NCOL, D);
        cudaDeviceSynchronize();
        k_fr_emul<<<(D + 63) / 64, 64>>>(Wtmp.gpu_data, AL_t.gpu_data, W.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t W_f = fold_public(W, ws_sum);
        if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp_sum.S_f2, hp_sum.U_f2, 2), 2)))
            RJ("SUM terminal identity");
    }
    absorb_fr(tr, "S_f2", hp_sum.S_f2); absorb_fr(tr, "U_f2", hp_sum.U_f2);
    vector<Fr_t> pt_s(ws_sum.rbegin(), ws_sum.rend());
    if (!fast_open_verify(com_S, gen, NCOL, Q, pt_s, hp_sum.S_f2, obdir + "/ipa_S_sum.bin", tr))
        RJ("IPA opening of S (SUM) vs com_S");

    // ---- MASK (rebuild+fold weight bcast(eq(u_m)) . (1-AL)) ----
    auto u_m = fs_challenge_vec(tr, logB);
    cur = F_ZERO;                                  // imposed protocol constant
    vector<Fr_t> ws_mask;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_mask.ev[4*k], hp_mask.ev[4*k+1], hp_mask.ev[4*k+2], hp_mask.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("MASK round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_mask.push_back(w);
        cur = lagrange4(e, w, inv6);
    }
    if (!fr_eq(hp_mask.U_f2, F_ONE)) RJ("MASK U_f2 != 1");
    {
        FrTensor eq_b = build_eq_tensor(u_m);
        FrTensor Wtmp(D), W(D);
        k_bcast_rows<<<(D + 255) / 256, 256>>>(eq_b.gpu_data, Wtmp.gpu_data, NCOL, D);
        cudaDeviceSynchronize();
        vector<int> onesh((size_t)D, 1);
        FrTensor ones_t(D, onesh.data());
        FrTensor NAL = ones_t - AL_t;
        k_fr_emul<<<(D + 63) / 64, 64>>>(Wtmp.gpu_data, NAL.gpu_data, W.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t W_f = fold_public(W, ws_mask);
        if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp_mask.S_f2, hp_mask.U_f2, 2), 2)))
            RJ("MASK terminal identity");
    }
    absorb_fr(tr, "S_f2", hp_mask.S_f2); absorb_fr(tr, "U_f2", hp_mask.U_f2);
    vector<Fr_t> pt_m(ws_mask.rbegin(), ws_mask.rend());
    if (!fast_open_verify(com_S, gen, NCOL, Q, pt_m, hp_mask.S_f2, obdir + "/ipa_S_mask.bin", tr))
        RJ("IPA opening of S (MASK) vs com_S");

    // ---- ATT (pure-broadcast weight: rmsnorm eq_acc shortcut, row rounds only) ----
    auto u_a = fs_challenge_vec(tr, logB);
    absorb_fr(tr, "ev_mx", hp_att.claim_H);
    cur = hp_att.claim_H;
    eq_acc = F_ONE;
    vector<Fr_t> ws_att;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_att.ev[4*k], hp_att.ev[4*k+1], hp_att.ev[4*k+2], hp_att.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("ATT round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_att.push_back(w);
        cur = lagrange4(e, w, inv6);
        if (k < logB)   // row bits are the MSBs; column rounds carry eq factor 1
            eq_acc = h_scalar(eq_acc, my_eq(u_a[logB - 1 - k], w), 2);
    }
    if (!fr_eq(cur, h_scalar(eq_acc, h_scalar(hp_att.S_f2, hp_att.U_f2, 2), 2)))
        RJ("ATT terminal identity");
    absorb_fr(tr, "S_f2", hp_att.S_f2); absorb_fr(tr, "U_f2", hp_att.U_f2);
    vector<Fr_t> pt_a(ws_att.rbegin(), ws_att.rend());
    if (!fast_open_verify(com_S, gen, NCOL, Q, pt_a, hp_att.S_f2, obdir + "/ipa_S_att.bin", tr))
        RJ("IPA opening of S (ATT) vs com_S");
    if (!fast_open_verify(com_z, gen, NCOL, Q, pt_a, hp_att.U_f2, obdir + "/ipa_z_att.bin", tr))
        RJ("IPA opening of z (ATT) vs com_z");
    if (!fast_open_verify(com_mx, genmx, B, Q, u_a, hp_att.claim_H, obdir + "/ipa_mx_att.bin", tr))
        RJ("IPA opening of ev_mx vs com_mx");

    // ---- DOM ----
    auto u_r = fs_challenge_vec(tr, logD);
    // c1
    absorb_fr(tr, "c1", hp_c1.claim_H);
    cur = hp_c1.claim_H;
    vector<Fr_t> ws_c1;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_c1.ev[4*k], hp_c1.ev[4*k+1], hp_c1.ev[4*k+2], hp_c1.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("c1 round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_c1.push_back(w);
        cur = lagrange4(e, w, inv6);
    }
    if (!fr_eq(hp_c1.U_f2, F_ONE)) RJ("c1 U_f2 != 1");
    {
        FrTensor eq_r = build_eq_tensor(u_r);
        FrTensor W(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(eq_r.gpu_data, AL_t.gpu_data, W.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t W_f = fold_public(W, ws_c1);
        if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp_c1.S_f2, hp_c1.U_f2, 2), 2)))
            RJ("c1 terminal identity");
    }
    absorb_fr(tr, "S_f2", hp_c1.S_f2); absorb_fr(tr, "U_f2", hp_c1.U_f2);
    vector<Fr_t> pt_c1(ws_c1.rbegin(), ws_c1.rend());
    if (!fast_open_verify(com_z, gen, NCOL, Q, pt_c1, hp_c1.S_f2, obdir + "/ipa_z_c1.bin", tr))
        RJ("IPA opening of z (c1) vs com_z");
    // c2
    absorb_fr(tr, "c2", hp_c2.claim_H);
    cur = hp_c2.claim_H;
    vector<Fr_t> ws_c2;
    for (uint k = 0; k < logD; k++) {
        array<Fr_t,4> e = {hp_c2.ev[4*k], hp_c2.ev[4*k+1], hp_c2.ev[4*k+2], hp_c2.ev[4*k+3]};
        if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
            RJ("c2 round " << k << " p(0)+p(1) != claim");
        absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
        absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
        Fr_t w = fs_challenge_fr(tr); ws_c2.push_back(w);
        cur = lagrange4(e, w, inv6);
    }
    if (!fr_eq(hp_c2.U_f2, F_ONE)) RJ("c2 U_f2 != 1");
    {
        FrTensor eq_r = build_eq_tensor(u_r);
        FrTensor W(D);
        k_fr_emul<<<(D + 63) / 64, 64>>>(eq_r.gpu_data, AL_t.gpu_data, W.gpu_data, D);
        cudaDeviceSynchronize();
        Fr_t W_f = fold_public(W, ws_c2);
        if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp_c2.S_f2, hp_c2.U_f2, 2), 2)))
            RJ("c2 terminal identity");
    }
    absorb_fr(tr, "S_f2", hp_c2.S_f2); absorb_fr(tr, "U_f2", hp_c2.U_f2);
    vector<Fr_t> pt_c2(ws_c2.rbegin(), ws_c2.rend());
    vector<Fr_t> pt_c2_rows(pt_c2.begin() + logC, pt_c2.end());
    // the never-committed broadcast is pinned to com_mx here: its MLE at pt_c2
    // equals the row vector's MLE at the row-bit suffix
    if (!fast_open_verify(com_mx, genmx, B, Q, pt_c2_rows, hp_c2.S_f2,
                          obdir + "/ipa_mx_c2.bin", tr))
        RJ("IPA opening of c2 terminal vs com_mx");
    // limb-plane openings
    if (NPL == 1) {
        absorb_fr(tr, "v0", lv[0]);
        if (!fast_open_verify(com_L, gen, NCOL, Q, u_r, lv[0], obdir + "/ipa_L_p0.bin", tr))
            RJ("IPA opening of L plane 0 vs com_L");
    } else {
        absorb_fr(tr, "v0", lv[0]);
        absorb_fr(tr, "v1", lv[1]);
        for (uint p = 0; p < 2; p++) {
            vector<Fr_t> u_pt(u_r);
            u_pt.push_back(p ? F_ONE : F_ZERO);
            if (!fast_open_verify(com_L, gen, NCOL, Q, u_pt, lv[p],
                                  obdir + (p ? "/ipa_L_p1.bin" : "/ipa_L_p0.bin"), tr))
                RJ("IPA opening of L plane " << p << " vs com_L");
        }
    }

    // ---- T-BIND ----
    if (tb) {
        auto u_t = fs_challenge_vec(tr, logB);
        cur = F_ONE;                               // imposed protocol constant
        vector<Fr_t> ws_tb;
        for (uint k = 0; k < logD; k++) {
            array<Fr_t,4> e = {hp_tb.ev[4*k], hp_tb.ev[4*k+1], hp_tb.ev[4*k+2], hp_tb.ev[4*k+3]};
            if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
                RJ("T-BIND round " << k << " p(0)+p(1) != claim");
            absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
            absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
            Fr_t w = fs_challenge_fr(tr); ws_tb.push_back(w);
            cur = lagrange4(e, w, inv6);
        }
        if (!fr_eq(hp_tb.U_f2, F_ONE)) RJ("T-BIND U_f2 != 1");
        {
            // W_t gathered from the verifier's OWN t* copy (one device upload)
            FrTensor eq_t = build_eq_tensor(u_t);
            vector<Fr_t> eqh(B);
            cudaMemcpy(eqh.data(), eq_t.gpu_data, sizeof(Fr_t) * B, cudaMemcpyDeviceToHost);
            vector<Fr_t> Wth((size_t)D, F_ZERO);
            for (uint i = 0; i < B; i++)
                Wth[(size_t)i * NCOL + (uint)(*tstar)[i]] = eqh[i];
            FrTensor W(D, Wth.data());
            Fr_t W_f = fold_public(W, ws_tb);
            if (!fr_eq(cur, h_scalar(W_f, h_scalar(hp_tb.S_f2, hp_tb.U_f2, 2), 2)))
                RJ("T-BIND terminal identity");
        }
        absorb_fr(tr, "S_f2", hp_tb.S_f2); absorb_fr(tr, "U_f2", hp_tb.U_f2);
        vector<Fr_t> pt_t(ws_tb.rbegin(), ws_tb.rend());
        if (!fast_open_verify(com_S, gen, NCOL, Q, pt_t, hp_tb.S_f2,
                              obdir + "/ipa_S_tbind.bin", tr))
            RJ("IPA opening of S (T-BIND) vs com_S");
    }

    // ---- DOM bracket identity (plain field, Schwartz-Zippel at u_r) ----
    {
        Fr_t rec = lv[0];
        if (NPL == 2)
            rec = h_scalar(lv[0], h_scalar(Fr_t{LEN_R,0,0,0,0,0,0,0}, lv[1], 2), 0);
        Fr_t lhs = h_scalar(hp_c2.claim_H, hp_c1.claim_H, 1);    // c2 - c1
        if (!fr_eq(lhs, rec)) RJ("DOM bracket identity");
    }
    cout << "ACCEPT" << endl;
    return true;
}

// ---------------------------------------------------------------------------
// selftest
static void tamper_byte(const string& path, long offset, int delta) {
    FILE* f = open_or_die(path, "rb+");
    fseek(f, offset, offset >= 0 ? SEEK_SET : SEEK_END);
    int c = fgetc(f);
    fseek(f, -1, SEEK_CUR);
    fputc((c + delta) & 0xff, f);
    fclose(f);
}
static long file_size(const string& path) {
    struct stat st;
    if (stat(path.c_str(), &st)) return 0;
    return (long)st.st_size;
}
static vector<string> proof_files(uint NPL, bool tb) {
    vector<string> v = {
        "dims.bin",
        "com_z.bin", "com_S.bin", "com_mx.bin", "com_L.bin", "com_m_L.bin",
        "com_A_L.bin",
        "lookup_L.bin",
        "hp_bin.bin", "hp_sum.bin", "hp_mask.bin", "hp_att.bin",
        "hp_c1.bin", "hp_c2.bin",
        "lvals.bin",
        "ipa_A_L.bin", "ipa_L_lk.bin", "ipa_m_L.bin",
        "ipa_S_bin.bin", "ipa_S_sum.bin", "ipa_S_mask.bin",
        "ipa_S_att.bin", "ipa_z_att.bin", "ipa_mx_att.bin",
        "ipa_z_c1.bin", "ipa_mx_c2.bin", "ipa_L_p0.bin"};
    if (NPL == 2) v.push_back("ipa_L_p1.bin");
    if (tb) { v.push_back("hp_tbind.bin"); v.push_back("ipa_S_tbind.bin"); }
    return v;
}
static long tamper_offset(const string& f) {
    if (f == "dims.bin") return 0;
    if (f == "lvals.bin") return 4;                 // inside v0
    if (f.substr(0, 4) == "ipa_") return -32;       // a_final
    if (f.substr(0, 4) == "com_") return 24;        // first point, x limbs
    if (f.substr(0, 3) == "hp_") return 36;         // round-0 evaluation
    return 4 + 32;                                  // lookup: round-0 evaluation
}

// GPU memory monitor (selftest only; cudaMemGetInfo from a second host thread)
static atomic<bool> mon_run{false};
static atomic<long long> mon_peak{0};
static void mem_monitor() {
    while (mon_run.load()) {
        size_t fr = 0, to = 0;
        cudaMemGetInfo(&fr, &to);
        long long used = (long long)(to - fr);
        long long pk = mon_peak.load();
        while (used > pk && !mon_peak.compare_exchange_weak(pk, used)) {}
        pk = mon_phase_peak.load();
        while (used > pk && !mon_phase_peak.compare_exchange_weak(pk, used)) {}
        this_thread::sleep_for(chrono::milliseconds(50));
    }
}

// honest lowest-index argmax t* (np.argmax convention, design §2.1)
static vector<int> host_tstar(const vector<int>& zh_unpadded, uint B, uint V) {
    vector<int> t(B);
    for (uint i = 0; i < B; i++) {
        long long best = LLONG_MIN; uint bj = 0;
        for (uint j = 0; j < V; j++)
            if ((long long)zh_unpadded[(size_t)i * V + j] > best) {
                best = zh_unpadded[(size_t)i * V + j]; bj = j;
            }
        t[i] = (int)bj;
    }
    return t;
}

struct EvilCase { int mode; uint i, j; const char* expect; const char* what; };

static bool selftest_case(uint B, uint NCOL, uint MODE, uint V, uint LEN_R, uint NPL,
                          bool with_tstar) {
    const uint D = B * NCOL;
    const uint COLS = (MODE == 1) ? V : NCOL;      // unpadded column count
    cout << "==== selftest case " << (MODE ? "vpad" : "causal")
         << " B=" << B << " NCOL=" << NCOL << " V=" << V << " LEN_R=" << LEN_R
         << " NPL=" << NPL << (with_tstar ? " +t*" : "")
         << " (n1=" << (ceilLog2(NPL * D) - ceilLog2(LEN_R)) << ") ====" << endl;
    // toy z: random ints with spread < LEN_R^NPL; rows 0/1/2 engineered for the
    // evil-mode setups (design §2.9: witness engineering is part of the setup)
    const int K = (NPL == 1) ? (int)LEN_R / 2 - 1 : 64;
    srand(31337 + B + NCOL);
    vector<int> zh((size_t)B * COLS);
    for (size_t k = 0; k < zh.size(); k++) zh[k] = (rand() % (2 * K + 1)) - K;
    if (B >= 3 && COLS >= 2) {
        if (NPL == 2) {                            // evil 6 needs an entry with Df >= LEN_R
            zh[0 * (size_t)COLS + 0] = 60; zh[0 * (size_t)COLS + 1] = -60;
        }
        // evil 2/3: row 1 has unique max at j=0 and second-distinct = max-1 at j=1
        zh[1 * (size_t)COLS + 0] = K; zh[1 * (size_t)COLS + 1] = K - 1;
        for (uint j = 2; j < COLS; j++)
            zh[1 * (size_t)COLS + j] = -(int)(j % (K > 2 ? (uint)K : 2u)) - 1;
        // evil 4: row 2 has an allowed zero at j=1, max positive at j=0
        zh[2 * (size_t)COLS + 0] = K; zh[2 * (size_t)COLS + 1] = 0;
    }
    vector<int> tst;
    if (with_tstar) tst = host_tstar(zh, B, V);
    Commitment gen = Commitment::random(NCOL);
    Commitment genmx = Commitment::random(B);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_rowmax_ob";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:rowmax";
    bool all = true;

    prove(obdir, seed, zh, B, NCOL, MODE, V, LEN_R, NPL, gen, genmx, Q,
          "/tmp/zkob_rowmax_mx.i32.bin", with_tstar ? &tst : nullptr);
    bool honest = verify(obdir, seed, B, NCOL, MODE, V, LEN_R, NPL, gen, genmx, Q,
                         with_tstar ? &tst : nullptr);
    cout << (honest ? "PASS" : "FAIL") << ": honest case "
         << (honest ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && honest;

    // semantic evil modes: each rejected by EXACTLY the named check
    vector<EvilCase> evils = {
        {1, 0, 0, "ATT round 0",
            "mx[0]+=1, S honest, Df/limbs recomputed from evil mx"},
        {2, 1, 0, "DOM bracket identity",
            "mx[1]-=1, S moved to second-distinct, residual -1 limbs truncated"},
        {3, 1, 0, "BIN round 0",
            "fractional selector S[r,j1]=c, S[r,j2]=1-c at a scanned row, mx honest"},
        {4, 2, 0, "SUM round 0",
            "two-hot: S[2,j2]+=1 at allowed z=0"},
        {7, 0, 0, "IPA opening of c2 terminal vs com_mx",
            "c2 broadcast buffer mx_bcast[0,0]+=1, c2 absorbed from that run"},
    };
    if (MODE == 1 && V < NCOL)
        evils.push_back({5, 0, 0, "MASK round 0", "selector on pad column S[0,V]+=1"});
    if (NPL == 2)
        evils.push_back({6, 0, 0, "limb lookup round 0",
            "lo+=LEN_R, hi-=1 (value unchanged, m_L honest)"});
    if (with_tstar)
        evils.push_back({8, 1, 0, "T-BIND round 0",
            "t*[1] = allowed non-argmax token, S honest"});
    string evdir = "/tmp/zkob_rowmax_evil";
    mkdir(evdir.c_str(), 0755);
    for (auto& ev : evils) {
        // t* handling: evil 2 moves the argmax of row 1 to j=1 -> a consistent
        // t* moves with it; evil 8 keeps S honest and ONLY corrupts t*[1].
        vector<int> tloc = tst;
        const vector<int>* tp = with_tstar ? &tloc : nullptr;
        if (with_tstar && (ev.mode == 2 || ev.mode == 8)) tloc[1] = 1;
        if (ev.mode == 3 && with_tstar) tp = nullptr;   // T-BIND off: the
        // fractional selector makes the T-BIND recursion inconsistent too;
        // strict=false stays on the targeted recursion only (BIN catches)
        prove(evdir, seed, zh, B, NCOL, MODE, V, LEN_R, NPL, gen, genmx, Q, "",
              tp, ev.mode, ev.i, ev.j);
        string reason;
        bool rejected = !verify(evdir, seed, B, NCOL, MODE, V, LEN_R, NPL,
                                gen, genmx, Q, tp, &reason);
        bool right = rejected && reason.find(ev.expect) != string::npos;
        cout << (right ? "PASS" : "FAIL") << ": evil=" << ev.mode << " (" << ev.what
             << ") rejected by [" << (rejected ? reason : string("NOT REJECTED"))
             << "], expected [" << ev.expect << "]" << endl;
        all = all && right;
    }

    // byte tampers on every proof file (tamper, must reject, restore)
    for (const string& fn : proof_files(NPL, with_tstar)) {
        long off = tamper_offset(fn);
        tamper_byte(obdir + "/" + fn, off, +1);
        bool rejected;
        try {
            rejected = !verify(obdir, seed, B, NCOL, MODE, V, LEN_R, NPL,
                               gen, genmx, Q, with_tstar ? &tst : nullptr);
        } catch (const exception& e) {
            rejected = true;                       // fail-closed (MINOR-4 posture)
            cout << "  (verify threw: " << e.what() << ")" << endl;
        }
        tamper_byte(obdir + "/" + fn, off, -1);
        cout << (rejected ? "PASS" : "FAIL") << ": byte tamper " << fn << "@" << off
             << " rejected: " << (rejected ? "YES" : "NO(!!)") << endl;
        all = all && rejected;
    }
    {   // serialized-claim_H forgery (audit MINOR-3): tamper hp_bin.bin@0
        // (inside claim_H, bytes 0-31) -- must be rejected by the EQUALITY half
        // of the constant-claim discipline, the exact named check
        tamper_byte(obdir + "/hp_bin.bin", 0, +1);
        string reason;
        bool rejected;
        try {
            rejected = !verify(obdir, seed, B, NCOL, MODE, V, LEN_R, NPL,
                               gen, genmx, Q, with_tstar ? &tst : nullptr, &reason);
        } catch (const exception& e) { rejected = true; reason = e.what(); }
        tamper_byte(obdir + "/hp_bin.bin", 0, -1);
        const char* expect = "BIN claim_H != protocol constant 0";
        bool right = rejected && reason.find(expect) != string::npos;
        cout << (right ? "PASS" : "FAIL") << ": claim_H forgery hp_bin.bin@0 rejected by ["
             << (rejected ? reason : string("NOT REJECTED")) << "], expected ["
             << expect << "]" << endl;
        all = all && right;
    }
    if (NPL == 2) {                                // v1 tamper (audit MINOR-4):
        // lvals.bin@36 is inside v1 (bytes 32-63), mirroring the v0@4 tamper
        tamper_byte(obdir + "/lvals.bin", 32 + 4, +1);
        string reason;
        bool rejected;
        try {
            rejected = !verify(obdir, seed, B, NCOL, MODE, V, LEN_R, NPL,
                               gen, genmx, Q, with_tstar ? &tst : nullptr, &reason);
        } catch (const exception& e) { rejected = true; reason = e.what(); }
        tamper_byte(obdir + "/lvals.bin", 32 + 4, -1);
        cout << (rejected ? "PASS" : "FAIL") << ": byte tamper lvals.bin@36 (v1)"
             << " rejected: " << (rejected ? "YES [" + reason + "]" : "NO(!!)") << endl;
        all = all && rejected;
    }
    bool restored = verify(obdir, seed, B, NCOL, MODE, V, LEN_R, NPL, gen, genmx, Q,
                           with_tstar ? &tst : nullptr);
    cout << (restored ? "PASS" : "FAIL") << ": restored verify "
         << (restored ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && restored;
    cout << (all ? "CASE PASS" : "CASE FAIL") << endl;
    return all;
}

// honest-prover throw guards (design §2.1): each setup must THROW
static bool selftest_guards() {
    cout << "==== selftest guard throws (design §2.1) ====" << endl;
    Commitment g8 = Commitment::random(8), g8b = Commitment::random(8);
    Commitment g16 = Commitment::random(16);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_rowmax_guard";
    mkdir(obdir.c_str(), 0755);
    vector<int> z64(64, 0), z88(88, 0);
    vector<int> tbad = {0, 0, 0, 0, 0, 0, 0, 99};
    vector<int> tok(8, 0);
    struct G { const char* what; function<void()> run; };
    vector<G> gs = {
        {"causal B != NCOL", [&]{ prove(obdir, "g", z88, 8, 16, 0, 0, 32, 1, g16, g8, Q, "", nullptr); }},
        {"causal V != 0",    [&]{ prove(obdir, "g", z64, 8, 8, 0, 4, 32, 1, g8, g8b, Q, "", nullptr); }},
        {"vpad V = 0",       [&]{ prove(obdir, "g", z64, 8, 8, 1, 0, 32, 2, g8, g8b, Q, "", nullptr); }},
        {"vpad V > NCOL",    [&]{ prove(obdir, "g", z88, 8, 8, 1, 11, 32, 2, g8, g8b, Q, "", nullptr); }},
        {"gen_grid size",    [&]{ prove(obdir, "g", z64, 8, 8, 0, 0, 32, 1, g16, g8b, Q, "", nullptr); }},
        {"gen_mx size",      [&]{ prove(obdir, "g", z64, 8, 8, 0, 0, 32, 1, g8, g16, Q, "", nullptr); }},
        {"t* in causal",     [&]{ prove(obdir, "g", z64, 8, 8, 0, 0, 32, 1, g8, g8b, Q, "", &tok); }},
        {"t* out of range",  [&]{ vector<int> zv(64, 0);
                                  prove(obdir, "g", zv, 8, 8, 1, 8, 32, 2, g8, g8b, Q, "", &tbad); }},
        {"Df >= LEN_R^NPL",  [&]{ vector<int> zv(z64); zv[8] = 40; zv[9] = -40;  // row 1 spread 80 >= 32
                                  prove(obdir, "g", zv, 8, 8, 0, 0, 32, 1, g8, g8b, Q, "", nullptr); }},
        {"vpad |z| >= 2^25", [&]{ vector<int> zv(64, 0); zv[0] = 1 << 25;
                                  prove(obdir, "g", zv, 8, 8, 1, 8, 32, 2, g8, g8b, Q, "", nullptr); }},
        {"LEN_R not pow2",   [&]{ prove(obdir, "g", z64, 8, 8, 0, 0, 33, 1, g8, g8b, Q, "", nullptr); }},
        {"NCOL > LEN_R",     [&]{ prove(obdir, "g", z64, 8, 8, 0, 0, 4, 1, g8, g8b, Q, "", nullptr); }},
        // audit MINOR-5: the three previously unexercised §2.1 guards
        {"B not pow2",       [&]{ prove(obdir, "g", z64, 6, 8, 0, 0, 32, 1, g8, g8b, Q, "", nullptr); }},
        {"NCOL not pow2",    [&]{ prove(obdir, "g", z64, 8, 12, 0, 0, 32, 1, g8, g8b, Q, "", nullptr); }},
        {"NPL not in {1,2}", [&]{ prove(obdir, "g", z64, 8, 8, 0, 0, 32, 3, g8, g8b, Q, "", nullptr); }},
    };
    bool all = true;
    for (auto& g : gs) {
        bool threw = false;
        string msg;
        try { g.run(); } catch (const exception& e) { threw = true; msg = e.what(); }
        cout << (threw ? "PASS" : "FAIL") << ": guard [" << g.what << "] "
             << (threw ? "threw: " + msg : string("DID NOT THROW(!!)")) << endl;
        all = all && threw;
    }
    return all;
}

// chunked-commit byte test (audit MINOR-6): the toy proof shapes all take
// commit_chunked's fall-through gen.commit() branch, so the §4 byte-identity
// diff never exercised the CHUNKED branch at byte level. Here rows is pushed
// just past CHUNK_ROWS (2^22/G = 1024 at G = 4096), forcing the chunked path
// with a full chunk plus a partial tail chunk, and the output buffer is
// compared byte-for-byte against the unchunked upstream Commitment::commit
// on the same tensor.
static bool selftest_chunked_commit() {
    const uint G = 4096, rows = 1025;              // CHUNK_ROWS = 1024 -> chunked
    cout << "==== selftest chunked-vs-unchunked commit byte test (G=" << G
         << ", rows=" << rows << ") ====" << endl;
    Commitment gen = Commitment::random(G);
    vector<long> vals((size_t)rows * G);
    mt19937_64 rng(20260611);
    for (auto& v : vals) v = (long)(rng() % 1000001) - 500000;
    FrTensor t((uint)vals.size(), vals.data());
    G1TensorJacobian a = commit_chunked(gen, t);
    G1TensorJacobian b = gen.commit(t);
    vector<unsigned char> ha((size_t)rows * sizeof(G1Jacobian_t));
    vector<unsigned char> hb((size_t)rows * sizeof(G1Jacobian_t));
    cudaMemcpy(ha.data(), a.gpu_data, ha.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b.gpu_data, hb.size(), cudaMemcpyDeviceToHost);
    bool ok = (memcmp(ha.data(), hb.data(), ha.size()) == 0);
    cout << (ok ? "PASS" : "FAIL") << ": chunked commit bytes "
         << (ok ? "IDENTICAL to unchunked" : "DIFFER from unchunked(!!)") << endl;
    return ok;
}

static bool selftest_real_causal() {
    const uint B = 1024, NCOL = 1024, LEN_R = 1u << 20, NPL = 1;
    cout << "==== selftest real-scale causal 1024x1024 LEN_R=2^20 NPL=1 ====" << endl;
    const string genpath = "/tmp/gen1024.bin";
    if (file_size(genpath) != 1024L * (long)sizeof(G1Jacobian_t)) {
        cout << "generating " << genpath << " via ppgen..." << endl;
        if (system(("./ppgen 1024 " + genpath).c_str()))
            throw runtime_error("ppgen 1024 failed");
    }
    Commitment gen(genpath);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    // scores-like data: round(N(0, 2^13)) clipped to +-(2^19-1) (design §2.9)
    mt19937_64 rng(20260611);
    normal_distribution<double> nd(0.0, 8192.0);
    vector<int> zh((size_t)B * NCOL);
    const long long CLIP = (1LL << 19) - 1;
    for (size_t k = 0; k < zh.size(); k++) {
        long long v = llround(nd(rng));
        if (v < -CLIP) v = -CLIP;
        if (v > CLIP) v = CLIP;
        zh[k] = (int)v;
    }
    string obdir = "/tmp/zkob_rowmax_real_causal";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:rowmax:real:causal";
    bool all = true;

    mon_peak = 0; mon_run = true;
    thread mt(mem_monitor);
    auto t0 = chrono::steady_clock::now();
    prove(obdir, seed, zh, B, NCOL, 0, 0, LEN_R, NPL, gen, gen, Q,
          "/tmp/zkob_rowmax_real_mx.i32.bin", nullptr);
    auto t1 = chrono::steady_clock::now();
    bool honest = verify(obdir, seed, B, NCOL, 0, 0, LEN_R, NPL, gen, gen, Q, nullptr);
    auto t2 = chrono::steady_clock::now();
    mon_run = false; mt.join();
    double prove_s = chrono::duration<double>(t1 - t0).count();
    double verify_s = chrono::duration<double>(t2 - t1).count();
    long bytes = 0;
    for (auto& fn : proof_files(NPL, false)) bytes += file_size(obdir + "/" + fn);
    cout << (honest ? "PASS" : "FAIL") << ": real-scale causal honest "
         << (honest ? "ACCEPT" : "REJECT(!!)") << "  prove " << prove_s
         << " s, verify " << verify_s << " s, proof+commitments " << bytes
         << " bytes, GPU peak " << (double)mon_peak.load() / (1u << 30) << " GiB" << endl;
    all = all && honest;

    tamper_byte(obdir + "/lookup_L.bin", 4 + 32, +1);
    bool rejected = !verify(obdir, seed, B, NCOL, 0, 0, LEN_R, NPL, gen, gen, Q, nullptr);
    tamper_byte(obdir + "/lookup_L.bin", 4 + 32, -1);
    cout << (rejected ? "PASS" : "FAIL") << ": real-scale causal byte tamper rejected: "
         << (rejected ? "YES" : "NO(!!)") << endl;
    all = all && rejected;
    cout << (all ? "REAL-SCALE CAUSAL: PASS" : "REAL-SCALE CAUSAL: FAIL") << endl;
    return all;
}

static bool selftest_real_vpad() {
    const uint B = 1024, NCOL = 32768, V = 32000, LEN_R = 1u << 20, NPL = 2;
    cout << "==== selftest real-scale vpad 1024x32768 V=32000 LEN_R=2^20 NPL=2 +t* ===="
         << endl;
    const string genpath = "/tmp/gen32768.bin";
    if (file_size(genpath) != 32768L * (long)sizeof(G1Jacobian_t)) {
        cout << "generating " << genpath << " via ppgen..." << endl;
        if (system(("./ppgen 32768 " + genpath).c_str()))
            throw runtime_error("ppgen 32768 failed");
    }
    const string genmxpath = "/tmp/gen1024.bin";
    if (file_size(genmxpath) != 1024L * (long)sizeof(G1Jacobian_t)) {
        cout << "generating " << genmxpath << " via ppgen..." << endl;
        if (system(("./ppgen 1024 " + genmxpath).c_str()))
            throw runtime_error("ppgen 1024 failed");
    }
    Commitment gen(genpath);
    Commitment genmx(genmxpath);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    // logits-like data: round(N(0, 2^21)) clipped to +-(2^25-1) (design §2.9)
    mt19937_64 rng(20260612);
    normal_distribution<double> nd(0.0, 2097152.0);
    vector<int> zh((size_t)B * V);
    const long long CLIP = (1LL << 25) - 1;
    for (size_t k = 0; k < zh.size(); k++) {
        long long v = llround(nd(rng));
        if (v < -CLIP) v = -CLIP;
        if (v > CLIP) v = CLIP;
        zh[k] = (int)v;
    }
    vector<int> tst = host_tstar(zh, B, V);        // t* = the true argmax
    string obdir = "/tmp/zkob_rowmax_real_vpad";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:rowmax:real:vpad";
    bool all = true;

    mon_peak = 0; mon_run = true;
    thread mt(mem_monitor);
    auto t0 = chrono::steady_clock::now();
    prove(obdir, seed, zh, B, NCOL, 1, V, LEN_R, NPL, gen, genmx, Q,
          "/tmp/zkob_rowmax_real_vpad_mx.i32.bin", &tst);
    auto t1 = chrono::steady_clock::now();
    bool honest = verify(obdir, seed, B, NCOL, 1, V, LEN_R, NPL, gen, genmx, Q, &tst);
    auto t2 = chrono::steady_clock::now();
    mon_run = false; mt.join();
    double prove_s = chrono::duration<double>(t1 - t0).count();
    double verify_s = chrono::duration<double>(t2 - t1).count();
    double peak_gib = (double)mon_peak.load() / (1u << 30);
    long bytes = 0;
    for (auto& fn : proof_files(NPL, true)) bytes += file_size(obdir + "/" + fn);
    cout << (honest ? "PASS" : "FAIL") << ": real-scale vpad honest "
         << (honest ? "ACCEPT" : "REJECT(!!)") << "  prove " << prove_s
         << " s, verify " << verify_s << " s, proof+commitments " << bytes
         << " bytes" << endl;
    cout << "GPU memory peak " << peak_gib << " GiB vs the ~18 GiB §6.3 gate: "
         << (peak_gib < 18.0 ? "WITHIN GATE" : "EXCEEDS GATE(!!)") << endl;
    all = all && honest && (peak_gib < 18.0);

    tamper_byte(obdir + "/lookup_L.bin", 4 + 32, +1);
    bool rejected = !verify(obdir, seed, B, NCOL, 1, V, LEN_R, NPL, gen, genmx, Q, &tst);
    tamper_byte(obdir + "/lookup_L.bin", 4 + 32, -1);
    cout << (rejected ? "PASS" : "FAIL") << ": real-scale vpad byte tamper rejected: "
         << (rejected ? "YES" : "NO(!!)") << endl;
    all = all && rejected;
    cout << (all ? "REAL-SCALE VPAD: PASS" : "REAL-SCALE VPAD: FAIL") << endl;
    return all;
}

static vector<int> load_i32(const string& path, size_t expect) {
    FILE* f = open_or_die(path, "rb");
    vector<int> v(expect);
    if (fread(v.data(), sizeof(int), expect, f) != expect)
        throw runtime_error("short read / wrong dims: " + path);
    fclose(f);
    return v;
}

int main(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    if (mode == "selftest") {
        bool a = selftest_case(8,  8,  0, 0,  32, 1, false);  // causal toy; diag-only row 0
        bool b = selftest_case(8,  16, 1, 11, 32, 2, true);   // vpad: pads + t* + plane bit
        bool c = selftest_case(16, 16, 0, 0,  64, 1, false);  // bigger causal grid
        bool d = selftest_case(4,  8,  1, 8,  32, 2, false);  // V == NCOL: MASK weight == 0
        bool g = selftest_guards();
        bool cc = selftest_chunked_commit();       // audit MINOR-6
        bool r1 = selftest_real_causal();
        bool r2 = selftest_real_vpad();
        bool ok = a && b && c && d && g && cc && r1 && r2;
        cout << (ok ? "ZKOB-ROWMAX SELFTEST: ALL PASS"
                    : "ZKOB-ROWMAX SELFTEST: FAIL") << endl;
        return ok ? 0 : 1;
    }
    auto parse_mode = [](const string& s) -> uint {
        if (s == "causal") return 0;
        if (s == "vpad") return 1;
        throw runtime_error("MODE must be 'causal' or 'vpad'");
    };
    if (mode == "prove" && argc >= 14 && argc <= 16) {
        string obdir = argv[2], seed = argv[3];
        uint B = (uint)stoul(argv[5]), NCOL = (uint)stoul(argv[6]);
        uint MODE = parse_mode(argv[7]);
        uint V = (uint)stoul(argv[8]);
        uint LEN_R = (uint)stoul(argv[9]), NPL = (uint)stoul(argv[10]);
        size_t cols = (MODE == 1) ? (size_t)V : (size_t)NCOL;
        vector<int> zh = load_i32(argv[4], (size_t)B * cols);
        Commitment gen(argv[11]), genmx(argv[12]), qg(argv[13]);
        string mx_out = (argc >= 15) ? argv[14] : "";
        vector<int> tst;
        bool have_t = false;
        if (argc == 16) { tst = load_i32(argv[15], B); have_t = true; }
        prove(obdir, seed, zh, B, NCOL, MODE, V, LEN_R, NPL, gen, genmx, qg(0),
              mx_out, have_t ? &tst : nullptr);
        return 0;
    }
    if (mode == "verify" && (argc == 13 || argc == 14)) {
        string obdir = argv[2], seed = argv[3];
        uint B = (uint)stoul(argv[4]), NCOL = (uint)stoul(argv[5]);
        uint MODE = parse_mode(argv[6]);
        uint V = (uint)stoul(argv[7]);
        uint LEN_R = (uint)stoul(argv[8]), NPL = (uint)stoul(argv[9]);
        Commitment gen(argv[10]), genmx(argv[11]), qg(argv[12]);
        vector<int> tst;
        bool have_t = false;
        if (argc == 14) { tst = load_i32(argv[13], B); have_t = true; }
        return verify(obdir, seed, B, NCOL, MODE, V, LEN_R, NPL, gen, genmx, qg(0),
                      have_t ? &tst : nullptr) ? 0 : 1;
    }
    cerr << "usage: zkob_rowmax selftest\n"
         << "       zkob_rowmax prove  <obdir> <seed> <z-int32> <B> <NCOL> <MODE> <V> <LEN_R> <NPL> <gen_grid> <gen_mx> <q> [mx-int32-out|-] [tstar-int32]\n"
         << "       zkob_rowmax verify <obdir> <seed> <B> <NCOL> <MODE> <V> <LEN_R> <NPL> <gen_grid> <gen_mx> <q> [tstar-int32]" << endl;
    return 2;
}
