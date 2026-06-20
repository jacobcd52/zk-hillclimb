// COORDINATOR-BUILT (do not let submission agents edit).
// Stage D of the transport rebuild (TRANSPORT_REBUILD_DESIGN §4.2, with the
// TRANSPORT_REVIEW F7 amendment): WEIGHT PRIVACY for the registered weight
// commitments. Included ONLY by zkob_batchopen.cu, zkob_fc.cu,
// zkob_rmsnorm.cu (the weight-touching TUs) — zkob_claims.cuh is NOT edited,
// so the public batch path is bit-identical to Stage C2.
//
//   D1  hiding Pedersen rows: registered com_W[r] = <g, W[r]> + s_r*H, fresh
//       per-row blinds s_r (urandom), stored prover-private (blinds file);
//       H = q.bin slot 1 (the §4.4 H-slot registered since Stage A).
//       NO new G1 kernel: the blind terms ride k_g1_scale + k_g1_add_pairs
//       (the long-probed dev_msm building blocks), cross-checked against the
//       1-thread h_mul/h_add path in wpriv_probe().
//   D2  hidden weight claims + hidden sumcheck rounds: weight-claim evals are
//       Pedersen scalar commitments C_v = v*Q + t*H (EvalVar=Committed — the
//       Stage-A tag, opened for the first time here); every sumcheck whose
//       round evals carry weight functionals (the fc zkip sumcheck and the
//       weight sub-batch's batch-eval sumcheck) switches to COMMITTED ROUND
//       MESSAGES with homomorphic round checks:
//          prover sends C_p0, C_p1, C_p2 (fresh blinds, one linear blind
//          constraint per round); verifier checks C_p0 + C_p1 == C_cur in G1
//          and folds C_cur' = lagrange3_g1(C_p0,C_p1,C_p2)(x).
//       This is the F7-mandated amendment of §4.2-D4 ("committed round
//       messages with homomorphic round checks"); it SUBSUMES the Libra-style
//       mask-polynomial formulation: the revealed round messages are
//       perfectly-hiding Pedersen commitments, so no mask polynomial (and no
//       extra mask opening) is needed. Terminal identities become Schnorr
//       sigma proofs of an H-discrete-log (soundness: a prover violating the
//       Q-component while satisfying the group equation computes a nontrivial
//       Q/H relation -> DLOG break).
//   D3  ZK final opening: the weight sub-batch's per-domain IPA carries the
//       folded blind beta in an H component; L/R points get fresh blinds
//       (beta_L, beta_R), beta folds as beta' = beta + x^2 beta_L + x^-2
//       beta_R, and the final a_final REVEAL is replaced by a two-base
//       Schnorr proof of knowledge of (a_f, beta_f) for
//       P_fin = a_f*(g_f + b_f*Q) + beta_f*H.
//   D4  the weight sub-batch itself (wbatch_prove / wbatch_verify): a second
//       accumulator (waccdir) holding ONLY Committed weight claims, run with
//       the SAME §2.2 reduction but with committed rounds (D2), committed
//       per-tensor terminals C_vfin_j (homomorphic G3 check), and the ZK IPA
//       (D3). Transcript seed = run_seed + ":opening_batch_w" (the review's
//       own-seed pin); its own claims_match (the review's own-claims_match
//       pin). The public batch never sees a weight claim (physical routing;
//       smuggling dies at claims_match on either side, and the public batch
//       still REJECTS Committed tags at the Stage-A evalvar check).
//
// Prover-private files (never read by the verifier, same authority split as
// witrefs): <registration>/..blinds.bin (row blinds), waccdir/cblinds.bin
// (per-claim (v, t)), waccdir/blindrefs.txt (comref -> blinds file).
// Proof artifacts (verifier-consumed): waccdir/claims.bin (Committed blobs),
// wbatch_sumcheck.bin (G1 round commitments), wbatch_vfin.bin (G1 terminal
// commitments), wipa_batch_<G>.bin (blinded L/R + Schnorr2).
#ifndef ZKOB_WPRIV_CUH
#define ZKOB_WPRIV_CUH
#include "zkob_claims.cuh"

// =====================  prover-side randomness  ============================
// Blinds/nonces come from /dev/urandom with the SAME limb distribution as
// fs_challenge_fr (top limb mod 1944954707): max point mass <= 3*2^-256,
// statistical distance from uniform < 2^-32 — inside the stated ZK bounds.
static Fr_t wp_rand() {
    static FILE* ur = nullptr;
    if (!ur) {
        ur = fopen("/dev/urandom", "rb");
        if (!ur) throw std::runtime_error("wp_rand: cannot open /dev/urandom");
    }
    while (true) {
        uint8_t buf[32];
        if (fread(buf, 1, 32, ur) != 32) throw std::runtime_error("wp_rand: short read");
        Fr_t x;
        for (int i = 0; i < 8; i++)
            x.val[i] = uint32_t(buf[4*i]) | (uint32_t(buf[4*i+1]) << 8)
                     | (uint32_t(buf[4*i+2]) << 16) | (uint32_t(buf[4*i+3]) << 24);
        x.val[7] %= 1944954707;
        for (int i = 0; i < 8; i++) if (x.val[i]) return x;
    }
}

// =====================  small G1 helpers  ==================================
static G1Jacobian_t h_sub(G1Jacobian_t a, G1Jacobian_t b) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_addsub<<<1,1>>>(a, b, 1, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
static G1Jacobian_t g1_identity() { G1Jacobian_t z; memset(&z, 0, sizeof(z)); return z; }
// Pedersen scalar commitment C = v*Q + t*H
static G1Jacobian_t ped_qh(const Fr_t& v, const Fr_t& t,
                           const G1Jacobian_t& Q, const G1Jacobian_t& H) {
    return h_add(h_mul(Q, v), h_mul(H, t));
}
// lagrange3 split out for the homomorphic fold (identical l_t formulas to
// vrf_common.cuh's lagrange3; cross-checked in wpriv_probe)
static void lagrange3_coeffs(const Fr_t& u, Fr_t& l0, Fr_t& l1, Fr_t& l2) {
    Fr_t um1 = h_scalar(u, F_ONE, 1);
    Fr_t um2 = h_scalar(u, F_TWO, 1);
    l0 = h_scalar(h_scalar(um1, um2, 2), F_INV2, 2);
    l1 = h_scalar(u, h_scalar(F_TWO, u, 1), 2);
    l2 = h_scalar(h_scalar(u, um1, 2), F_INV2, 2);
}
static G1Jacobian_t lagrange3_g1(const G1Jacobian_t& C0, const G1Jacobian_t& C1,
                                 const G1Jacobian_t& C2, const Fr_t& u) {
    Fr_t l0, l1, l2; lagrange3_coeffs(u, l0, l1, l2);
    return h_add(h_add(h_mul(C0, l0), h_mul(C1, l1)), h_mul(C2, l2));
}

// =====================  D1: hiding row commitments  ========================
// com[r] += s_r * H in-place, via the long-probed k_g1_scale/k_g1_add_pairs
// shapes (one scale launch + one add launch; no new G1 kernel).
static void wp_hide_rows(G1TensorJacobian& com, const std::vector<Fr_t>& s,
                         const G1Jacobian_t& H) {
    const uint n = com.size;
    if (s.size() != n) throw std::runtime_error("wp_hide_rows: blind count != rows");
    G1Jacobian_t *d_buf, *d_out, *d_hrep;
    Fr_t* d_s;
    cudaMalloc(&d_buf, 2ull * n * sizeof(G1Jacobian_t));
    cudaMalloc(&d_out, n * sizeof(G1Jacobian_t));
    cudaMalloc(&d_hrep, n * sizeof(G1Jacobian_t));
    cudaMalloc(&d_s, n * sizeof(Fr_t));
    std::vector<G1Jacobian_t> hrep(n, H);
    cudaMemcpy(d_hrep, hrep.data(), n * sizeof(G1Jacobian_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf, com.gpu_data, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    k_g1_scale<<<(n + 63) / 64, 64>>>(d_hrep, d_s, d_buf + n, n);
    cudaDeviceSynchronize();
    k_g1_add_pairs<<<(n + 63) / 64, 64>>>(d_buf, d_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(com.gpu_data, d_out, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    cudaFree(d_buf); cudaFree(d_out); cudaFree(d_hrep); cudaFree(d_s);
}

// row-blind file (prover-private; magic ZKWB)
static const char WP_BLINDS_MAGIC[4] = {'Z','K','W','B'};
static void wp_blinds_save(const std::string& path, const std::vector<Fr_t>& s) {
    FILE* f = open_or_die(path, "wb");
    fwrite(WP_BLINDS_MAGIC, 1, 4, f);
    write_pod_vec(f, s);
    fclose(f);
}
static std::vector<Fr_t> wp_blinds_load(const std::string& path) {
    FILE* f = open_or_die(path, "rb");
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, WP_BLINDS_MAGIC, 4)) {
        fclose(f); throw std::runtime_error("wp_blinds_load: bad magic " + path); }
    auto v = read_pod_vec<Fr_t>(f);
    fclose(f);
    return v;
}

// per-claim eval blinds (prover-private; id-keyed)
static void wp_cblind_emit(const std::string& waccdir, const std::string& id,
                           const Fr_t& v, const Fr_t& t) {
    FILE* f = open_or_die(waccdir + "/cblinds.bin", "ab");
    uint32_t n = (uint32_t)id.size();
    fwrite(&n, sizeof(n), 1, f);
    fwrite(id.data(), 1, n, f);
    fwrite(&v, sizeof(Fr_t), 1, f);
    fwrite(&t, sizeof(Fr_t), 1, f);
    fclose(f);
}
static std::map<std::string, std::pair<Fr_t, Fr_t>> wp_cblinds_load(const std::string& waccdir) {
    std::map<std::string, std::pair<Fr_t, Fr_t>> m;
    auto bytes = bo_read_file(waccdir + "/cblinds.bin");
    BoReader r(bytes);
    while (r.off < bytes.size()) {
        std::string id = r.str(r.u32());
        Fr_t v, t;
        r.bytes(&v, sizeof(Fr_t));
        r.bytes(&t, sizeof(Fr_t));
        m[id] = {v, t};
    }
    return m;
}

// comref -> row-blind file (prover-private; witrefs.txt format, separate file)
static void wp_blindref_emit(const std::string& waccdir, const std::string& comref,
                             const std::string& blindpath) {
    std::string path = waccdir + "/blindrefs.txt";
    if (bo_file_exists(path)) {
        auto bytes = bo_read_file(path);
        std::string all((const char*)bytes.data(), bytes.size());
        if (all.find(comref + "\t") != std::string::npos) return;
    }
    FILE* f = open_or_die(path, "ab");
    fprintf(f, "%s\t%s\n", comref.c_str(), blindpath.c_str());
    fclose(f);
}
static std::map<std::string, std::string> wp_blindrefs_load(const std::string& waccdir) {
    std::map<std::string, std::string> m;
    auto bytes = bo_read_file(waccdir + "/blindrefs.txt");
    std::string all((const char*)bytes.data(), bytes.size());
    size_t pos = 0;
    while (pos < all.size()) {
        size_t nl = all.find('\n', pos); if (nl == std::string::npos) nl = all.size();
        std::string line = all.substr(pos, nl - pos); pos = nl + 1;
        size_t tab = line.find('\t'); if (tab == std::string::npos) continue;
        m[line.substr(0, tab)] = line.substr(tab + 1);
    }
    return m;
}

// =====================  sigma proofs  ======================================
// Schnorr PoK of delta with E = delta*H. Challenge from the caller's
// transcript (E itself is a function of already-absorbed values).
struct SchnorrH { G1Jacobian_t A; Fr_t z; };
static SchnorrH schnorr_h_prove(const Fr_t& delta, const G1Jacobian_t& H,
                                fs::Transcript& tr) {
    Fr_t k = wp_rand();
    SchnorrH p;
    p.A = h_mul(H, k);
    absorb_g1(tr, "wschA", p.A);
    Fr_t e = fs_challenge_fr(tr);
    p.z = h_scalar(k, h_scalar(e, delta, 2), 0);   // z = k + e*delta
    return p;
}
static bool schnorr_h_verify(const G1Jacobian_t& E, const SchnorrH& p,
                             const G1Jacobian_t& H, fs::Transcript& tr) {
    absorb_g1(tr, "wschA", p.A);
    Fr_t e = fs_challenge_fr(tr);
    return g1_eq(h_mul(H, p.z), h_add(p.A, h_mul(E, e)));
}
// two-base Schnorr PoK of (a, b) with P = a*U + b*H (the ZK-IPA final round)
struct Schnorr2 { G1Jacobian_t A; Fr_t za, zb; };
static Schnorr2 schnorr2_prove(const Fr_t& a, const Fr_t& b,
                               const G1Jacobian_t& U, const G1Jacobian_t& H,
                               fs::Transcript& tr) {
    Fr_t k1 = wp_rand(), k2 = wp_rand();
    Schnorr2 p;
    p.A = h_add(h_mul(U, k1), h_mul(H, k2));
    absorb_g1(tr, "wsch2A", p.A);
    Fr_t e = fs_challenge_fr(tr);
    p.za = h_scalar(k1, h_scalar(e, a, 2), 0);
    p.zb = h_scalar(k2, h_scalar(e, b, 2), 0);
    return p;
}
static bool schnorr2_verify(const G1Jacobian_t& P, const Schnorr2& p,
                            const G1Jacobian_t& U, const G1Jacobian_t& H,
                            fs::Transcript& tr) {
    absorb_g1(tr, "wsch2A", p.A);
    Fr_t e = fs_challenge_fr(tr);
    return g1_eq(h_add(h_mul(U, p.za), h_mul(H, p.zb)),
                 h_add(p.A, h_mul(P, e)));
}

// D5: product proof — prove the committed value of C_z is the PRODUCT of the
// committed values of C_x and C_w, in ZK (no pairing). Used for the fc terminal
// cur = claim_X * claim_W once BOTH claim_X and claim_W are hidden (so the
// one-hidden-factor Schnorr no longer applies). 3-move sigma; HVZK; soundness
// reduces to DLOG. Checks (verifier):
//   z_x*Q + s_x*H == t_x + e*C_x
//   z_w*Q + s_w*H == t_w + e*C_w
//   z_x*C_w + s_z*H == t_z + e*C_z      (holds iff z = x*w)
struct ProdProof { G1Jacobian_t t_x, t_w, t_z; Fr_t z_x, z_w, s_x, s_w, s_z; };
static ProdProof prod_prove(const Fr_t& x, const Fr_t& r_x,
                            const Fr_t& w, const Fr_t& r_w,
                            const Fr_t& z, const Fr_t& r_z,
                            const G1Jacobian_t& C_w,
                            const G1Jacobian_t& Q, const G1Jacobian_t& H,
                            fs::Transcript& tr) {
    Fr_t b_x = wp_rand(), b_w = wp_rand(), s1 = wp_rand(), s2 = wp_rand(), s3 = wp_rand();
    ProdProof p;
    p.t_x = ped_qh(b_x, s1, Q, H);
    p.t_w = ped_qh(b_w, s2, Q, H);
    p.t_z = h_add(h_mul(C_w, b_x), h_mul(H, s3));     // b_x*C_w + s3*H
    absorb_g1(tr, "prtx", p.t_x); absorb_g1(tr, "prtw", p.t_w); absorb_g1(tr, "prtz", p.t_z);
    Fr_t e = fs_challenge_fr(tr);
    p.z_x = h_scalar(b_x, h_scalar(e, x, 2), 0);      // b_x + e*x
    p.z_w = h_scalar(b_w, h_scalar(e, w, 2), 0);
    p.s_x = h_scalar(s1, h_scalar(e, r_x, 2), 0);
    p.s_w = h_scalar(s2, h_scalar(e, r_w, 2), 0);
    Fr_t cross = h_scalar(r_z, h_scalar(x, r_w, 2), 1);  // r_z - x*r_w
    p.s_z = h_scalar(s3, h_scalar(e, cross, 2), 0);
    return p;
}
static bool prod_verify(const G1Jacobian_t& C_x, const G1Jacobian_t& C_w,
                        const G1Jacobian_t& C_z, const ProdProof& p,
                        const G1Jacobian_t& Q, const G1Jacobian_t& H,
                        fs::Transcript& tr) {
    absorb_g1(tr, "prtx", p.t_x); absorb_g1(tr, "prtw", p.t_w); absorb_g1(tr, "prtz", p.t_z);
    Fr_t e = fs_challenge_fr(tr);
    bool c1 = g1_eq(ped_qh(p.z_x, p.s_x, Q, H), h_add(p.t_x, h_mul(C_x, e)));
    bool c2 = g1_eq(ped_qh(p.z_w, p.s_w, Q, H), h_add(p.t_w, h_mul(C_w, e)));
    bool c3 = g1_eq(h_add(h_mul(C_w, p.z_x), h_mul(H, p.s_z)), h_add(p.t_z, h_mul(C_z, e)));
    return c1 && c2 && c3;
}

// =====================  D3: the ZK IPA  ====================================
// Relation: P0 = <g, a> + <a, b>*Q + beta*H  (b public ME weights).
// Rounds are the header IPA's rounds plus fresh blinds on L/R; the final
// a_final reveal is replaced by schnorr2 over P_fin = a_f*(g_f + b_f*Q) +
// beta_f*H. Same absorb schedule as ipa_prove for L/R; the proof leaks
// nothing about a: L/R are uniform given beta_L/beta_R, schnorr2 is HVZK.
struct ZkIpaProof {
    std::vector<G1Jacobian_t> L, R;
    Schnorr2 fin;
};
static void write_zkipa(const std::string& path, const ZkIpaProof& pf) {
    FILE* f = open_or_die(path, "wb");
    write_pod_vec(f, pf.L);
    write_pod_vec(f, pf.R);
    fwrite(&pf.fin.A, sizeof(G1Jacobian_t), 1, f);
    fwrite(&pf.fin.za, sizeof(Fr_t), 1, f);
    fwrite(&pf.fin.zb, sizeof(Fr_t), 1, f);
    fclose(f);
}
static ZkIpaProof read_zkipa(const std::string& path) {
    FILE* f = open_or_die(path, "rb");
    ZkIpaProof pf;
    pf.L = read_pod_vec<G1Jacobian_t>(f);
    pf.R = read_pod_vec<G1Jacobian_t>(f);
    if (fread(&pf.fin.A, sizeof(G1Jacobian_t), 1, f) != 1 ||
        fread(&pf.fin.za, sizeof(Fr_t), 1, f) != 1 ||
        fread(&pf.fin.zb, sizeof(Fr_t), 1, f) != 1)
        throw std::runtime_error("read_zkipa: final");
    fclose(f);
    return pf;
}
static ZkIpaProof zk_ipa_prove(const Fr_t* d_a_in, const std::vector<Fr_t>& b_in,
                               const G1Jacobian_t* d_g_in, const G1Jacobian_t& Q,
                               const G1Jacobian_t& H, const Fr_t& beta0,
                               uint n, fs::Transcript& tr) {
    if (n & (n - 1)) throw std::runtime_error("zk_ipa_prove: n not a power of two");
    Fr_t *d_a, *d_a2, *d_b, *d_b2;
    G1Jacobian_t *d_g, *d_g2;
    cudaMalloc(&d_a, n * sizeof(Fr_t));   cudaMalloc(&d_a2, (n/2+1) * sizeof(Fr_t));
    cudaMalloc(&d_b, n * sizeof(Fr_t));   cudaMalloc(&d_b2, (n/2+1) * sizeof(Fr_t));
    cudaMalloc(&d_g, n * sizeof(G1Jacobian_t)); cudaMalloc(&d_g2, (n/2+1) * sizeof(G1Jacobian_t));
    cudaMemcpy(d_a, d_a_in, n * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_b, b_in.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, d_g_in, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    ZkIpaProof pf;
    Fr_t beta = beta0;
    uint sz = n;
    while (sz > 1) {
        uint half = sz / 2;
        Fr_t bL = wp_rand(), bR = wp_rand();
        G1Jacobian_t Lp = h_add(h_add(dev_msm(d_g + half, d_a, half),
                                      h_mul(Q, dev_ip(d_a, d_b + half, half))),
                                h_mul(H, bL));
        G1Jacobian_t Rp = h_add(h_add(dev_msm(d_g, d_a + half, half),
                                      h_mul(Q, dev_ip(d_a + half, d_b, half))),
                                h_mul(H, bR));
        absorb_g1(tr, "L", Lp); absorb_g1(tr, "R", Rp);
        Fr_t x = fs_challenge_fr(tr), xi = inv(x);
        k_fr_fold2<<<(half + 63) / 64, 64>>>(d_a, x, xi, d_a2, half);
        k_fr_fold2<<<(half + 63) / 64, 64>>>(d_b, xi, x, d_b2, half);
        k_g1_fold2<<<(half + 63) / 64, 64>>>(d_g, xi, x, d_g2, half);
        cudaDeviceSynchronize();
        std::swap(d_a, d_a2); std::swap(d_b, d_b2); std::swap(d_g, d_g2);
        // beta' = beta + x^2 bL + x^-2 bR
        beta = h_scalar(beta, h_scalar(h_scalar(x, x, 2), bL, 2), 0);
        beta = h_scalar(beta, h_scalar(h_scalar(xi, xi, 2), bR, 2), 0);
        pf.L.push_back(Lp); pf.R.push_back(Rp);
        sz = half;
    }
    Fr_t a_f, b_f;
    G1Jacobian_t g_f;
    cudaMemcpy(&a_f, d_a, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b_f, d_b, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&g_f, d_g, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_a2); cudaFree(d_b); cudaFree(d_b2); cudaFree(d_g); cudaFree(d_g2);
    G1Jacobian_t U = h_add(g_f, h_mul(Q, b_f));
    pf.fin = schnorr2_prove(a_f, beta, U, H, tr);
    return pf;
}
// verify: fast-path g_f/b_f via the Stage-B s-vector helpers (probed
// element-exact in bo_probe_kernels); the round fold of P is the header
// ipa_verify's equation unchanged (the H components ride inside the points).
static bool zk_ipa_verify(const G1Jacobian_t* d_g, uint n, const G1Jacobian_t& Q,
                          const G1Jacobian_t& H, const G1Jacobian_t& P0,
                          const std::vector<Fr_t>& u_b, const ZkIpaProof& pf,
                          fs::Transcript& tr) {
    const uint rounds = (uint)pf.L.size();
    if (rounds != pf.R.size()) return false;
    if (n != (1u << rounds)) return false;
    if ((1u << u_b.size()) != n) return false;
    std::vector<Fr_t> xs(rounds), xis(rounds);
    G1Jacobian_t P = P0;
    for (uint r = 0; r < rounds; r++) {
        absorb_g1(tr, "L", pf.L[r]);
        absorb_g1(tr, "R", pf.R[r]);
        xs[r] = fs_challenge_fr(tr);
        xis[r] = inv(xs[r]);
        P = h_add(h_add(h_mul(pf.L[r], h_scalar(xs[r], xs[r], 2)), P),
                  h_mul(pf.R[r], h_scalar(xis[r], xis[r], 2)));
    }
    Fr_t* d_s = bo_fast_s_vector_dev(xs, xis);
    Fr_t* d_b = bo_fast_me_weights_dev(u_b);
    Fr_t b_f = dev_ip(d_b, d_s, n);
    G1Jacobian_t g_f = dev_msm(d_g, d_s, n);
    cudaFree(d_s);
    cudaFree(d_b);
    G1Jacobian_t U = h_add(g_f, h_mul(Q, b_f));
    return schnorr2_verify(P, pf.fin, U, H, tr);
}

// =====================  weight sub-batch artifacts  ========================
static const char WP_WS_MAGIC[4] = {'Z','K','W','S'};
static const char WP_WV_MAGIC[4] = {'Z','K','W','V'};
struct WBatchSumcheck {
    uint32_t n_claims = 0, m_max = 0;        // redundant; cross-checked (F6)
    std::vector<G1Jacobian_t> cp;            // 3 round commitments per round
};
static void write_wbatch_sumcheck(const std::string& path, const WBatchSumcheck& p) {
    FILE* f = open_or_die(path, "wb");
    fwrite(WP_WS_MAGIC, 1, 4, f);
    fwrite(&p.n_claims, sizeof(uint32_t), 1, f);
    fwrite(&p.m_max, sizeof(uint32_t), 1, f);
    write_pod_vec(f, p.cp);
    fclose(f);
}
static WBatchSumcheck read_wbatch_sumcheck(const std::string& path) {
    FILE* f = open_or_die(path, "rb");
    char magic[4];
    WBatchSumcheck p;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, WP_WS_MAGIC, 4) ||
        fread(&p.n_claims, sizeof(uint32_t), 1, f) != 1 ||
        fread(&p.m_max, sizeof(uint32_t), 1, f) != 1) {
        fclose(f); throw std::runtime_error("read_wbatch_sumcheck: header"); }
    p.cp = read_pod_vec<G1Jacobian_t>(f);
    fclose(f);
    return p;
}
static void write_wbatch_vfin(const std::string& path, const std::vector<G1Jacobian_t>& v) {
    FILE* f = open_or_die(path, "wb");
    fwrite(WP_WV_MAGIC, 1, 4, f);
    write_pod_vec(f, v);
    fclose(f);
}
static std::vector<G1Jacobian_t> read_wbatch_vfin(const std::string& path) {
    FILE* f = open_or_die(path, "rb");
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, WP_WV_MAGIC, 4)) {
        fclose(f); throw std::runtime_error("read_wbatch_vfin: header"); }
    auto v = read_pod_vec<G1Jacobian_t>(f);
    fclose(f);
    return v;
}

// =====================  weight sub-batch: prove  ===========================
// evil modes (selftest/battery forgery construction ONLY; 0 in production):
//   1 honest PROCEDURE over a (possibly false) committed claim list (the
//     driver-level lie: cblinds carry the false v; the witness is honest)
//     -> dies at wround0 (homomorphic round-0 check)
//   2 adaptive: per round force p1 = cur - p0 (and tau1 = taucur - tau0 so
//     the round check PASSES), then poison the LAST tensor's C_vfin so the
//     homomorphic G3 passes -> forced into that tensor's group ZK-IPA
//   3 doctor the transcript/RLC inputs (claim 0 ceval + Q) but emit honest
//     artifacts -> verifier rho differs -> wround0
//   4 wrong blind bookkeeping: shift the IPA beta by 1 -> wipa<G>
static void wbatch_prove(const std::string& waccdir, const std::string& run_seed,
                         const std::map<uint32_t, std::string>& genpaths,
                         const std::string& qpath, int evil = 0) {
    BoTimer T("wbatch_prove");
    std::vector<BoClaim> cs = claims_load(waccdir + "/claims.bin");
    if (cs.empty()) throw std::runtime_error("wbatch_prove: zero claims");
    std::vector<BoClaim> cs_tr = cs;
    Commitment qg(qpath);
    if (qg.size < 2) throw std::runtime_error("wbatch_prove: q.bin has no H slot");
    G1Jacobian_t Q = qg(0), H = qg(1);
    if (evil == 3) cs_tr[0].ceval = h_add(cs_tr[0].ceval, Q);
    for (auto& c : cs_tr)
        if (c.tag != BO_EVAL_COMMITTED)
            throw std::runtime_error("wbatch_prove: plain claim in weight accumulator: " + c.id);
    std::vector<TensorInfo> tensors;
    std::string err;
    if (!derive_tensors(cs_tr, tensors, err)) throw std::runtime_error("wbatch_prove: " + err);
    std::vector<DrvState> dss;
    if (bo_file_exists(waccdir + "/drvstates.bin")) dss = drvstates_load(waccdir + "/drvstates.bin");
    auto wits = witrefs_load(waccdir);
    auto cblinds = wp_cblinds_load(waccdir);
    auto blindrefs = wp_blindrefs_load(waccdir);

    uint m_max = 0;
    for (auto& t : tensors) m_max = std::max(m_max, t.vars);
    if (m_max < 1) throw std::runtime_error("wbatch_prove: m_max < 1");
    const uint nT = (uint)tensors.size();

    // per-claim (v, t) from the prover-private blind store; self-check each
    // against the shipped ceval (catches blind-store desync LOUDLY)
    std::vector<Fr_t> ev(cs.size()), et(cs.size());
    for (uint i = 0; i < cs.size(); i++) {
        auto it = cblinds.find(cs[i].id);
        if (it == cblinds.end())
            throw std::runtime_error("wbatch_prove: no cblind for claim " + cs[i].id);
        ev[i] = it->second.first;
        et[i] = it->second.second;
        if (evil == 0 && !g1_eq(ped_qh(ev[i], et[i], Q, H), cs[i].ceval))
            throw std::runtime_error("wbatch_prove: cblind != ceval for claim " + cs[i].id);
    }

    // G0w + rho (the review pins: own seed suffix, own claims_match)
    fs::Transcript tr(run_seed + ":opening_batch_w");
    batch_absorb_g0(tr, cs_tr, tensors, dss);
    Fr_t rho = fs_challenge_fr(tr);
    std::vector<Fr_t> rho_pows(cs.size());
    rho_pows[0] = F_ONE;
    for (uint i = 1; i < cs.size(); i++) rho_pows[i] = h_scalar(rho_pows[i - 1], rho, 2);

    std::vector<std::string> witpath(nT);
    for (uint j = 0; j < nT; j++) {
        auto it = wits.find(tensors[j].comref);
        if (it == wits.end()) throw std::runtime_error("wbatch_prove: no witref for " + tensors[j].comref);
        witpath[j] = it->second;
    }

    // M/P tables (the §2.2 machinery, transient witness loads as in C2)
    std::vector<Fr_t*> d_M(nT), d_P(nT);
    std::vector<uint> cur_size(nT);
    std::vector<Fr_t> c2(nT, F_ONE), S(nT, F_ZERO);
    Fr_t* d_eq = nullptr;
    bo_malloc(&d_eq, 1u << m_max, "w eq table");
    for (uint j = 0; j < nT; j++) {
        uint n = 1u << tensors[j].vars;
        bo_malloc(&d_M[j], n, ("wM[" + tensors[j].comref + "]").c_str());
        bo_malloc(&d_P[j], n, ("wP[" + tensors[j].comref + "]").c_str());
        cudaMemset(d_M[j], 0, n * sizeof(Fr_t));
        {
            FrTensor w(witpath[j]);
            if (w.size != (size_t)tensors[j].n_rows * tensors[j].domain)
                throw std::runtime_error("wbatch_prove: witness size mismatch for " + tensors[j].comref);
            cudaMemcpy(d_P[j], w.gpu_data, n * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
        }
        cur_size[j] = n;
        for (uint32_t idx : tensors[j].claim_idx) {
            bo_build_eq(d_eq, cs_tr[idx].point);
            k_bo_axpy<<<(n + 255) / 256, 256>>>(d_M[j], d_eq, rho_pows[idx], n);
            cudaDeviceSynchronize();
        }
        if (tensors[j].vars < m_max)
            S[j] = dev_ip(d_M[j], d_P[j], n);
    }
    cudaFree(d_eq);
    T.lap("setup");

    // initial committed claim: cur/tau tracked scalar-side; the verifier
    // forms C_cur0 homomorphically from the cevals
    Fr_t cur = F_ZERO, tau = F_ZERO;
    for (uint i = 0; i < cs.size(); i++) {
        cur = h_scalar(cur, h_scalar(rho_pows[i], ev[i], 2), 0);
        tau = h_scalar(tau, h_scalar(rho_pows[i], et[i], 2), 0);
    }

    // committed rounds (D2/F7): C_pt = p_t*Q + tau_t*H with tau0, tau2 fresh
    // and tau1 = tau_cur - tau0 (the one linear constraint the homomorphic
    // round check enforces); fold cur' = lagrange3(p)(x), tau' = lagrange3(tau)(x)
    WBatchSumcheck ws;
    ws.n_claims = (uint32_t)cs.size();
    ws.m_max = m_max;
    std::vector<Fr_t> r(m_max, F_ZERO);
    std::vector<Fr_t*> scratch(3);
    bo_malloc(&scratch[0], 1u << (m_max - 1), "w round scratch 0");
    bo_malloc(&scratch[1], 1u << (m_max - 1), "w round scratch 1");
    bo_malloc(&scratch[2], 1u << (m_max - 1), "w round scratch 2");
    Fr_t* d_fold;
    bo_malloc(&d_fold, 1u << (m_max - 1), "w fold scratch");
    for (uint t = 0; t < m_max; t++) {
        uint v = m_max - 1 - t;
        Fr_t p0 = F_ZERO, p1 = F_ZERO, p2 = F_ZERO;
        for (uint j = 0; j < nT; j++) {
            if (tensors[j].vars <= v) {
                Fr_t term = h_scalar(c2[j], S[j], 2);
                p0 = h_scalar(p0, term, 0);
                p2 = h_scalar(p2, term, 0);
            } else {
                uint half = cur_size[j] / 2;
                k_bo_hp2<<<(half + 63) / 64, 64>>>(d_M[j], d_P[j],
                    scratch[0], scratch[1], scratch[2], half);
                cudaDeviceSynchronize();
                p0 = h_scalar(p0, h_scalar(c2[j], bo_dev_sum(scratch[0], half), 2), 0);
                p1 = h_scalar(p1, h_scalar(c2[j], bo_dev_sum(scratch[1], half), 2), 0);
                p2 = h_scalar(p2, h_scalar(c2[j], bo_dev_sum(scratch[2], half), 2), 0);
            }
        }
        if (evil == 0 && !fr_eq(cur, h_scalar(p0, p1, 0)))
            throw std::runtime_error("wbatch_prove: round " + std::to_string(t) +
                                     " inconsistency (witness/claims bug)");
        if (evil == 2) p1 = h_scalar(cur, p0, 1);
        Fr_t tau0 = wp_rand(), tau2 = wp_rand();
        Fr_t tau1 = h_scalar(tau, tau0, 1);
        G1Jacobian_t C_p0 = ped_qh(p0, tau0, Q, H);
        G1Jacobian_t C_p1 = ped_qh(p1, tau1, Q, H);
        G1Jacobian_t C_p2 = ped_qh(p2, tau2, Q, H);
        absorb_g1(tr, "wp0", C_p0); absorb_g1(tr, "wp1", C_p1); absorb_g1(tr, "wp2", C_p2);
        Fr_t x = fs_challenge_fr(tr);
        r[v] = x;
        ws.cp.push_back(C_p0); ws.cp.push_back(C_p1); ws.cp.push_back(C_p2);
        cur = lagrange3(p0, p1, p2, x);
        tau = lagrange3(tau0, tau1, tau2, x);
        Fr_t one_minus_x = h_scalar(F_ONE, x, 1);
        Fr_t omx2 = h_scalar(one_minus_x, one_minus_x, 2);
        for (uint j = 0; j < nT; j++) {
            if (tensors[j].vars <= v) {
                c2[j] = h_scalar(c2[j], omx2, 2);
            } else {
                uint half = cur_size[j] / 2;
                k_bo_fold<<<(half + 255) / 256, 256>>>(d_M[j], x, d_fold, half);
                cudaDeviceSynchronize();
                cudaMemcpy(d_M[j], d_fold, half * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
                k_bo_fold<<<(half + 255) / 256, 256>>>(d_P[j], x, d_fold, half);
                cudaDeviceSynchronize();
                cudaMemcpy(d_P[j], d_fold, half * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
                cur_size[j] = half;
            }
        }
    }
    write_wbatch_sumcheck(waccdir + "/wbatch_sumcheck.bin", ws);
    T.lap("rounds");

    // committed terminals: C_vfin_j = vfin_j*Q + t'_j*H with fresh t'_j and
    // the LAST nonzero-coefficient tensor's t' solved so that
    // sum_j c_j*t'_j == tau (the homomorphic G3 identity's H component)
    std::vector<Fr_t> vfin(nT), cj(nT);
    for (uint j = 0; j < nT; j++) {
        cudaMemcpy(&vfin[j], d_P[j], sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cj[j] = h_scalar(bo_Mj_at_r(tensors[j], cs_tr, rho_pows, r),
                         bo_kappa(tensors[j].vars, r), 2);
    }
    if (evil == 2) {
        // poison the LAST tensor's vfin so the scalar G3 identity passes
        Fr_t need = cur;
        for (uint j = 0; j + 1 < nT; j++)
            need = h_scalar(need, h_scalar(cj[j], vfin[j], 2), 1);
        vfin[nT - 1] = h_scalar(need, inv(cj[nT - 1]), 2);
    }
    int J = -1;
    for (int j = (int)nT - 1; j >= 0; j--)
        if (!fr_eq(cj[j], F_ZERO)) { J = j; break; }
    if (J < 0) throw std::runtime_error("wbatch_prove: all terminal coefficients zero");
    std::vector<Fr_t> tfin(nT);
    Fr_t acc = F_ZERO;
    for (uint j = 0; j < nT; j++) {
        if ((int)j == J) continue;
        tfin[j] = wp_rand();
        acc = h_scalar(acc, h_scalar(cj[j], tfin[j], 2), 0);
    }
    tfin[J] = h_scalar(h_scalar(tau, acc, 1), inv(cj[J]), 2);
    if (evil == 0) {
        Fr_t chk = F_ZERO;
        for (uint j = 0; j < nT; j++)
            chk = h_scalar(chk, h_scalar(cj[j], vfin[j], 2), 0);
        if (!fr_eq(chk, cur))
            throw std::runtime_error("wbatch_prove: G3 terminal self-check failed");
    }
    std::vector<G1Jacobian_t> cvfin(nT);
    for (uint j = 0; j < nT; j++) {
        cvfin[j] = ped_qh(vfin[j], tfin[j], Q, H);
        absorb_g1(tr, "wvfin", cvfin[j]);
    }
    write_wbatch_vfin(waccdir + "/wbatch_vfin.bin", cvfin);
    for (uint j = 0; j < nT; j++) { cudaFree(d_M[j]); cudaFree(d_P[j]); }
    cudaFree(scratch[0]); cudaFree(scratch[1]); cudaFree(scratch[2]); cudaFree(d_fold);
    T.lap("terminal");

    // G4w/G5w: per domain group, the SAME RLC fold for the a-vector; the
    // blind side beta_g = S*_g (folded registration row blinds) + T*_g
    // (folded terminal blinds); ZK IPA discharges
    //   P0_g = <gen, A_g> + v*_g*Q + beta_g*H
    Fr_t rhop = fs_challenge_fr(tr);
    std::vector<Fr_t> rhop_pows(nT);
    rhop_pows[0] = F_ONE;
    for (uint j = 1; j < nT; j++) rhop_pows[j] = h_scalar(rhop_pows[j - 1], rhop, 2);
    std::set<uint32_t> domains;
    for (auto& t : tensors) domains.insert(t.domain);
    for (uint32_t G : domains) {
        uint logG = ceilLog2(G);
        Fr_t* d_a;
        bo_malloc(&d_a, G, "w group RLC vector");
        cudaMemset(d_a, 0, G * sizeof(Fr_t));
        Fr_t beta = F_ZERO;
        for (uint j = 0; j < nT; j++) {
            if (tensors[j].domain != G) continue;
            Fr_t coef = h_scalar(rhop_pows[j], bo_kappa(tensors[j].vars, r), 2);
            std::vector<Fr_t> u_row(r.begin() + logG, r.begin() + tensors[j].vars);
            {
                FrTensor w(witpath[j]);
                if (u_row.empty()) {
                    k_bo_axpy<<<(G + 255) / 256, 256>>>(d_a, w.gpu_data, coef, G);
                    cudaDeviceSynchronize();
                } else {
                    FrTensor rf = w.partial_me(u_row, G);
                    k_bo_axpy<<<(G + 255) / 256, 256>>>(d_a, rf.gpu_data, coef, G);
                    cudaDeviceSynchronize();
                }
            }
            // S*: folded registration row blinds <s_j, me_weights(u_row)>
            auto bit = blindrefs.find(tensors[j].comref);
            if (bit == blindrefs.end())
                throw std::runtime_error("wbatch_prove: no blindref for " + tensors[j].comref);
            auto s = wp_blinds_load(bit->second);
            if (s.size() != tensors[j].n_rows)
                throw std::runtime_error("wbatch_prove: blind count mismatch for " + tensors[j].comref);
            Fr_t sfold;
            if (u_row.empty()) {
                sfold = s[0];
            } else {
                Fr_t* d_s;
                bo_malloc(&d_s, s.size(), "w row blinds");
                cudaMemcpy(d_s, s.data(), s.size() * sizeof(Fr_t), cudaMemcpyHostToDevice);
                Fr_t* d_w = bo_fast_me_weights_dev(u_row);
                sfold = dev_ip(d_s, d_w, (uint)s.size());
                cudaFree(d_s); cudaFree(d_w);
            }
            // T*: folded terminal blinds
            beta = h_scalar(beta, h_scalar(coef, h_scalar(sfold, tfin[j], 0), 2), 0);
        }
        if (evil == 4) beta = h_scalar(beta, F_ONE, 0);
        auto git = genpaths.find(G);
        if (git == genpaths.end())
            throw std::runtime_error("wbatch_prove: no generator file for domain " + std::to_string(G));
        Commitment gen(git->second);
        if (gen.size != G)
            throw std::runtime_error("wbatch_prove: generator size mismatch for domain " + std::to_string(G));
        if (getenv("ZKOB_BATCH_SELFCHECK") && evil == 0) {
            // <gen, a> + v* Q + beta H must equal the verifier's
            // C*(folded hiding coms) + V*(folded C_vfin)
            G1Jacobian_t lhs = dev_msm(gen.gpu_data, d_a, G);
            Fr_t vstar = F_ZERO;
            G1Jacobian_t rhs = g1_identity();
            for (uint j = 0; j < nT; j++) {
                if (tensors[j].domain != G) continue;
                Fr_t coef = h_scalar(rhop_pows[j], bo_kappa(tensors[j].vars, r), 2);
                vstar = h_scalar(vstar, h_scalar(coef, vfin[j], 2), 0);
                G1TensorJacobian com(tensors[j].comref);
                std::vector<Fr_t> u_row(r.begin() + logG, r.begin() + tensors[j].vars);
                G1Jacobian_t cjp = fold_chain(com.gpu_data, com.size, u_row, 0);
                rhs = h_add(rhs, h_mul(h_add(cjp, cvfin[j]), coef));
            }
            lhs = h_add(lhs, ped_qh(vstar, beta, Q, H));
            if (!g1_eq(lhs, rhs))
                throw std::runtime_error("wbatch_prove: P0 self-check failed (domain " +
                                         std::to_string(G) + ")");
        }
        ZkIpaProof pf = zk_ipa_prove(d_a,
            bo_fast_me_weights(std::vector<Fr_t>(r.begin(), r.begin() + logG)),
            gen.gpu_data, Q, H, beta, G, tr);
        write_zkipa(waccdir + "/wipa_batch_" + std::to_string(G) + ".bin", pf);
        cudaFree(d_a);
    }
    T.lap("zkipa");
}

// =====================  weight sub-batch: verify  ==========================
// Witness-free AND blind-free: every check is a G1 equation over absorbed
// commitments. REJECT loci: opening_batch_w.{wempty,wevalvar,wclaims_match,
// wshape,wxcheck,wround<k>,wvfin_count,wterminal,wgroup_missing,wipa<G>}.
static bool wbatch_verify(const std::string& pwaccdir, const std::string& vwaccdir,
                          const std::string& run_seed,
                          const std::map<uint32_t, std::string>& genpaths,
                          const std::string& qpath, std::string* locus_out = nullptr) {
    auto reject = [&](const std::string& locus, const std::string& detail) {
        printf("REJECT[opening_batch_w.%s]: %s\n", locus.c_str(), detail.c_str());
        if (locus_out) *locus_out = locus;
        return false;
    };
    BoTimer T("wbatch_verify");
    std::vector<BoClaim> cs = claims_load(vwaccdir + "/claims.bin");
    std::vector<DrvState> dss;
    if (bo_file_exists(vwaccdir + "/drvstates.bin")) dss = drvstates_load(vwaccdir + "/drvstates.bin");
    if (cs.empty()) return reject("wempty", "zero weight claims");
    for (auto& c : cs)
        if (c.tag != BO_EVAL_COMMITTED)
            return reject("wevalvar", "plain-eval claim in the weight batch (claim " + c.id + ")");
    {
        std::vector<uint8_t> mine = claims_serialize(cs);
        if (!bo_file_exists(pwaccdir + "/claims.bin"))
            return reject("wclaims_match", "prover weight claims.bin missing");
        std::vector<uint8_t> theirs = bo_read_file(pwaccdir + "/claims.bin");
        if (mine.size() != theirs.size() || memcmp(mine.data(), theirs.data(), mine.size()))
            return reject("wclaims_match", "prover weight claim list != verifier-recomputed list");
    }
    std::vector<TensorInfo> tensors;
    std::string err;
    if (!derive_tensors(cs, tensors, err)) return reject("wshape", err);
    const uint nT = (uint)tensors.size();
    uint m_max = 0;
    for (auto& t : tensors) m_max = std::max(m_max, t.vars);
    if (m_max < 1) return reject("wshape", "m_max < 1");
    const bool require_rel = getenv("ZKOB_REQUIRE_RELATIVE_COMREF") != nullptr;
    std::vector<G1TensorJacobian> coms;
    coms.reserve(nT);
    for (auto& t : tensors) {
        if (require_rel && !t.comref.empty() && t.comref[0] == '/')
            return reject("wshape", "absolute comref under relative-comref policy: " + t.comref);
        if (!bo_file_exists(t.comref))
            return reject("wshape", "commitment file missing: " + t.comref);
        coms.emplace_back(t.comref);
        if (coms.back().size != t.n_rows)
            return reject("wshape", "commitment row count " + std::to_string(coms.back().size) +
                          " != n_rows " + std::to_string(t.n_rows) + " (" + t.comref + ")");
    }
    std::map<uint32_t, Commitment> gens;
    for (auto& t : tensors) {
        if (gens.count(t.domain)) continue;
        auto git = genpaths.find(t.domain);
        if (git == genpaths.end())
            return reject("wshape", "no generator file mapped for domain " + std::to_string(t.domain));
        gens.emplace(t.domain, Commitment(git->second));
        if (gens.at(t.domain).size != t.domain)
            return reject("wshape", "generator file size != domain " + std::to_string(t.domain));
    }
    Commitment qg(qpath);
    if (qg.size < 2) return reject("wshape", "q.bin has no H slot");
    G1Jacobian_t Q = qg(0), H = qg(1);
    T.lap("shape");

    fs::Transcript tr(run_seed + ":opening_batch_w");
    batch_absorb_g0(tr, cs, tensors, dss);
    Fr_t rho = fs_challenge_fr(tr);
    std::vector<Fr_t> rho_pows(cs.size());
    rho_pows[0] = F_ONE;
    for (uint i = 1; i < cs.size(); i++) rho_pows[i] = h_scalar(rho_pows[i - 1], rho, 2);

    // homomorphic initial claim
    G1Jacobian_t C_cur = g1_identity();
    for (uint i = 0; i < cs.size(); i++)
        C_cur = h_add(C_cur, h_mul(cs[i].ceval, rho_pows[i]));
    T.lap("absorb");

    WBatchSumcheck ws;
    try { ws = read_wbatch_sumcheck(pwaccdir + "/wbatch_sumcheck.bin"); }
    catch (const std::exception& e) { return reject("wxcheck", e.what()); }
    if (ws.n_claims != cs.size())
        return reject("wxcheck", "wbatch_sumcheck.bin n_claims field != derived claim count");
    if (ws.m_max != m_max)
        return reject("wxcheck", "wbatch_sumcheck.bin m_max field != derived m_max");
    if (ws.cp.size() != 3u * m_max)
        return reject("wxcheck", "wbatch_sumcheck.bin round-commitment count != 3*m_max");
    std::vector<Fr_t> r(m_max, F_ZERO);
    for (uint t = 0; t < m_max; t++) {
        const G1Jacobian_t &C0 = ws.cp[3*t], &C1 = ws.cp[3*t+1], &C2 = ws.cp[3*t+2];
        if (!g1_eq(h_add(C0, C1), C_cur))
            return reject("wround" + std::to_string(t), "C_p0 + C_p1 != C_cur (homomorphic round check)");
        absorb_g1(tr, "wp0", C0); absorb_g1(tr, "wp1", C1); absorb_g1(tr, "wp2", C2);
        Fr_t x = fs_challenge_fr(tr);
        r[m_max - 1 - t] = x;
        C_cur = lagrange3_g1(C0, C1, C2, x);
    }
    T.lap("rounds");

    std::vector<G1Jacobian_t> cvfin;
    try { cvfin = read_wbatch_vfin(pwaccdir + "/wbatch_vfin.bin"); }
    catch (const std::exception& e) { return reject("wvfin_count", e.what()); }
    if (cvfin.size() != nT)
        return reject("wvfin_count", "wbatch_vfin.bin entry count != distinct tensor count");
    for (uint j = 0; j < nT; j++) absorb_g1(tr, "wvfin", cvfin[j]);
    {
        G1Jacobian_t chk = g1_identity();
        for (uint j = 0; j < nT; j++) {
            Fr_t cj = h_scalar(bo_Mj_at_r(tensors[j], cs, rho_pows, r),
                               bo_kappa(tensors[j].vars, r), 2);
            chk = h_add(chk, h_mul(cvfin[j], cj));
        }
        if (!g1_eq(chk, C_cur))
            return reject("wterminal", "sum_j c_j*C_vfin_j != C_cur (homomorphic G3)");
    }
    T.lap("terminal");

    Fr_t rhop = fs_challenge_fr(tr);
    std::vector<Fr_t> rhop_pows(nT);
    rhop_pows[0] = F_ONE;
    for (uint j = 1; j < nT; j++) rhop_pows[j] = h_scalar(rhop_pows[j - 1], rhop, 2);
    std::set<uint32_t> domains;
    for (auto& t : tensors) domains.insert(t.domain);
    const bool slow_fold = getenv("ZKOB_SLOW_FOLD") != nullptr;
    const bool fold_xchk = getenv("ZKOB_FOLD_CROSSCHECK") != nullptr;
    for (uint32_t G : domains) {
        uint logG = ceilLog2(G);
        std::string ipath = pwaccdir + "/wipa_batch_" + std::to_string(G) + ".bin";
        if (!bo_file_exists(ipath))
            return reject("wgroup_missing", "wipa_batch_" + std::to_string(G) + ".bin missing");
        std::vector<uint32_t> gj;
        std::vector<Fr_t> gcoef;
        G1Jacobian_t Vstar = g1_identity();
        for (uint j = 0; j < nT; j++) {
            if (tensors[j].domain != G) continue;
            Fr_t coef = h_scalar(rhop_pows[j], bo_kappa(tensors[j].vars, r), 2);
            gj.push_back(j);
            gcoef.push_back(coef);
            Vstar = h_add(Vstar, h_mul(cvfin[j], coef));
        }
        G1Jacobian_t Cstar = g1_identity();
        if (!slow_fold) {
            std::vector<G1Jacobian_t> per_tensor;
            Cstar = bo_batched_group_fold(coms, tensors, gj, gcoef, r, logG,
                                          fold_xchk ? &per_tensor : nullptr);
            if (fold_xchk) {
                G1Jacobian_t Cslow = g1_identity();
                for (size_t t = 0; t < gj.size(); t++) {
                    uint j = gj[t];
                    std::vector<Fr_t> u_row(r.begin() + logG, r.begin() + tensors[j].vars);
                    G1Jacobian_t cjp = h_mul(fold_chain(coms[j].gpu_data, coms[j].size, u_row, 0),
                                             gcoef[t]);
                    if (!g1_eq(per_tensor[t], cjp))
                        throw std::runtime_error("STOP: w batched fold != per-tensor fold_chain ("
                                                 + tensors[j].comref + ")");
                    Cslow = h_add(Cslow, cjp);
                }
                if (!g1_eq(Cstar, Cslow))
                    throw std::runtime_error("STOP: w batched group fold != fold_chain RLC (domain "
                                             + std::to_string(G) + ")");
            }
        } else {
            for (size_t t = 0; t < gj.size(); t++) {
                uint j = gj[t];
                std::vector<Fr_t> u_row(r.begin() + logG, r.begin() + tensors[j].vars);
                G1Jacobian_t cjp = fold_chain(coms[j].gpu_data, coms[j].size, u_row, 0);
                Cstar = h_add(Cstar, h_mul(cjp, gcoef[t]));
            }
        }
        T.lap(("wfold_" + std::to_string(G)).c_str());
        G1Jacobian_t P0 = h_add(Cstar, Vstar);
        ZkIpaProof pf;
        try { pf = read_zkipa(ipath); }
        catch (const std::exception& e) { return reject("wipa" + std::to_string(G), e.what()); }
        if (!zk_ipa_verify(gens.at(G).gpu_data, G, Q, H, P0,
                           std::vector<Fr_t>(r.begin(), r.begin() + logG), pf, tr))
            return reject("wipa" + std::to_string(G),
                          "ZK batched IPA for domain " + std::to_string(G) + " failed");
        T.lap(("wipa_" + std::to_string(G)).c_str());
    }
    printf("opening_batch_w ACCEPT (%u claims, %u tensors, %zu domains; all evals committed)\n",
           (uint)cs.size(), nT, domains.size());
    if (locus_out) *locus_out = "accept";
    return true;
}

// =====================  D4 leak scan  ======================================
// Byte-scan a file for a 32-byte Fr pattern (the known weight-MLE eval).
static bool wp_file_contains(const std::string& path, const Fr_t& pat) {
    if (!bo_file_exists(path)) return false;
    auto b = bo_read_file(path);
    if (b.size() < sizeof(Fr_t)) return false;
    const uint8_t* p = (const uint8_t*)&pat;
    for (size_t i = 0; i + sizeof(Fr_t) <= b.size(); i++)
        if (!memcmp(b.data() + i, p, sizeof(Fr_t))) return true;
    return false;
}
// scan a list of artifact files; returns the paths that contain the pattern
static std::vector<std::string> wp_leak_scan(const std::vector<std::string>& files,
                                             const Fr_t& secret) {
    std::vector<std::string> hits;
    for (auto& f : files)
        if (wp_file_contains(f, secret)) hits.push_back(f);
    return hits;
}

// =====================  runtime probe  =====================================
// D1's only kernel-shape novelty is wp_hide_rows' use of k_g1_scale /
// k_g1_add_pairs at row scale — cross-checked against the 1-thread
// h_mul/h_add path; plus the lagrange3_g1 / lagrange3 consistency identity
// (commit-then-fold == fold-then-commit) the homomorphic rounds rely on.
static void wpriv_probe() {
    fs::Transcript tr("wpriv_probe");
    // hide_rows vs 1-thread path on 5 rows
    {
        const uint n = 5;
        Commitment gen = Commitment::random(8);
        FrTensor t = FrTensor::random_int(n * 8, 8);
        G1TensorJacobian com = gen.commit(t);
        std::vector<G1Jacobian_t> before(n);
        cudaMemcpy(before.data(), com.gpu_data, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        Commitment hq = Commitment::random(1);
        G1Jacobian_t H = hq(0);
        std::vector<Fr_t> s(n);
        for (auto& x : s) x = fs_challenge_fr(tr);
        wp_hide_rows(com, s, H);
        std::vector<G1Jacobian_t> after(n);
        cudaMemcpy(after.data(), com.gpu_data, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        for (uint i = 0; i < n; i++)
            if (!g1_eq(after[i], h_add(before[i], h_mul(H, s[i]))))
                throw std::runtime_error("PROBE FAIL: wp_hide_rows");
    }
    // commit-then-fold == fold-then-commit (the homomorphic round identity)
    {
        Commitment qg = Commitment::random(2);
        G1Jacobian_t Q = qg(0), H = qg(1);
        Fr_t p0 = fs_challenge_fr(tr), p1 = fs_challenge_fr(tr), p2 = fs_challenge_fr(tr);
        Fr_t t0 = fs_challenge_fr(tr), t1 = fs_challenge_fr(tr), t2 = fs_challenge_fr(tr);
        Fr_t x = fs_challenge_fr(tr);
        G1Jacobian_t lhs = lagrange3_g1(ped_qh(p0, t0, Q, H), ped_qh(p1, t1, Q, H),
                                        ped_qh(p2, t2, Q, H), x);
        G1Jacobian_t rhs = ped_qh(lagrange3(p0, p1, p2, x), lagrange3(t0, t1, t2, x), Q, H);
        if (!g1_eq(lhs, rhs))
            throw std::runtime_error("PROBE FAIL: lagrange3_g1 homomorphism");
    }
    // schnorr roundtrips
    {
        Commitment qg = Commitment::random(2);
        G1Jacobian_t Q = qg(0), H = qg(1);
        Fr_t d = fs_challenge_fr(tr);
        fs::Transcript tp("wpriv_probe_s"), tv("wpriv_probe_s");
        SchnorrH p = schnorr_h_prove(d, H, tp);
        if (!schnorr_h_verify(h_mul(H, d), p, H, tv))
            throw std::runtime_error("PROBE FAIL: schnorr_h roundtrip");
        Fr_t a = fs_challenge_fr(tr), b = fs_challenge_fr(tr);
        fs::Transcript tp2("wpriv_probe_s2"), tv2("wpriv_probe_s2");
        Schnorr2 p2 = schnorr2_prove(a, b, Q, H, tp2);
        if (!schnorr2_verify(h_add(h_mul(Q, a), h_mul(H, b)), p2, Q, H, tv2))
            throw std::runtime_error("PROBE FAIL: schnorr2 roundtrip");
    }
}

#endif // ZKOB_WPRIV_CUH
