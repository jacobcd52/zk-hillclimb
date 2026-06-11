// COORDINATOR-BUILT (do not let submission agents edit).
// Stage A of the transport rebuild (TRANSPORT_REBUILD_DESIGN §2.1/§2.2 + the
// TRANSPORT_REVIEW required pins F3/F4/F5/F6):
//   - Claim objects + claims.bin accumulator serialization (EvalVar tag from
//     day one; Committed claims REJECTED until Stage D — §4.4 forward-compat)
//   - the F4 absorb encoding (every variable-length field length-prefixed
//     INSIDE the claim blob; EvalVar tag absorbed explicitly)
//   - batch-evaluation sumcheck (RLC-of-eq reduction) + ONE IPA per generator
//     domain (batch_prove / batch_verify)
//   - F3: per distinct tensor, com_file_point_count == n_rows ==
//     2^{vars-logG} checked BEFORE fold_chain (restores open_verify's
//     zkob_lookup.cuh:132 size check at the new location)
//   - F5: batch_verify consumes the VERIFIER-recomputed claim list only; the
//     prover's claims.bin is byte-compared (claims_match), never parsed or
//     consumed; claims_match is a component of the opening_batch verdict
//   - F6: the drvstate list absorbed into the batch transcript is read from
//     the VERIFIER's own accumulator dir, never from a prover artifact;
//     batch_sumcheck.bin's redundant n_claims/m_max fields are cross-checked;
//     n_claims == 0 REJECTS explicitly
//
// Orientation pins (TRANSPORT_REVIEW §9, validated in vrf_toy_batchopen.cu):
//   - flat index = row*G + col; point = u_col[0..logG) ++ u_row (col bits
//     first, LSB-first within each, me_weights orientation)
//   - k_bo_fold binds the current MSB => batch round t binds VARIABLE
//     m_max-1-t; r[] is indexed BY VARIABLE: r[m_max-1-t] = round-t challenge
//   - u-hat padding is in the high variables (zero-extension); kappa_j =
//     prod_{v >= vars_j} (1 - r[v])
//   - M_j(r) includes its own high-variable (1-r_v) factors, so the G3
//     terminal coefficient is M_j(r)*kappa_j (kappa squared in total); the G5
//     group RLC uses kappa_j ONCE (consistent on both C* and v* sides)
//
// All Fr arithmetic in the kernels' "mont limbs as integers" view. New Fr
// kernels (k_bo_eq_expand, k_bo_hp2, k_bo_fold, k_bo_axpy) are presumed
// -dlto miscompile bait until probed: bo_probe_kernels() cross-checks every
// shape against the proven 1-thread h_scalar helpers at runtime; the toy and
// both selftests call it at startup, and the per-round p(0)+p(1)==cur strict
// checks re-probe every prove.
#ifndef ZKOB_CLAIMS_CUH
#define ZKOB_CLAIMS_CUH
#include "vrf_common.cuh"
#include <chrono>
#include <map>
#include <set>
#include <sys/stat.h>

// =====================  profiling (env-guarded, ZKOB_PROF=1)  ==============
static bool bo_prof_on() { static int v = -1;
    if (v < 0) v = getenv("ZKOB_PROF") ? 1 : 0; return v == 1; }
static double bo_now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct BoTimer {
    double t0; const char* what;
    explicit BoTimer(const char* w) : t0(bo_now()), what(w) {}
    void lap(const char* sub) {
        if (bo_prof_on()) { cudaDeviceSynchronize();
            printf("PROF %s.%s %.4f s\n", what, sub, bo_now() - t0); }
        t0 = bo_now(); }
};

// =====================  claim objects + serialization  =====================
// EvalVar tags (§4.4): the format carries the variant from day one; the plain
// batch rejects Committed until Stage D.
enum : uint8_t { BO_EVAL_PLAIN = 0, BO_EVAL_COMMITTED = 1 };

struct BoClaim {
    std::string id;        // canonical: "<manifest_id>[/<sub>]:<tensor>"
    std::string comref;    // path of the commitment file it opens against
    uint32_t domain = 0;   // generator domain size G
    uint32_t n_rows = 0;   // R_pad (row count of the commitment vector)
    std::vector<Fr_t> point;  // u_col[0..logG) ++ u_row  (col bits first)
    uint8_t tag = BO_EVAL_PLAIN;
    Fr_t eval = {0,0,0,0,0,0,0,0};
    G1Jacobian_t ceval = {};  // tag == BO_EVAL_COMMITTED only (Stage D)
};

// canonical per-claim byte encoding — used BOTH for claims.bin and for the
// G0 absorb (F4: id/comref/point are length-prefixed inside the blob, the
// EvalVar tag is an explicit byte; two different claims can never serialize
// to the same bytes).
static void bo_put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; i++) b.push_back(uint8_t(v >> (8 * i)));
}
static void bo_put_bytes(std::vector<uint8_t>& b, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*)p;
    b.insert(b.end(), q, q + n);
}
static std::vector<uint8_t> claim_blob(const BoClaim& c) {
    std::vector<uint8_t> b;
    bo_put_u32(b, (uint32_t)c.id.size());     bo_put_bytes(b, c.id.data(), c.id.size());
    bo_put_u32(b, (uint32_t)c.comref.size()); bo_put_bytes(b, c.comref.data(), c.comref.size());
    bo_put_u32(b, c.domain);
    bo_put_u32(b, c.n_rows);
    bo_put_u32(b, (uint32_t)c.point.size());
    bo_put_bytes(b, c.point.data(), c.point.size() * sizeof(Fr_t));
    b.push_back(c.tag);
    if (c.tag == BO_EVAL_PLAIN) bo_put_bytes(b, &c.eval, sizeof(Fr_t));
    else                        bo_put_bytes(b, &c.ceval, sizeof(G1Jacobian_t));
    return b;
}

static const char BO_CLAIMS_MAGIC[4] = {'Z','K','C','L'};
static const uint32_t BO_CLAIMS_VERSION = 1;

static std::vector<uint8_t> claims_serialize(const std::vector<BoClaim>& cs) {
    std::vector<uint8_t> b;
    bo_put_bytes(b, BO_CLAIMS_MAGIC, 4);
    bo_put_u32(b, BO_CLAIMS_VERSION);
    bo_put_u32(b, (uint32_t)cs.size());
    for (auto& c : cs) { auto cb = claim_blob(c); bo_put_bytes(b, cb.data(), cb.size()); }
    return b;
}
static void claims_save(const std::string& path, const std::vector<BoClaim>& cs) {
    auto b = claims_serialize(cs);
    FILE* f = open_or_die(path, "wb");
    fwrite(b.data(), 1, b.size(), f);
    fclose(f);
}

struct BoReader {
    const std::vector<uint8_t>& b; size_t off = 0;
    explicit BoReader(const std::vector<uint8_t>& bb) : b(bb) {}
    uint32_t u32() {
        if (off + 4 > b.size()) throw std::runtime_error("claims: truncated u32");
        uint32_t v = 0; for (int i = 0; i < 4; i++) v |= uint32_t(b[off + i]) << (8 * i);
        off += 4; return v; }
    void bytes(void* p, size_t n) {
        if (off + n > b.size()) throw std::runtime_error("claims: truncated bytes");
        memcpy(p, b.data() + off, n); off += n; }
    std::string str(uint32_t n) {
        if (off + n > b.size()) throw std::runtime_error("claims: truncated str");
        std::string s((const char*)b.data() + off, n); off += n; return s; }
};
static std::vector<uint8_t> bo_read_file(const std::string& path) {
    FILE* f = open_or_die(path, "rb");
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(n);
    if (n && fread(b.data(), 1, n, f) != (size_t)n) { fclose(f); throw std::runtime_error("short read: " + path); }
    fclose(f); return b;
}
static bool bo_file_exists(const std::string& path) {
    struct stat st; return stat(path.c_str(), &st) == 0;
}
static std::vector<BoClaim> claims_parse(const std::vector<uint8_t>& bytes) {
    BoReader r(bytes);
    char magic[4]; r.bytes(magic, 4);
    if (memcmp(magic, BO_CLAIMS_MAGIC, 4)) throw std::runtime_error("claims: bad magic");
    if (r.u32() != BO_CLAIMS_VERSION) throw std::runtime_error("claims: bad version");
    uint32_t n = r.u32();
    std::vector<BoClaim> cs(n);
    for (auto& c : cs) {
        c.id = r.str(r.u32());
        c.comref = r.str(r.u32());
        c.domain = r.u32();
        c.n_rows = r.u32();
        uint32_t np = r.u32();
        if (np > 64) throw std::runtime_error("claims: point too long");
        c.point.resize(np);
        r.bytes(c.point.data(), np * sizeof(Fr_t));
        r.bytes(&c.tag, 1);
        if (c.tag == BO_EVAL_PLAIN) r.bytes(&c.eval, sizeof(Fr_t));
        else if (c.tag == BO_EVAL_COMMITTED) r.bytes(&c.ceval, sizeof(G1Jacobian_t));
        else throw std::runtime_error("claims: unknown EvalVar tag");
    }
    if (r.off != bytes.size()) throw std::runtime_error("claims: trailing bytes");
    return cs;
}
static std::vector<BoClaim> claims_load(const std::string& path) {
    return claims_parse(bo_read_file(path));
}

// ---- driver-state list (F6: the verifier absorbs ITS OWN drvstates) ----
struct DrvState { std::string id; uint8_t state[32]; };
static const char BO_DS_MAGIC[4] = {'Z','K','D','S'};
static void drvstates_save(const std::string& path, const std::vector<DrvState>& ds) {
    std::vector<uint8_t> b;
    bo_put_bytes(b, BO_DS_MAGIC, 4); bo_put_u32(b, 1); bo_put_u32(b, (uint32_t)ds.size());
    for (auto& d : ds) {
        bo_put_u32(b, (uint32_t)d.id.size()); bo_put_bytes(b, d.id.data(), d.id.size());
        bo_put_bytes(b, d.state, 32);
    }
    FILE* f = open_or_die(path, "wb");
    fwrite(b.data(), 1, b.size(), f); fclose(f);
}
static std::vector<DrvState> drvstates_load(const std::string& path) {
    auto bytes = bo_read_file(path);
    BoReader r(bytes);
    char magic[4]; r.bytes(magic, 4);
    if (memcmp(magic, BO_DS_MAGIC, 4)) throw std::runtime_error("drvstates: bad magic");
    if (r.u32() != 1) throw std::runtime_error("drvstates: bad version");
    uint32_t n = r.u32();
    std::vector<DrvState> ds(n);
    for (auto& d : ds) { d.id = r.str(r.u32()); r.bytes(d.state, 32); }
    return ds;
}

// ---- emission API (drivers call these at their old ipa_prove/verify sites) ----
static void claim_emit(const std::string& accdir, const BoClaim& c) {
    std::string path = accdir + "/claims.bin";
    std::vector<BoClaim> cs;
    if (bo_file_exists(path)) cs = claims_load(path);
    cs.push_back(c);
    claims_save(path, cs);
}
// prover-side witness pointer (NOT a protocol artifact; batch_prove input only)
static void witref_emit(const std::string& accdir, const std::string& comref,
                        const std::string& witpath) {
    std::string path = accdir + "/witrefs.txt";
    // one entry per distinct tensor (comref key)
    if (bo_file_exists(path)) {
        auto bytes = bo_read_file(path);
        std::string all((const char*)bytes.data(), bytes.size());
        if (all.find(comref + "\t") != std::string::npos) return;
    }
    FILE* f = open_or_die(path, "ab");
    fprintf(f, "%s\t%s\n", comref.c_str(), witpath.c_str());
    fclose(f);
}
static std::map<std::string, std::string> witrefs_load(const std::string& accdir) {
    std::map<std::string, std::string> m;
    auto bytes = bo_read_file(accdir + "/witrefs.txt");
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
static void drvstate_emit(const std::string& accdir, const std::string& runid,
                          const fs::Transcript& tr) {
    std::string path = accdir + "/drvstates.bin";
    std::vector<DrvState> ds;
    if (bo_file_exists(path)) ds = drvstates_load(path);
    DrvState d; d.id = runid; memcpy(d.state, tr.state, 32);
    ds.push_back(d);
    drvstates_save(path, ds);
}

// =====================  distinct-tensor derivation  ========================
// canonical tensor identity key = comref (the recomputed path string); canonical
// tensor order = first appearance in the canonical claim list.
struct TensorInfo {
    std::string comref;
    uint32_t domain = 0, n_rows = 0, vars = 0, logG = 0;
    std::vector<uint32_t> claim_idx;   // indices into the claim list
};
// structural consistency of the claim list itself (claims sharing a tensor
// must agree on shape; per-claim n_rows == 2^{vars - logG}). Returns false
// with err set on violation. F3's com-FILE check is separate (needs disk).
static bool derive_tensors(const std::vector<BoClaim>& cs, std::vector<TensorInfo>& out,
                           std::string& err) {
    out.clear();
    std::map<std::string, uint32_t> key2idx;
    for (uint32_t i = 0; i < cs.size(); i++) {
        const BoClaim& c = cs[i];
        if (c.domain < 2 || (c.domain & (c.domain - 1))) { err = "claim " + c.id + ": domain not a power of two >= 2"; return false; }
        uint32_t logG = ceilLog2(c.domain);
        if (c.point.size() < logG) { err = "claim " + c.id + ": point shorter than logG"; return false; }
        uint32_t vars = (uint32_t)c.point.size();
        if (vars - logG >= 32 || c.n_rows != (1u << (vars - logG))) {
            err = "claim " + c.id + ": n_rows != 2^{vars-logG}"; return false; }
        auto it = key2idx.find(c.comref);
        if (it == key2idx.end()) {
            TensorInfo t; t.comref = c.comref; t.domain = c.domain;
            t.n_rows = c.n_rows; t.vars = vars; t.logG = logG;
            t.claim_idx.push_back(i);
            key2idx[c.comref] = (uint32_t)out.size();
            out.push_back(t);
        } else {
            TensorInfo& t = out[it->second];
            if (t.domain != c.domain || t.n_rows != c.n_rows || t.vars != vars) {
                err = "claim " + c.id + ": shape disagrees with earlier claim on same tensor"; return false; }
            t.claim_idx.push_back(i);
        }
    }
    return true;
}

// =====================  G0 absorb schedule  ================================
static void bo_file_sha256(const std::string& path, uint8_t out[32]) {
    FILE* f = open_or_die(path, "rb");
    fs::Sha256Ctx c; fs::sha256_init(c);
    uint8_t buf[65536]; size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) fs::sha256_update(c, buf, n);
    fclose(f);
    fs::sha256_final(c, out);
}
// G0: absorb n, every claim (canonical order), every distinct comref hash
// (canonical tensor order), every drvstate (walk order) — all BEFORE rho.
static void batch_absorb_g0(fs::Transcript& tr, const std::vector<BoClaim>& cs,
                            const std::vector<TensorInfo>& tensors,
                            const std::vector<DrvState>& dss) {
    absorb_u32(tr, "n_claims", (uint32_t)cs.size());
    for (auto& c : cs) {
        auto b = claim_blob(c);
        tr.absorb("claim", b.data(), b.size());
    }
    for (auto& t : tensors) {
        std::vector<uint8_t> b;
        bo_put_u32(b, (uint32_t)t.comref.size());
        bo_put_bytes(b, t.comref.data(), t.comref.size());
        uint8_t h[32]; bo_file_sha256(t.comref, h);
        bo_put_bytes(b, h, 32);
        tr.absorb("comref", b.data(), b.size());
    }
    for (auto& d : dss) {
        std::vector<uint8_t> b;
        bo_put_u32(b, (uint32_t)d.id.size());
        bo_put_bytes(b, d.id.data(), d.id.size());
        bo_put_bytes(b, d.state, 32);
        tr.absorb("drvstate", b.data(), b.size());
    }
}

// =====================  Fr kernels (new shapes; -dlto probed)  =============
// eq-tensor doubling: bit i is the NEW top bit (LSB-first me_weights order);
// same shape as zkob_lookup.cuh's k_eq_expand (distinct name: this header
// must not collide with zkob_lookup.cuh in TUs that include both).
KERNEL void k_bo_eq_expand(GLOBAL Fr_t* in, Fr_t c, GLOBAL Fr_t* out, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    Fr_t cm = blstrs__scalar__Scalar_mont(c);
    Fr_t one_minus = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ONE, cm);
    out[gid]     = blstrs__scalar__Scalar_mul(one_minus, in[gid]);
    out[gid + n] = blstrs__scalar__Scalar_mul(cm, in[gid]);
}
// front/back-half fold (binds current MSB), k_fr_fold shape
KERNEL void k_bo_fold(GLOBAL Fr_t* a, Fr_t v, GLOBAL Fr_t* out, uint N_out) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;
    out[gid] = blstrs__scalar__Scalar_add(a[gid],
        blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(v),
            blstrs__scalar__Scalar_sub(a[gid + N_out], a[gid])));
}
// degree-2 round evals of sum_x M(x)*P(x): p(t) per element for t = 0,1,2,
// X_t[i] = X[i] + t*(X[i+h]-X[i]) (front/back orientation, binds current MSB)
KERNEL void k_bo_hp2(GLOBAL Fr_t* M, GLOBAL Fr_t* P,
                     GLOBAL Fr_t* o0, GLOBAL Fr_t* o1, GLOBAL Fr_t* o2, uint h) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= h) return;
    Fr_t m = M[gid], dm = blstrs__scalar__Scalar_sub(M[gid + h], M[gid]);
    Fr_t p = P[gid], dp = blstrs__scalar__Scalar_sub(P[gid + h], P[gid]);
    #pragma unroll
    for (int t = 0; t < 3; t++) {
        // plain m*p in the integer view: mont(m) * p
        Fr_t v = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(m), p);
        if (t == 0) o0[gid] = v;
        else if (t == 1) o1[gid] = v;
        else o2[gid] = v;
        m = blstrs__scalar__Scalar_add(m, dm);
        p = blstrs__scalar__Scalar_add(p, dp);
    }
}
// out += c * in (integer view)
KERNEL void k_bo_axpy(GLOBAL Fr_t* out, GLOBAL Fr_t* in, Fr_t c, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    out[gid] = blstrs__scalar__Scalar_add(out[gid],
        blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(c), in[gid]));
}

// pairwise-add sum over pow2 n (k_fr_add_pairs reduce, source preserved)
static Fr_t bo_dev_sum(const Fr_t* d_src, uint n) {
    Fr_t *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(Fr_t));
    cudaMalloc(&d_y, (n / 2 + 1) * sizeof(Fr_t));
    cudaMemcpy(d_x, d_src, n * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    uint sz = n;
    while (sz > 1) {
        uint half = sz / 2;
        k_fr_add_pairs<<<(half + 63) / 64, 64>>>(d_x, d_y, half);
        cudaDeviceSynchronize();
        std::swap(d_x, d_y); sz = half;
    }
    Fr_t out; cudaMemcpy(&out, d_x, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y);
    return out;
}

// eq(u,v) = 2uv - (u+v) + 1, integer view (matches upstream eqEvalKernel /
// zkob_lookup.cuh my_eq; redefined here so this header stands alone)
static Fr_t bo_eq1(const Fr_t& u, const Fr_t& v) {
    Fr_t uv = h_scalar(u, v, 2);
    Fr_t e = h_scalar(h_scalar(uv, uv, 0), h_scalar(u, v, 0), 1);
    return h_scalar(e, F_ONE, 0);
}

// device eq table E[b] = prod_i (b>>i & 1 ? u[i] : 1-u[i])  (raw pointers)
static void bo_build_eq(Fr_t* d_out /* size 2^|u| */, const std::vector<Fr_t>& u) {
    uint L = (uint)u.size();
    Fr_t* d_a; Fr_t* d_b;
    cudaMalloc(&d_a, sizeof(Fr_t) << L);
    cudaMalloc(&d_b, sizeof(Fr_t) << L);
    Fr_t one = F_ONE;
    cudaMemcpy(d_a, &one, sizeof(Fr_t), cudaMemcpyHostToDevice);
    uint n = 1;
    for (uint i = 0; i < L; i++) {
        k_bo_eq_expand<<<(n + 255) / 256, 256>>>(d_a, u[i], d_b, n);
        cudaDeviceSynchronize();
        std::swap(d_a, d_b);
        n <<= 1;
    }
    cudaMemcpy(d_out, d_a, sizeof(Fr_t) << L, cudaMemcpyDeviceToDevice);
    cudaFree(d_a); cudaFree(d_b);
}

// runtime -dlto miscompile probe: every new kernel shape vs the proven
// 1-thread h_scalar helpers, at a non-trivial size. Throws on mismatch.
static void bo_probe_kernels() {
    const uint L = 5, n = 1u << L;
    std::vector<Fr_t> u(L);
    fs::Transcript tr("bo_probe");
    for (auto& x : u) x = fs_challenge_fr(tr);
    // k_bo_eq_expand vs host me_weights (same orientation by construction)
    Fr_t* d_e; cudaMalloc(&d_e, n * sizeof(Fr_t));
    bo_build_eq(d_e, u);
    std::vector<Fr_t> h_e(n);
    cudaMemcpy(h_e.data(), d_e, n * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    auto w = me_weights(u);
    for (uint i = 0; i < n; i++)
        if (!fr_eq(h_e[i], w[i])) throw std::runtime_error("PROBE FAIL: k_bo_eq_expand");
    // random-ish test vectors
    std::vector<Fr_t> hM(n), hP(n);
    for (uint i = 0; i < n; i++) { hM[i] = fs_challenge_fr(tr); hP[i] = fs_challenge_fr(tr); }
    Fr_t *d_M, *d_P; cudaMalloc(&d_M, n * sizeof(Fr_t)); cudaMalloc(&d_P, n * sizeof(Fr_t));
    cudaMemcpy(d_M, hM.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, hP.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);
    // k_bo_hp2 vs host
    uint h = n / 2;
    Fr_t *d_o0, *d_o1, *d_o2;
    cudaMalloc(&d_o0, h * sizeof(Fr_t)); cudaMalloc(&d_o1, h * sizeof(Fr_t));
    cudaMalloc(&d_o2, h * sizeof(Fr_t));
    k_bo_hp2<<<(h + 63) / 64, 64>>>(d_M, d_P, d_o0, d_o1, d_o2, h);
    cudaDeviceSynchronize();
    std::vector<Fr_t> o0(h), o1(h), o2(h);
    cudaMemcpy(o0.data(), d_o0, h * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(o1.data(), d_o1, h * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(o2.data(), d_o2, h * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    for (uint i = 0; i < h; i++) {
        Fr_t dm = h_scalar(hM[i + h], hM[i], 1), dp = h_scalar(hP[i + h], hP[i], 1);
        Fr_t m = hM[i], p = hP[i];
        for (int t = 0; t < 3; t++) {
            Fr_t v = h_scalar(m, p, 2);
            const Fr_t& got = (t == 0) ? o0[i] : (t == 1) ? o1[i] : o2[i];
            if (!fr_eq(got, v)) throw std::runtime_error("PROBE FAIL: k_bo_hp2");
            m = h_scalar(m, dm, 0); p = h_scalar(p, dp, 0);
        }
    }
    // k_bo_fold vs host
    Fr_t v = fs_challenge_fr(tr);
    Fr_t* d_f; cudaMalloc(&d_f, h * sizeof(Fr_t));
    k_bo_fold<<<(h + 63) / 64, 64>>>(d_M, v, d_f, h);
    cudaDeviceSynchronize();
    std::vector<Fr_t> f(h);
    cudaMemcpy(f.data(), d_f, h * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    for (uint i = 0; i < h; i++) {
        Fr_t want = h_scalar(hM[i], h_scalar(v, h_scalar(hM[i + h], hM[i], 1), 2), 0);
        if (!fr_eq(f[i], want)) throw std::runtime_error("PROBE FAIL: k_bo_fold");
    }
    // k_bo_axpy vs host
    Fr_t c = fs_challenge_fr(tr);
    k_bo_axpy<<<(n + 63) / 64, 64>>>(d_M, d_P, c, n);
    cudaDeviceSynchronize();
    std::vector<Fr_t> a(n);
    cudaMemcpy(a.data(), d_M, n * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    for (uint i = 0; i < n; i++) {
        Fr_t want = h_scalar(hM[i], h_scalar(c, hP[i], 2), 0);
        if (!fr_eq(a[i], want)) throw std::runtime_error("PROBE FAIL: k_bo_axpy");
    }
    cudaFree(d_e); cudaFree(d_M); cudaFree(d_P);
    cudaFree(d_o0); cudaFree(d_o1); cudaFree(d_o2); cudaFree(d_f);
}

// =====================  verifier-side terminal helpers  ====================
// kappa_j = prod_{v >= vars_j} (1 - r[v])    (r indexed by VARIABLE)
static Fr_t bo_kappa(uint vars_j, const std::vector<Fr_t>& r) {
    Fr_t k = F_ONE;
    for (uint v = vars_j; v < r.size(); v++)
        k = h_scalar(k, h_scalar(F_ONE, r[v], 1), 2);
    return k;
}
// M_j(r) = sum_{i in j} rho^i * prod_{v < vars_j} eq(point_i[v], r[v]) * kappa_j
// (the high-variable factors eq(0, r_v) = 1 - r_v ARE included: this is the
// honest full m_max-variable M_j at r)
static Fr_t bo_Mj_at_r(const TensorInfo& tj, const std::vector<BoClaim>& cs,
                       const std::vector<Fr_t>& rho_pows, const std::vector<Fr_t>& r) {
    Fr_t kap = bo_kappa(tj.vars, r);
    Fr_t acc = F_ZERO;
    for (uint32_t idx : tj.claim_idx) {
        Fr_t term = rho_pows[idx];
        for (uint v = 0; v < tj.vars; v++)
            term = h_scalar(term, bo_eq1(cs[idx].point[v], r[v]), 2);
        acc = h_scalar(acc, h_scalar(term, kap, 2), 0);
    }
    return acc;
}

// =====================  batch artifacts  ===================================
static const char BO_BS_MAGIC[4] = {'Z','K','B','S'};
static const char BO_BV_MAGIC[4] = {'Z','K','B','V'};
struct BatchSumcheck {
    uint32_t n_claims = 0, m_max = 0;   // redundant; cross-checked (F6)
    std::vector<Fr_t> ev;               // 3 per round
};
static void write_batch_sumcheck(const std::string& path, const BatchSumcheck& p) {
    FILE* f = open_or_die(path, "wb");
    fwrite(BO_BS_MAGIC, 1, 4, f);
    fwrite(&p.n_claims, sizeof(uint32_t), 1, f);
    fwrite(&p.m_max, sizeof(uint32_t), 1, f);
    write_pod_vec(f, p.ev);
    fclose(f);
}
static BatchSumcheck read_batch_sumcheck(const std::string& path) {
    FILE* f = open_or_die(path, "rb");
    char magic[4];
    BatchSumcheck p;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, BO_BS_MAGIC, 4) ||
        fread(&p.n_claims, sizeof(uint32_t), 1, f) != 1 ||
        fread(&p.m_max, sizeof(uint32_t), 1, f) != 1) {
        fclose(f); throw std::runtime_error("read_batch_sumcheck: header"); }
    p.ev = read_pod_vec<Fr_t>(f);
    fclose(f);
    return p;
}
static void write_batch_vfin(const std::string& path, const std::vector<Fr_t>& v) {
    FILE* f = open_or_die(path, "wb");
    fwrite(BO_BV_MAGIC, 1, 4, f);
    write_pod_vec(f, v);
    fclose(f);
}
static std::vector<Fr_t> read_batch_vfin(const std::string& path) {
    FILE* f = open_or_die(path, "rb");
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, BO_BV_MAGIC, 4)) {
        fclose(f); throw std::runtime_error("read_batch_vfin: header"); }
    auto v = read_pod_vec<Fr_t>(f);
    fclose(f);
    return v;
}

// =====================  batch prove  =======================================
// evil modes (selftest forgery construction ONLY; 0 in production):
//   0 honest (strict round/terminal self-checks)
//   1 honest PROCEDURE over a (possibly false) claim list, self-checks off
//     (BO-1a: a false claim makes round 0 fail verifier-side)
//   2 adaptive: per round force p(1) = cur - p(0), then poison the LAST
//     tensor's v' to satisfy G3 (BO-1b: dies at that tensor's group IPA)
//   3 doctor the transcript/RLC inputs (claim 0 eval +1) but write honest
//     artifacts w.r.t. the witness (BO-4: verifier rho differs -> round 0)
static void batch_prove(const std::string& accdir, const std::string& run_seed,
                        const std::map<uint32_t, std::string>& genpaths,
                        const std::string& qpath, int evil = 0) {
    BoTimer T("batch_prove");
    std::vector<BoClaim> cs = claims_load(accdir + "/claims.bin");
    if (cs.empty()) throw std::runtime_error("batch_prove: zero claims");
    std::vector<BoClaim> cs_tr = cs;            // list used for transcript/RLC
    if (evil == 3) cs_tr[0].eval = h_scalar(cs_tr[0].eval, F_ONE, 0);
    std::vector<TensorInfo> tensors;
    std::string err;
    if (!derive_tensors(cs_tr, tensors, err)) throw std::runtime_error("batch_prove: " + err);
    std::vector<DrvState> dss;
    if (bo_file_exists(accdir + "/drvstates.bin")) dss = drvstates_load(accdir + "/drvstates.bin");
    auto wits = witrefs_load(accdir);
    Commitment qg(qpath);
    G1Jacobian_t Q = qg(0);

    uint m_max = 0;
    for (auto& t : tensors) m_max = std::max(m_max, t.vars);
    if (m_max < 1) throw std::runtime_error("batch_prove: m_max < 1");
    const uint nT = (uint)tensors.size();

    // G0 + rho
    fs::Transcript tr(run_seed + ":opening_batch");
    batch_absorb_g0(tr, cs_tr, tensors, dss);
    Fr_t rho = fs_challenge_fr(tr);
    std::vector<Fr_t> rho_pows(cs.size());
    rho_pows[0] = F_ONE;
    for (uint i = 1; i < cs.size(); i++) rho_pows[i] = h_scalar(rho_pows[i - 1], rho, 2);

    // witness tensors (kept alive; reused for the G5 row-folds)
    std::vector<FrTensor> wit;
    wit.reserve(nT);
    for (auto& t : tensors) {
        auto it = wits.find(t.comref);
        if (it == wits.end()) throw std::runtime_error("batch_prove: no witref for " + t.comref);
        wit.emplace_back(it->second);
        if (wit.back().size != (size_t)t.n_rows * t.domain)
            throw std::runtime_error("batch_prove: witness size mismatch for " + t.comref);
    }

    // per-tensor M and P tables (size 2^vars_j), c2 factors, lazy S
    std::vector<Fr_t*> d_M(nT), d_P(nT);
    std::vector<uint> cur_size(nT);
    std::vector<Fr_t> c2(nT, F_ONE), S(nT, F_ZERO);
    Fr_t* d_eq = nullptr; uint eq_cap = 1u << m_max;
    cudaMalloc(&d_eq, eq_cap * sizeof(Fr_t));
    for (uint j = 0; j < nT; j++) {
        uint n = 1u << tensors[j].vars;
        cudaMalloc(&d_M[j], n * sizeof(Fr_t));
        cudaMalloc(&d_P[j], n * sizeof(Fr_t));
        cudaMemset(d_M[j], 0, n * sizeof(Fr_t));
        cudaMemcpy(d_P[j], wit[j].gpu_data, n * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
        cur_size[j] = n;
        for (uint32_t idx : tensors[j].claim_idx) {
            bo_build_eq(d_eq, cs_tr[idx].point);
            k_bo_axpy<<<(n + 255) / 256, 256>>>(d_M[j], d_eq, rho_pows[idx], n);
            cudaDeviceSynchronize();
        }
        if (tensors[j].vars < m_max)        // case-1 rounds use a constant S_j
            S[j] = dev_ip(d_M[j], d_P[j], n);
    }
    cudaFree(d_eq);
    T.lap("setup");

    // initial claim
    Fr_t cur = F_ZERO;
    for (uint i = 0; i < cs.size(); i++)
        cur = h_scalar(cur, h_scalar(rho_pows[i], cs_tr[i].eval, 2), 0);

    // G2 rounds: round t binds VARIABLE v = m_max-1-t
    BatchSumcheck bs; bs.n_claims = (uint32_t)cs.size(); bs.m_max = m_max;
    std::vector<Fr_t> r(m_max, F_ZERO);
    std::vector<Fr_t*> scratch(3);
    cudaMalloc(&scratch[0], (1u << (m_max - 1)) * sizeof(Fr_t));
    cudaMalloc(&scratch[1], (1u << (m_max - 1)) * sizeof(Fr_t));
    cudaMalloc(&scratch[2], (1u << (m_max - 1)) * sizeof(Fr_t));
    Fr_t* d_fold; cudaMalloc(&d_fold, (1u << (m_max - 1)) * sizeof(Fr_t));
    for (uint t = 0; t < m_max; t++) {
        uint v = m_max - 1 - t;
        Fr_t p0 = F_ZERO, p1 = F_ZERO, p2 = F_ZERO;
        for (uint j = 0; j < nT; j++) {
            if (tensors[j].vars <= v) {
                // case 1: both M_j and P-hat_j carry (1-x_v); contribution
                // c2_j * S_j * (1-T)^2  ->  +S at T=0, 0 at T=1, +S at T=2
                Fr_t term = h_scalar(c2[j], S[j], 2);
                p0 = h_scalar(p0, term, 0);
                p2 = h_scalar(p2, term, 0);
            } else {
                uint half = cur_size[j] / 2;
                k_bo_hp2<<<(half + 63) / 64, 64>>>(d_M[j], d_P[j],
                    scratch[0], scratch[1], scratch[2], half);
                cudaDeviceSynchronize();
                Fr_t q0 = bo_dev_sum(scratch[0], half);
                Fr_t q1 = bo_dev_sum(scratch[1], half);
                Fr_t q2 = bo_dev_sum(scratch[2], half);
                p0 = h_scalar(p0, h_scalar(c2[j], q0, 2), 0);
                p1 = h_scalar(p1, h_scalar(c2[j], q1, 2), 0);
                p2 = h_scalar(p2, h_scalar(c2[j], q2, 2), 0);
            }
        }
        if (evil == 0 && !fr_eq(cur, h_scalar(p0, p1, 0)))
            throw std::runtime_error("batch_prove: round " + std::to_string(t) +
                                     " inconsistency (witness/claims bug)");
        if (evil == 2) p1 = h_scalar(cur, p0, 1);   // adaptive lie
        absorb_fr(tr, "bp0", p0); absorb_fr(tr, "bp1", p1); absorb_fr(tr, "bp2", p2);
        Fr_t x = fs_challenge_fr(tr);
        r[v] = x;
        bs.ev.push_back(p0); bs.ev.push_back(p1); bs.ev.push_back(p2);
        cur = lagrange3(p0, p1, p2, x);
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
    write_batch_sumcheck(accdir + "/batch_sumcheck.bin", bs);
    T.lap("rounds");

    // G3: terminal evals v'_j = P_j(r[0..vars_j)), canonical tensor order
    std::vector<Fr_t> vfin(nT);
    for (uint j = 0; j < nT; j++)
        cudaMemcpy(&vfin[j], d_P[j], sizeof(Fr_t), cudaMemcpyDeviceToHost);
    if (evil == 2) {
        // poison the LAST tensor's v' so the G3 identity passes; the lie is
        // then forced into that tensor's group IPA (the BO-1b locus)
        Fr_t need = cur, clast = F_ZERO;
        for (uint j = 0; j < nT; j++) {
            Fr_t cj = h_scalar(bo_Mj_at_r(tensors[j], cs_tr, rho_pows, r),
                               bo_kappa(tensors[j].vars, r), 2);
            if (j + 1 == nT) { clast = cj; break; }
            need = h_scalar(need, h_scalar(cj, vfin[j], 2), 1);
        }
        vfin[nT - 1] = h_scalar(need, inv(clast), 2);
    }
    for (uint j = 0; j < nT; j++) absorb_fr(tr, "vfin", vfin[j]);
    write_batch_vfin(accdir + "/batch_vfin.bin", vfin);
    if (evil == 0) {
        Fr_t chk = F_ZERO;
        for (uint j = 0; j < nT; j++) {
            Fr_t cj = h_scalar(bo_Mj_at_r(tensors[j], cs_tr, rho_pows, r),
                               bo_kappa(tensors[j].vars, r), 2);
            chk = h_scalar(chk, h_scalar(cj, vfin[j], 2), 0);
        }
        if (!fr_eq(chk, cur))
            throw std::runtime_error("batch_prove: G3 terminal self-check failed");
    }
    T.lap("terminal");

    // G4 + G5: per generator-domain group, ascending; weights rho'^j by the
    // GLOBAL canonical tensor index j; coefficient rho'_j * kappa_j (once)
    Fr_t rhop = fs_challenge_fr(tr);
    std::vector<Fr_t> rhop_pows(nT);
    rhop_pows[0] = F_ONE;
    for (uint j = 1; j < nT; j++) rhop_pows[j] = h_scalar(rhop_pows[j - 1], rhop, 2);
    std::set<uint32_t> domains;
    for (auto& t : tensors) domains.insert(t.domain);
    for (uint32_t G : domains) {
        uint logG = ceilLog2(G);
        Fr_t* d_a; cudaMalloc(&d_a, G * sizeof(Fr_t));
        cudaMemset(d_a, 0, G * sizeof(Fr_t));
        Fr_t vstar = F_ZERO;
        for (uint j = 0; j < nT; j++) {
            if (tensors[j].domain != G) continue;
            Fr_t coef = h_scalar(rhop_pows[j], bo_kappa(tensors[j].vars, r), 2);
            std::vector<Fr_t> u_row(r.begin() + logG, r.begin() + tensors[j].vars);
            if (u_row.empty()) {
                k_bo_axpy<<<(G + 255) / 256, 256>>>(d_a, wit[j].gpu_data, coef, G);
                cudaDeviceSynchronize();
            } else {
                FrTensor rf = wit[j].partial_me(u_row, G);
                k_bo_axpy<<<(G + 255) / 256, 256>>>(d_a, rf.gpu_data, coef, G);
                cudaDeviceSynchronize();
            }
            vstar = h_scalar(vstar, h_scalar(coef, vfin[j], 2), 0);
        }
        auto git = genpaths.find(G);
        if (git == genpaths.end()) throw std::runtime_error("batch_prove: no generator file for domain " + std::to_string(G));
        Commitment gen(git->second);
        if (gen.size != G) throw std::runtime_error("batch_prove: generator size mismatch for domain " + std::to_string(G));
        if (getenv("ZKOB_BATCH_SELFCHECK")) {
            // <gen, a_g> must equal the homomorphic RLC of the folded coms
            G1Jacobian_t lhs = dev_msm(gen.gpu_data, d_a, G);
            G1Jacobian_t rhs; memset(&rhs, 0, sizeof(rhs));
            for (uint j = 0; j < nT; j++) {
                if (tensors[j].domain != G) continue;
                G1TensorJacobian com(tensors[j].comref);
                std::vector<Fr_t> u_row(r.begin() + logG, r.begin() + tensors[j].vars);
                G1Jacobian_t cj = fold_chain(com.gpu_data, com.size, u_row, 0);
                Fr_t coef = h_scalar(rhop_pows[j], bo_kappa(tensors[j].vars, r), 2);
                rhs = h_add(rhs, h_mul(cj, coef));
            }
            if (!g1_eq(lhs, rhs))
                throw std::runtime_error("batch_prove: C* self-check failed (domain " + std::to_string(G) + ")");
        }
        IpaProof pf = ipa_prove(d_a, me_weights(std::vector<Fr_t>(r.begin(), r.begin() + logG)),
                                gen.gpu_data, Q, G, tr);
        write_ipa(accdir + "/ipa_batch_" + std::to_string(G) + ".bin", pf);
        cudaFree(d_a);
    }
    for (uint j = 0; j < nT; j++) { cudaFree(d_M[j]); cudaFree(d_P[j]); }
    cudaFree(scratch[0]); cudaFree(scratch[1]); cudaFree(scratch[2]); cudaFree(d_fold);
    T.lap("ipa");
}

// =====================  batch verify  ======================================
// F5: the verifier-recomputed claim list (vaccdir) is the INPUT; the prover's
// claims.bin is byte-compared only, never parsed. Every REJECT names its
// locus: opening_batch.{empty,evalvar,claims_match,shape,xcheck,round<k>,
// vfin_count,terminal,group_missing,ipa<G>}.
static bool batch_verify(const std::string& paccdir, const std::string& vaccdir,
                         const std::string& run_seed,
                         const std::map<uint32_t, std::string>& genpaths,
                         const std::string& qpath, std::string* locus_out = nullptr) {
    auto reject = [&](const std::string& locus, const std::string& detail) {
        printf("REJECT[opening_batch.%s]: %s\n", locus.c_str(), detail.c_str());
        if (locus_out) *locus_out = locus;
        return false;
    };
    BoTimer T("batch_verify");
    // verifier-recomputed inputs (F5/F6)
    std::vector<BoClaim> cs = claims_load(vaccdir + "/claims.bin");
    std::vector<DrvState> dss;
    if (bo_file_exists(vaccdir + "/drvstates.bin")) dss = drvstates_load(vaccdir + "/drvstates.bin");
    if (cs.empty()) return reject("empty", "zero claims (a batch over no claims is not a valid obligation)");
    for (auto& c : cs)
        if (c.tag != BO_EVAL_PLAIN)
            return reject("evalvar", "Committed eval claims are not accepted until Stage D (claim " + c.id + ")");
    // claims_match: byte-compare the prover artifact against the recomputed list
    {
        std::vector<uint8_t> mine = claims_serialize(cs);
        if (!bo_file_exists(paccdir + "/claims.bin"))
            return reject("claims_match", "prover claims.bin missing");
        std::vector<uint8_t> theirs = bo_read_file(paccdir + "/claims.bin");
        if (mine.size() != theirs.size() || memcmp(mine.data(), theirs.data(), mine.size()))
            return reject("claims_match", "prover claim list != verifier-recomputed list");
    }
    T.lap("claims_match");
    std::vector<TensorInfo> tensors;
    std::string err;
    if (!derive_tensors(cs, tensors, err)) return reject("shape", err);
    const uint nT = (uint)tensors.size();
    uint m_max = 0;
    for (auto& t : tensors) m_max = std::max(m_max, t.vars);
    if (m_max < 1) return reject("shape", "m_max < 1");
    // F3: per distinct tensor, com_file_point_count == n_rows == 2^{vars-logG},
    // checked BEFORE any fold_chain (restores open_verify's size check; without
    // it trailing prover-chosen commitment rows are silently accepted)
    std::vector<G1TensorJacobian> coms;
    coms.reserve(nT);
    for (auto& t : tensors) {
        if (!bo_file_exists(t.comref))
            return reject("shape", "commitment file missing: " + t.comref);
        coms.emplace_back(t.comref);
        if (coms.back().size != t.n_rows)
            return reject("shape", "commitment row count " + std::to_string(coms.back().size) +
                          " != n_rows " + std::to_string(t.n_rows) + " (" + t.comref + ")");
    }
    // generators per domain (registration invariant: ONE generator file per
    // domain size; the RLC grouping is only a commitment under that invariant)
    std::map<uint32_t, Commitment> gens;
    for (auto& t : tensors) {
        if (gens.count(t.domain)) continue;
        auto git = genpaths.find(t.domain);
        if (git == genpaths.end())
            return reject("shape", "no generator file mapped for domain " + std::to_string(t.domain));
        gens.emplace(t.domain, Commitment(git->second));
        if (gens.at(t.domain).size != t.domain)
            return reject("shape", "generator file size != domain " + std::to_string(t.domain));
    }
    Commitment qg(qpath);
    G1Jacobian_t Q = qg(0);   // qg(1) = H slot when present (registered, unused until Stage D)
    T.lap("shape");

    // G0 + rho
    fs::Transcript tr(run_seed + ":opening_batch");
    batch_absorb_g0(tr, cs, tensors, dss);
    Fr_t rho = fs_challenge_fr(tr);
    std::vector<Fr_t> rho_pows(cs.size());
    rho_pows[0] = F_ONE;
    for (uint i = 1; i < cs.size(); i++) rho_pows[i] = h_scalar(rho_pows[i - 1], rho, 2);
    T.lap("absorb");

    // G2 round checks
    BatchSumcheck bs;
    try { bs = read_batch_sumcheck(paccdir + "/batch_sumcheck.bin"); }
    catch (const std::exception& e) { return reject("xcheck", e.what()); }
    if (bs.n_claims != cs.size())
        return reject("xcheck", "batch_sumcheck.bin n_claims field != derived claim count");
    if (bs.m_max != m_max)
        return reject("xcheck", "batch_sumcheck.bin m_max field != derived m_max");
    if (bs.ev.size() != 3u * m_max)
        return reject("xcheck", "batch_sumcheck.bin round-eval count != 3*m_max");
    Fr_t cur = F_ZERO;
    for (uint i = 0; i < cs.size(); i++)
        cur = h_scalar(cur, h_scalar(rho_pows[i], cs[i].eval, 2), 0);
    std::vector<Fr_t> r(m_max, F_ZERO);
    for (uint t = 0; t < m_max; t++) {
        Fr_t p0 = bs.ev[3 * t], p1 = bs.ev[3 * t + 1], p2 = bs.ev[3 * t + 2];
        if (!fr_eq(cur, h_scalar(p0, p1, 0)))
            return reject("round" + std::to_string(t), "p(0)+p(1) != running claim");
        absorb_fr(tr, "bp0", p0); absorb_fr(tr, "bp1", p1); absorb_fr(tr, "bp2", p2);
        Fr_t x = fs_challenge_fr(tr);
        r[m_max - 1 - t] = x;
        cur = lagrange3(p0, p1, p2, x);
    }
    T.lap("rounds");

    // G3 terminal: cur == sum_j M_j(r) * kappa_j * v'_j, with M_j(r) and
    // kappa_j computed by the VERIFIER from (rho, points, r)
    std::vector<Fr_t> vfin;
    try { vfin = read_batch_vfin(paccdir + "/batch_vfin.bin"); }
    catch (const std::exception& e) { return reject("vfin_count", e.what()); }
    if (vfin.size() != nT)
        return reject("vfin_count", "batch_vfin.bin entry count != distinct tensor count");
    for (uint j = 0; j < nT; j++) absorb_fr(tr, "vfin", vfin[j]);
    {
        Fr_t chk = F_ZERO;
        for (uint j = 0; j < nT; j++) {
            Fr_t cj = h_scalar(bo_Mj_at_r(tensors[j], cs, rho_pows, r),
                               bo_kappa(tensors[j].vars, r), 2);
            chk = h_scalar(chk, h_scalar(cj, vfin[j], 2), 0);
        }
        if (!fr_eq(chk, cur))
            return reject("terminal", "sum_j M_j(r)*kappa_j*v'_j != running claim");
    }
    T.lap("terminal");

    // G4 + G5
    Fr_t rhop = fs_challenge_fr(tr);
    std::vector<Fr_t> rhop_pows(nT);
    rhop_pows[0] = F_ONE;
    for (uint j = 1; j < nT; j++) rhop_pows[j] = h_scalar(rhop_pows[j - 1], rhop, 2);
    std::set<uint32_t> domains;
    for (auto& t : tensors) domains.insert(t.domain);
    for (uint32_t G : domains) {
        uint logG = ceilLog2(G);
        std::string ipath = paccdir + "/ipa_batch_" + std::to_string(G) + ".bin";
        if (!bo_file_exists(ipath))
            return reject("group_missing", "ipa_batch_" + std::to_string(G) + ".bin missing");
        G1Jacobian_t Cstar; memset(&Cstar, 0, sizeof(Cstar));   // identity (z = 0)
        Fr_t vstar = F_ZERO;
        for (uint j = 0; j < nT; j++) {
            if (tensors[j].domain != G) continue;
            std::vector<Fr_t> u_row(r.begin() + logG, r.begin() + tensors[j].vars);
            G1Jacobian_t cj = fold_chain(coms[j].gpu_data, coms[j].size, u_row, 0);
            Fr_t coef = h_scalar(rhop_pows[j], bo_kappa(tensors[j].vars, r), 2);
            Cstar = h_add(Cstar, h_mul(cj, coef));
            vstar = h_scalar(vstar, h_scalar(coef, vfin[j], 2), 0);
        }
        T.lap(("fold_" + std::to_string(G)).c_str());
        G1Jacobian_t P0 = h_add(Cstar, h_mul(Q, vstar));
        IpaProof pf;
        try { pf = read_ipa(ipath); }
        catch (const std::exception& e) { return reject("ipa" + std::to_string(G), e.what()); }
        if (!ipa_verify(gens.at(G).gpu_data, G, Q, P0,
                        std::vector<Fr_t>(r.begin(), r.begin() + logG), pf, tr))
            return reject("ipa" + std::to_string(G), "batched IPA for domain " + std::to_string(G) + " failed");
        T.lap(("ipa_" + std::to_string(G)).c_str());
    }
    printf("opening_batch ACCEPT (%u claims, %u tensors, %zu domains)\n",
           (uint)cs.size(), nT, domains.size());
    if (locus_out) *locus_out = "accept";
    return true;
}

#endif // ZKOB_CLAIMS_CUH
