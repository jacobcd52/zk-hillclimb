// COORDINATOR-BUILT (do not let submission agents edit).
// Toy ground-truth harness for the Stage-A batched-opening protocol
// (TRANSPORT_REBUILD_DESIGN §2.1/§2.2; required pins from TRANSPORT_REVIEW).
// Pins, per case, against INDEPENDENT brute-force recomputation:
//   P1  flat layout / point orientation: brute-force MLE over the flat table
//       with point = u_col ++ u_row (col bits first, LSB-first) equals
//       upstream multi_dim_me({u_row, u_col}, {rows, G}) bit-exact
//   P2  old-primitive cross-pin: open_prove/open_verify (the audited §10
//       primitive) ACCEPTs each emitted (point, eval) tuple as-is
//   P3  batch honest prove -> verify ACCEPT
//   P4  completeness identity: sum_i rho^i v_i == round-0 p(0)+p(1) == the
//       FULL brute-force sum over the embedded hypercube
//       sum_j sum_{x_low} M_j^bf(x_low‖0) * T_j[x_low]
//       (M_j^bf uses explicit zero padding over ALL m_max variables —
//        independent of both the prover's GPU tables and the verifier's
//        split low/high formula); this is also the "batched == sum of the
//        individual opens" statement, with each individual open pinned by P2
//   P5  fold orientation: batch_vfin v'_j == brute-force P_j(r[0..vars_j))
//       (k_bo_fold MSB-first <-> r indexed by variable)
//   P6  terminal coefficients: bo_Mj_at_r / bo_kappa == brute-force
//       full-variable eq products with explicit zero padding
//   P7  G5 split: <a_g, me_weights(r_col)> == v*_g with a_g recomputed by a
//       host brute-force row-fold of the original tensors
//   P8  F9 rho-sensitivity: a 1-byte change in the LAST claim's eval, or in
//       the LAST drvstate, changes rho (G0 absorb-before-squeeze pin)
//   P9  forgery smoke: false eval -> round 0; vfin tamper -> terminal;
//       ipa tamper -> that group's IPA (full battery in zkob_batchopen)
// Every new kernel shape is probed at startup (bo_probe_kernels; the -dlto
// miscompile rule). Prints PASS/FAIL per case.
#include "vrf_common.cuh"
#include "zkob_claims.cuh"
#include "zkob_lookup.cuh"   // open_prove/open_verify for the P2 cross-pin
#include <iostream>
#include <sys/stat.h>
using namespace std;

// host brute-force MLE in the me_weights orientation over the FLAT table:
// val = sum_flat T[flat] * prod_v (flat>>v & 1 ? u[v] : 1 - u[v])
static Fr_t bf_mle(const vector<Fr_t>& T, const vector<Fr_t>& u) {
    uint n = (uint)T.size();
    if (n != (1u << u.size())) throw runtime_error("bf_mle: size mismatch");
    Fr_t acc = F_ZERO;
    for (uint b = 0; b < n; b++) {
        Fr_t w = F_ONE;
        for (uint v = 0; v < u.size(); v++)
            w = h_scalar(w, ((b >> v) & 1) ? u[v] : h_scalar(F_ONE, u[v], 1), 2);
        acc = h_scalar(acc, h_scalar(w, T[b], 2), 0);
    }
    return acc;
}
// brute-force M_j at an arbitrary point x of ALL m_max variables, with the
// claim points zero-padded explicitly (no low/high split anywhere)
static Fr_t bf_Mj(const TensorInfo& tj, const vector<BoClaim>& cs,
                  const vector<Fr_t>& rho_pows, const vector<Fr_t>& x) {
    Fr_t acc = F_ZERO;
    for (uint32_t idx : tj.claim_idx) {
        vector<Fr_t> uhat = cs[idx].point;
        while (uhat.size() < x.size()) uhat.push_back(F_ZERO);   // high-var zero padding
        Fr_t term = rho_pows[idx];
        for (uint v = 0; v < x.size(); v++)
            term = h_scalar(term, bo_eq1(uhat[v], x[v]), 2);
        acc = h_scalar(acc, term, 0);
    }
    return acc;
}

struct ToyTensor {
    string name;
    uint rows, G, vars;
    FrTensor* data = nullptr;
    vector<Fr_t> host;
    string compath, witpath;
};

static bool gPASS = true;
static void check(bool ok, const string& what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) gPASS = false;
}

// replay the batch transcript (verifier-style) to extract rho and r for the
// brute-force pins; uses only the §10-pinned fs primitives
static void replay_transcript(const string& accdir, const string& run_seed,
                              const vector<BoClaim>& cs, const vector<TensorInfo>& tensors,
                              const vector<DrvState>& dss, uint m_max,
                              Fr_t& rho, vector<Fr_t>& r) {
    fs::Transcript tr(run_seed + ":opening_batch");
    batch_absorb_g0(tr, cs, tensors, dss);
    rho = fs_challenge_fr(tr);
    BatchSumcheck bs = read_batch_sumcheck(accdir + "/batch_sumcheck.bin");
    r.assign(m_max, F_ZERO);
    for (uint t = 0; t < m_max; t++) {
        absorb_fr(tr, "bp0", bs.ev[3 * t]);
        absorb_fr(tr, "bp1", bs.ev[3 * t + 1]);
        absorb_fr(tr, "bp2", bs.ev[3 * t + 2]);
        r[m_max - 1 - t] = fs_challenge_fr(tr);
    }
}

static void tamper_byte(const string& path, long offset, int delta) {
    FILE* f = open_or_die(path, "rb+");
    fseek(f, offset, offset >= 0 ? SEEK_SET : SEEK_END);
    int c = fgetc(f);
    fseek(f, -1, SEEK_CUR);
    fputc((c + delta) & 0xff, f);
    fclose(f);
}

// specs: (name, rows, G, n_claims_on_it)
struct Spec { const char* name; uint rows, G, nclaims; };

static bool toy_case(const string& tag, const vector<Spec>& specs) {
    printf("=== toy case %s ===\n", tag.c_str());
    gPASS = true;
    string dir = "/tmp/vrf_toy_batchopen_" + tag;
    string vdir = dir + "_v";
    for (auto& d : {dir, vdir}) { string c = "rm -rf " + d; system(c.c_str()); mkdir(d.c_str(), 0755); }
    string run_seed = "toy:" + tag;

    // generators: ONE per domain size (the registration invariant the
    // grouping relies on), plus a 2-slot q.bin exercising the H slot
    map<uint32_t, string> genpaths;
    map<uint32_t, Commitment*> gens;
    for (auto& s : specs)
        if (!gens.count(s.G)) {
            Commitment* g = new Commitment(Commitment::random(s.G));
            string p = dir + "/gen" + to_string(s.G) + ".bin";
            g->save(p);
            gens[s.G] = g; genpaths[s.G] = p;
        }
    string qpath = dir + "/q.bin";
    Commitment qg = Commitment::random(2);     // [Q, H] — H registered, unused (§4.4)
    qg.save(qpath);
    G1Jacobian_t Q = qg(0);

    // tensors + commitments
    vector<ToyTensor> tts;
    fs::Transcript ptr("toy:points:" + tag);   // deterministic toy point source
    for (auto& s : specs) {
        ToyTensor t;
        t.name = s.name; t.rows = s.rows; t.G = s.G;
        t.vars = ceilLog2(s.G) + ceilLog2(s.rows);
        t.data = new FrTensor(FrTensor::random_int(s.rows * s.G, 8));
        t.host.resize(s.rows * s.G);
        cudaMemcpy(t.host.data(), t.data->gpu_data, t.host.size() * sizeof(Fr_t),
                   cudaMemcpyDeviceToHost);
        G1TensorJacobian com = gens[s.G]->commit(*t.data);
        t.compath = dir + "/com_" + t.name + ".bin";
        com.save(t.compath);
        t.witpath = dir + "/wit_" + t.name + ".fr";
        t.data->save(t.witpath);
        tts.push_back(t);
    }

    // claims: eval via upstream multi_dim_me; P1 pins it against bf_mle
    vector<BoClaim> cs;
    for (auto& t : tts) {
        uint logG = ceilLog2(t.G), logR = ceilLog2(t.rows);
        uint nclaims = 0;
        for (auto& s : specs) if (t.name == s.name) nclaims = s.nclaims;
        for (uint k = 0; k < nclaims; k++) {
            BoClaim c;
            c.id = "toy." + t.name + ":c" + to_string(k);
            c.comref = t.compath;
            c.domain = t.G; c.n_rows = t.rows;
            vector<Fr_t> u_col(logG), u_row(logR);
            for (auto& x : u_col) x = fs_challenge_fr(ptr);
            for (auto& x : u_row) x = fs_challenge_fr(ptr);
            c.point = u_col;
            c.point.insert(c.point.end(), u_row.begin(), u_row.end());
            c.eval = t.data->multi_dim_me({u_row, u_col}, {t.rows, t.G});
            check(fr_eq(c.eval, bf_mle(t.host, c.point)),
                  "P1 layout: multi_dim_me == flat brute force (" + c.id + ")");
            // P2: the audited per-claim primitive accepts the same tuple
            {
                string ipath = dir + "/toyipa_" + t.name + to_string(k) + ".bin";
                fs::Transcript tp("toyopen:" + c.id), tv("toyopen:" + c.id);
                open_prove(*t.data, t.G, *gens[t.G], Q, c.point, ipath, tp);
                G1TensorJacobian com(t.compath);
                check(open_verify(com, *gens[t.G], t.G, Q, c.point, c.eval, ipath, tv),
                      "P2 old-primitive open_verify ACCEPTs (" + c.id + ")");
            }
            cs.push_back(c);
        }
    }

    // fake driver states (two "runs"), identical on both sides (honest)
    vector<DrvState> dss;
    for (int d = 0; d < 2; d++) {
        fs::Transcript dtr("toy:drv" + to_string(d) + ":" + tag);
        absorb_u32(dtr, "x", 41 + d);
        DrvState s; s.id = "toy.drv" + to_string(d);
        memcpy(s.state, dtr.state, 32);
        dss.push_back(s);
    }

    // accumulators: prover side (with witrefs) and verifier side
    claims_save(dir + "/claims.bin", cs);
    drvstates_save(dir + "/drvstates.bin", dss);
    for (auto& t : tts) witref_emit(dir, t.compath, t.witpath);
    claims_save(vdir + "/claims.bin", cs);
    drvstates_save(vdir + "/drvstates.bin", dss);

    // P3: honest batch
    setenv("ZKOB_BATCH_SELFCHECK", "1", 1);
    batch_prove(dir, run_seed, genpaths, qpath);
    string locus;
    check(batch_verify(dir, vdir, run_seed, genpaths, qpath, &locus),
          "P3 honest batch verify ACCEPT");

    // structural data for the pins
    vector<TensorInfo> tensors; string err;
    if (!derive_tensors(cs, tensors, err)) throw runtime_error("toy: " + err);
    uint m_max = 0;
    for (auto& t : tensors) m_max = max(m_max, t.vars);
    Fr_t rho; vector<Fr_t> r;
    replay_transcript(dir, run_seed, cs, tensors, dss, m_max, rho, r);
    vector<Fr_t> rho_pows(cs.size());
    rho_pows[0] = F_ONE;
    for (uint i = 1; i < cs.size(); i++) rho_pows[i] = h_scalar(rho_pows[i - 1], rho, 2);

    // P4: completeness identity, three independent ways
    {
        Fr_t lhs = F_ZERO;   // sum_i rho^i * v_i (the sum of the individual opens)
        for (uint i = 0; i < cs.size(); i++)
            lhs = h_scalar(lhs, h_scalar(rho_pows[i], cs[i].eval, 2), 0);
        BatchSumcheck bs = read_batch_sumcheck(dir + "/batch_sumcheck.bin");
        Fr_t r0 = h_scalar(bs.ev[0], bs.ev[1], 0);
        check(fr_eq(lhs, r0), "P4 sum_i rho^i v_i == round-0 p(0)+p(1)");
        // full brute force over the embedded hypercube: on hypercube points
        // P-hat_j(x) = T_j[x_low] iff all high bits 0, else 0
        Fr_t bf = F_ZERO;
        for (uint j = 0; j < tensors.size(); j++) {
            const ToyTensor* tt = nullptr;
            for (auto& t : tts) if (t.compath == tensors[j].comref) tt = &t;
            for (uint b = 0; b < (1u << tensors[j].vars); b++) {
                vector<Fr_t> x(m_max, F_ZERO);
                for (uint v = 0; v < m_max; v++)
                    if (v < tensors[j].vars && ((b >> v) & 1)) x[v] = F_ONE;
                bf = h_scalar(bf, h_scalar(bf_Mj(tensors[j], cs, rho_pows, x),
                                           tt->host[b], 2), 0);
            }
        }
        check(fr_eq(lhs, bf), "P4 == full hypercube brute force sum_j sum_x M_j(x)*T_j[x]");
    }

    // P5/P6: terminal pins
    vector<Fr_t> vfin = read_batch_vfin(dir + "/batch_vfin.bin");
    for (uint j = 0; j < tensors.size(); j++) {
        const ToyTensor* tt = nullptr;
        for (auto& t : tts) if (t.compath == tensors[j].comref) tt = &t;
        vector<Fr_t> r_low(r.begin(), r.begin() + tensors[j].vars);
        check(fr_eq(vfin[j], bf_mle(tt->host, r_low)),
              "P5 v'_j == brute-force P_j(r_low) (" + tts[j].name + ")");
        Fr_t mj_split = bo_Mj_at_r(tensors[j], cs, rho_pows, r);
        Fr_t mj_bf = bf_Mj(tensors[j], cs, rho_pows, r);
        check(fr_eq(mj_split, mj_bf),
              "P6 M_j(r) split formula == zero-padded brute force (" + tts[j].name + ")");
        // kappa cross-check: full bf at r with low vars from r equals
        // M_j^low * kappa by construction of the split — covered by P6;
        // additionally pin kappa itself:
        Fr_t kap = bo_kappa(tensors[j].vars, r);
        Fr_t kap_bf = F_ONE;
        for (uint v = tensors[j].vars; v < m_max; v++)
            kap_bf = h_scalar(kap_bf, bo_eq1(F_ZERO, r[v]), 2);   // eq(0, r_v) = 1 - r_v
        check(fr_eq(kap, kap_bf), "P6 kappa_j == prod eq(0, r_v) (" + tts[j].name + ")");
    }

    // P7: G5 split — host brute-force row-fold a_g, then <a_g, me_weights(r_col)>
    {
        fs::Transcript tr2(run_seed + ":opening_batch");
        batch_absorb_g0(tr2, cs, tensors, dss);
        fs_challenge_fr(tr2);   // rho
        BatchSumcheck bs = read_batch_sumcheck(dir + "/batch_sumcheck.bin");
        for (uint t = 0; t < m_max; t++) {
            absorb_fr(tr2, "bp0", bs.ev[3 * t]); absorb_fr(tr2, "bp1", bs.ev[3 * t + 1]);
            absorb_fr(tr2, "bp2", bs.ev[3 * t + 2]);
            fs_challenge_fr(tr2);
        }
        for (uint j = 0; j < tensors.size(); j++) absorb_fr(tr2, "vfin", vfin[j]);
        Fr_t rhop = fs_challenge_fr(tr2);
        vector<Fr_t> rhop_pows(tensors.size());
        rhop_pows[0] = F_ONE;
        for (uint j = 1; j < tensors.size(); j++) rhop_pows[j] = h_scalar(rhop_pows[j - 1], rhop, 2);
        set<uint32_t> domains;
        for (auto& t : tensors) domains.insert(t.domain);
        for (uint32_t G : domains) {
            uint logG = ceilLog2(G);
            vector<Fr_t> a(G, F_ZERO);
            Fr_t vstar = F_ZERO;
            for (uint j = 0; j < tensors.size(); j++) {
                if (tensors[j].domain != G) continue;
                const ToyTensor* tt = nullptr;
                for (auto& t : tts) if (t.compath == tensors[j].comref) tt = &t;
                // brute-force row fold: a[col] += coef * sum_row T[row,col]*w_row(row)
                uint rows = tensors[j].n_rows, logR = tensors[j].vars - logG;
                vector<Fr_t> u_row(r.begin() + logG, r.begin() + tensors[j].vars);
                Fr_t coef = h_scalar(rhop_pows[j], bo_kappa(tensors[j].vars, r), 2);
                vstar = h_scalar(vstar, h_scalar(coef, vfin[j], 2), 0);
                for (uint col = 0; col < G; col++) {
                    Fr_t acc = F_ZERO;
                    for (uint row = 0; row < rows; row++) {
                        Fr_t w = F_ONE;
                        for (uint v = 0; v < logR; v++)
                            w = h_scalar(w, ((row >> v) & 1) ? u_row[v]
                                                             : h_scalar(F_ONE, u_row[v], 1), 2);
                        acc = h_scalar(acc, h_scalar(w, tt->host[row * G + col], 2), 0);
                    }
                    a[col] = h_scalar(a[col], h_scalar(coef, acc, 2), 0);
                }
            }
            auto b = me_weights(vector<Fr_t>(r.begin(), r.begin() + logG));
            Fr_t ip = F_ZERO;
            for (uint k = 0; k < G; k++) ip = h_scalar(ip, h_scalar(a[k], b[k], 2), 0);
            check(fr_eq(ip, vstar),
                  "P7 <a_g, me_weights(r_col)> == v*_g (domain " + to_string(G) + ")");
        }
    }

    // P8: F9 rho-sensitivity
    {
        Fr_t rho0 = rho;
        vector<BoClaim> cs2 = cs;
        cs2.back().eval.val[0] ^= 1;        // last claim, low byte of eval
        fs::Transcript ta(run_seed + ":opening_batch");
        vector<TensorInfo> t2; string e2;
        derive_tensors(cs2, t2, e2);
        batch_absorb_g0(ta, cs2, t2, dss);
        check(!fr_eq(rho0, fs_challenge_fr(ta)), "P8 rho differs on last-claim eval bit flip");
        vector<DrvState> ds2 = dss;
        ds2.back().state[31] ^= 1;          // last drvstate, last byte
        fs::Transcript tb(run_seed + ":opening_batch");
        batch_absorb_g0(tb, cs, tensors, ds2);
        check(!fr_eq(rho0, fs_challenge_fr(tb)), "P8 rho differs on last drvstate bit flip");
    }

    // P9: forgery smoke (full battery lives in zkob_batchopen selftest)
    {
        // false eval in BOTH lists, honest-procedure batch -> round 0
        vector<BoClaim> cs2 = cs;
        cs2[0].eval = h_scalar(cs2[0].eval, F_ONE, 0);
        claims_save(dir + "/claims.bin", cs2);
        claims_save(vdir + "/claims.bin", cs2);
        batch_prove(dir, run_seed, genpaths, qpath, /*evil=*/1);
        string l;
        bool rej = !batch_verify(dir, vdir, run_seed, genpaths, qpath, &l);
        check(rej && l == "round0", "P9 false eval (honest-procedure batch) dies at round 0, got: " + l);
        claims_save(dir + "/claims.bin", cs);
        claims_save(vdir + "/claims.bin", cs);
        batch_prove(dir, run_seed, genpaths, qpath);    // restore artifacts
        check(batch_verify(dir, vdir, run_seed, genpaths, qpath, &l), "P9 restored ACCEPT");
        tamper_byte(dir + "/batch_vfin.bin", -32, +1);
        rej = !batch_verify(dir, vdir, run_seed, genpaths, qpath, &l);
        check(rej && l == "terminal", "P9 vfin tamper dies at G3 terminal, got: " + l);
        tamper_byte(dir + "/batch_vfin.bin", -32, -1);
        uint32_t Gmax = 0; for (auto& t : tensors) Gmax = max(Gmax, t.domain);
        string ipath = dir + "/ipa_batch_" + to_string(Gmax) + ".bin";
        tamper_byte(ipath, -32, +1);        // a_final
        rej = !batch_verify(dir, vdir, run_seed, genpaths, qpath, &l);
        check(rej && l == "ipa" + to_string(Gmax),
              "P9 ipa a_final tamper dies at that group's IPA, got: " + l);
        tamper_byte(ipath, -32, -1);
        check(batch_verify(dir, vdir, run_seed, genpaths, qpath, &l), "P9 restored ACCEPT (2)");
    }

    for (auto& t : tts) delete t.data;
    for (auto& g : gens) delete g.second;
    printf("=== toy case %s: %s ===\n", tag.c_str(), gPASS ? "PASS" : "FAIL");
    return gPASS;
}

int main() {
    vrf_selfcheck();
    bo_probe_kernels();
    printf("kernel -dlto probes: PASS\n");
    bool ok = true;
    // mixed domains, multi-claim tensor (rope/headslice shape: two claims on
    // the same tensor at different points), single-row tensor, max-vars tensor
    ok &= toy_case("mixed", {{"A", 4, 4, 2}, {"B", 2, 8, 1}, {"C", 1, 4, 1}, {"D", 8, 8, 1}});
    // single domain, two tensors (one group; exercises same-G multi-tensor RLC)
    ok &= toy_case("onedom", {{"E", 4, 8, 1}, {"F", 2, 8, 2}});
    // minimal: one single-row tensor, one claim (u_row empty end-to-end)
    ok &= toy_case("minimal", {{"G", 1, 4, 1}});
    printf(ok ? "TOY-BATCHOPEN: ALL PASS\n" : "TOY-BATCHOPEN: FAIL\n");
    return ok ? 0 : 1;
}
