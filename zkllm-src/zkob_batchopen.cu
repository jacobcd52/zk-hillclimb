// COORDINATOR-BUILT (do not let submission agents edit).
// Real driver for the opening_batch obligation (TRANSPORT_REBUILD_DESIGN
// §2.1/§2.2, Stage A): ONE batch-evaluation sumcheck over every claim in the
// run accumulator + ONE homomorphically-combined IPA per generator domain.
// Protocol core lives in zkob_claims.cuh; algebra pinned at toy scale against
// brute force in vrf_toy_batchopen.cu (TOY-BATCHOPEN: ALL PASS).
//
// Audit pins implemented (TRANSPORT_REVIEW): F3 (per-tensor commitment
// row-count check before fold_chain — the covert-channel pin), F4 (length-
// prefixed claim absorb + explicit EvalVar tag), F5 (verify consumes the
// VERIFIER-recomputed claim list; the prover's claims.bin is byte-compared
// only; claims_match is a component of the opening_batch verdict), F6
// (drvstates verifier-internal; redundant header fields cross-checked;
// n_claims == 0 rejected explicitly), F8 (BO-1 split into BO-1a round-0
// locus / BO-1b adaptive G3-or-group-IPA locus), F9 (rho-sensitivity, in
// vrf_toy_batchopen + re-pinned here), F10 (structural-shape battery:
// trailing rows / truncation / n_rows lie).
//
// Usage:
//   zkob_batchopen prove  <accdir> <run_seed> <q.bin> <G>=<gen.bin> ...
//   zkob_batchopen verify <prover-accdir> <verifier-accdir> <run_seed>
//                         <q.bin> <G>=<gen.bin> ...
//   zkob_batchopen genq   <out.bin>     (2-slot q: [Q, H] — the §4.4 H-slot
//                                        registration touch; H unused until D)
//   zkob_batchopen selftest
// The batch transcript is seeded "<run_seed>:opening_batch" internally.
// Verify exit 0 = opening_batch ACCEPT; per-run driver verdicts remain
// ACCEPT-conditional until this passes (orchestrator wiring is Stage C).
#include "vrf_common.cuh"
#include "zkob_claims.cuh"
#include "zkob_wpriv.cuh"   // Stage D: weight sub-batch (wprove/wverify/wselftest)
#include <iostream>
#include <sys/stat.h>
using namespace std;

static map<uint32_t, string> parse_genspecs(int argc, char* argv[], int from) {
    map<uint32_t, string> m;
    for (int i = from; i < argc; i++) {
        string s = argv[i];
        size_t eq = s.find('=');
        if (eq == string::npos) throw runtime_error("bad gen spec (want <G>=<file>): " + s);
        m[(uint32_t)stoul(s.substr(0, eq))] = s.substr(eq + 1);
    }
    return m;
}

// ===========================  selftest  ====================================
static void tamper_byte(const string& path, long offset, int delta) {
    FILE* f = open_or_die(path, "rb+");
    fseek(f, offset, offset >= 0 ? SEEK_SET : SEEK_END);
    int c = fgetc(f);
    fseek(f, -1, SEEK_CUR);
    fputc((c + delta) & 0xff, f);
    fclose(f);
}
static void copy_file(const string& a, const string& b) {
    auto bytes = bo_read_file(a);
    FILE* f = open_or_die(b, "wb");
    fwrite(bytes.data(), 1, bytes.size(), f); fclose(f);
}

struct MiniTensor {
    string name, compath, witpath;
    uint rows, G;
    FrTensor* data;
    vector<Fr_t> host;
};
struct MiniRun {
    string base, acc, vacc, qpath, run_seed;
    map<uint32_t, string> genpaths;
    map<uint32_t, Commitment*> gens;
    vector<MiniTensor> tts;
    vector<BoClaim> claims;
    vector<DrvState> dss;
};

// build a small honest run: 3 simulated drivers over 2 domains, one tensor
// carrying TWO claims (the rope/headslice multi-claim shape), and batch-prove
// it. Claims are emitted through the same claim_emit/drvstate_emit API the
// real drivers use.
static MiniRun build_minirun(const string& base, const string& seedtag) {
    MiniRun M;
    M.base = base; M.acc = base + "/acc"; M.vacc = base + "/vacc";
    M.run_seed = "selftest:" + seedtag;
    { string c = "rm -rf " + base; system(c.c_str()); }
    mkdir(base.c_str(), 0755); mkdir(M.acc.c_str(), 0755); mkdir(M.vacc.c_str(), 0755);

    struct Spec { const char* name; uint rows, G, nclaims; };
    vector<Spec> specs = {{"A", 4, 4, 2}, {"B", 2, 8, 1}, {"C", 8, 8, 1}};
    for (auto& s : specs)
        if (!M.gens.count(s.G)) {
            Commitment* g = new Commitment(Commitment::random(s.G));
            string p = base + "/gen" + to_string(s.G) + ".bin";
            g->save(p);
            M.gens[s.G] = g; M.genpaths[s.G] = p;
        }
    // q.bin produced by the genq path (the §4.4 H-slot registration touch):
    // 2 points [Q, H]; only Q is consumed until Stage D.
    M.qpath = base + "/q.bin";
    Commitment::random(2).save(M.qpath);

    fs::Transcript ptr("selftest:points:" + seedtag);
    for (auto& s : specs) {
        MiniTensor t;
        t.name = s.name; t.rows = s.rows; t.G = s.G;
        t.data = new FrTensor(FrTensor::random_int(s.rows * s.G, 8));
        t.host.resize(s.rows * s.G);
        cudaMemcpy(t.host.data(), t.data->gpu_data, t.host.size() * sizeof(Fr_t),
                   cudaMemcpyDeviceToHost);
        G1TensorJacobian com = M.gens[s.G]->commit(*t.data);
        t.compath = base + "/com_" + t.name + ".bin";
        com.save(t.compath);
        t.witpath = base + "/wit_" + t.name + ".fr";
        t.data->save(t.witpath);

        // one simulated driver per tensor: a tiny transcript that absorbs the
        // com and squeezes the claim points, then emits its claims + drvstate
        fs::Transcript dtr(M.run_seed + ":drv." + t.name);
        absorb_g1_tensor(dtr, "com", com);
        for (uint k = 0; k < s.nclaims; k++) {
            uint logG = ceilLog2(t.G), logR = ceilLog2(t.rows);
            BoClaim c;
            c.id = "drv." + t.name + ":c" + to_string(k);
            c.comref = t.compath;
            c.domain = t.G; c.n_rows = t.rows;
            c.point = fs_challenge_vec(dtr, logG + logR);
            vector<Fr_t> u_col(c.point.begin(), c.point.begin() + logG);
            vector<Fr_t> u_row(c.point.begin() + logG, c.point.end());
            c.eval = t.data->multi_dim_me({u_row, u_col}, {t.rows, t.G});
            absorb_fr(dtr, "claim", c.eval);
            M.claims.push_back(c);
            claim_emit(M.acc, c);              // prover-side accumulator
            claim_emit(M.vacc, c);             // verifier-recomputed list
        }
        witref_emit(M.acc, t.compath, t.witpath);
        drvstate_emit(M.acc, "drv." + t.name, dtr);
        drvstate_emit(M.vacc, "drv." + t.name, dtr);   // verifier's own run, F6
        DrvState d; d.id = "drv." + t.name; memcpy(d.state, dtr.state, 32);
        M.dss.push_back(d);
        M.tts.push_back(t);
    }
    batch_prove(M.acc, M.run_seed, M.genpaths, M.qpath);
    return M;
}
// restore the honest accumulator contents + batch artifacts
static void restore(MiniRun& M) {
    claims_save(M.acc + "/claims.bin", M.claims);
    claims_save(M.vacc + "/claims.bin", M.claims);
    drvstates_save(M.acc + "/drvstates.bin", M.dss);
    drvstates_save(M.vacc + "/drvstates.bin", M.dss);
    batch_prove(M.acc, M.run_seed, M.genpaths, M.qpath);
}

static int g_total = 0, g_fail = 0;
static void expect(MiniRun& M, const string& what, const string& want_locus) {
    string locus;
    bool acc = batch_verify(M.acc, M.vacc, M.run_seed, M.genpaths, M.qpath, &locus);
    bool ok = !acc && locus == want_locus;
    if (want_locus == "accept") ok = acc;
    g_total++;
    if (!ok) g_fail++;
    printf("  [%s] %s -> expected %s, got %s\n", ok ? "PASS" : "FAIL",
           what.c_str(), want_locus.c_str(), locus.c_str());
}

static bool selftest() {
    vrf_selfcheck();
    bo_probe_kernels();
    printf("kernel -dlto probes: PASS\n");
    setenv("ZKOB_BATCH_SELFCHECK", "1", 1);
    // Stage B: every selftest verify cross-checks the batched G5 fold against
    // per-tensor fold_chain element-exact (THROWS on divergence)
    setenv("ZKOB_FOLD_CROSSCHECK", "1", 1);
    MiniRun M = build_minirun("/tmp/zkob_batchopen_selftest", "bo");

    printf("=== honest ===\n");
    expect(M, "honest batch", "accept");

    printf("=== BO-1a/BO-1b (F8 split) + RLC cancellation ===\n");
    {
        // BO-1a: a driver emitted a false eval, driver-locally compensated, so
        // BOTH lists carry it; the batch prover runs the honest PROCEDURE over
        // the committed tensors -> round-0 evals sum to the TRUE total while
        // cur_0 is the FALSE RLC -> dies at batch sumcheck round 0.
        vector<BoClaim> bad = M.claims;
        bad[1].eval = h_scalar(bad[1].eval, F_ONE, 0);
        claims_save(M.acc + "/claims.bin", bad);
        claims_save(M.vacc + "/claims.bin", bad);
        batch_prove(M.acc, M.run_seed, M.genpaths, M.qpath, /*evil=*/1);
        expect(M, "BO-1a false eval, honest-procedure batch prover", "round0");
        // BO-1b: fully adaptive prover — lies coherently through every round
        // (p1 := cur - p0) and poisons the last tensor's v' so G3 passes; the
        // false running value is then forced into that tensor's group IPA.
        batch_prove(M.acc, M.run_seed, M.genpaths, M.qpath, /*evil=*/2);
        // last tensor in canonical order is C (domain 8)
        expect(M, "BO-1b adaptive prover (forged v' to pass G3)", "ipa8");
        // RLC-cancellation attempt: pick delta on claim 1 and a compensating
        // delta' = -delta*rho^{i1-i2} on claim 3 USING the rho derived from
        // the already-doctored list — but re-serializing the doctored list
        // changes rho (P8/F9), so the cancellation never holds.
        {
            vector<BoClaim> can = M.claims;
            Fr_t delta = F_ONE;
            can[1].eval = h_scalar(can[1].eval, delta, 0);
            // derive rho from THIS list (the attacker's best information)
            vector<TensorInfo> ts; string err;
            derive_tensors(can, ts, err);
            fs::Transcript ta(M.run_seed + ":opening_batch");
            batch_absorb_g0(ta, can, ts, M.dss);
            Fr_t rho = fs_challenge_fr(ta);
            // delta' = -delta * rho^{1} / rho^{3} would cancel IF rho stayed
            // fixed; apply it, which re-randomizes rho through G0
            Fr_t rho2 = h_scalar(rho, rho, 2);
            can[3].eval = h_scalar(can[3].eval, h_scalar(delta, inv(rho2), 2), 1);
            claims_save(M.acc + "/claims.bin", can);
            claims_save(M.vacc + "/claims.bin", can);
            batch_prove(M.acc, M.run_seed, M.genpaths, M.qpath, /*evil=*/1);
            expect(M, "RLC-cancellation attempt (rho re-randomizes)", "round0");
        }
        restore(M);
    }

    printf("=== BO-2/BO-3/BO-10 + reorder (claims_match) ===\n");
    {
        vector<BoClaim> omit(M.claims.begin(), M.claims.end() - 1);
        claims_save(M.acc + "/claims.bin", omit);
        expect(M, "BO-2 claim omitted from prover claims.bin", "claims_match");
        vector<BoClaim> swp = M.claims;
        std::swap(swp[2].comref, swp[3].comref);   // B <-> C comrefs
        claims_save(M.acc + "/claims.bin", swp);
        expect(M, "BO-3 comrefs of two claims swapped (list only)", "claims_match");
        vector<BoClaim> dup = M.claims;
        dup.push_back(dup.back());
        claims_save(M.acc + "/claims.bin", dup);
        expect(M, "BO-10 duplicate claim appended", "claims_match");
        vector<BoClaim> ro = M.claims;
        std::swap(ro[0], ro[1]);
        claims_save(M.acc + "/claims.bin", ro);
        expect(M, "reordered claims in prover claims.bin", "claims_match");
        restore(M);
    }

    printf("=== BO-4 (rho from doctored list) ===\n");
    {
        // prover derives rho/RLC from a doctored list but ships the honest
        // claims.bin -> claims_match passes, batch transcript diverges
        batch_prove(M.acc, M.run_seed, M.genpaths, M.qpath, /*evil=*/3);
        expect(M, "BO-4 transcript built over doctored list", "round0");
        restore(M);
    }

    printf("=== BO-5/BO-6/BO-8 (artifact byte tampers) ===\n");
    {
        // batch_sumcheck.bin: 4 magic + 4 n_claims + 4 m_max + 4 vec-count
        tamper_byte(M.acc + "/batch_sumcheck.bin", 16 + 32, +1);   // round-0 p(1)
        expect(M, "BO-5 round-0 p(1) tamper", "round0");
        tamper_byte(M.acc + "/batch_sumcheck.bin", 16 + 32, -1);
        tamper_byte(M.acc + "/batch_sumcheck.bin", 16 + 3 * 32, +1); // round-1 p(0)
        expect(M, "BO-5 round-1 p(0) tamper", "round1");
        tamper_byte(M.acc + "/batch_sumcheck.bin", 16 + 3 * 32, -1);
        tamper_byte(M.acc + "/batch_sumcheck.bin", 4, +1);         // n_claims field
        expect(M, "BO-8 batch_sumcheck n_claims field tamper", "xcheck");
        tamper_byte(M.acc + "/batch_sumcheck.bin", 4, -1);
        tamper_byte(M.acc + "/batch_vfin.bin", -32, +1);           // last v'_j
        expect(M, "BO-6 batch_vfin tamper", "terminal");
        tamper_byte(M.acc + "/batch_vfin.bin", -32, -1);
        tamper_byte(M.acc + "/ipa_batch_4.bin", -32, +1);          // a_final
        expect(M, "BO-8 ipa_batch_4 a_final tamper", "ipa4");
        tamper_byte(M.acc + "/ipa_batch_4.bin", -32, -1);
        tamper_byte(M.acc + "/ipa_batch_8.bin", 8 + 16, +1);       // L[0] body
        expect(M, "BO-8 ipa_batch_8 round-0 L tamper", "ipa8");
        tamper_byte(M.acc + "/ipa_batch_8.bin", 8 + 16, -1);
        // per-field claims.bin tampers (all die at the byte-compare; the
        // prover file is never parsed — F5)
        long off_id = 4 + 4 + 4 + 4;                 // first claim's id bytes
        tamper_byte(M.acc + "/claims.bin", off_id, +1);
        expect(M, "BO-8 claims.bin id byte tamper", "claims_match");
        tamper_byte(M.acc + "/claims.bin", off_id, -1);
        tamper_byte(M.acc + "/claims.bin", -1, +1);  // last byte = last eval byte
        expect(M, "BO-8 claims.bin eval byte tamper", "claims_match");
        tamper_byte(M.acc + "/claims.bin", -1, -1);
        // VERIFIER-side drvstate divergence (a per-driver transcript that does
        // not match what the prover batched over -> rho diverges)
        vector<DrvState> ds2 = M.dss;
        ds2.back().state[0] ^= 1;
        drvstates_save(M.vacc + "/drvstates.bin", ds2);
        expect(M, "BO-8 verifier drvstate divergence", "round0");
        restore(M);
    }

    printf("=== BO-7 + F10 (structural shape, the F3 pin) ===\n");
    {
        // vars lie, forged CONSISTENTLY in both lists: point one var longer
        // than n_rows allows
        vector<BoClaim> lie = M.claims;
        lie[2].point.push_back(F_ONE);
        claims_save(M.acc + "/claims.bin", lie);
        claims_save(M.vacc + "/claims.bin", lie);
        expect(M, "BO-7 vars lie (junk-padded point, both lists)", "shape");
        // n_rows lie consistent with the longer point: com FILE count differs
        lie[2].n_rows *= 2;
        // claim 2 is tensor B's only claim, so the tensor shape follows the lie
        claims_save(M.acc + "/claims.bin", lie);
        claims_save(M.vacc + "/claims.bin", lie);
        expect(M, "BO-7 consistent n_rows lie (com file count differs)", "shape");
        restore(M);
        // F10: commitment file with extra trailing rows (prover-chosen bytes
        // after the real rows) must die at the named structural check BEFORE
        // any fold_chain — this is the F3 covert-channel pin
        string com = M.tts[0].compath, bak = com + ".bak";
        copy_file(com, bak);
        { auto extra = bo_read_file(com);
          FILE* f = open_or_die(com, "ab");
          fwrite(extra.data(), 1, sizeof(G1Jacobian_t), f); fclose(f); }
        expect(M, "F10 com file with extra trailing row", "shape");
        copy_file(bak, com);
        // F10: truncated com file
        { auto bytes = bo_read_file(com);
          FILE* f = open_or_die(com, "wb");
          fwrite(bytes.data(), 1, bytes.size() - sizeof(G1Jacobian_t), f); fclose(f); }
        expect(M, "F10 truncated com file", "shape");
        copy_file(bak, com);
        remove(bak.c_str());
        expect(M, "restored after F10", "accept");
    }

    printf("=== BO-9 (cross-run replay) + BO-11 (substituted com) ===\n");
    {
        MiniRun M2 = build_minirun("/tmp/zkob_batchopen_selftest2", "bo2");
        // (a) full replay incl. claims.bin -> the recomputed list differs
        copy_file(M2.acc + "/claims.bin", M.acc + "/claims.bin");
        copy_file(M2.acc + "/batch_sumcheck.bin", M.acc + "/batch_sumcheck.bin");
        copy_file(M2.acc + "/batch_vfin.bin", M.acc + "/batch_vfin.bin");
        copy_file(M2.acc + "/ipa_batch_4.bin", M.acc + "/ipa_batch_4.bin");
        copy_file(M2.acc + "/ipa_batch_8.bin", M.acc + "/ipa_batch_8.bin");
        expect(M, "BO-9a full cross-run replay", "claims_match");
        restore(M);
        // (b) replay only the batch artifacts (self-consistent foreign proof)
        copy_file(M2.acc + "/batch_sumcheck.bin", M.acc + "/batch_sumcheck.bin");
        copy_file(M2.acc + "/batch_vfin.bin", M.acc + "/batch_vfin.bin");
        copy_file(M2.acc + "/ipa_batch_4.bin", M.acc + "/ipa_batch_4.bin");
        copy_file(M2.acc + "/ipa_batch_8.bin", M.acc + "/ipa_batch_8.bin");
        expect(M, "BO-9b batch artifacts replayed from another run", "round0");
        restore(M);
        // BO-11 (single-artifact variant of F11 too): substitute ONE group's
        // IPA file from the other honest run
        copy_file(M2.acc + "/ipa_batch_8.bin", M.acc + "/ipa_batch_8.bin");
        expect(M, "F11 single foreign ipa_batch_8.bin", "ipa8");
        restore(M);
        // BO-11: substitute a commitment file post-prove (same shape, honest
        // hash of the substituted bytes on the verifier side) -> the G0
        // comref-hash absorb diverges from what the prover batched over.
        // (The string-level comref misattribution variant is claims_match —
        // covered by BO-3; the producing-driver divergence is pinned in the
        // zkob_fc claim-mode selftest com tampers.)
        string com = M.tts[1].compath, bak = com + ".bak";
        copy_file(com, bak);
        FrTensor sub = FrTensor::random_int(M.tts[1].rows * M.tts[1].G, 8);
        M.gens[M.tts[1].G]->commit(sub).save(com);
        expect(M, "BO-11 substituted commitment file (post-prove)", "round0");
        copy_file(bak, com);
        remove(bak.c_str());
        expect(M, "restored after BO-11", "accept");
    }

    printf("=== BO-12 (group structure) ===\n");
    {
        string p = M.acc + "/ipa_batch_4.bin", bak = p + ".bak";
        copy_file(p, bak);
        remove(p.c_str());
        expect(M, "BO-12 missing domain-4 group IPA", "group_missing");
        copy_file(bak, p);
        remove(bak.c_str());
        expect(M, "restored after BO-12", "accept");
    }

    printf("=== EvalVar / empty-accumulator structural edges (F6) ===\n");
    {
        vector<BoClaim> cv = M.claims;
        cv[0].tag = BO_EVAL_COMMITTED;
        memset(&cv[0].ceval, 0x11, sizeof(G1Jacobian_t));
        claims_save(M.acc + "/claims.bin", cv);
        claims_save(M.vacc + "/claims.bin", cv);
        expect(M, "Committed EvalVar claim (path closed until Stage D)", "evalvar");
        claims_save(M.acc + "/claims.bin", {});
        claims_save(M.vacc + "/claims.bin", {});
        expect(M, "n_claims == 0", "empty");
        restore(M);
        expect(M, "final restored honest", "accept");
    }

    for (auto& t : M.tts) delete t.data;
    for (auto& g : M.gens) delete g.second;
    printf("ZKOB-BATCHOPEN SELFTEST: %d/%d %s\n", g_total - g_fail, g_total,
           g_fail == 0 ? "ALL PASS" : "FAIL");
    return g_fail == 0;
}

// ===========================  Stage-D wselftest  ===========================
// Weight sub-batch battery: hiding registration (D1), Committed claims +
// committed-round sumcheck (D2), ZK IPA (D3), leak regression (D4) at mini
// scale, mirroring build_minirun's shapes (3 tensors / 2 domains / one
// multi-claim tensor).
struct WMiniRun {
    string base, wacc, wvacc, qpath, run_seed;
    map<uint32_t, string> genpaths;
    map<uint32_t, Commitment*> gens;
    vector<MiniTensor> tts;
    vector<BoClaim> claims;          // Committed claims (cevals)
    vector<Fr_t> evals, eblinds;     // the hidden evals + their blinds
    vector<DrvState> dss;
};
static WMiniRun build_wminirun(const string& base, const string& seedtag) {
    WMiniRun M;
    M.base = base; M.wacc = base + "/wacc"; M.wvacc = base + "/wvacc";
    M.run_seed = "selftest:" + seedtag;
    { string c = "rm -rf " + base; system(c.c_str()); }
    mkdir(base.c_str(), 0755); mkdir(M.wacc.c_str(), 0755); mkdir(M.wvacc.c_str(), 0755);

    struct Spec { const char* name; uint rows, G, nclaims; };
    vector<Spec> specs = {{"A", 4, 4, 2}, {"B", 2, 8, 1}, {"C", 8, 8, 1}};
    for (auto& s : specs)
        if (!M.gens.count(s.G)) {
            Commitment* g = new Commitment(Commitment::random(s.G));
            string p = base + "/gen" + to_string(s.G) + ".bin";
            g->save(p);
            M.gens[s.G] = g; M.genpaths[s.G] = p;
        }
    M.qpath = base + "/q.bin";
    Commitment::random(2).save(M.qpath);   // [Q, H] — H is CONSUMED here (Stage D)
    Commitment qg(M.qpath);
    G1Jacobian_t Q = qg(0), H = qg(1);

    for (auto& s : specs) {
        MiniTensor t;
        t.name = s.name; t.rows = s.rows; t.G = s.G;
        t.data = new FrTensor(FrTensor::random_int(s.rows * s.G, 8));
        t.host.resize(s.rows * s.G);
        cudaMemcpy(t.host.data(), t.data->gpu_data, t.host.size() * sizeof(Fr_t),
                   cudaMemcpyDeviceToHost);
        // D1: hiding registration — fresh per-row blinds, prover-private file
        G1TensorJacobian com = M.gens[s.G]->commit(*t.data);
        vector<Fr_t> rb(s.rows);
        for (auto& x : rb) x = wp_rand();
        wp_hide_rows(com, rb, H);
        t.compath = base + "/com_" + t.name + ".bin";
        com.save(t.compath);
        string blindpath = base + "/blinds_" + t.name + ".bin";
        wp_blinds_save(blindpath, rb);
        t.witpath = base + "/wit_" + t.name + ".fr";
        t.data->save(t.witpath);

        // simulated weight-touching driver: absorbs the (hiding) registered
        // com, squeezes points, emits COMMITTED claims (C_v absorbed like a
        // driver's C_W)
        fs::Transcript dtr(M.run_seed + ":wdrv." + t.name);
        absorb_g1_tensor(dtr, "com", com);
        for (uint k = 0; k < s.nclaims; k++) {
            uint logG = ceilLog2(t.G), logR = ceilLog2(t.rows);
            BoClaim c;
            c.id = "wdrv." + t.name + ":c" + to_string(k);
            c.comref = t.compath;
            c.domain = t.G; c.n_rows = t.rows;
            c.point = fs_challenge_vec(dtr, logG + logR);
            vector<Fr_t> u_col(c.point.begin(), c.point.begin() + logG);
            vector<Fr_t> u_row(c.point.begin() + logG, c.point.end());
            Fr_t v = t.data->multi_dim_me({u_row, u_col}, {t.rows, t.G});
            Fr_t tb = wp_rand();
            c.tag = BO_EVAL_COMMITTED;
            c.ceval = ped_qh(v, tb, Q, H);
            absorb_g1(dtr, "C_claim", c.ceval);
            M.claims.push_back(c);
            M.evals.push_back(v);
            M.eblinds.push_back(tb);
            claim_emit(M.wacc, c);
            claim_emit(M.wvacc, c);
            wp_cblind_emit(M.wacc, c.id, v, tb);
        }
        witref_emit(M.wacc, t.compath, t.witpath);
        wp_blindref_emit(M.wacc, t.compath, blindpath);
        drvstate_emit(M.wacc, "wdrv." + t.name, dtr);
        drvstate_emit(M.wvacc, "wdrv." + t.name, dtr);
        DrvState d; d.id = "wdrv." + t.name; memcpy(d.state, dtr.state, 32);
        M.dss.push_back(d);
        M.tts.push_back(t);
    }
    wbatch_prove(M.wacc, M.run_seed, M.genpaths, M.qpath);
    return M;
}
static void wrestore(WMiniRun& M) {
    claims_save(M.wacc + "/claims.bin", M.claims);
    claims_save(M.wvacc + "/claims.bin", M.claims);
    drvstates_save(M.wacc + "/drvstates.bin", M.dss);
    drvstates_save(M.wvacc + "/drvstates.bin", M.dss);
    wbatch_prove(M.wacc, M.run_seed, M.genpaths, M.qpath);
}
static void wexpect(WMiniRun& M, const string& what, const string& want_locus) {
    string locus;
    bool acc = wbatch_verify(M.wacc, M.wvacc, M.run_seed, M.genpaths, M.qpath, &locus);
    bool ok = !acc && locus == want_locus;
    if (want_locus == "accept") ok = acc;
    g_total++;
    if (!ok) g_fail++;
    printf("  [%s] %s -> expected %s, got %s\n", ok ? "PASS" : "FAIL",
           what.c_str(), want_locus.c_str(), locus.c_str());
}
static void wcheck(const string& what, bool ok) {
    g_total++;
    if (!ok) g_fail++;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
}

static bool wselftest() {
    vrf_selfcheck();
    bo_probe_kernels();
    wpriv_probe();
    printf("kernel -dlto probes (incl. wpriv shapes): PASS\n");
    setenv("ZKOB_BATCH_SELFCHECK", "1", 1);
    setenv("ZKOB_FOLD_CROSSCHECK", "1", 1);
    WMiniRun M = build_wminirun("/tmp/zkob_wbatch_selftest", "wbo");

    printf("=== honest (D1 hiding coms + D2 committed rounds + D3 ZK IPA) ===\n");
    wexpect(M, "honest weight batch", "accept");

    printf("=== D4 leak regression (mini scale) ===\n");
    {
        // every verifier-consumed artifact of the weight path must be free of
        // every hidden eval's byte pattern
        vector<string> artifacts = {
            M.wacc + "/claims.bin", M.wvacc + "/claims.bin",
            M.wacc + "/wbatch_sumcheck.bin", M.wacc + "/wbatch_vfin.bin",
            M.wacc + "/drvstates.bin"};
        for (auto& g : M.genpaths) artifacts.push_back(M.wacc + "/wipa_batch_" + to_string(g.first) + ".bin");
        bool clean = true;
        for (uint i = 0; i < M.evals.size(); i++) {
            auto hits = wp_leak_scan(artifacts, M.evals[i]);
            for (auto& h : hits) {
                printf("    LEAK: eval %u found in %s\n", i, h.c_str());
                clean = false;
            }
        }
        wcheck("no hidden weight eval appears in any weight-path artifact", clean);
        // scanner positive control: a PLAIN claim list with the same eval
        // must be flagged
        {
            vector<BoClaim> plain = M.claims;
            plain[0].tag = BO_EVAL_PLAIN;
            plain[0].eval = M.evals[0];
            claims_save(M.base + "/control_plain_claims.bin", plain);
            wcheck("leak scanner positive control (plain list flags the eval)",
                   wp_file_contains(M.base + "/control_plain_claims.bin", M.evals[0]));
        }
    }

    printf("=== ZK re-randomization sanity ===\n");
    {
        auto before = bo_read_file(M.wacc + "/wbatch_sumcheck.bin");
        auto ipa_before = bo_read_file(M.wacc + "/wipa_batch_4.bin");
        wbatch_prove(M.wacc, M.run_seed, M.genpaths, M.qpath);
        auto after = bo_read_file(M.wacc + "/wbatch_sumcheck.bin");
        auto ipa_after = bo_read_file(M.wacc + "/wipa_batch_4.bin");
        wcheck("re-prove changes round commitments (fresh blinds)",
               before.size() == after.size() && memcmp(before.data(), after.data(), before.size()) != 0);
        wcheck("re-prove changes the ZK IPA (fresh blinds)",
               ipa_before.size() == ipa_after.size() &&
               memcmp(ipa_before.data(), ipa_after.data(), ipa_before.size()) != 0);
        wexpect(M, "re-randomized honest proof still accepts", "accept");
    }

    printf("=== WBO-1a/1b (false committed eval; adaptive prover) ===\n");
    {
        // driver-level lie: ceval commits v+1 (consistently in both lists +
        // the prover blind store); honest-PROCEDURE batch dies at wround0
        Commitment qg(M.qpath);
        vector<BoClaim> bad = M.claims;
        Fr_t vbad = h_scalar(M.evals[1], F_ONE, 0);
        bad[1].ceval = ped_qh(vbad, M.eblinds[1], qg(0), qg(1));
        claims_save(M.wacc + "/claims.bin", bad);
        claims_save(M.wvacc + "/claims.bin", bad);
        { remove((M.wacc + "/cblinds.bin").c_str());
          for (uint i = 0; i < bad.size(); i++)
              wp_cblind_emit(M.wacc, bad[i].id, i == 1 ? vbad : M.evals[i], M.eblinds[i]); }
        wbatch_prove(M.wacc, M.run_seed, M.genpaths, M.qpath, /*evil=*/1);
        wexpect(M, "WBO-1a false committed eval, honest-procedure prover", "wround0");
        // adaptive: coherent commitment-level lies through every round +
        // poisoned last C_vfin -> forced into that tensor's group ZK-IPA
        wbatch_prove(M.wacc, M.run_seed, M.genpaths, M.qpath, /*evil=*/2);
        wexpect(M, "WBO-1b adaptive prover (forged C_vfin to pass G3)", "wipa8");
        { remove((M.wacc + "/cblinds.bin").c_str());
          for (uint i = 0; i < M.claims.size(); i++)
              wp_cblind_emit(M.wacc, M.claims[i].id, M.evals[i], M.eblinds[i]); }
        wrestore(M);
    }

    printf("=== WBO transcript/blind forgeries ===\n");
    {
        wbatch_prove(M.wacc, M.run_seed, M.genpaths, M.qpath, /*evil=*/3);
        wexpect(M, "rho derived from doctored ceval list", "wround0");
        wbatch_prove(M.wacc, M.run_seed, M.genpaths, M.qpath, /*evil=*/4);
        wexpect(M, "IPA blind beta shifted by 1 (blind bookkeeping lie)", "wipa4");
        wrestore(M);
    }

    printf("=== artifact byte tampers ===\n");
    {
        // wbatch_sumcheck.bin: 4 magic + 4 n_claims + 4 m_max + 4 vec-count
        long g1sz = (long)sizeof(G1Jacobian_t);
        tamper_byte(M.wacc + "/wbatch_sumcheck.bin", 16 + g1sz, +1);  // round-0 C_p1
        wexpect(M, "round-0 C_p1 tamper", "wround0");
        tamper_byte(M.wacc + "/wbatch_sumcheck.bin", 16 + g1sz, -1);
        tamper_byte(M.wacc + "/wbatch_sumcheck.bin", 16 + 3 * g1sz, +1); // round-1 C_p0
        wexpect(M, "round-1 C_p0 tamper", "wround1");
        tamper_byte(M.wacc + "/wbatch_sumcheck.bin", 16 + 3 * g1sz, -1);
        tamper_byte(M.wacc + "/wbatch_sumcheck.bin", 4, +1);
        wexpect(M, "wbatch_sumcheck n_claims field tamper", "wxcheck");
        tamper_byte(M.wacc + "/wbatch_sumcheck.bin", 4, -1);
        tamper_byte(M.wacc + "/wbatch_vfin.bin", -1, +1);
        wexpect(M, "wbatch_vfin C_vfin tamper", "wterminal");
        tamper_byte(M.wacc + "/wbatch_vfin.bin", -1, -1);
        tamper_byte(M.wacc + "/wipa_batch_4.bin", -1, +1);            // zb of schnorr2
        wexpect(M, "wipa_batch_4 Schnorr response tamper", "wipa4");
        tamper_byte(M.wacc + "/wipa_batch_4.bin", -1, -1);
        tamper_byte(M.wacc + "/wipa_batch_8.bin", 8 + 16, +1);        // L[0] body
        wexpect(M, "wipa_batch_8 round-0 L tamper", "wipa8");
        tamper_byte(M.wacc + "/wipa_batch_8.bin", 8 + 16, -1);
        tamper_byte(M.wacc + "/claims.bin", -1, +1);                  // last ceval byte
        wexpect(M, "weight claims.bin ceval byte tamper", "wclaims_match");
        tamper_byte(M.wacc + "/claims.bin", -1, -1);
        vector<DrvState> ds2 = M.dss;
        ds2.back().state[0] ^= 1;
        drvstates_save(M.wvacc + "/drvstates.bin", ds2);
        wexpect(M, "verifier drvstate divergence", "wround0");
        wrestore(M);
    }

    printf("=== structure: omission, routing (BO-13), empty, substitution ===\n");
    {
        vector<BoClaim> omit(M.claims.begin(), M.claims.end() - 1);
        claims_save(M.wacc + "/claims.bin", omit);
        wexpect(M, "weight claim omitted from prover list", "wclaims_match");
        wrestore(M);
        // BO-13a: a PLAIN claim smuggled into the weight batch (both lists)
        vector<BoClaim> pl = M.claims;
        pl[0].tag = BO_EVAL_PLAIN;
        pl[0].eval = M.evals[0];
        claims_save(M.wacc + "/claims.bin", pl);
        claims_save(M.wvacc + "/claims.bin", pl);
        wexpect(M, "BO-13a plain-eval claim inside the weight batch", "wevalvar");
        wrestore(M);
        // BO-13b: the weight claims fed to the PUBLIC batch verifier must
        // die at the Stage-A evalvar gate (the path stays closed)
        {
            string locus;
            bool acc = batch_verify(M.wacc, M.wvacc, M.run_seed, M.genpaths, M.qpath, &locus);
            wcheck("BO-13b committed claims rejected by the PUBLIC batch (evalvar)",
                   !acc && locus == "evalvar");
        }
        claims_save(M.wacc + "/claims.bin", {});
        claims_save(M.wvacc + "/claims.bin", {});
        wexpect(M, "n_claims == 0", "wempty");
        wrestore(M);
        // substituted (re-blinded) commitment file post-prove: same tensor,
        // fresh blinds -> comref hash diverges from what the prover batched
        string com = M.tts[1].compath, bak = com + ".bak";
        copy_file(com, bak);
        {
            Commitment qg(M.qpath);
            G1TensorJacobian sub = M.gens[M.tts[1].G]->commit(*M.tts[1].data);
            vector<Fr_t> rb(M.tts[1].rows);
            for (auto& x : rb) x = wp_rand();
            wp_hide_rows(sub, rb, qg(1));
            sub.save(com);
        }
        wexpect(M, "substituted (re-blinded) commitment file post-prove", "wround0");
        copy_file(bak, com);
        remove(bak.c_str());
        wexpect(M, "restored after substitution", "accept");
        string p = M.wacc + "/wipa_batch_4.bin";
        copy_file(p, p + ".bak");
        remove(p.c_str());
        wexpect(M, "missing domain-4 group ZK-IPA", "wgroup_missing");
        copy_file(p + ".bak", p);
        remove((p + ".bak").c_str());
        wexpect(M, "final restored honest", "accept");
    }

    for (auto& t : M.tts) delete t.data;
    for (auto& g : M.gens) delete g.second;
    printf("ZKOB-WBATCH SELFTEST: %d/%d %s\n", g_total - g_fail, g_total,
           g_fail == 0 ? "ALL PASS" : "FAIL");
    return g_fail == 0;
}

#include "zkob_serve.cuh"
static int zkw_run1(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    try {
        if (mode == "selftest") return selftest() ? 0 : 1;
        if (mode == "wselftest") return wselftest() ? 0 : 1;
        if (mode == "wprove" && argc >= 5) {
            int evil = getenv("ZKOB_EVIL") ? atoi(getenv("ZKOB_EVIL")) : 0;
            wbatch_prove(argv[2], argv[3], parse_genspecs(argc, argv, 5), argv[4], evil);
            cout << "PROVED opening_batch_w" << (evil ? " (EVIL mode, selftest only)" : "")
                 << " -> " << argv[2] << endl;
            return 0;
        }
        if (mode == "wverify" && argc >= 6) {
            bool ok = wbatch_verify(argv[2], argv[3], argv[4],
                                    parse_genspecs(argc, argv, 6), argv[5]);
            return ok ? 0 : 1;
        }
        if (mode == "genq" && argc == 3) {
            // §4.4 H-slot registration touch: q.bin = [Q, H]; every existing
            // consumer reads index 0 only; H is registered now, used in Stage D
            Commitment::random(2).save(argv[2]);
            cout << "wrote 2-slot q file (Q + H slot) -> " << argv[2] << endl;
            return 0;
        }
        if (mode == "prove" && argc >= 5) {
            // ZKOB_EVIL=1|2|3: selftest/battery-only forgery construction
            // (the header's batch_prove evil modes, exposed for the pair-scale
            // BO battery; unset in production — mode 0)
            int evil = getenv("ZKOB_EVIL") ? atoi(getenv("ZKOB_EVIL")) : 0;
            batch_prove(argv[2], argv[3], parse_genspecs(argc, argv, 5), argv[4], evil);
            cout << "PROVED opening_batch" << (evil ? " (EVIL mode, selftest only)" : "")
                 << " -> " << argv[2] << endl;
            return 0;
        }
        if (mode == "verify" && argc >= 6) {
            bool ok = batch_verify(argv[2], argv[3], argv[4],
                                   parse_genspecs(argc, argv, 6), argv[5]);
            return ok ? 0 : 1;
        }
    } catch (const std::exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 2;
    }
    cerr << "usage: zkob_batchopen selftest | wselftest | genq <out>\n"
            "       zkob_batchopen prove   <accdir> <run_seed> <q.bin> <G>=<gen.bin> ...\n"
            "       zkob_batchopen verify  <prover-accdir> <verifier-accdir> <run_seed> <q.bin> <G>=<gen.bin> ...\n"
            "       zkob_batchopen wprove  <waccdir> <run_seed> <q.bin> <G>=<gen.bin> ...\n"
            "       zkob_batchopen wverify <prover-waccdir> <verifier-waccdir> <run_seed> <q.bin> <G>=<gen.bin> ..."
         << endl;
    return 2;
}

// Stage C2 single-process transport: `serve` keeps this driver resident (one
// CUDA init for the whole walk); every request runs the same zkw_run1 entry.
int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "serve")
        return zkw_serve(argv[0], zkw_run1);
    return zkw_run1(argc, argv);
}
