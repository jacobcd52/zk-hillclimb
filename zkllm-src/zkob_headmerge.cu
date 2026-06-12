// Real driver for ONE head-merge obligation (layer{l}.attn.merge), per
// ROPE_ATTENTION_DESIGN.md (DESIGN FINAL 2026-06-10) section 4.6 / 5.3.
// Binds the NH per-head value outputs out_h (B x HD int32 @2^16, chained from
// the values rescales, edge A14) to com_attn_out = com_O2 INCLUDING the
// pipeline's line-156/157 double transpose+reshape permutation pi:
//   O2[i,j] = M[t,e],  m = i*C + j,  t = m mod B,  e = m div B   (pi, section 1.3)
//   M[t, HD*h+d] = out_h[t,d]                                    (head concat)
//   pi^-1(t,e): m = e*B + t; i = m div C; j = m mod C            (real index sets)
// (B == C makes pi the plain transpose -- legal, NOT thrown; selftest case.)
//
// Binding: FS challenge u (logB + logC_pad vars over the padded O2 grid),
// claim ev = O2~(u), then NH public-weight hadamards (logB + logHD vars each):
//   c_h = sum_{(t,d)} Wm_h[t*HD+d] * out_h[t*HD+d],
//   Wm_h[t*HD+d] = E_u[ i*C_pad + j ] with (i,j) = pi^-1(t, HD*h+d)
// run as fs_hadamard(E = Wm_h, S = out_h, U = 1); terminal opens out_h at
// reverse(ws_h) vs com_O{hh}; the verifier rebuilds each Wm_h ITSELF
// (build_eq_tensor(u) once, one host copy, NH gathers with the pinned pi^-1
// formula, upload, fold) and REQUIRES U_f2 == 1 per head. Final: IPA of O2 at
// u vs com_O2 (eval = ev) and the plain-field check sum_h c_h == ev.
// Each real (i,j) is hit by exactly one (h,t,d), so sum_h c_h is the MLE at u
// of "pi(concat) on real entries, 0 on padding" -- concat order, pi AND
// padding hygiene certified in one check. Zero advice; ZERO new CUDA kernels.
//
// FS schedule (design section 5.3; labels exact, {hh} = 2-digit decimal):
//   absorb B, C, HD; com_O2; per hh com_O{hh} -> u
//   absorb ev
//   per hh: absorb c{hh}; hadamard rounds ("hp0".."hp3" -> w);
//           absorb S_f2, U_f2; IPA(out_h)@reverse(ws_hh)
//   IPA(O2)@u
//
// Files in <obdir>: dims.bin, com_O2.bin, com_O{hh}.bin, ev.bin (1 Fr_t, raw),
//   hp{hh}.bin, ipa_O{hh}.bin, ipa_O2.bin
//
// PERM mode (STAGE3_FAITHFUL_DESIGN.md section 4.2): positional <perm> after
// <HD>, literal pi157 (0) | concat (1). pi157 keeps the formulas above
// verbatim; concat uses the identity layout (the faithful-arch line-157 fix):
//   Wm_h[t*HD+d] = E_u[t*C_pad + (HD*h+d)];  O2[t, HD*h+d] = out_h[t,d].
// PERM is recorded in dims.bin (u32, 4th field) and absorbed into the
// transcript ("PERM", immediately after "HD") -- REQUIRED, else a proof made
// in one mode could be replayed against a verifier in the other.
//
// Usage:
//   zkob_headmerge prove   <obdir> <seed> <Oh-prefix> <B> <C> <HD> <perm>
//                          <gen_big.bin> <gen_small.bin> <q.bin> [O2-int32-out.bin]
//   zkob_headmerge verify  <obdir> <seed> <B> <C> <HD> <perm>
//                          <gen_big.bin> <gen_small.bin> <q.bin>
//   zkob_headmerge selftest
// <Oh-prefix>{hh}.i32.bin is read for hh = 00..NH-1 (file sizes exact).
// The driver does NOT mkdir the obdir.
// Both prove and verify accept a trailing [--claims <(v)accdir> <obid>].
//
// CLAIM MODE (Stage C of the transport rebuild, flag-selected; the old
// inline-IPA tail stays compilable and is the DEFAULT): with --claims, prove
// EMITS its NH+1 terminal claims at the exact old open_prove sites —
//   per head h: out_h vs com_O{hh} at reverse(ws_h)  (eval S_f2_h, domain HD)
//   then:       O2    vs com_O2    at u              (eval ev, domain C_pad)
// — into <accdir>/claims.bin plus witrefs and drvstate (a TWO-domain batch:
// gen_small=HD and gen_big=C_pad). Verify keeps every round check, the
// verifier-rebuilt Wm folds, the U_f2 == 1 pins and the sum_h c_h == ev
// identity UNCHANGED, then recomputes the NH+1 claims from its own FS replay
// into <vaccdir>; verdict becomes ACCEPT-conditional. All five semantic evil
// modes die DRIVER-side exactly as before (their loci are not openings).
#include "zkob_lookup.cuh"
#include "zkob_claims.cuh"
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <sys/stat.h>
using namespace std;

static_assert(sizeof(long) == 8, "host math needs 64-bit long");

static string hh2(uint h) { char b[8]; snprintf(b, sizeof b, "%02u", h); return string(b); }

// layout guards (design section 6, pinned; B == C is allowed)
static void layout_guards(uint B, uint C, uint HD,
                          const Commitment& gen_big, const Commitment& gen_small) {
    const uint C_pad = 1u << ceilLog2(C);
    if (B != (1u << ceilLog2(B))) throw runtime_error("B not a power of two");
    if (HD < 2 || HD != (1u << ceilLog2(HD))) throw runtime_error("HD must be a power of two >= 2");
    if (C % HD) throw runtime_error("HD must divide C");
    if (C_pad % HD) throw runtime_error("HD must divide C_pad");
    if (C / HD < 2) throw runtime_error("NH = C/HD must be >= 2");
    if (gen_big.size != C_pad) throw runtime_error("gen_big size != C_pad");
    if (gen_small.size != HD) throw runtime_error("gen_small size != HD");
}

// strict-size int32 loader (input file sizes exact; short read throws)
static vector<int> load_i32_exact(const string& path, uint expect) {
    FILE* f = open_or_die(path, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz != (long)expect * 4)
        { fclose(f); throw runtime_error("file size != expected int32 count: " + path); }
    fseek(f, 0, SEEK_SET);
    vector<int> buf(expect);
    if (fread(buf.data(), sizeof(int), expect, f) != expect)
        { fclose(f); throw runtime_error("short read: " + path); }
    fclose(f);
    return buf;
}

// gather of the host eq tensor into head h's weight (the pinned formulas;
// gather_h = h is the honest case, evil=3 passes h+1). perm = 0 (pi157): the
// section-1.3 pi^-1; perm = 1 (concat): the identity layout (section 4.2).
static vector<Fr_t> gather_Wm(const vector<Fr_t>& Eh, uint B, uint C, uint C_pad,
                              uint HD, uint gather_h, uint perm) {
    vector<Fr_t> Wm((size_t)B * HD);
    for (uint t = 0; t < B; t++)
        for (uint d = 0; d < HD; d++) {
            const size_t e = (size_t)HD * gather_h + d;
            size_t i, j;
            if (perm == 0) { const size_t m = e * B + t; i = m / C; j = m % C; }
            else           { i = t; j = e; }
            Wm[(size_t)t * HD + d] = Eh[i * C_pad + j];
        }
    return Wm;
}
static uint parse_perm(const string& s) {
    if (s == "pi157") return 0;
    if (s == "concat") return 1;
    throw runtime_error("perm must be 'pi157' or 'concat'");
}

// fold a public device tensor over the round challenges (k_fr_fold chain)
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

// ---------------- prove ----------------
// evil (selftest only; honest PROCEDURE on an inconsistent witness):
//  1: O2 assembled WITHOUT the line-157 permutation (O2 := M, the natural
//     implementation bug); ev honest from the evil O2; hadamards honest
//     -> "sum of head claims != ev". THE certifying evil for pi.
//  2: O2 honest on real entries but junk in a padding column; ev honest from
//     it -> "sum of head claims != ev" (weights' real-index support forces
//     padding = 0). Only run when C_pad > C.
//  3: Wm of head evil_h gathered with h off by one (h+1), that head's
//     hadamard run honestly on it, c absorbed from that run
//     -> "merge hadamard {hh} terminal" (verifier-rebuilt Wm fold differs).
//  4: ones-buffer bump in head evil_h's hadamard, c computed from it
//     -> "merge hadamard {hh} U_f2 != 1".
//  5: O2 assembled with the WRONG mode's layout (concat-mode run with the
//     pi157 gather and vice versa -- the exact regression the perm flag could
//     introduce), ev honest from the evil O2 -> "sum of head claims != ev".
static void prove(const string& obdir, const string& seed,
                  const vector<vector<int>>& outs, uint B, uint C, uint HD,
                  uint perm,
                  const Commitment& gen_big, const Commitment& gen_small,
                  const G1Jacobian_t& Q, const string& o2_out,
                  int evil = 0, uint evil_h = 0,
                  const string& accdir = "", const string& obid = "") {
    const bool claim_mode = !accdir.empty();
    layout_guards(B, C, HD, gen_big, gen_small);
    const uint C_pad = 1u << ceilLog2(C);
    const uint logB = ceilLog2(B), logCp = ceilLog2(C_pad), logHD = ceilLog2(HD);
    const uint NH = C / HD, D = B * C_pad, DH = B * HD;
    if (outs.size() != NH) throw runtime_error("head count");
    for (auto& o : outs) if (o.size() != (size_t)DH) throw runtime_error("out_h dims");

    // ---- O2 on the padded grid (host integer gather): pi157 = pi(concat),
    //      concat = plain head-concat M (evil 5 swaps the two layouts) ----
    const uint asm_perm = (evil == 5) ? 1 - perm : perm;
    vector<int> O2g(D, 0);
    if (asm_perm == 0)
        for (uint i = 0; i < B; i++)
            for (uint j = 0; j < C; j++) {
                const size_t m = (size_t)i * C + j;
                const uint t = (uint)(m % B), e = (uint)(m / B);
                O2g[(size_t)i * C_pad + j] = outs[e / HD][(size_t)t * HD + (e % HD)];
            }
    else
        for (uint i = 0; i < B; i++)
            for (uint j = 0; j < C; j++)
                O2g[(size_t)i * C_pad + j] = outs[j / HD][(size_t)i * HD + (j % HD)];
    if (evil == 1) {        // identity layout: O2 := M (no pi); pi157-mode only
        if (perm != 0) throw runtime_error("evil 1 setup: pi157 mode only (vacuous in concat)");
        for (uint i = 0; i < B; i++)
            for (uint j = 0; j < C; j++)
                O2g[(size_t)i * C_pad + j] = outs[j / HD][(size_t)i * HD + (j % HD)];
    }
    if (evil == 2) {        // junk in a padding column (requires C_pad > C)
        if (C_pad <= C) throw runtime_error("evil 2 setup: no padding columns");
        O2g[(size_t)0 * C_pad + C] = 17;
    }
    if (!o2_out.empty()) {   // chain file: UNPADDED B x C int32, scale 2^16
        vector<int> oo((size_t)B * C);
        for (uint i = 0; i < B; i++)
            for (uint j = 0; j < C; j++) oo[(size_t)i * C + j] = O2g[(size_t)i * C_pad + j];
        FILE* f = open_or_die(o2_out, "wb");
        fwrite(oo.data(), sizeof(int), oo.size(), f); fclose(f);
    }

    // ---- tensors + commitments ----
    FrTensor O2_t(D, O2g.data());
    vector<FrTensor> out_t;
    for (uint h = 0; h < NH; h++) out_t.emplace_back(DH, outs[h].data());
    G1TensorJacobian com_O2 = gen_big.commit(O2_t);
    com_O2.save(obdir + "/com_O2.bin");
    vector<G1TensorJacobian> com_O;
    for (uint h = 0; h < NH; h++) {
        com_O.push_back(gen_small.commit(out_t[h]));
        com_O[h].save(obdir + "/com_O" + hh2(h) + ".bin");
    }
    { FILE* f = open_or_die(obdir + "/dims.bin", "wb");
      uint32_t d[4] = {B, C, HD, perm}; fwrite(d, sizeof(uint32_t), 4, f); fclose(f); }

    // ---- transcript ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C); absorb_u32(tr, "HD", HD);
    absorb_u32(tr, "PERM", perm);
    absorb_g1_tensor(tr, "com_O2", com_O2);
    for (uint h = 0; h < NH; h++) absorb_g1_tensor(tr, "com_O" + hh2(h), com_O[h]);
    auto u = fs_challenge_vec(tr, logB + logCp);
    vector<Fr_t> u_col(u.begin(), u.begin() + logCp);
    vector<Fr_t> u_row(u.begin() + logCp, u.end());

    Fr_t ev = O2_t.multi_dim_me({u_row, u_col}, {B, C_pad});
    absorb_fr(tr, "ev", ev);
    { FILE* f = open_or_die(obdir + "/ev.bin", "wb");
      fwrite(&ev, sizeof(Fr_t), 1, f); fclose(f); }

    // eq tensor built ONCE (device), copied to host once, gathered NH times --
    // prover and verifier use the identical pinned pi^-1 formula
    FrTensor E_u = build_eq_tensor(u);
    vector<Fr_t> Eh(D);
    cudaMemcpy(Eh.data(), E_u.gpu_data, D * sizeof(Fr_t), cudaMemcpyDeviceToHost);

    vector<int> onesh(DH, 1);
    Fr_t csum = F_ZERO;
    for (uint h = 0; h < NH; h++) {
        const uint gather_h = (evil == 3 && h == evil_h) ? h + 1 : h;
        vector<Fr_t> Wm = gather_Wm(Eh, B, C, C_pad, HD, gather_h, perm);
        FrTensor Wm_t(DH, Wm.data());
        vector<int> uh(onesh);
        if (evil == 4 && h == evil_h) uh[1] += 1;
        FrTensor U_t(DH, uh.data());
        HadamardProof hp;
        hp.claim_H = (Wm_t * out_t[h] * U_t).sum();                 // c_h
        absorb_fr(tr, "c" + hh2(h), hp.claim_H);
        csum = h_scalar(csum, hp.claim_H, 0);
        vector<Fr_t> ws;
        {
            FrTensor Ec(Wm_t), Sc(out_t[h]), Uc(U_t);
            fs_hadamard(hp.claim_H, Ec, Sc, Uc, tr, ws, hp, true);
        }
        absorb_fr(tr, "S_f2", hp.S_f2); absorb_fr(tr, "U_f2", hp.U_f2);
        write_hp(obdir + "/hp" + hh2(h) + ".bin", hp);
        vector<Fr_t> pt(ws.rbegin(), ws.rend());
        if (evil == 0) {    // convention sanity: fold terminals == ME evaluations
            vector<Fr_t> p_col(pt.begin(), pt.begin() + logHD);
            vector<Fr_t> p_row(pt.begin() + logHD, pt.end());
            if (!fr_eq(hp.S_f2, out_t[h].multi_dim_me({p_row, p_col}, {B, HD})) ||
                !fr_eq(hp.U_f2, F_ONE))
                throw runtime_error("merge hadamard terminal != multi_dim_me / 1 (convention bug)");
        }
        if (claim_mode) {
            // claim emission at the exact old open_prove site (order: heads
            // ascending, then O2)
            BoClaim c;
            c.id = obid + ":O" + hh2(h);
            c.comref = obdir + "/com_O" + hh2(h) + ".bin";
            c.domain = HD; c.n_rows = B;
            c.point = pt;
            c.eval = hp.S_f2;
            claim_emit(accdir, c);
            string wit = accdir + "/wit_" + obid + "_O" + hh2(h) + ".fr";
            out_t[h].save(wit);
            witref_emit(accdir, c.comref, wit);
        } else {
            open_prove(out_t[h], HD, gen_small, Q, pt, obdir + "/ipa_O" + hh2(h) + ".bin", tr);
        }
    }
    if (evil == 0 && !fr_eq(csum, ev))   // the section-4.6 identity, honest case
        throw runtime_error("sum of head claims != ev (pi formula bug)");
    if (claim_mode) {
        BoClaim c;
        c.id = obid + ":O2";
        c.comref = obdir + "/com_O2.bin";
        c.domain = C_pad; c.n_rows = B;
        c.point = u;
        c.eval = ev;
        claim_emit(accdir, c);
        string wit = accdir + "/wit_" + obid + "_O2.fr";
        O2_t.save(wit);
        witref_emit(accdir, c.comref, wit);
        drvstate_emit(accdir, obid, tr);
        cout << "PROVED headmerge obligation (claim mode, " << NH + 1
             << " claims emitted) -> " << obdir << endl;
        return;
    }
    open_prove(O2_t, C_pad, gen_big, Q, u, obdir + "/ipa_O2.bin", tr);
    cout << "PROVED headmerge obligation -> " << obdir << endl;
}

// ---------------- verify (witness-free) ----------------
#define RJ(msg) do { ostringstream oss_; oss_ << msg; \
    cout << "REJECT: " << oss_.str() << endl; \
    if (reason) *reason = oss_.str(); return false; } while (0)

static bool verify(const string& obdir, const string& seed,
                   uint B, uint C, uint HD, uint perm,
                   const Commitment& gen_big, const Commitment& gen_small,
                   const G1Jacobian_t& Q, string* reason = nullptr,
                   const string& vaccdir = "", const string& obid = "") {
    const bool claim_mode = !vaccdir.empty();
    BoTimer prof("headmerge_verify");
    const uint C_pad = 1u << ceilLog2(C);
    const uint logB = ceilLog2(B), logCp = ceilLog2(C_pad), logHD = ceilLog2(HD);
    const uint NH = C / HD, D = B * C_pad, DH = B * HD;
    const uint logDH = logB + logHD;
    if (B != (1u << logB) || HD < 2 || HD != (1u << logHD) ||
        (C % HD) || (C_pad % HD) || NH < 2 ||
        gen_big.size != C_pad || gen_small.size != HD)
        RJ("bad layout params");
    { FILE* f = open_or_die(obdir + "/dims.bin", "rb");
      uint32_t d[4];
      if (fread(d, sizeof(uint32_t), 4, f) != 4) { fclose(f); RJ("dims.bin short read"); }
      fclose(f);
      if (d[0] != B || d[1] != C || d[2] != HD || d[3] != perm) RJ("dims.bin mismatch"); }

    G1TensorJacobian com_O2(obdir + "/com_O2.bin");
    if (com_O2.size != B) RJ("com_O2 row count");
    vector<G1TensorJacobian> com_O;
    vector<HadamardProof> hps;
    for (uint h = 0; h < NH; h++) {
        com_O.emplace_back(obdir + "/com_O" + hh2(h) + ".bin");
        if (com_O[h].size != B) RJ("com_O" << hh2(h) << " row count");
        hps.push_back(read_hp(obdir + "/hp" + hh2(h) + ".bin"));
        if (hps[h].ev.size() != 4 * logDH) RJ("hadamard " << hh2(h) << " round count");
    }
    Fr_t ev;
    { FILE* f = open_or_die(obdir + "/ev.bin", "rb");
      if (fread(&ev, sizeof(Fr_t), 1, f) != 1) { fclose(f); RJ("ev.bin short read"); }
      fclose(f); }

    // ---- transcript replay ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C); absorb_u32(tr, "HD", HD);
    absorb_u32(tr, "PERM", perm);
    absorb_g1_tensor(tr, "com_O2", com_O2);
    for (uint h = 0; h < NH; h++) absorb_g1_tensor(tr, "com_O" + hh2(h), com_O[h]);
    auto u = fs_challenge_vec(tr, logB + logCp);
    absorb_fr(tr, "ev", ev);

    const Fr_t inv6 = inv(F_SIX);
    // eq tensor built ONCE, host copy ONCE, gathered per head (pinned pi^-1)
    FrTensor E_u = build_eq_tensor(u);
    vector<Fr_t> Eh(D);
    cudaMemcpy(Eh.data(), E_u.gpu_data, D * sizeof(Fr_t), cudaMemcpyDeviceToHost);

    Fr_t csum = F_ZERO;
    vector<vector<Fr_t>> pts;    // per-head opening points (claim mode)
    for (uint h = 0; h < NH; h++) {
        absorb_fr(tr, "c" + hh2(h), hps[h].claim_H);
        csum = h_scalar(csum, hps[h].claim_H, 0);
        Fr_t cur = hps[h].claim_H;
        vector<Fr_t> ws;
        for (uint k = 0; k < logDH; k++) {
            array<Fr_t,4> e = {hps[h].ev[4*k], hps[h].ev[4*k+1],
                               hps[h].ev[4*k+2], hps[h].ev[4*k+3]};
            if (!fr_eq(cur, h_scalar(e[0], e[1], 0)))
                RJ("merge hadamard " << hh2(h) << " round " << k << " p(0)+p(1) != claim");
            absorb_fr(tr, "hp0", e[0]); absorb_fr(tr, "hp1", e[1]);
            absorb_fr(tr, "hp2", e[2]); absorb_fr(tr, "hp3", e[3]);
            Fr_t w = fs_challenge_fr(tr); ws.push_back(w);
            cur = lagrange4(e, w, inv6);
        }
        if (!fr_eq(hps[h].U_f2, F_ONE)) RJ("merge hadamard " << hh2(h) << " U_f2 != 1");
        {
            vector<Fr_t> Wm = gather_Wm(Eh, B, C, C_pad, HD, h, perm);
            FrTensor Wm_t(DH, Wm.data());
            Fr_t W_f = fold_public(Wm_t, ws);
            if (!fr_eq(cur, h_scalar(W_f, h_scalar(hps[h].S_f2, hps[h].U_f2, 2), 2)))
                RJ("merge hadamard " << hh2(h) << " terminal");
        }
        absorb_fr(tr, "S_f2", hps[h].S_f2); absorb_fr(tr, "U_f2", hps[h].U_f2);
        vector<Fr_t> pt(ws.rbegin(), ws.rend());
        if (claim_mode) {
            pts.push_back(pt);   // claim recomputation deferred past csum == ev
        } else {
            if (!open_verify(com_O[h], gen_small, HD, Q, pt, hps[h].S_f2,
                             obdir + "/ipa_O" + hh2(h) + ".bin", tr))
                RJ("IPA opening of head " << hh2(h) << " terminal vs com_O" << hh2(h));
        }
    }
    if (!claim_mode) {
        if (!open_verify(com_O2, gen_big, C_pad, Q, u, ev, obdir + "/ipa_O2.bin", tr))
            RJ("IPA opening of ev vs com_O2");
    }
    if (!fr_eq(csum, ev)) RJ("sum of head claims != ev");
    if (claim_mode) {
        // ---- claim recomputation (canonical order: heads ascending, then
        // O2 — the prover's emission order); ACCEPT-conditional verdict ----
        for (uint h = 0; h < NH; h++) {
            BoClaim c;
            c.id = obid + ":O" + hh2(h);
            c.comref = obdir + "/com_O" + hh2(h) + ".bin";
            c.domain = HD; c.n_rows = B;
            c.point = pts[h];
            c.eval = hps[h].S_f2;
            claim_emit(vaccdir, c);
        }
        BoClaim c;
        c.id = obid + ":O2";
        c.comref = obdir + "/com_O2.bin";
        c.domain = C_pad; c.n_rows = B;
        c.point = u;
        c.eval = ev;
        claim_emit(vaccdir, c);
        drvstate_emit(vaccdir, obid, tr);
        prof.lap("claim_emit");
        cout << "ACCEPT-conditional (" << NH + 1
             << " claims emitted; final verdict gated on opening_batch)" << endl;
        return true;
    }
    cout << "ACCEPT" << endl;
    return true;
}

// ---------------- selftest ----------------
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
static vector<string> proof_files(uint NH) {
    vector<string> fs = {"dims.bin", "com_O2.bin", "ev.bin", "ipa_O2.bin"};
    for (uint h = 0; h < NH; h++) {
        fs.push_back("com_O" + hh2(h) + ".bin");
        fs.push_back("hp" + hh2(h) + ".bin");
        fs.push_back("ipa_O" + hh2(h) + ".bin");
    }
    return fs;
}
static long tamper_offset(const string& f) {
    if (f == "dims.bin") return 0;
    if (f == "ev.bin") return 4;                    // raw Fr sequence
    if (f.substr(0, 4) == "ipa_") return -32;       // a_final
    if (f.substr(0, 4) == "com_") return 24;        // first point, x limbs
    return 36;                                      // hp: claim_H 0-31, count 32-35
}

static bool selftest_case(uint B, uint C, uint HD, uint perm) {
    const uint NH = C / HD, C_pad = 1u << ceilLog2(C);
    cout << "==== selftest case B=" << B << " C=" << C << " HD=" << HD
         << " perm=" << (perm ? "concat" : "pi157")
         << " (NH=" << NH << ", C_pad=" << C_pad
         << (B == C ? ", pi = plain transpose" : "") << ") ====" << endl;
    srand(31337 + B * 100 + C * 10 + HD);
    vector<vector<int>> outs(NH);
    for (uint h = 0; h < NH; h++) {
        outs[h].resize((size_t)B * HD);
        for (auto& x : outs[h]) x = rand() % 257 - 128;
    }
    Commitment gen_big = Commitment::random(C_pad);
    Commitment gen_small = Commitment::random(HD);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_headmerge_ob";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:merge";
    bool all = true;

    prove(obdir, seed, outs, B, C, HD, perm, gen_big, gen_small, Q, "/tmp/zkob_merge_O2.i32.bin");
    bool honest = verify(obdir, seed, B, C, HD, perm, gen_big, gen_small, Q);
    cout << (honest ? "PASS" : "FAIL") << ": honest case "
         << (honest ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && honest;

    // chain-file sanity via the INVERSE formula: pi157: O2[pi^-1(t,e)] == M[t,e];
    // concat: O2[t,e] == M[t,e]
    {
        vector<int> oo((size_t)B * C);
        FILE* f = open_or_die("/tmp/zkob_merge_O2.i32.bin", "rb");
        if (fread(oo.data(), sizeof(int), oo.size(), f) != oo.size())
            throw runtime_error("O2 chain file short read");
        fclose(f);
        bool ok = true;
        for (uint t = 0; t < B && ok; t++)
            for (uint e = 0; e < C && ok; e++) {
                size_t i, j;
                if (perm == 0) { const size_t m = (size_t)e * B + t; i = m / C; j = m % C; }
                else           { i = t; j = e; }
                ok = (oo[i * C + j] == outs[e / HD][(size_t)t * HD + (e % HD)]);
            }
        cout << (ok ? "PASS" : "FAIL") << ": O2 chain file matches "
             << (perm ? "concat" : "pi (checked via pi^-1)") << endl;
        all = all && ok;
    }

    // semantic evil modes; head indices clamped to the toy NH (design 8.2 uses
    // heads 3 and 7, which only exist at real scale)
    const uint eh3 = min(3u, NH - 2);   // gather h+1 must stay a real head
    const uint eh4 = min(7u, NH - 1);
    struct Evil { int mode; uint h; string expect; const char* what; };
    vector<Evil> evils = {
        {3, eh3, "merge hadamard " + hh2(eh3) + " terminal",
            "Wm gathered with h off by one, hadamard honest on it"},
        {4, eh4, "merge hadamard " + hh2(eh4) + " U_f2 != 1",
            "ones-buffer bump, c computed from it"},
        {5, 0, "sum of head claims != ev",
            "O2 assembled with the WRONG mode's layout, ev honest from evil O2"},
    };
    if (perm == 0)
        evils.insert(evils.begin(), Evil{1, 0, "sum of head claims != ev",
            "O2 := M (no line-157 permutation), ev honest from evil O2"});
    else
        cout << "(evil=1 skipped: O2 := M IS the honest concat layout)" << endl;
    if (C_pad > C)
        evils.insert(evils.begin(), Evil{2, 0, "sum of head claims != ev",
            "junk in a padding column, ev honest from it"});
    else
        cout << "(evil=2 skipped: C == C_pad, no padding columns)" << endl;
    string evdir = "/tmp/zkob_headmerge_evil";
    mkdir(evdir.c_str(), 0755);
    for (auto& ev : evils) {
        prove(evdir, seed, outs, B, C, HD, perm, gen_big, gen_small, Q, "", ev.mode, ev.h);
        string reason;
        bool rejected = !verify(evdir, seed, B, C, HD, perm, gen_big, gen_small, Q, &reason);
        bool right = rejected && reason.find(ev.expect) != string::npos;
        cout << (right ? "PASS" : "FAIL") << ": evil=" << ev.mode << " (" << ev.what
             << ") rejected by [" << (rejected ? reason : string("NOT REJECTED"))
             << "], expected [" << ev.expect << "]" << endl;
        all = all && right;
    }

    // byte tampers on every proof file
    for (const string& fn : proof_files(NH)) {
        long off = tamper_offset(fn);
        tamper_byte(obdir + "/" + fn, off, +1);
        bool rejected = !verify(obdir, seed, B, C, HD, perm, gen_big, gen_small, Q);
        tamper_byte(obdir + "/" + fn, off, -1);
        cout << (rejected ? "PASS" : "FAIL") << ": byte tamper " << fn << "@" << off
             << " rejected: " << (rejected ? "YES" : "NO(!!)") << endl;
        all = all && rejected;
    }
    bool restored = verify(obdir, seed, B, C, HD, perm, gen_big, gen_small, Q);
    cout << (restored ? "PASS" : "FAIL") << ": restored verify "
         << (restored ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && restored;
    cout << (all ? "CASE PASS" : "CASE FAIL") << endl;
    return all;
}

static bool selftest_real(uint perm) {
    const uint B = 1024, C = 768, HD = 64, NH = C / HD;
    cout << "==== selftest real-scale case B=1024 C=768 HD=64 (NH=12) perm="
         << (perm ? "concat" : "pi157") << " ====" << endl;
    mt19937_64 rng(20260612);
    normal_distribution<double> nd(0.0, 262144.0);
    vector<vector<int>> outs(NH);
    for (uint h = 0; h < NH; h++) {
        outs[h].resize((size_t)B * HD);
        for (auto& x : outs[h]) x = (int)llround(nd(rng));
    }
    const string genbig_path = "/tmp/gen1024.bin", gensmall_path = "/tmp/gen64.bin";
    if (file_size(genbig_path) != 1024L * (long)sizeof(G1Jacobian_t)) {
        cout << "generating " << genbig_path << " via ppgen..." << endl;
        if (system(("./ppgen 1024 " + genbig_path).c_str()))
            throw runtime_error("ppgen failed");
    }
    if (file_size(gensmall_path) != 64L * (long)sizeof(G1Jacobian_t)) {
        cout << "generating " << gensmall_path << " via ppgen..." << endl;
        if (system(("./ppgen 64 " + gensmall_path).c_str()))
            throw runtime_error("ppgen failed");
    }
    Commitment gen_big(genbig_path), gen_small(gensmall_path);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_headmerge_real";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:merge:real";
    bool all = true;

    auto t0 = chrono::steady_clock::now();
    prove(obdir, seed, outs, B, C, HD, perm, gen_big, gen_small, Q, "/tmp/zkob_merge_real_O2.i32.bin");
    auto t1 = chrono::steady_clock::now();
    bool honest = verify(obdir, seed, B, C, HD, perm, gen_big, gen_small, Q);
    auto t2 = chrono::steady_clock::now();
    double prove_s = chrono::duration<double>(t1 - t0).count();
    double verify_s = chrono::duration<double>(t2 - t1).count();
    long bytes = 0;
    for (const string& fn : proof_files(NH)) bytes += file_size(obdir + "/" + fn);
    cout << (honest ? "PASS" : "FAIL") << ": real-scale honest "
         << (honest ? "ACCEPT" : "REJECT(!!)") << "  prove " << prove_s
         << " s, verify " << verify_s << " s, proof+commitments " << bytes
         << " bytes" << endl;
    all = all && honest;

    tamper_byte(obdir + "/hp07.bin", 36, +1);
    bool rejected = !verify(obdir, seed, B, C, HD, perm, gen_big, gen_small, Q);
    tamper_byte(obdir + "/hp07.bin", 36, -1);
    cout << (rejected ? "PASS" : "FAIL") << ": real-scale byte tamper rejected: "
         << (rejected ? "YES" : "NO(!!)") << endl;
    all = all && rejected;
    cout << (all ? "REAL-SCALE CASE: PASS" : "REAL-SCALE CASE: FAIL") << endl;
    return all;
}

// claim-mode selftest (Stage C): honest ACCEPT goes conditional-ACCEPT +
// batch ACCEPT over a TWO-domain batch (gen_small=HD + gen_big=C_pad);
// every forgery still rejects at a NAMED locus.
static bool selftest_case_claims(uint B, uint C, uint HD, uint perm) {
    const uint NH = C / HD, C_pad = 1u << ceilLog2(C);
    cout << "==== selftest (claim mode) B=" << B << " C=" << C << " HD=" << HD
         << " perm=" << (perm ? "concat" : "pi157") << " ====" << endl;
    srand(61337 + B * 100 + C * 10 + HD + perm);
    vector<vector<int>> outs(NH);
    for (uint h = 0; h < NH; h++) {
        outs[h].resize((size_t)B * HD);
        for (auto& x : outs[h]) x = rand() % 257 - 128;
    }
    string dir = "/tmp/zkob_headmerge_cm";
    { string c = "rm -rf " + dir; system(c.c_str()); }
    mkdir(dir.c_str(), 0755);
    string acc = dir + "/acc";
    mkdir(acc.c_str(), 0755);
    string run_seed = "selftest";
    string seed = "selftest:merge";
    string obid = "selftest.merge";

    map<uint32_t, string> genpaths;
    Commitment gen_big = Commitment::random(C_pad);
    Commitment gen_small = Commitment::random(HD);
    genpaths[C_pad] = dir + "/gen" + to_string(C_pad) + ".bin";
    genpaths[HD] = dir + "/gen" + to_string(HD) + ".bin";
    gen_big.save(genpaths[C_pad]);
    gen_small.save(genpaths[HD]);
    string qpath = dir + "/q.bin";
    Commitment::random(2).save(qpath);
    Commitment qg(qpath);
    G1Jacobian_t Q = qg(0);

    prove(dir, seed, outs, B, C, HD, perm, gen_big, gen_small, Q, "", 0, 0, acc, obid);
    batch_prove(acc, run_seed, genpaths, qpath);

    int vacc_n = 0;
    auto pipeline = [&](string& locus) -> bool {
        string vacc = dir + "/vacc" + to_string(vacc_n++);
        mkdir(vacc.c_str(), 0755);
        string reason;
        if (!verify(dir, seed, B, C, HD, perm, gen_big, gen_small, Q, &reason, vacc, obid)) {
            locus = "driver:" + reason; return false;
        }
        return batch_verify(acc, vacc, run_seed, genpaths, qpath, &locus);
    };
    int total = 0, fail = 0;
    auto expect = [&](const string& what, const string& want) {
        string locus;
        bool acc_ok = pipeline(locus);
        bool ok = (want == "accept") ? acc_ok
                                     : (!acc_ok && locus.find(want) != string::npos);
        total++; if (!ok) fail++;
        cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << what
             << " -> expected " << want << ", got " << (acc_ok ? "accept" : locus) << endl;
    };

    expect("honest (conditional ACCEPT + 2-domain batch ACCEPT)", "accept");

    // forgeries the DRIVER still catches (unchanged loci)
    tamper_byte(dir + "/hp01.bin", 36, +1);
    expect("hp01 round-0 eval tamper (driver round check)", "driver:");
    tamper_byte(dir + "/hp01.bin", 36, -1);
    tamper_byte(dir + "/ev.bin", 4, +1);
    expect("ev.bin tamper (driver sum/terminal via divergence)", "driver:");
    tamper_byte(dir + "/ev.bin", 4, -1);
    tamper_byte(dir + "/com_O2.bin", 24, +1);
    expect("com_O2 tamper (driver transcript divergence)", "driver:");
    tamper_byte(dir + "/com_O2.bin", 24, -1);

    // forgeries that now die in the BATCH
    tamper_byte(acc + "/claims.bin", -1, +1);
    expect("prover claims.bin eval tamper", "claims_match");
    tamper_byte(acc + "/claims.bin", -1, -1);
    {
        auto cs = claims_load(acc + "/claims.bin");
        auto omitted = vector<BoClaim>(cs.begin(), cs.end() - 1);
        claims_save(acc + "/claims.bin", omitted);
        expect("claim dropped from prover accumulator", "claims_match");
        claims_save(acc + "/claims.bin", cs);
    }
    tamper_byte(acc + "/ipa_batch_" + to_string(HD) + ".bin", -32, +1);
    expect("small-domain batched IPA a_final tamper", "ipa" + to_string(HD));
    tamper_byte(acc + "/ipa_batch_" + to_string(HD) + ".bin", -32, -1);
    tamper_byte(acc + "/ipa_batch_" + to_string(C_pad) + ".bin", -32, +1);
    expect("big-domain batched IPA a_final tamper", "ipa" + to_string(C_pad));
    tamper_byte(acc + "/ipa_batch_" + to_string(C_pad) + ".bin", -32, -1);

    expect("restored", "accept");

    // semantic evils — all die DRIVER-side, unchanged loci (none was an IPA)
    const uint eh3 = min(3u, NH - 2);
    struct EvilCM { int mode; uint h; string want; const char* what; };
    vector<EvilCM> evils = {
        {3, eh3, "driver:merge hadamard " + hh2(eh3) + " terminal",
            "Wm gathered with h off by one"},
        {4, min(7u, NH - 1), "driver:merge hadamard " + hh2(min(7u, NH - 1)) + " U_f2 != 1",
            "ones-buffer bump"},
        {5, 0, "driver:sum of head claims != ev", "wrong mode's layout"},
    };
    if (perm == 0)
        evils.push_back({1, 0, "driver:sum of head claims != ev", "O2 := M (no pi)"});
    for (auto& ev : evils) {
        string edir = dir + "/evil", eacc = dir + "/eacc", evacc = dir + "/evacc";
        { string c = "rm -rf " + edir + " " + eacc + " " + evacc; system(c.c_str()); }
        mkdir(edir.c_str(), 0755); mkdir(eacc.c_str(), 0755); mkdir(evacc.c_str(), 0755);
        prove(edir, seed, outs, B, C, HD, perm, gen_big, gen_small, Q, "", ev.mode, ev.h, eacc, obid);
        string reason;
        bool rej = !verify(edir, seed, B, C, HD, perm, gen_big, gen_small, Q, &reason, evacc, obid);
        string locus = "driver:" + reason;
        bool ok = rej && locus.find(ev.want) != string::npos;
        total++; if (!ok) fail++;
        cout << "  [" << (ok ? "PASS" : "FAIL") << "] evil=" << ev.mode << " (" << ev.what
             << ") -> expected " << ev.want << ", got "
             << (rej ? locus : string("accept")) << endl;
    }

    bool ok = fail == 0;
    cout << (ok ? "CASE PASS" : "CASE FAIL") << " (claim mode, " << (total - fail)
         << "/" << total << ")" << endl;
    return ok;
}

// cross-mode splice (design 4.2): an honest proof made in one mode presented
// to a verifier running the other mode must REJECT -- first at the dims.bin
// PERM cross-check, and, with dims.bin forged to match, at the "PERM" absorb
// (transcript divergence). Both directions, both layers.
static bool selftest_splice() {
    const uint B = 8, C = 6, HD = 2, NH = C / HD, C_pad = 8;
    cout << "==== selftest cross-mode splice B=" << B << " C=" << C
         << " HD=" << HD << " ====" << endl;
    srand(424242);
    vector<vector<int>> outs(NH);
    for (uint h = 0; h < NH; h++) {
        outs[h].resize((size_t)B * HD);
        for (auto& x : outs[h]) x = rand() % 257 - 128;
    }
    Commitment gen_big = Commitment::random(C_pad);
    Commitment gen_small = Commitment::random(HD);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string seed = "selftest:merge:splice";
    bool all = true;
    for (uint pv = 0; pv < 2; pv++) {           // pv = prover's mode
        const uint other = 1 - pv;
        string obdir = "/tmp/zkob_headmerge_splice" + to_string(pv);
        mkdir(obdir.c_str(), 0755);
        prove(obdir, seed, outs, B, C, HD, pv, gen_big, gen_small, Q, "");
        string reason;
        // layer 1: the dims.bin cross-check
        bool rej1 = !verify(obdir, seed, B, C, HD, other, gen_big, gen_small, Q, &reason);
        bool ok1 = rej1 && reason.find("dims.bin mismatch") != string::npos;
        cout << (ok1 ? "PASS" : "FAIL") << ": splice " << (pv ? "concat" : "pi157")
             << " proof vs " << (other ? "concat" : "pi157") << " verifier rejected by ["
             << (rej1 ? reason : string("NOT REJECTED")) << "], expected [dims.bin mismatch]"
             << endl;
        all = all && ok1;
        // layer 2: forge dims.bin's PERM field to match the verifier; the
        // "PERM" absorb must still diverge the transcript
        { FILE* f = open_or_die(obdir + "/dims.bin", "rb+");
          fseek(f, 12, SEEK_SET); uint32_t o = other;
          fwrite(&o, sizeof(uint32_t), 1, f); fclose(f); }
        reason.clear();
        bool rej2 = !verify(obdir, seed, B, C, HD, other, gen_big, gen_small, Q, &reason);
        bool ok2 = rej2 && reason.find("dims.bin") == string::npos;
        cout << (ok2 ? "PASS" : "FAIL") << ": splice with forged dims PERM rejected by ["
             << (rej2 ? reason : string("NOT REJECTED"))
             << "] (transcript divergence, not the dims guard)" << endl;
        all = all && ok2;
        // restore and re-verify honestly
        { FILE* f = open_or_die(obdir + "/dims.bin", "rb+");
          fseek(f, 12, SEEK_SET); uint32_t o = pv;
          fwrite(&o, sizeof(uint32_t), 1, f); fclose(f); }
        bool restored = verify(obdir, seed, B, C, HD, pv, gen_big, gen_small, Q);
        cout << (restored ? "PASS" : "FAIL") << ": restored splice dir verify "
             << (restored ? "ACCEPT" : "REJECT(!!)") << endl;
        all = all && restored;
    }
    cout << (all ? "SPLICE PASS" : "SPLICE FAIL") << endl;
    return all;
}

int main(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    // strip the optional claim-mode flag block: --claims <accdir> <obid>
    int base_argc = argc;
    string cm_a, cm_b;
    for (int i = 2; i < argc; i++)
        if (string(argv[i]) == "--claims") {
            base_argc = i;
            if (i + 1 < argc) cm_a = argv[i + 1];
            if (i + 2 < argc) cm_b = argv[i + 2];
            break;
        }
    if (mode == "selftest") {
        bo_probe_kernels();
        cout << "kernel -dlto probes: PASS" << endl;
        setenv("ZKOB_FOLD_CROSSCHECK", "1", 1);
        bool ok = true;
        for (uint perm = 0; perm < 2; perm++) {
            ok = selftest_case(8, 6, 2, perm) && ok;   // generic pi, padded grid
            ok = selftest_case(4, 4, 2, perm) && ok;   // B == C: pi = plain transpose
            ok = selftest_case(16, 12, 4, perm) && ok; // generic pi, padded 16x16 grid
            ok = selftest_case(16, 6, 2, perm) && ok;  // B != C_pad (audit MINOR-2): a
                                                       // B<->C_pad symbol confusion in
                                                       // the gather is non-vacuous here
        }
        ok = selftest_splice() && ok;
        ok = selftest_real(0) && ok;
        ok = selftest_real(1) && ok;
        for (uint perm = 0; perm < 2; perm++) {
            ok = selftest_case_claims(8, 6, 2, perm) && ok;
            ok = selftest_case_claims(16, 12, 4, perm) && ok;
        }
        cout << (ok ? "ZKOB-HEADMERGE SELFTEST: ALL PASS"
                    : "ZKOB-HEADMERGE SELFTEST: FAIL") << endl;
        return ok ? 0 : 1;
    }
    if (mode == "prove" && (base_argc == 12 || base_argc == 13)) {
        string obdir = argv[2], seed = argv[3], prefix = argv[4];
        uint B = stoi(argv[5]), C = stoi(argv[6]), HD = stoi(argv[7]);
        uint perm = parse_perm(argv[8]);
        if (HD == 0 || C % HD) throw runtime_error("HD must divide C");
        const uint NH = C / HD;
        vector<vector<int>> outs(NH);
        for (uint h = 0; h < NH; h++)
            outs[h] = load_i32_exact(prefix + hh2(h) + ".i32.bin", B * HD);
        Commitment gen_big(argv[9]), gen_small(argv[10]), qg(argv[11]);
        if (!cm_a.empty() && cm_b.empty())
            throw runtime_error("prove --claims needs <accdir> <obid>");
        prove(obdir, seed, outs, B, C, HD, perm, gen_big, gen_small, qg(0),
              base_argc == 13 ? argv[12] : "", 0, 0, cm_a, cm_b);
        return 0;
    }
    if (mode == "verify" && base_argc == 11) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[4]), C = stoi(argv[5]), HD = stoi(argv[6]);
        uint perm = parse_perm(argv[7]);
        Commitment gen_big(argv[8]), gen_small(argv[9]), qg(argv[10]);
        if (!cm_a.empty() && cm_b.empty())
            throw runtime_error("verify --claims needs <vaccdir> <obid>");
        return verify(obdir, seed, B, C, HD, perm, gen_big, gen_small, qg(0),
                      nullptr, cm_a, cm_b) ? 0 : 1;
    }
    cerr << "usage: zkob_headmerge selftest\n"
         << "       zkob_headmerge prove  <obdir> <seed> <Oh-prefix> <B> <C> <HD> <perm> <gen_big> <gen_small> <q> [O2-int32-out] [--claims <accdir> <obid>]\n"
         << "       zkob_headmerge verify <obdir> <seed> <B> <C> <HD> <perm> <gen_big> <gen_small> <q> [--claims <vaccdir> <obid>]\n"
         << "       <perm> = pi157 | concat" << endl;
    return 2;
}
