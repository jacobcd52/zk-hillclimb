// Real driver for ONE per-head slicing obligation (layer{l}.attn.slice), per
// ROPE_ATTENTION_DESIGN.md (DESIGN FINAL 2026-06-10) section 4.3 / 5.2.
// Binds, per layer, all 3*NH per-head operand commitments (Qh, KhT, Vh per
// head) to the chained full-tensor commitments com_Q, com_K, com_V (re-committed
// here from the chain files; pinned upstream by edges A3v/A5):
//   Qh{hh}[t,d]  = Q[t, HD*h + d]    (B x HD,  committed with gen_small, B rows)
//   KhT{hh}[d,t] = K[t, HD*h + d]    (HD x B,  committed with the B-sized gens, HD rows)
//   Vh{hh}[t,d]  = V[t, HD*h + d]    (B x HD,  committed with gen_small, B rows)
// One FS challenge v of logHD + logB variables, v_d = v[0..logHD),
// v_t = v[logHD..). Per head, three claimed evaluations, each discharged by
// TWO IPA openings that must both verify against the SAME absorbed eval:
//   eQ_h: com_Qh{hh}  at (v_d || v_t)        and com_Q at (v_d || bits(h) || v_t)
//   eK_h: com_KhT{hh} at (v_t || v_d)        and com_K at (v_d || bits(h) || v_t)
//   eV_h: com_Vh{hh}  at (v_d || v_t)        and com_V at (v_d || bits(h) || v_t)
// bits(h) = the nhb = logC_pad - logHD LSB-first Boolean bits of h at point
// positions logHD..logC_pad (the head-selector column bits); the KhT point is
// the same (v_d, v_t) pair with the coordinate blocks swapped (KhT flat index
// = d*B + t) -- the transpose is a pure reordering of the opening point.
// Equality of each pair at the random v forces (Schwartz-Zippel, 16 vars at
// real scale) the slice tensor to equal the head-h column block of the full
// tensor as integer tensors. Zero prover advice; ZERO new CUDA kernels.
//
// FS schedule (design section 5.2; labels exact, {hh} = 2-digit decimal):
//   absorb B, C, HD; com_Q, com_K, com_V;
//   per hh: com_Qh{hh}, com_KhT{hh}, com_Vh{hh}
//   -> v (logHD + logB vars)
//   per hh: absorb eQ{hh}; IPA(Qh); IPA(Q);
//           absorb eK{hh}; IPA(KhT); IPA(K);
//           absorb eV{hh}; IPA(Vh); IPA(V)
//
// Files in <obdir>: dims.bin, com_Q/K/V.bin, com_Qh{hh}/com_KhT{hh}/com_Vh{hh}.bin,
//   evals.bin (3*NH Fr_t, raw: eQ00,eK00,eV00,eQ01,...),
//   ipa_Qh{hh}/ipa_Qf{hh}/ipa_Kh{hh}/ipa_Kf{hh}/ipa_Vh{hh}/ipa_Vf{hh}.bin
// [slice-out-dir]: prover-only witness files Qh{hh}.i32.bin (B x HD),
//   KhT{hh}.i32.bin (HD x B, transposed layout), Vh{hh}.i32.bin (B x HD) for
//   the downstream zkob_fc runs. The driver does NOT mkdir anything.
//
// Usage:
//   zkob_headslice prove   <obdir> <seed> <Q-int32.bin> <K-int32.bin> <V-int32.bin>
//                          <B> <C> <HD> <gen_big.bin> <gen_small.bin> <q.bin> [slice-out-dir]
//   zkob_headslice verify  <obdir> <seed> <B> <C> <HD> <gen_big.bin> <gen_small.bin> <q.bin>
//   zkob_headslice selftest
#include "zkob_lookup.cuh"
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <sys/stat.h>
using namespace std;

static_assert(sizeof(long) == 8, "host math needs 64-bit long");

static string hh2(uint h) { char b[8]; snprintf(b, sizeof b, "%02u", h); return string(b); }

// layout guards (design section 6, pinned)
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
    if (gen_big.size != B && gen_small.size != B)
        throw runtime_error("no generator set of size B for the KhT commitments");
}
// the KhT rows have width B: pick whichever provided gen set has size B
static const Commitment& gen_for_B(uint B, const Commitment& gen_big,
                                   const Commitment& gen_small) {
    if (gen_big.size == B) return gen_big;
    if (gen_small.size == B) return gen_small;
    throw runtime_error("no generator set of size B");
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

// full-tensor opening point (v_d || bits(h) || v_t), bits LSB-first
static vector<Fr_t> full_point(const vector<Fr_t>& v_d, const vector<Fr_t>& v_t,
                               uint h, uint nhb) {
    vector<Fr_t> pt(v_d);
    for (uint k = 0; k < nhb; k++) pt.push_back(((h >> k) & 1) ? F_ONE : F_ZERO);
    pt.insert(pt.end(), v_t.begin(), v_t.end());
    return pt;
}

// ---------------- prove ----------------
// evil (selftest only; the evil prover computes the claimed eval from its evil
// slice so the slice-side IPA passes; the full-tensor side must catch it).
// evil = the error FAMILY, evil_t = the target tensor (0=Q, 1=K, 2=V); across
// the three toy cases every family hits every tensor once (audit MINOR-4):
//  1: slice {1} filled from head (2 % NH)'s columns (wrong head, honest
//     layout) -> "IPA opening of e?01 vs com_? (head-selector point)".
//  2: slice {0} gathered with a one-column offset (cols 1..HD, honest layout)
//     -> "IPA opening of e?00 vs com_? (head-selector point)".
//  3: slice {0} = head 0's data in the WRONG layout (K: row-major B x HD bytes
//     reinterpreted as HD x B without transposing; Q/V: transposed when they
//     must not be) -> "IPA opening of e?00 vs com_? (head-selector point)".
//     (Honest-case pass + this family jointly pin the (v_t || v_d) swap.)
static void prove(const string& obdir, const string& seed,
                  const vector<int>& Qh_in, const vector<int>& Kh_in,
                  const vector<int>& Vh_in, uint B, uint C, uint HD,
                  const Commitment& gen_big, const Commitment& gen_small,
                  const G1Jacobian_t& Q, const string& slice_dir,
                  int evil = 0, int evil_t = 0) {
    layout_guards(B, C, HD, gen_big, gen_small);
    const uint C_pad = 1u << ceilLog2(C);
    const uint logB = ceilLog2(B), logCp = ceilLog2(C_pad), logHD = ceilLog2(HD);
    const uint nhb = logCp - logHD, NH = C / HD;
    const Commitment& genB = gen_for_B(B, gen_big, gen_small);
    if (Qh_in.size() != (size_t)B * C || Kh_in.size() != (size_t)B * C ||
        Vh_in.size() != (size_t)B * C)
        throw runtime_error("input dims");

    // ---- slice gathers (deterministic, no arithmetic) ----
    vector<vector<int>> Qs(NH), Ks(NH), Vs(NH);
    for (uint h = 0; h < NH; h++) {
        Qs[h].resize((size_t)B * HD); Ks[h].resize((size_t)B * HD);
        Vs[h].resize((size_t)B * HD);
        for (uint t = 0; t < B; t++)
            for (uint d = 0; d < HD; d++) {
                Qs[h][(size_t)t * HD + d] = Qh_in[(size_t)t * C + HD * h + d];
                Ks[h][(size_t)d * B + t] = Kh_in[(size_t)t * C + HD * h + d];   // transposed
                Vs[h][(size_t)t * HD + d] = Vh_in[(size_t)t * C + HD * h + d];
            }
    }
    if (evil) {             // family `evil` planted on tensor family `evil_t`
        vector<vector<int>>& S = (evil_t == 0 ? Qs : evil_t == 1 ? Ks : Vs);
        const vector<int>& full = (evil_t == 0 ? Qh_in : evil_t == 1 ? Kh_in : Vh_in);
        const bool tK = (evil_t == 1);  // K's HONEST layout is the transposed one
        if (evil == 1) {    // wrong head: slice {1} from head (2 % NH)'s columns
            uint src = 2 % NH;
            for (uint t = 0; t < B; t++)
                for (uint d = 0; d < HD; d++)
                    S[1][tK ? (size_t)d * B + t : (size_t)t * HD + d]
                        = full[(size_t)t * C + HD * src + d];
        }
        if (evil == 2) {    // slice {0} with the classic one-column offset
            for (uint t = 0; t < B; t++)
                for (uint d = 0; d < HD; d++)
                    S[0][tK ? (size_t)d * B + t : (size_t)t * HD + d]
                        = full[(size_t)t * C + d + 1];
        }
        if (evil == 3) {    // slice {0} = head 0's data in the WRONG layout
            for (uint t = 0; t < B; t++)
                for (uint d = 0; d < HD; d++)
                    S[0][tK ? (size_t)t * HD + d : (size_t)d * B + t]
                        = full[(size_t)t * C + d];
        }
    }
    if (!slice_dir.empty()) {     // prover-only witness files for the fc runs
        for (uint h = 0; h < NH; h++) {
            auto wr = [&](const string& name, const vector<int>& v) {
                FILE* f = open_or_die(slice_dir + "/" + name, "wb");
                fwrite(v.data(), sizeof(int), v.size(), f); fclose(f);
            };
            wr("Qh" + hh2(h) + ".i32.bin", Qs[h]);
            wr("KhT" + hh2(h) + ".i32.bin", Ks[h]);
            wr("Vh" + hh2(h) + ".i32.bin", Vs[h]);
        }
    }

    // ---- tensors + commitments ----
    FrTensor Qt((size_t)B * C, Qh_in.data()), Kt((size_t)B * C, Kh_in.data()),
             Vt((size_t)B * C, Vh_in.data());
    FrTensor Q_pad = Qt.pad({B, C}), K_pad = Kt.pad({B, C}), V_pad = Vt.pad({B, C});
    G1TensorJacobian com_Q = gen_big.commit(Q_pad);
    G1TensorJacobian com_K = gen_big.commit(K_pad);
    G1TensorJacobian com_V = gen_big.commit(V_pad);
    com_Q.save(obdir + "/com_Q.bin");
    com_K.save(obdir + "/com_K.bin");
    com_V.save(obdir + "/com_V.bin");
    vector<FrTensor> Qt_h, Kt_h, Vt_h;
    vector<G1TensorJacobian> com_Qh, com_Kh, com_Vh;
    for (uint h = 0; h < NH; h++) {
        Qt_h.emplace_back((uint)((size_t)B * HD), Qs[h].data());
        Kt_h.emplace_back((uint)((size_t)B * HD), Ks[h].data());
        Vt_h.emplace_back((uint)((size_t)B * HD), Vs[h].data());
        com_Qh.push_back(gen_small.commit(Qt_h[h]));
        com_Kh.push_back(genB.commit(Kt_h[h]));
        com_Vh.push_back(gen_small.commit(Vt_h[h]));
        com_Qh[h].save(obdir + "/com_Qh" + hh2(h) + ".bin");
        com_Kh[h].save(obdir + "/com_KhT" + hh2(h) + ".bin");
        com_Vh[h].save(obdir + "/com_Vh" + hh2(h) + ".bin");
    }
    { FILE* f = open_or_die(obdir + "/dims.bin", "wb");
      uint32_t d[3] = {B, C, HD}; fwrite(d, sizeof(uint32_t), 3, f); fclose(f); }

    // ---- transcript ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C); absorb_u32(tr, "HD", HD);
    absorb_g1_tensor(tr, "com_Q", com_Q);
    absorb_g1_tensor(tr, "com_K", com_K);
    absorb_g1_tensor(tr, "com_V", com_V);
    for (uint h = 0; h < NH; h++) {
        absorb_g1_tensor(tr, "com_Qh" + hh2(h), com_Qh[h]);
        absorb_g1_tensor(tr, "com_KhT" + hh2(h), com_Kh[h]);
        absorb_g1_tensor(tr, "com_Vh" + hh2(h), com_Vh[h]);
    }
    auto v = fs_challenge_vec(tr, logHD + logB);
    vector<Fr_t> v_d(v.begin(), v.begin() + logHD);
    vector<Fr_t> v_t(v.begin() + logHD, v.end());
    vector<Fr_t> vdt(v);                                   // (v_d || v_t)
    vector<Fr_t> vtd(v_t); vtd.insert(vtd.end(), v_d.begin(), v_d.end());  // (v_t || v_d)

    vector<Fr_t> evals;
    for (uint h = 0; h < NH; h++) {
        vector<Fr_t> fpt = full_point(v_d, v_t, h, nhb);
        vector<Fr_t> fpt_col(fpt.begin(), fpt.begin() + logCp);

        Fr_t eQ = Qt_h[h].multi_dim_me({v_t, v_d}, {B, HD});
        Fr_t eK = Kt_h[h].multi_dim_me({v_d, v_t}, {HD, B});
        Fr_t eV = Vt_h[h].multi_dim_me({v_t, v_d}, {B, HD});
        if (evil == 0) {    // convention sanity: slice MLE == full MLE at the
                            // head-selector point (both layouts, design 1.4)
            if (!fr_eq(eQ, Q_pad.multi_dim_me({v_t, fpt_col}, {B, C_pad})) ||
                !fr_eq(eK, K_pad.multi_dim_me({v_t, fpt_col}, {B, C_pad})) ||
                !fr_eq(eV, V_pad.multi_dim_me({v_t, fpt_col}, {B, C_pad})))
                throw runtime_error("slice MLE != full-tensor MLE (convention bug)");
        }
        absorb_fr(tr, "eQ" + hh2(h), eQ);
        open_prove(Qt_h[h], HD, gen_small, Q, vdt, obdir + "/ipa_Qh" + hh2(h) + ".bin", tr);
        open_prove(Q_pad, C_pad, gen_big, Q, fpt, obdir + "/ipa_Qf" + hh2(h) + ".bin", tr);
        absorb_fr(tr, "eK" + hh2(h), eK);
        open_prove(Kt_h[h], B, genB, Q, vtd, obdir + "/ipa_Kh" + hh2(h) + ".bin", tr);
        open_prove(K_pad, C_pad, gen_big, Q, fpt, obdir + "/ipa_Kf" + hh2(h) + ".bin", tr);
        absorb_fr(tr, "eV" + hh2(h), eV);
        open_prove(Vt_h[h], HD, gen_small, Q, vdt, obdir + "/ipa_Vh" + hh2(h) + ".bin", tr);
        open_prove(V_pad, C_pad, gen_big, Q, fpt, obdir + "/ipa_Vf" + hh2(h) + ".bin", tr);
        evals.push_back(eQ); evals.push_back(eK); evals.push_back(eV);
    }
    { FILE* f = open_or_die(obdir + "/evals.bin", "wb");
      fwrite(evals.data(), sizeof(Fr_t), evals.size(), f); fclose(f); }
    cout << "PROVED headslice obligation -> " << obdir << endl;
}

// ---------------- verify (witness-free) ----------------
#define RJ(msg) do { ostringstream oss_; oss_ << msg; \
    cout << "REJECT: " << oss_.str() << endl; \
    if (reason) *reason = oss_.str(); return false; } while (0)

static bool verify(const string& obdir, const string& seed,
                   uint B, uint C, uint HD,
                   const Commitment& gen_big, const Commitment& gen_small,
                   const G1Jacobian_t& Q, string* reason = nullptr) {
    const uint C_pad = 1u << ceilLog2(C);
    const uint logB = ceilLog2(B), logCp = ceilLog2(C_pad), logHD = ceilLog2(HD);
    const uint nhb = logCp - logHD, NH = C / HD;
    if (B != (1u << logB) || HD < 2 || HD != (1u << logHD) ||
        (C % HD) || (C_pad % HD) || NH < 2 ||
        gen_big.size != C_pad || gen_small.size != HD ||
        (gen_big.size != B && gen_small.size != B))
        RJ("bad layout params");
    const Commitment& genB = gen_for_B(B, gen_big, gen_small);
    { FILE* f = open_or_die(obdir + "/dims.bin", "rb");
      uint32_t d[3];
      if (fread(d, sizeof(uint32_t), 3, f) != 3) { fclose(f); RJ("dims.bin short read"); }
      fclose(f);
      if (d[0] != B || d[1] != C || d[2] != HD) RJ("dims.bin mismatch"); }

    G1TensorJacobian com_Q(obdir + "/com_Q.bin");
    G1TensorJacobian com_K(obdir + "/com_K.bin");
    G1TensorJacobian com_V(obdir + "/com_V.bin");
    if (com_Q.size != B || com_K.size != B || com_V.size != B)
        RJ("full-tensor commitment row counts");
    vector<G1TensorJacobian> com_Qh, com_Kh, com_Vh;
    for (uint h = 0; h < NH; h++) {
        com_Qh.emplace_back(obdir + "/com_Qh" + hh2(h) + ".bin");
        com_Kh.emplace_back(obdir + "/com_KhT" + hh2(h) + ".bin");
        com_Vh.emplace_back(obdir + "/com_Vh" + hh2(h) + ".bin");
        if (com_Qh[h].size != B || com_Kh[h].size != HD || com_Vh[h].size != B)
            RJ("slice commitment row counts (head " << hh2(h) << ")");
    }
    vector<Fr_t> evals(3 * (size_t)NH);
    { FILE* f = open_or_die(obdir + "/evals.bin", "rb");
      fseek(f, 0, SEEK_END);
      if (ftell(f) != (long)evals.size() * (long)sizeof(Fr_t))
          { fclose(f); RJ("evals.bin size"); }
      fseek(f, 0, SEEK_SET);
      if (fread(evals.data(), sizeof(Fr_t), evals.size(), f) != evals.size())
          { fclose(f); RJ("evals.bin short read"); }
      fclose(f); }

    // ---- transcript replay ----
    fs::Transcript tr(seed);
    absorb_u32(tr, "B", B); absorb_u32(tr, "C", C); absorb_u32(tr, "HD", HD);
    absorb_g1_tensor(tr, "com_Q", com_Q);
    absorb_g1_tensor(tr, "com_K", com_K);
    absorb_g1_tensor(tr, "com_V", com_V);
    for (uint h = 0; h < NH; h++) {
        absorb_g1_tensor(tr, "com_Qh" + hh2(h), com_Qh[h]);
        absorb_g1_tensor(tr, "com_KhT" + hh2(h), com_Kh[h]);
        absorb_g1_tensor(tr, "com_Vh" + hh2(h), com_Vh[h]);
    }
    auto v = fs_challenge_vec(tr, logHD + logB);
    vector<Fr_t> v_d(v.begin(), v.begin() + logHD);
    vector<Fr_t> v_t(v.begin() + logHD, v.end());
    vector<Fr_t> vdt(v);
    vector<Fr_t> vtd(v_t); vtd.insert(vtd.end(), v_d.begin(), v_d.end());

    // per head: each pair verifies BOTH IPAs against the SAME absorbed eval
    for (uint h = 0; h < NH; h++) {
        vector<Fr_t> fpt = full_point(v_d, v_t, h, nhb);
        const Fr_t eQ = evals[3 * h], eK = evals[3 * h + 1], eV = evals[3 * h + 2];
        absorb_fr(tr, "eQ" + hh2(h), eQ);
        if (!open_verify(com_Qh[h], gen_small, HD, Q, vdt, eQ,
                         obdir + "/ipa_Qh" + hh2(h) + ".bin", tr))
            RJ("IPA opening of eQ" << hh2(h) << " vs com_Qh" << hh2(h));
        if (!open_verify(com_Q, gen_big, C_pad, Q, fpt, eQ,
                         obdir + "/ipa_Qf" + hh2(h) + ".bin", tr))
            RJ("IPA opening of eQ" << hh2(h) << " vs com_Q (head-selector point)");
        absorb_fr(tr, "eK" + hh2(h), eK);
        if (!open_verify(com_Kh[h], genB, B, Q, vtd, eK,
                         obdir + "/ipa_Kh" + hh2(h) + ".bin", tr))
            RJ("IPA opening of eK" << hh2(h) << " vs com_KhT" << hh2(h));
        if (!open_verify(com_K, gen_big, C_pad, Q, fpt, eK,
                         obdir + "/ipa_Kf" + hh2(h) + ".bin", tr))
            RJ("IPA opening of eK" << hh2(h) << " vs com_K (head-selector point)");
        absorb_fr(tr, "eV" + hh2(h), eV);
        if (!open_verify(com_Vh[h], gen_small, HD, Q, vdt, eV,
                         obdir + "/ipa_Vh" + hh2(h) + ".bin", tr))
            RJ("IPA opening of eV" << hh2(h) << " vs com_Vh" << hh2(h));
        if (!open_verify(com_V, gen_big, C_pad, Q, fpt, eV,
                         obdir + "/ipa_Vf" + hh2(h) + ".bin", tr))
            RJ("IPA opening of eV" << hh2(h) << " vs com_V (head-selector point)");
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
    vector<string> fs = {"dims.bin", "com_Q.bin", "com_K.bin", "com_V.bin", "evals.bin"};
    for (uint h = 0; h < NH; h++) {
        fs.push_back("com_Qh" + hh2(h) + ".bin");
        fs.push_back("com_KhT" + hh2(h) + ".bin");
        fs.push_back("com_Vh" + hh2(h) + ".bin");
        fs.push_back("ipa_Qh" + hh2(h) + ".bin");
        fs.push_back("ipa_Qf" + hh2(h) + ".bin");
        fs.push_back("ipa_Kh" + hh2(h) + ".bin");
        fs.push_back("ipa_Kf" + hh2(h) + ".bin");
        fs.push_back("ipa_Vh" + hh2(h) + ".bin");
        fs.push_back("ipa_Vf" + hh2(h) + ".bin");
    }
    return fs;
}
static long tamper_offset(const string& f) {
    if (f == "dims.bin") return 0;
    if (f == "evals.bin") return 4;                 // raw Fr sequence
    if (f.substr(0, 4) == "ipa_") return -32;       // a_final
    return 24;                                      // com_*: first point, x limbs
}

// fam_t[k] = the tensor (0=Q, 1=K, 2=V) that error family k+1 targets in this
// case; the three toy cases use a Latin square so every family hits every
// tensor at least once across the selftest (audit MINOR-4)
static bool selftest_case(uint B, uint C, uint HD, const int fam_t[3]) {
    const uint NH = C / HD, C_pad = 1u << ceilLog2(C);
    cout << "==== selftest case B=" << B << " C=" << C << " HD=" << HD
         << " (NH=" << NH << ", C_pad=" << C_pad << ") ====" << endl;
    srand(777 + B * 100 + C * 10 + HD);
    vector<int> Qv((size_t)B * C), Kv((size_t)B * C), Vv((size_t)B * C);
    for (auto& x : Qv) x = rand() % 257 - 128;
    for (auto& x : Kv) x = rand() % 257 - 128;
    for (auto& x : Vv) x = rand() % 257 - 128;
    Commitment gen_big = Commitment::random(C_pad);
    Commitment gen_small = Commitment::random(HD);
    Commitment qg = Commitment::random(1);
    G1Jacobian_t Q = qg(0);
    string obdir = "/tmp/zkob_headslice_ob";
    mkdir(obdir.c_str(), 0755);
    string seed = "selftest:slice";
    bool all = true;

    prove(obdir, seed, Qv, Kv, Vv, B, C, HD, gen_big, gen_small, Q, "");
    bool honest = verify(obdir, seed, B, C, HD, gen_big, gen_small, Q);
    cout << (honest ? "PASS" : "FAIL") << ": honest case "
         << (honest ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && honest;

    // semantic evil modes (full-tensor side must catch each); family x tensor
    // per the case's fam_t map
    static const char* FAM_WHAT[3] = {
        "slice {1} filled from head (2 % NH)'s columns",
        "slice {0} gathered with a one-column offset",
        "slice {0} in the wrong (un/transposed) layout"};
    struct Evil { int mode; int tensor; string expect; string what; };
    vector<Evil> evils;
    for (int fam = 1; fam <= 3; fam++) {
        const int tt = fam_t[fam - 1];
        const char tn = "QKV"[tt];
        const char* idx = (fam == 1) ? "01" : "00";
        evils.push_back({fam, tt,
            string("IPA opening of e") + tn + idx + " vs com_" + tn
                + " (head-selector point)",
            string(FAM_WHAT[fam - 1]) + " [target " + tn + "]"});
    }
    string evdir = "/tmp/zkob_headslice_evil";
    mkdir(evdir.c_str(), 0755);
    for (auto& ev : evils) {
        prove(evdir, seed, Qv, Kv, Vv, B, C, HD, gen_big, gen_small, Q, "", ev.mode, ev.tensor);
        string reason;
        bool rejected = !verify(evdir, seed, B, C, HD, gen_big, gen_small, Q, &reason);
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
        bool rejected = !verify(obdir, seed, B, C, HD, gen_big, gen_small, Q);
        tamper_byte(obdir + "/" + fn, off, -1);
        cout << (rejected ? "PASS" : "FAIL") << ": byte tamper " << fn << "@" << off
             << " rejected: " << (rejected ? "YES" : "NO(!!)") << endl;
        all = all && rejected;
    }
    bool restored = verify(obdir, seed, B, C, HD, gen_big, gen_small, Q);
    cout << (restored ? "PASS" : "FAIL") << ": restored verify "
         << (restored ? "ACCEPT" : "REJECT(!!)") << endl;
    all = all && restored;
    cout << (all ? "CASE PASS" : "CASE FAIL") << endl;
    return all;
}

static bool selftest_real() {
    const uint B = 1024, C = 768, HD = 64, NH = C / HD;
    cout << "==== selftest real-scale case B=1024 C=768 HD=64 (NH=12) ====" << endl;
    mt19937_64 rng(20260611);
    normal_distribution<double> nd(0.0, 262144.0);
    vector<int> Qv((size_t)B * C), Kv((size_t)B * C), Vv((size_t)B * C);
    for (auto& x : Qv) x = (int)llround(nd(rng));
    for (auto& x : Kv) x = (int)llround(nd(rng));
    for (auto& x : Vv) x = (int)llround(nd(rng));
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
    string obdir = "/tmp/zkob_headslice_real";
    string sldir = "/tmp/zkob_headslice_data";
    mkdir(obdir.c_str(), 0755);
    mkdir(sldir.c_str(), 0755);
    string seed = "selftest:slice:real";
    bool all = true;

    auto t0 = chrono::steady_clock::now();
    prove(obdir, seed, Qv, Kv, Vv, B, C, HD, gen_big, gen_small, Q, sldir);
    auto t1 = chrono::steady_clock::now();
    bool honest = verify(obdir, seed, B, C, HD, gen_big, gen_small, Q);
    auto t2 = chrono::steady_clock::now();
    double prove_s = chrono::duration<double>(t1 - t0).count();
    double verify_s = chrono::duration<double>(t2 - t1).count();
    long bytes = 0;
    for (const string& fn : proof_files(NH)) bytes += file_size(obdir + "/" + fn);
    cout << (honest ? "PASS" : "FAIL") << ": real-scale honest "
         << (honest ? "ACCEPT" : "REJECT(!!)") << "  prove " << prove_s
         << " s, verify " << verify_s << " s, proof+commitments " << bytes
         << " bytes" << endl;
    cout << "GATE (design 9.1, ~30 s): prove " << prove_s << " s, verify "
         << verify_s << " s -> "
         << ((prove_s <= 30.0 && verify_s <= 30.0) ? "WITHIN GATE" : "EXCEEDS GATE")
         << endl;
    all = all && honest;

    // slice witness-file sanity: Qh05 file content == the head-5 column block
    {
        vector<int> qh = load_i32_exact(sldir + "/Qh05.i32.bin", B * HD);
        bool ok = true;
        for (uint t = 0; t < B && ok; t++)
            for (uint d = 0; d < HD && ok; d++)
                ok = (qh[(size_t)t * HD + d] == Qv[(size_t)t * C + HD * 5 + d]);
        vector<int> kh = load_i32_exact(sldir + "/KhT05.i32.bin", B * HD);
        for (uint t = 0; t < B && ok; t++)
            for (uint d = 0; d < HD && ok; d++)
                ok = (kh[(size_t)d * B + t] == Kv[(size_t)t * C + HD * 5 + d]);
        cout << (ok ? "PASS" : "FAIL") << ": slice witness files match the spec gathers" << endl;
        all = all && ok;
    }

    tamper_byte(obdir + "/ipa_Qf05.bin", -32, +1);
    bool rejected = !verify(obdir, seed, B, C, HD, gen_big, gen_small, Q);
    tamper_byte(obdir + "/ipa_Qf05.bin", -32, -1);
    cout << (rejected ? "PASS" : "FAIL") << ": real-scale byte tamper rejected: "
         << (rejected ? "YES" : "NO(!!)") << endl;
    all = all && rejected;
    cout << (all ? "REAL-SCALE CASE: PASS" : "REAL-SCALE CASE: FAIL") << endl;
    return all;
}

int main(int argc, char* argv[]) {
    vrf_selfcheck();
    string mode = argc > 1 ? argv[1] : "";
    if (mode == "selftest") {
        // family->tensor Latin square (audit MINOR-4): wrong-head hits Q,K,V;
        // column-offset hits V,Q,K; wrong-layout hits K,V,Q across the cases
        static const int fa[3] = {0, 2, 1};  // the original assignment
        static const int fb[3] = {1, 0, 2};
        static const int fc[3] = {2, 1, 0};
        bool a = selftest_case(8, 6, 2, fa);     // NH=3, padded heads present
        bool b = selftest_case(4, 8, 4, fb);     // NH=2, no padding
        bool c = selftest_case(16, 12, 4, fc);   // NH=3, padded heads present
        bool d = selftest_real();
        bool ok = a && b && c && d;
        cout << (ok ? "ZKOB-HEADSLICE SELFTEST: ALL PASS"
                    : "ZKOB-HEADSLICE SELFTEST: FAIL") << endl;
        return ok ? 0 : 1;
    }
    if (mode == "prove" && (argc == 13 || argc == 14)) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[7]), C = stoi(argv[8]), HD = stoi(argv[9]);
        vector<int> Qv = load_i32_exact(argv[4], B * C);
        vector<int> Kv = load_i32_exact(argv[5], B * C);
        vector<int> Vv = load_i32_exact(argv[6], B * C);
        Commitment gen_big(argv[10]), gen_small(argv[11]), qg(argv[12]);
        prove(obdir, seed, Qv, Kv, Vv, B, C, HD, gen_big, gen_small, qg(0),
              argc == 14 ? argv[13] : "");
        return 0;
    }
    if (mode == "verify" && argc == 10) {
        string obdir = argv[2], seed = argv[3];
        uint B = stoi(argv[4]), C = stoi(argv[5]), HD = stoi(argv[6]);
        Commitment gen_big(argv[7]), gen_small(argv[8]), qg(argv[9]);
        return verify(obdir, seed, B, C, HD, gen_big, gen_small, qg(0)) ? 0 : 1;
    }
    cerr << "usage: zkob_headslice selftest\n"
         << "       zkob_headslice prove  <obdir> <seed> <Q-int32> <K-int32> <V-int32> <B> <C> <HD> <gen_big> <gen_small> <q> [slice-out-dir]\n"
         << "       zkob_headslice verify <obdir> <seed> <B> <C> <HD> <gen_big> <gen_small> <q>" << endl;
    return 2;
}
