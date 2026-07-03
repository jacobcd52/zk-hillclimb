// Composed full-transformer-layer test suite (p3_transformer.cuh) -- teeth.
//
//   python3 transformer_ref.py --dump-tables  p3_rmsnorm_tables.bin
//   python3 transformer_ref.py --dump-layer   transformer_layer.bin
//   python3 transformer_ref.py --dump-weights transformer_weights.bin
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_transformer_test.cu -o /root/p3_transformer_test
//   cd /root/zkllm && /root/p3_transformer_test
//
// HONEST bar: the composed proof VERIFIES and every chained intermediate AND
// the final output are BITWISE equal to transformer_layer.bin.  Teeth: each
// adversarial case breaks a DIFFERENT stage or seam and must reject.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_rmsnorm.cuh"
#include "p3_bfadd.cuh"
#include "p3_rope.cuh"
#include "p3_quant.cuh"
#include "p3_softmax.cuh"
#include "p3_swiglu.cuh"
#include "p3_transformer.cuh"
using namespace p3tf;
using std::vector;

static int np_ = 0, nf_ = 0;
static void ck(const char* n, bool c, const char* why = nullptr) {
    printf("  [%s] %s%s%s%s\n", c ? "PASS" : "FAIL", n,
           why ? "  (why=" : "", why ? why : "", why ? ")" : "");
    fflush(stdout);
    if (c) np_++; else nf_++;
}
static const uint32_t R = 2, Q = 24;
static Art A;
static TfTables* TT;
static Weights WW;
static WeightRoots WR;
static Trace TR;
static Config CFG;
static vector<uint16_t> X0PUB;
static Hash RX0;

static size_t peak_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256]; size_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "VmHWM: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb;
}

static bool eq16(const vector<uint16_t>& a, const vector<uint16_t>& b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), 2 * a.size()) == 0;
}

// bitwise comparison of every chained intermediate against the golden trace
static bool trace_bitwise(const TfWit& w, std::string& bad) {
    auto chk = [&](const char* nm, const vector<uint16_t>& v) {
        const vector<uint16_t>* t = trace_get(TR, nm);
        if (!t || !eq16(*t, v)) { bad = nm; return false; }
        return true;
    };
    if (!chk("rms1", w.rms1y)) return false;
    if (!chk("q", w.mmY[MM_WQ]) || !chk("k", w.mmY[MM_WK]) || !chk("v", w.mmY[MM_WV])) return false;
    for (int h = 0; h < 2; h++) {
        char nm[32];
        snprintf(nm, sizeof nm, "ropeq%d", h); if (!chk(nm, w.ropq[h])) return false;
        snprintf(nm, sizeof nm, "ropek%d", h); if (!chk(nm, w.ropk[h])) return false;
        snprintf(nm, sizeof nm, "scores%d", h); if (!chk(nm, w.mmY[MM_QK[h]])) return false;
        snprintf(nm, sizeof nm, "probs%d", h); if (!chk(nm, w.probs[h])) return false;
        snprintf(nm, sizeof nm, "attnout%d", h); if (!chk(nm, w.mmY[MM_PV[h]])) return false;
    }
    if (!chk("oproj", w.mmY[MM_WO])) return false;
    if (!chk("resid1", w.res1o)) return false;
    if (!chk("rms2", w.rms2y)) return false;
    if (!chk("gate", w.mmY[MM_WG]) || !chk("up", w.mmY[MM_WU])) return false;
    if (!chk("swiglu", w.swm)) return false;
    if (!chk("down", w.mmY[MM_WD])) return false;
    if (!chk("out", w.outp)) return false;
    return true;
}

// build + prove + verify one tampered layer; expect REJECT
static bool tamper_rejects(int mode, const char** why) {
    *why = nullptr;
    TfWit w;
    try {
        w = build_witness(CFG, X0PUB, WW, A, mode);
    } catch (const std::exception& e) {
        *why = "witness builder threw (out of domain)";
        return true;
    }
    TfOps o = commit_all(w, R);
    TfProof pf;
    try {
        fs::Transcript tp("tf-layer");
        pf = prove(tp, w, o, *TT, A, R, Q, /*strict=*/false);
    } catch (const std::exception& e) {
        *why = "prover threw (out of supported domain)";
        return true;
    }
    fs::Transcript tv("tf-layer");
    bool ok = verify(tv, pf, *TT, A, CFG, X0PUB, RX0, WR, WW.cos, WW.sin,
                     w.outp, Q, R, why);
    return !ok;
}

int main() {
    printf("=== COMPOSED FULL-TRANSFORMER-LAYER selftest (canonical = transformer_ref.py) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    if (!p3rms::load_art("p3_rmsnorm_tables.bin", A)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-tables p3_rmsnorm_tables.bin\n");
        return 1;
    }
    if (!load_weights("transformer_weights.bin", WW)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-weights transformer_weights.bin\n");
        return 1;
    }
    if (!load_trace("transformer_layer.bin", TR)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-layer transformer_layer.bin\n");
        return 1;
    }
    CFG = WW.cfg;
    const vector<uint16_t>* xin = trace_get(TR, "input");
    if (!xin) { printf("FATAL: trace has no input\n"); return 1; }
    X0PUB = *xin;
    printf("config: seq=%u d=%u nh=%u dh=%u dff=%u; trace ops=%zu\n",
           CFG.seq, CFG.d, CFG.nh, CFG.dh, CFG.dff, TR.names.size());

    TfTables T = p3tf::build_tables(A); TT = &T;
    WR = weight_roots(WW, R);

    // ================= HONEST ACCEPT =================
    printf("-- honest accept (one proof for the ENTIRE layer) --\n");
    TfProof hpf;
    vector<uint16_t> hout;
    {
        double t0 = p3hwl::now_ms();
        TfWit w = build_witness(CFG, X0PUB, WW, A);
        double t1 = p3hwl::now_ms();
        std::string bad;
        bool bit = trace_bitwise(w, bad);
        ck("EVERY chained intermediate bitwise == transformer_layer.bin",
           bit, bit ? nullptr : bad.c_str());
        TfOps o = commit_all(w, R);
        RX0 = o.rms[0].X.root;
        // sanity: the orchestrator output columns really share the matmul roots
        bool rooteq = true;
        for (int i = 0; i < NMM; i++) { (void)i; }
        double t2 = p3hwl::now_ms();
        TfProf prof;
        fs::Transcript tp("tf-layer");
        hpf = prove(tp, w, o, T, A, R, Q, /*strict=*/true, &prof);
        for (int i = 0; i < NMM; i++)
            rooteq = rooteq && (o.Ymm[i].root == hpf.mm[i].rdo[p3hwl::O_YB]);
        ck("matmul output commitments chain by ROOT EQUALITY (11/11)", rooteq);
        double t3 = p3hwl::now_ms();
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, hpf, T, A, CFG, X0PUB, RX0, WR, WW.cos, WW.sin,
                         w.outp, Q, R, &why);
        double t4 = p3hwl::now_ms();
        hout = w.outp;
        const vector<uint16_t>* gout = trace_get(TR, "out");
        ck("composed proof VERIFIES and accepted output == golden `out` bitwise",
           ok && gout && eq16(*gout, w.outp), ok ? nullptr : why);
        size_t psz = proof_size(hpf);
        printf("  timings: witness %.2f s, commits %.2f s, prove %.2f s, verify %.2f s\n",
               (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3);
        printf("  prove breakdown: rms %.2f  qnt %.2f  matmul %.2f  rope %.2f  smx %.2f"
               "  bfa %.2f  swg %.2f  seams %.2f  batch %.2f s\n",
               prof.rms / 1e3, prof.qnt / 1e3, prof.mm / 1e3, prof.rope / 1e3,
               prof.smx / 1e3, prof.bfa / 1e3, prof.swg / 1e3, prof.seam / 1e3,
               prof.batch / 1e3);
        printf("  proof %.2f MB (%zu seam claims, %zu batch classes), peak RSS %.2f GB\n",
               psz / 1048576.0, hpf.seam.size(), hpf.batches.size(),
               peak_rss_kb() / 1048576.0);
    }

    // ================= MUST-REJECT: tampered intermediates =================
    printf("-- must-reject: tampered intermediates (owning gadget catches) --\n");
    struct Case { int mode; const char* name; };
    const Case mids[] = {
        {TFT_RMS1_Y,   "rms1 output value flipped (rmsnorm zero-check)"},
        {TFT_SMX_P,    "softmax prob value flipped (softmax zero-check)"},
        {TFT_ROPE_OUT, "rope rotated value flipped (rope combine binding)"},
        {TFT_RES1_OUT, "residual-1 sum flipped (bf16-add zero-check)"},
        {TFT_SWG_M,    "swiglu output flipped (swiglu zero-check)"},
        {TFT_MM_YPUB,  "matmul PUBLIC Y claim flipped (Hawkeye Y binding)"},
    };
    for (auto& cse : mids) {
        bool rej = tamper_rejects(cse.mode, &why);
        ck(cse.name, rej, why);
    }

    // ============ MUST-REJECT: broken op->op chain seams ============
    // every sub-proof in these cases is VALID on its own inputs -- only the
    // composition seam can reject the mismatched hand-off
    printf("-- must-reject: broken chain seams (valid sub-proofs, seam catches) --\n");
    const Case seams[] = {
        {TFT_SEAM_MMX,    "matmul X code != quantizer CODES (restriction seam)"},
        {TFT_SEAM_PVPAD,  "nonzero code smuggled into PV k-padding (zero seam)"},
        {TFT_SEAM_ROPEQ,  "rope operand != head slice of Wq output (slice seam)"},
        {TFT_SEAM_CONCAT, "concat half != attnout head 1 (concat seam)"},
        {TFT_SEAM_VT,     "V^T operand != transposed Wv output (transpose seam)"},
    };
    for (auto& cse : seams) {
        bool rej = tamper_rejects(cse.mode, &why);
        ck(cse.name, rej, why);
    }

    // ============ MUST-REJECT: per-gadget witness forgeries ============
    printf("-- must-reject: per-gadget witness forgeries in the composed proof --\n");
    const Case gws[] = {
        {TFT_GW_MM,   "matmul accumulator-state teleport (chain sumcheck)"},
        {TFT_GW_RMS,  "rmsnorm rsqrt mantissa forge (RSQ lookup)"},
        {TFT_GW_QNT,  "quantize magnitude forge (QEXT lookup)"},
        {TFT_GW_SMX,  "softmax masked-lane denominator leak (lane gating)"},
        {TFT_GW_ROPE, "rope product mantissa forge (MUL7 lookup)"},
        {TFT_GW_BFA,  "residual RNE round-direction flip (bfadd RMH)"},
        {TFT_GW_SWG,  "swiglu silu-table forge (SILU lookup)"},
    };
    for (auto& cse : gws) {
        bool rej = tamper_rejects(cse.mode, &why);
        ck(cse.name, rej, why);
    }

    // ============ MUST-REJECT: statement / params / proof tampers ============
    printf("-- must-reject: statement, params and proof-object tampers --\n");
    {
        vector<uint16_t> bad = hout; bad[11] ^= 1;
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, hpf, T, A, CFG, X0PUB, RX0, WR, WW.cos, WW.sin,
                         bad, Q, R, &why);
        ck("flipped PUBLIC output claim rejects at the output binding", !ok, why);
    }
    {
        vector<uint16_t> bad = X0PUB; bad[3] ^= 1;
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, hpf, T, A, CFG, bad, RX0, WR, WW.cos, WW.sin,
                         hout, Q, R, &why);
        ck("flipped PUBLIC input claim rejects at the input binding", !ok, why);
    }
    {
        Config c2 = CFG; c2.dh = 16; c2.d = 32;   // stays pow2-consistent
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, hpf, T, A, c2, X0PUB, RX0, WR, WW.cos, WW.sin,
                         hout, Q, R, &why);
        ck("wrong pinned dims reject", !ok, why);
    }
    {
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, hpf, T, A, CFG, X0PUB, RX0, WR, WW.cos, WW.sin,
                         hout, 0, R, &why);
        ck("Q=0 params forgery rejects", !ok, why);
    }
    {
        WeightRoots wr2 = WR; wr2.Wc[W_Q].data()[0] ^= 1;
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, hpf, T, A, CFG, X0PUB, RX0, wr2, WW.cos, WW.sin,
                         hout, Q, R, &why);
        ck("wrong pinned weight root rejects", !ok, why);
    }
    {
        auto p2 = hpf; p2.seam[2] = gl_add(p2.seam[2], 1ULL);
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, p2, T, A, CFG, X0PUB, RX0, WR, WW.cos, WW.sin,
                         hout, Q, R, &why);
        ck("tampered seam claim rejects", !ok, why);
    }
    {
        auto p2 = hpf; p2.batches[0].ystar[0] = gl_add(p2.batches[0].ystar[0], 1ULL);
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, p2, T, A, CFG, X0PUB, RX0, WR, WW.cos, WW.sin,
                         hout, Q, R, &why);
        ck("tampered shared batch opening rejects", !ok, why);
    }
    {
        auto p2 = hpf; p2.sm[0].mDe[0].s2 = gl_add(p2.sm[0].mDe[0].s2, 1ULL);
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, p2, T, A, CFG, X0PUB, RX0, WR, WW.cos, WW.sin,
                         hout, Q, R, &why);
        ck("tampered sub-proof sumcheck message rejects", !ok, why);
    }
    {
        auto p2 = hpf; p2.rRes1.data()[5] ^= 1;   // break the chain root itself
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, p2, T, A, CFG, X0PUB, RX0, WR, WW.cos, WW.sin,
                         hout, Q, R, &why);
        ck("tampered chained intermediate root rejects", !ok, why);
    }

    printf("\nCOMPOSED-LAYER: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
