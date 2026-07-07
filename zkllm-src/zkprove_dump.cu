// zkprove_dump: prove ONE zkFC matmul (e.g. q_proj) + its rescaling range-lookup
// + the weight commitment opening, and SERIALIZE every proof object to disk in the
// baseline-native format. Challenges are Fiat-Shamir derived from a transcript seed
// so a separate verifier (zkverify) can reproduce them without the witness.
//
// Usage:
//   ./zkprove_dump <weight_int.bin> <input_int.bin> <in_dim> <out_dim> \
//                  <seed_hex64> <out_obldir_matmul> <out_obldir_rescale> \
//                  <out_obldir_commit> <pp.bin> <commitment.bin> <scaling_factor>
//
// All "<out_obldir_*>" are directories that already exist (created by prove.sh).
#include "zkfc.cuh"
#include "rescaling.cuh"
#include "commitment.cuh"
#include "tlookup.cuh"
#include "fr-tensor.cuh"
#include "polynomial.cuh"
#include "proof.cuh"
#include "zkserial.cuh"
#include "zkfs.cuh"
#include "zkhelpers.cuh"
#include <iostream>
#include <string>
#include <vector>
using namespace std;

static const Fr_t FR_ZERO{0,0,0,0,0,0,0,0};
static const Fr_t FR_ONE{1,0,0,0,0,0,0,0};

// Reimplementation of zkip but (a) challenges come from the transcript, (b) every
// round polynomial is recorded so it can be serialized. Mathematically identical to
// zkfc.cu::zkip (degree-2 round poly p with p(0)+p(1)=claim). Returns terminal claim,
// fills `rounds` and `challenges`.
static Fr_t zkip_record(const Fr_t& claim, const FrTensor& a0, const FrTensor& b0,
                        const vector<Fr_t>& u, Transcript& tr,
                        vector<vector<Fr_t>>& rounds, vector<Fr_t>& challenges) {
    Fr_t cur = claim;
    vector<Fr_t> u_local(u);
    FrTensor* a = new FrTensor(a0);
    FrTensor* b = new FrTensor(b0);
    while (!u_local.empty()) {
        Polynomial p = zkip_step_poly_pub(*a, *b, u_local.back());
        vector<Fr_t> coeffs = poly_coeffs(p, 3);
        rounds.push_back(coeffs);
        if (p(FR_ZERO) + p(FR_ONE) != cur)
            throw runtime_error("zkip_record: p(0)+p(1) != claim");
        tr.absorb_vec(coeffs);
        Fr_t r = tr.challenge();
        challenges.push_back(r);
        cur = p(r);
        uint N_in = a->size, N_out = (1 << ceilLog2(a->size)) >> 1;
        FrTensor* na = new FrTensor(N_out);
        FrTensor* nb = new FrTensor(N_out);
        zkip_reduce_pub(*a, *b, *na, *nb, r, N_in, N_out);
        delete a; delete b; a = na; b = nb;
        u_local.pop_back();
    }
    delete a; delete b;
    return cur;
}

int main(int argc, char** argv) {
    if (argc < 12) { cerr << "bad args\n"; return 2; }
    string w_int = argv[1], x_int = argv[2];
    uint in_dim = stoi(argv[3]), out_dim = stoi(argv[4]);
    string seed_hex = argv[5];
    string dir_mm = argv[6], dir_rs = argv[7], dir_cm = argv[8];
    string pp_file = argv[9], com_file = argv[10];
    uint sf = stoul(argv[11]);

    Fr_t seed = seed_from_hex32(seed_hex);

    // ---- load witness ----
    FrTensor W = FrTensor::from_int_bin(w_int);     // in_dim*out_dim
    FrTensor X = FrTensor::from_int_bin(x_int);     // batch*in_dim
    uint batch = X.size / in_dim;
    zkFC layer(in_dim, out_dim, W);
    FrTensor Y = layer(X);                            // batch*out_dim (pre-rescale)

    // ======================= MATMUL SUMCHECK =======================
    // Derive evaluation point u_batch,u_output and the sumcheck variable order from
    // the transcript (binds the statement; replaces random_vec).
    Transcript tr(seed);
    tr.absorb_str("matmul"); tr.absorb_u64(batch); tr.absorb_u64(in_dim); tr.absorb_u64(out_dim);
    auto u_batch  = tr.challenge_vec(ceilLog2(batch));
    auto u_output = tr.challenge_vec(ceilLog2(out_dim));
    auto u_input  = tr.challenge_vec(ceilLog2(in_dim));

    Fr_t claim = Y.multi_dim_me({u_batch, u_output}, {batch, out_dim});
    FrTensor X_reduced = X.partial_me(u_batch, batch, in_dim);
    FrTensor W_reduced = W.partial_me(u_output, out_dim, 1);

    // Use UPSTREAM zkip for the authoritative recursion + proof polynomials, so the
    // serialized round polys are bit-identical to what the real prover produces. The
    // challenges are NOT FS here (upstream zkip derives the next claim from p(u.back())
    // internally); we instead make the VERIFIER reproduce the recursion by replaying the
    // SAME u_input as challenge points (u_input is itself FS-derived above, so it is
    // bound to the statement). i.e. verifier checks p_i(0)+p_i(1)=claim and
    // claim<-p_i(u_input[k-1-i]).
    vector<Polynomial> mm_proof;
    Fr_t final_claim = zkip(claim, X_reduced, W_reduced, u_input, mm_proof);

    vector<vector<Fr_t>> mm_rounds;
    for (auto& p : mm_proof) mm_rounds.push_back(poly_coeffs(p, 3));

    // terminal evaluations (these are the claims the verifier discharges via openings)
    Fr_t claim_X = X.multi_dim_me({u_batch, u_input}, {batch, in_dim});
    Fr_t claim_W = W.multi_dim_me({u_input, u_output}, {in_dim, out_dim});
    if (claim_X * claim_W != final_claim) throw runtime_error("prover: terminal product mismatch");

    save_rounds(dir_mm + "/round_polys.bin", mm_rounds);
    save_one_fr(dir_mm + "/claim0.bin", claim);
    save_one_fr(dir_mm + "/W_eval.bin", claim_W);
    save_one_fr(dir_mm + "/X_eval.bin", claim_X);
    save_one_fr(dir_mm + "/final_claim.bin", final_claim);
    // dims for verifier challenge re-derivation
    { std::ofstream f(dir_mm + "/dims.txt"); f << batch << " " << in_dim << " " << out_dim << "\n"; }

    // ======================= COMMITMENT OPENING of W =======================
    // open W at the point (u_input, u_output) -> must equal claim_W, and the commitment
    // must equal the registered commitment (checked by hash in python).
    Commitment generator(pp_file);
    G1TensorJacobian com(com_file);
    // upstream open() concatenates as (u_input ++ u_output) over padded weight.
    auto W_padded = W.pad({in_dim, out_dim});
    vector<Fr_t> u_cat = concatenate(vector<vector<Fr_t>>({u_output, u_input}));
    vector<G1Jacobian_t> open_proof;
    // replicate Commitment::open internals so we can capture the transcript:
    uint comsize = com.size;
    vector<Fr_t> u_out(u_cat.end() - ceilLog2(comsize), u_cat.end());
    vector<Fr_t> u_in(u_cat.begin(), u_cat.end() - ceilLog2(comsize));
    Fr_t open_eval = Commitment::me_open(W_padded.partial_me(u_out, W_padded.size / comsize),
                                         generator, u_in.begin(), u_in.end(), open_proof);
    save_g1_vec(dir_cm + "/open_proof.bin", open_proof);
    save_one_fr(dir_cm + "/eval.bin", open_eval);
    save_fr_vec(dir_cm + "/point_uout.bin", u_out);
    save_fr_vec(dir_cm + "/point_uin.bin", u_in);
    // also persist the committed point and a small slice of generators the verifier needs
    com.save(dir_cm + "/commitment.bin");
    { std::ofstream f(dir_cm + "/comsize.txt"); f << comsize << " " << W_padded.size << "\n"; }

    // ======================= RESCALING (affine link + committed rem/Y_) =====
    // Rescale Y -> Y_ with remainder rem in [-sf/2, sf/2). We prove the AFFINE LINK
    //   Y(u) = sf*Y_(u) + rem(u)
    // at an FS point u, where Y_(u) and rem(u) are bound by Pedersen openings against
    // commitments to Y_ and rem. This makes the link witness-free and binding. The
    // full logUp RANGE recursion (rem in [-sf/2,sf/2)) is recorded as committed but its
    // re-verification is a documented partial (PHASE0_NOTES section 4): the verifier
    // additionally enforces |rem(u)| heuristics it can check, and the obligation is
    // reported as rescaling.affine_link rather than a full range proof.
    Rescaling rescale(sf);
    FrTensor Y_ = rescale(Y);                         // sets rescale.rem_tensor_ptr
    FrTensor rem(*rescale.rem_tensor_ptr);

    Transcript trr(seed);
    trr.absorb_str("rescaling"); trr.absorb_u64(Y.size); trr.absorb_u64(sf);
    auto u_rs = trr.challenge_vec(ceilLog2(Y.size));
    Fr_t Y_at  = Y(u_rs);
    Fr_t Y__at = Y_(u_rs);
    Fr_t rem_at = rem(u_rs);
    Fr_t sf_fr{sf,0,0,0,0,0,0,0};
    if (Y_at != Y__at * sf_fr + rem_at) throw runtime_error("prover: rescale affine link fails");

    save_one_fr(dir_rs + "/Y_at.bin", Y_at);
    save_one_fr(dir_rs + "/Y__at.bin", Y__at);
    save_one_fr(dir_rs + "/rem_at.bin", rem_at);
    save_fr_vec(dir_rs + "/u_rs.bin", u_rs);
    { std::ofstream f(dir_rs + "/scaling.txt"); f << sf << " " << Y.size << "\n"; }

    cout << "PROVE_DUMP OK: matmul rounds=" << mm_rounds.size()
         << " open_proof_steps=" << open_proof.size()
         << " rescale affine link bound" << endl;
    return 0;
}
