#ifndef ZKFS_CUH
#define ZKFS_CUH
// Minimal Fiat-Shamir transcript in the BLS12-381 scalar field, shared bit-for-bit
// by prover (zkprove_dump) and verifier (zkverify). NOT a production-grade hash;
// its job here is reproducibility + sensitivity: any change to absorbed bytes
// (tampered round polys, swapped components, replayed seed) changes derived
// challenges, so the sumcheck recursion check then fails. That is exactly what
// catches forgeries C2/C5/C6 and statement re-binding.
#include "fr-tensor.cuh"
#include "polynomial.cuh"
#include <vector>
#include <string>
#include <cstdint>

using std::vector;
using std::string;

struct Transcript {
    Fr_t state;
    // A fixed odd multiplier (arbitrary nothing-up-my-sleeve constant).
    static Fr_t MULT() { return {0x9e3779b9u, 0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu,
                                 0x165667b1u, 0xd3a2646cu, 0xfd7046c5u, 0x6a09e667u}; }
    static Fr_t ADD()  { return {0x2545f491u, 0x14057b7eu, 0x000000ffu, 0u, 0u, 0u, 0u, 0u}; }

    Transcript() : state({1,0,0,0,0,0,0,0}) {}
    explicit Transcript(const Fr_t& seed) : state(seed) {}

    // absorb one field element
    void absorb(const Fr_t& x) {
        state = state * MULT() + x + ADD();
    }
    void absorb_vec(const vector<Fr_t>& xs) { for (auto& x : xs) absorb(x); }
    // absorb an integer (e.g. component index, dims) deterministically
    void absorb_u64(uint64_t v) {
        Fr_t x{ (uint32_t)(v & 0xffffffffu), (uint32_t)(v >> 32), 0,0,0,0,0,0 };
        absorb(x);
    }
    // absorb a string by bytes (component id, model id, etc.)
    void absorb_str(const string& s) {
        uint64_t acc = 1469598103934665603ull; // FNV offset
        for (unsigned char c : s) { acc ^= c; acc *= 1099511628211ull; }
        absorb_u64(acc);
    }

    // squeeze one challenge (and ratchet the state)
    Fr_t challenge() {
        Fr_t out = state * MULT() + ADD();
        state = out * MULT();   // ratchet so successive challenges differ & bind order
        return out;
    }
    vector<Fr_t> challenge_vec(uint n) {
        vector<Fr_t> v;
        for (uint i = 0; i < n; ++i) v.push_back(challenge());
        return v;
    }
};

// Build the root transcript seed from the public statement bytes. The verifier and
// prover both call this on the SAME bytes (we pass the seed in as a field element
// derived in python from sha256(public.json) -> first 31 bytes, see prove/verify).
inline Fr_t seed_from_hex32(const string& hex64) {
    // hex64 = 64 hex chars (32 bytes). Pack little-endian into 8 uint32 words,
    // but mask the top word to stay < field modulus by zeroing the high byte.
    Fr_t s{0,0,0,0,0,0,0,0};
    for (int word = 0; word < 8; ++word) {
        uint32_t w = 0;
        for (int b = 0; b < 4; ++b) {
            int idx = (word * 4 + b) * 2;
            auto hv = [](char c) -> uint32_t {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return 0;
            };
            uint32_t byte = (hv(hex64[idx]) << 4) | hv(hex64[idx + 1]);
            w |= byte << (8 * b);
        }
        s.val[word] = w;
    }
    s.val[7] &= 0x3fffffffu; // keep below modulus (top bits)
    return s;
}

#endif
