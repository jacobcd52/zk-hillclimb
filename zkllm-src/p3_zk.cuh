// P3.5 privacy building block: salted (hiding) Merkle commitment.
//
// The integrity Merkle hashes leaf_i = SHA256(8 bytes of value_i).  For small-domain
// data (int8 activations/weights) that leaf is brute-forceable -> the commitment and
// its paths leak values (the "guess-and-confirm" weakness).  Salting each leaf with a
// fresh 256-bit nonce makes the commitment hiding: the root and unopened sibling hashes
// reveal nothing, and an opened position discloses only (value, salt) for that index.
//
// This is the commitment-hiding layer.  Full zero-knowledge of the OPENINGS (query
// values + sumcheck messages + the evaluation claims) needs the additional masking /
// eval-claim-commitment machinery described in P3_PRIVACY_DESIGN.md.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include "p3_goldilocks.cuh"
#include "fs_transcript.hpp"

namespace p3zk {

typedef std::array<uint8_t, 32> Hash;
typedef std::array<uint8_t, 32> Salt;

static inline Hash leaf_hash_salt(gl_t v, const Salt& s) {
    uint8_t b[40];
    for (int k = 0; k < 8; k++) b[k] = (uint8_t)(v >> (8 * k));
    memcpy(b + 8, s.data(), 32);
    Hash h; fs::sha256(b, 40, h.data()); return h;
}
static inline Hash node_hash(const Hash& l, const Hash& r) {
    uint8_t b[64]; memcpy(b, l.data(), 32); memcpy(b + 32, r.data(), 32);
    Hash h; fs::sha256(b, 64, h.data()); return h;
}

struct SaltedMerkle {
    std::vector<std::vector<Hash>> levels;
    std::vector<Salt> salts;                       // one per leaf (secret)
    void build(const std::vector<gl_t>& cw, const std::vector<Salt>& salt_in) {
        salts = salt_in;
        std::vector<Hash> lv(cw.size());
        for (size_t i = 0; i < cw.size(); i++) lv[i] = leaf_hash_salt(cw[i], salts[i]);
        levels.assign(1, lv);
        while (levels.back().size() > 1) {
            const auto& cur = levels.back();
            std::vector<Hash> nxt(cur.size() / 2);
            for (size_t i = 0; i < nxt.size(); i++) nxt[i] = node_hash(cur[2*i], cur[2*i+1]);
            levels.push_back(nxt);
        }
    }
    Hash root() const { return levels.back()[0]; }
    std::vector<Hash> path(size_t idx) const {
        std::vector<Hash> p;
        for (size_t d = 0; d + 1 < levels.size(); d++) { p.push_back(levels[d][idx ^ 1]); idx >>= 1; }
        return p;
    }
};

// open reveals only (value, salt) for one position + the authentication path.
static inline bool verify_open(gl_t v, const Salt& s, size_t idx, const std::vector<Hash>& path, const Hash& root) {
    Hash h = leaf_hash_salt(v, s);
    for (const auto& sib : path) { h = (idx & 1) ? node_hash(sib, h) : node_hash(h, sib); idx >>= 1; }
    return h == root;
}

} // namespace p3zk
