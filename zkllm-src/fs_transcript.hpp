// COORDINATOR-BUILT (do not let submission agents edit).
// Minimal dependency-free Fiat-Shamir transcript over SHA-256.
//
// state_0 = SHA256(seed)
// absorb(label, data):  state = SHA256(state || len32(label) || label || len64(data) || data)
// challenge_bytes():    out   = SHA256(state || "chal"); state = SHA256(state || "rtch")
//
// All challenges MUST be derived only after absorbing the messages they are
// supposed to bind (per sumcheck round / per IPA round). Deriving a round's
// challenge before its round message is absorbed makes the round forgeable
// (see PHASE0_NOTES §10: the me_open steering attack).
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace fs {

// ---- compact SHA-256 (FIPS 180-4) ----
struct Sha256Ctx {
    uint32_t h[8];
    uint64_t len;        // total bytes absorbed
    uint8_t buf[64];
    size_t buflen;
};

inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void sha256_init(Sha256Ctx& c) {
    static const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                   0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(c.h, iv, sizeof(iv)); c.len = 0; c.buflen = 0;
}

inline void sha256_block(Sha256Ctx& c, const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t(p[4*i]) << 24) | (uint32_t(p[4*i+1]) << 16) |
               (uint32_t(p[4*i+2]) << 8) | uint32_t(p[4*i+3]);
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ror(w[i-15],7) ^ ror(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = ror(w[i-2],17) ^ ror(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=c.h[0],b=c.h[1],cc=c.h[2],d=c.h[3],e=c.h[4],f=c.h[5],g=c.h[6],h=c.h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22);
        uint32_t mj = (a & b) ^ (a & cc) ^ (b & cc);
        uint32_t t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=cc; cc=b; b=a; a=t1+t2;
    }
    c.h[0]+=a; c.h[1]+=b; c.h[2]+=cc; c.h[3]+=d; c.h[4]+=e; c.h[5]+=f; c.h[6]+=g; c.h[7]+=h;
}

inline void sha256_update(Sha256Ctx& c, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    c.len += n;
    while (n) {
        size_t take = 64 - c.buflen; if (take > n) take = n;
        memcpy(c.buf + c.buflen, p, take);
        c.buflen += take; p += take; n -= take;
        if (c.buflen == 64) { sha256_block(c, c.buf); c.buflen = 0; }
    }
}

inline void sha256_final(Sha256Ctx& c, uint8_t out[32]) {
    uint64_t bitlen = c.len * 8;
    uint8_t pad = 0x80;
    sha256_update(c, &pad, 1);
    uint8_t z = 0;
    while (c.buflen != 56) sha256_update(c, &z, 1);
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) lenb[i] = uint8_t(bitlen >> (56 - 8*i));
    sha256_update(c, lenb, 8);
    for (int i = 0; i < 8; i++) {
        out[4*i]   = uint8_t(c.h[i] >> 24); out[4*i+1] = uint8_t(c.h[i] >> 16);
        out[4*i+2] = uint8_t(c.h[i] >> 8);  out[4*i+3] = uint8_t(c.h[i]);
    }
}

inline void sha256(const void* data, size_t n, uint8_t out[32]) {
    Sha256Ctx c; sha256_init(c); sha256_update(c, data, n); sha256_final(c, out);
}

// ---- challenge tape (hiding-battery experimental control; nullptr = normal FS) ----
// RECORD: challenges computed normally and logged.  REPLAY: challenges are read
// back from the tape in global draw order, so a prover re-run with different
// masks/blinds/salts sees the SAME public challenges (the fixed-challenge
// setting the chi-square battery requires; such transcripts are for
// distribution analysis only and are not FS-verifiable).
struct Tape {
    std::vector<std::array<uint8_t, 32>> ch;
    size_t pos = 0;
    bool record = false;
};
static Tape* g_tape = nullptr;

// ---- transcript ----
struct Transcript {
    uint8_t state[32];

    explicit Transcript(const std::string& seed) {
        sha256(seed.data(), seed.size(), state);
    }
    void absorb(const std::string& label, const void* data, size_t len) {
        Sha256Ctx c; sha256_init(c);
        sha256_update(c, state, 32);
        uint32_t ll = (uint32_t)label.size();
        uint8_t llb[4] = {uint8_t(ll), uint8_t(ll>>8), uint8_t(ll>>16), uint8_t(ll>>24)};
        sha256_update(c, llb, 4);
        sha256_update(c, label.data(), label.size());
        uint64_t dl = (uint64_t)len;
        uint8_t dlb[8]; for (int i = 0; i < 8; i++) dlb[i] = uint8_t(dl >> (8*i));
        sha256_update(c, dlb, 8);
        sha256_update(c, data, len);
        sha256_final(c, state);
        ablog(label.c_str(), len);
    }
    // env-gated absorb trace (P3_ABLOG=path): label, length and the running
    // state prefix -- diffing the prover and verifier halves pinpoints the
    // first diverging absorb.  Debug-only; no protocol effect.
    void ablog(const char* label, size_t len) {
        static FILE* f = [] {
            const char* p = getenv("P3_ABLOG");
            return p ? fopen(p, "w") : nullptr;
        }();
        if (!f) return;
        fprintf(f, "%s %zu %02x%02x%02x%02x%02x%02x%02x%02x\n", label, len,
                state[0], state[1], state[2], state[3],
                state[4], state[5], state[6], state[7]);
        fflush(f);
    }
    void challenge_bytes(uint8_t out[32]) {
        if (g_tape && !g_tape->record) {
            if (g_tape->pos >= g_tape->ch.size()) { memset(out, 0, 32); return; }
            memcpy(out, g_tape->ch[g_tape->pos++].data(), 32);
            return;
        }
        { Sha256Ctx c; sha256_init(c); sha256_update(c, state, 32);
          sha256_update(c, "chal", 4); sha256_final(c, out); }
        { Sha256Ctx c; sha256_init(c); sha256_update(c, state, 32);
          uint8_t tmp[32];
          sha256_update(c, "rtch", 4); sha256_final(c, tmp);
          memcpy(state, tmp, 32); }
        if (g_tape && g_tape->record) {
            std::array<uint8_t, 32> a; memcpy(a.data(), out, 32);
            g_tape->ch.push_back(a);
        }
    }
};

} // namespace fs
