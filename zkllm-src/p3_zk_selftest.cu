// P3.5 selftest: salted (hiding) Merkle commitment.
//   ./p3_zk_selftest
// Demonstrates: (1) opened path verifies; (2) tamper value/salt/path rejects (binding);
// (3) fresh salts -> different root for identical data (hiding); (4) an int8 leaf is
// brute-forceable WITHOUT salt but not WITH salt (the concrete reason salting matters).
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_zk.cuh"
#include "fs_transcript.hpp"
using namespace p3zk;

static uint64_t rs_=42; static uint64_t rng(){ rs_=rs_*6364136223846793005ULL+1; uint64_t z=rs_; z^=z>>31; return z; }
static Salt rand_salt(){ Salt s; for(int i=0;i<32;i++) s[i]=(uint8_t)rng(); return s; }

static int npass=0,nfail=0; static void check(const char* n,bool c){ printf("  [%s] %s\n",c?"PASS":"FAIL",n); if(c)npass++; else nfail++; }

int main(){
    printf("=== P3.5  salted hiding Merkle selftest ===\n");
    uint32_t N=1024;
    std::vector<gl_t> data(N); for(auto&x:data) x=rng()%256;          // int8-domain values
    std::vector<Salt> salts(N); for(auto&s:salts) s=rand_salt();
    SaltedMerkle mk; mk.build(data, salts);
    Hash root=mk.root();

    // (1) honest open
    size_t idx=137;
    check("salted path open verifies", verify_open(data[idx], salts[idx], idx, mk.path(idx), root));

    // (2) binding: tamper value / salt / path
    check("tamper value -> reject", !verify_open(gl_add(data[idx],1), salts[idx], idx, mk.path(idx), root));
    { Salt s2=salts[idx]; s2[0]^=1; check("tamper salt -> reject", !verify_open(data[idx], s2, idx, mk.path(idx), root)); }
    { auto p=mk.path(idx); p[0][0]^=1; check("tamper path -> reject", !verify_open(data[idx], salts[idx], idx, p, root)); }

    // (3) hiding: identical data, fresh salts -> different root
    { std::vector<Salt> s2(N); for(auto&s:s2) s=rand_salt(); SaltedMerkle mk2; mk2.build(data,s2);
      check("fresh salts -> different root (hiding)", !(mk2.root()==root)); }

    // binding still holds: changing one data entry changes the root (same salts)
    { auto d2=data; d2[0]=gl_add(d2[0],1); SaltedMerkle mk3; mk3.build(d2,salts);
      check("changed data -> different root (binding)", !(mk3.root()==root)); }

    // (4) brute-force demonstration on a single leaf in {0..255}
    {   gl_t secret = 173; Salt s = rand_salt();
        // unsalted leaf (as in the integrity Merkle): SHA256(8 bytes) -> brute-forceable
        uint8_t bb[8]; for(int k=0;k<8;k++) bb[k]=(uint8_t)(secret>>(8*k));
        Hash unsalted; fs::sha256(bb,8,unsalted.data());
        bool found_unsalted=false;
        for(uint32_t g=0; g<256; g++){ uint8_t gb[8]; for(int k=0;k<8;k++) gb[k]=(uint8_t)((gl_t)g>>(8*k));
            Hash h; fs::sha256(gb,8,h.data()); if(h==unsalted){ found_unsalted=(g==secret); break; } }
        // salted leaf: same 256 guesses (salt unknown) -> no match
        Hash salted = leaf_hash_salt(secret, s);
        bool found_salted=false;
        for(uint32_t g=0; g<256; g++){ if(leaf_hash_salt((gl_t)g, Salt{})==salted){ found_salted=true; break; } }
        check("unsalted int8 leaf is brute-forced", found_unsalted);
        check("salted int8 leaf resists brute force", !found_salted);
    }

    printf("\nP3.5 SALTED-MERKLE: %d passed, %d failed -> %s\n", npass,nfail, nfail==0?"ALL PASS":"FAIL");
    return nfail==0?0:1;
}
