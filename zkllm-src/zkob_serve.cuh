// zkob_serve.cuh — Stage C2 single-process verifier transport
// (TRANSPORT_REBUILD_DESIGN §1.3/§2.7/§6 Stage C: "single-process
// zkverify_walk ... one CUDA context, gens loaded once"). Every driver
// binary gains a `serve` mode: it reads one request per line from stdin
// (the argv tail of a normal one-shot invocation, whitespace-split), runs it
// through the SAME zkw_run1() entry the one-shot CLI uses — byte-identical
// FS schedules, checks and verdicts — and prints a "ZKW-RC <rc>" sentinel
// line after flushing all output. The orchestrator keeps ONE serve process
// per driver alive for the whole walk, so CUDA init is paid once per DRIVER
// instead of once per OBLIGATION (~235 inits -> ~12). Verification
// PACKAGING, not protocol.
//
// Exceptions from a request are caught and reported as rc=3 (the one-shot
// CLI would have died with a nonzero exit; the orchestrator treats both as
// crash, fail-closed). A crashed serve process is respawned by the caller.
#ifndef ZKOB_SERVE_CUH
#define ZKOB_SERVE_CUH

#include <cstdio>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int zkw_serve(const char* argv0, int (*run1)(int, char**)) {
    std::string line;
    std::cout << "ZKW-READY" << std::endl;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::vector<std::string> toks;
        std::string t;
        while (iss >> t) toks.push_back(t);
        if (toks.empty()) continue;
        std::vector<char*> av;
        av.push_back(const_cast<char*>(argv0));
        for (auto& s : toks) av.push_back(const_cast<char*>(s.c_str()));
        int rc;
        try {
            rc = run1((int)av.size(), av.data());
        } catch (const std::exception& e) {
            std::cout << "serve: request threw: " << e.what() << std::endl;
            rc = 3;
        } catch (...) {
            std::cout << "serve: request threw (non-std exception)" << std::endl;
            rc = 3;
        }
        std::fflush(stdout);            // printf-side buffer
        std::cout << "ZKW-RC " << rc << std::endl;   // endl flushes cout
    }
    return 0;
}

#endif
