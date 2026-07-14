#!/usr/bin/env python3
"""Ratio-under-accounting table for the fp8-vs-int comparison audit.
Stage data from the zkrun_* logs (i2d binary, zk=1), int_s from bench_sat.json /
bench_final.json.  All derived variants documented inline."""
import json

# tag: (prove, stages dict) from logs
S = {
 's64':  dict(rms=.096, qnt=.265, mm=5.418, rope=.243, smx=.287, bfa=.214, swg=.055, lug=1.404, seam=.039, batch=2.004, prove=10.025),
 's128': dict(rms=.171, qnt=.569, mm=9.216, rope=.468, smx=.917, bfa=.380, swg=.099, lug=2.691, seam=.066, batch=3.449, prove=18.026),
 's256': dict(rms=.285, qnt=.803, mm=15.351, rope=1.014, smx=1.009, bfa=.671, swg=.072, lug=6.010, seam=.087, batch=6.942, prove=32.248),
 's512': dict(rms=.225, qnt=1.600, mm=29.625, rope=1.758, smx=2.929, bfa=.456, swg=.109, lug=12.288, seam=.049, batch=12.851, prove=61.972),
 's1024':dict(rms=.337, qnt=6.221, mm=91.802, rope=3.017, smx=13.732, bfa=.681, swg=.166, lug=29.097, seam=.016, batch=33.581, prove=178.991),
 'b4':   dict(rms=.221, qnt=1.619, mm=27.317, rope=1.864, smx=3.677, bfa=.444, swg=.111, lug=7.951, seam=.126, batch=10.118, prove=53.455),
 'b16':  dict(rms=.540, qnt=6.124, mm=86.452, rope=7.495, smx=14.609, bfa=1.002, swg=.321, lug=22.074, seam=.523, batch=25.579, prove=164.914),
 'b64':  dict(rms=3.185, qnt=26.342, mm=433.652, rope=30.000, smx=58.064, bfa=4.292, swg=1.899, lug=207.011, seam=2.266, batch=214.738, prove=981.815),
 'p64':  dict(rms=.096, qnt=.265, mm=5.479, rope=.244, smx=.305, bfa=.210, swg=.055, lug=1.395, seam=.039, batch=2.120, prove=10.207),
 'p128': dict(rms=.160, qnt=.481, mm=14.194, rope=.484, smx=.612, bfa=.389, swg=.070, lug=5.987, seam=.075, batch=6.582, prove=29.037),
 'p256': dict(rms=.281, qnt=.785, mm=35.514, rope=.925, smx=.608, bfa=.664, swg=.090, lug=12.449, seam=.112, batch=13.489, prove=65.031),
 'p512': dict(rms=.222, qnt=1.214, mm=128.002, rope=1.859, smx=1.206, bfa=.420, swg=.128, lug=39.262, seam=.126, batch=66.344, prove=239.249),
}
INT = {r['tag']: r['int_s'] for r in json.load(open('/root/zkllm/bench_sat.json')) if r['int_s']}
NONMM_KEYS = ('rms','qnt','rope','smx','bfa','swg','seam')
ZKP = 2.1        # measured zk1/zk0 premium 2.02-2.22 (BENCH_ZK_AT_SCALE C)
COMP_SHARE = .03 # nonmm share of lug+batch by committed-data volume (upper-ish; see audit)
BO_FIXED = 0.3   # per-class batch-open fixed cost for an int composed layer (s)

print(f"{'cfg':6} {'fp8':>7} {'int':>6} {'i:as':>6} {'ii:noqnt':>8} {'iii:mm':>7} {'iv:int+zk+comp':>14} {'v:nonzk':>8}")
for tag, st in S.items():
    if tag not in INT: continue
    nonmm = sum(st[k] for k in NONMM_KEYS)
    fp8 = st['prove']; int_s = INT[tag]; int_mm = int_s - nonmm
    r_as   = fp8 / int_s
    r_noq  = fp8 / (int_s - st['qnt'])
    r_mm   = (st['mm'] + st['lug'] + st['batch']) / int_mm          # matmul machinery only
    int_iv = ZKP*int_mm + nonmm + COMP_SHARE*(st['lug']+st['batch']) + BO_FIXED
    r_iv   = fp8 / int_iv
    # v: both sides integrity-only: fp8_zk0 ~ prove/ZKP ; int = int_mm + nonmm/ZKP
    r_v    = (fp8/ZKP) / (int_mm + nonmm/ZKP)
    print(f"{tag:6} {fp8:7.1f} {int_s:6.2f} {r_as:6.1f} {r_noq:8.1f} {r_mm:7.0f} {r_iv:14.1f} {r_v:8.1f}")
