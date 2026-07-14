#!/usr/bin/env python3
"""Utilization-corrected overhead sweep.

Forward baseline per config = min over a batch ladder of (batched_fwd_ms / batch)
-- the effective per-sequence GPU time at saturation (user-proposed methodology).
Prover: composed layer zk=1 with the current binary; params panel extended to
d=512 (s128) and d=1024 (s64). Writes bench_sat.json incrementally."""
import json, re, subprocess, os
ZK = '/root/zkllm'; TB = '/root/p3_tb_i2d'; IMM = '/root/p3_matmul_bench2'
def lg(x): return max(0, (x-1).bit_length())
def sh(c, to=3600): return subprocess.run(c, capture_output=True, text=True, cwd=ZK, timeout=to)

def fwd_sat(seq, d, nh, dh, dff):
    best = None; ladder = []
    b = 32
    while True:
        p = sh(['python3', 'tf_fwd_bench.py', str(seq), str(d), str(nh), str(dh),
                str(dff), str(b), '20', '1'], to=900)
        m = re.search(r'fwd_ms=([\d.]+)', p.stdout)
        if not m: break                      # OOM or failure: stop the ladder
        per = float(m.group(1)) / b
        ladder.append((b, per))
        if best is None or per < best * 0.97: best = min(best or per, per)
        else: break                          # plateaued
        if b >= 4096: break
        b *= 4
    return best, ladder

_imm = {}
def imm(B, IN, OUT):
    k = (lg(B), lg(IN), lg(OUT))
    if k not in _imm:
        p = sh([IMM, str(k[0]), str(k[1]), str(k[2])], to=1800)
        m = re.search(r'prove_ms=([\d.]+)', p.stdout)
        _imm[k] = float(m.group(1))/1000 if m else None
    return _imm[k]
def int_mm(seq, d, nh, dh, dff, batch):
    T = batch*seq; A = batch*nh
    parts = [4*imm(T, d, d), 2*imm(T, d, dff), imm(T, dff, d),
             A*imm(seq, dh, seq), A*imm(seq, seq, dh)]
    return None if any(p is None for p in parts) else sum(parts)

def parse(text):
    b = re.search(r'BENCH .*verify_ok=1.*prove=([\d.]+).*rss_gb=([\d.]+)', text)
    s = re.search(r'STAGES (.*)', text)
    if not (b and s): return None
    st = dict(kv.split('=') for kv in s.group(1).split())
    nonmm = sum(float(st[k]) for k in ('rms','qnt','rope','smx','bfa','swg','seam'))
    return float(b.group(1)), nonmm, float(b.group(2))

def run_fp8(tag, seq, d, nh, dh, dff, batch, tables):
    log = f'/root/zkrun_sat_{tag}.log'
    if not os.path.exists(log) or parse(open(log,'rb').read().decode('utf8','replace')) is None:
        with open(log, 'w') as f:
            subprocess.run(['setsid','bash','-c',
                f'echo 1000 > /proc/self/oom_score_adj; exec env P3_MEMLOG=1 P3_ZKPROF=1 '
                f'timeout 7200 {TB} {seq} {d} {nh} {dh} {dff} {batch} 1 {tables}'],
                stdout=f, stderr=subprocess.STDOUT, cwd=ZK, timeout=7300)
    return parse(open(log, 'rb').read().decode('utf8', 'replace'))

OLD = json.load(open(f'{ZK}/bench_final.json'))          # reuse proven fp8/int numbers
old = {r['tag']: r for r in OLD}
CFG = [
    # tag, grp, seq, d, nh, dh, dff, batch, tables (None table => reuse old fp8/int)
    ('s64',   'seq',   64,   64, 2, 32, 128,  1, None),
    ('s128',  'seq',   128,  64, 2, 32, 128,  1, None),
    ('s256',  'seq',   256,  64, 2, 32, 128,  1, None),
    ('s512',  'seq',   512,  64, 2, 32, 128,  1, None),
    ('s1024', 'seq',   1024, 64, 2, 32, 128,  1, None),
    ('b4',    'batch', 128,  64, 2, 32, 128,  4, None),
    ('b16',   'batch', 128,  64, 2, 32, 128, 16, None),
    ('b64',   'batch', 128,  64, 2, 32, 128, 64, None),
    # params panel at seq=64, batch=1 (uniform axis; d=1024 only fits at s64)
    ('p64',   'model', 64,   64, 2, 32, 128,  1, 'tables_ld6.bin'),
    ('p128',  'model', 64,  128, 4, 32, 512,  1, 'p3_rmsnorm_tables_ld7.bin'),
    ('p256',  'model', 64,  256, 4, 64, 1024, 1, 'tables_ld8.bin'),
    ('p512',  'model', 64,  512, 8, 64, 2048, 1, 'tables_ld9.bin'),
    ('p1024', 'model', 64, 1024,16, 64, 4096, 1, 'tables_ld10.bin'),
]
rows = []
for tag, grp, seq, d, nh, dh, dff, batch, tables in CFG:
    if tables is None:
        o = old[tag]; prove, ints, rss = o['fp8_s'], o['int_s'], None
    else:
        pr = run_fp8(tag, seq, d, nh, dh, dff, batch, tables)
        if pr is None:
            print(f'{tag}: FP8 FAILED (see /root/zkrun_sat_{tag}.log)', flush=True); continue
        prove, nonmm, rss = pr
        mm = int_mm(seq, d, nh, dh, dff, batch)
        ints = (mm + nonmm) if mm is not None else None
    per_seq_fwd, ladder = fwd_sat(seq, d, nh, dh, dff)
    fwd_eff = per_seq_fwd * batch            # effective forward time for THIS config
    rows.append(dict(tag=tag, grp=grp, seq=seq, d=d, dff=dff, batch=batch,
                     tokens=batch*seq, params=4*d*d + 3*d*dff, fp8_s=prove,
                     int_s=ints, fwd_eff_ms=fwd_eff, ladder=ladder, rss=rss))
    io = f'{ints*1000/fwd_eff:,.0f}x' if ints else '-'
    print(f'{tag}: fp8={prove:.1f}s int={ints if ints else 0:.2f}s fwd_eff={fwd_eff:.3f}ms '
          f'ovh fp8={prove*1000/fwd_eff:,.0f}x int={io} (ladder {ladder})', flush=True)
    json.dump(rows, open(f'{ZK}/bench_sat.json', 'w'), indent=1)
print('SAT SWEEP DONE', flush=True)
