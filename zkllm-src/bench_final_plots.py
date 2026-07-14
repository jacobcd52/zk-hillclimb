#!/usr/bin/env python3
"""Fresh like-for-like overhead sweep with the CURRENT prover (p3_tb_i2d):
fp8 zk=1 composed layer vs integerized estimate (int GEMM proofs for the same
matmul shapes + the same run's measured non-matmul gadget stages), both over
the batched-attention native forward. Writes bench_final.json.
Reuses the two long endgame runs (d256s128, s128b64) instead of rerunning."""
import json, re, subprocess, os
ZK = '/root/zkllm'
TB = '/root/p3_tb_i2d'
IMM = '/root/p3_matmul_bench2'
def lg(x): return max(0, (x-1).bit_length())
def sh(c, to=3600):
    return subprocess.run(c, capture_output=True, text=True, cwd=ZK, timeout=to)

_imm = {}
def imm(B, IN, OUT):
    k = (lg(B), lg(IN), lg(OUT))
    if k not in _imm:
        p = sh([IMM, str(k[0]), str(k[1]), str(k[2])], to=900)
        m = re.search(r'prove_ms=([\d.]+)', p.stdout)
        _imm[k] = float(m.group(1))/1000 if m else None
    return _imm[k]

def int_mm(seq, d, nh, dh, dff, batch):
    T = batch*seq; A = batch*nh
    return (4*imm(T, d, d) + 2*imm(T, d, dff) + imm(T, dff, d)
            + A*imm(seq, dh, seq) + A*imm(seq, seq, dh))

def fwd(seq, d, nh, dh, dff, batch):
    p = sh(['python3', 'tf_fwd_bench.py', str(seq), str(d), str(nh), str(dh),
            str(dff), str(batch), '30', '1'], to=600)
    m = re.search(r'fwd_ms=([\d.]+)', p.stdout)
    return float(m.group(1)) if m else None

def parse(text):
    b = re.search(r'BENCH .*verify_ok=1.*prove=([\d.]+)', text)
    s = re.search(r'STAGES (.*)', text)
    if not (b and s): return None
    st = dict(kv.split('=') for kv in s.group(1).split())
    nonmm = sum(float(st[k]) for k in ('rms','qnt','rope','smx','bfa','swg','seam'))
    return float(b.group(1)), nonmm

def run_fp8(tag, seq, d, nh, dh, dff, batch, tables):
    log = f'/root/zkrun_fp_{tag}.log'
    if not os.path.exists(log):
        with open(log, 'w') as f:
            subprocess.run(['env', 'P3_ZKPROF=1', 'timeout', '7200', TB, str(seq),
                            str(d), str(nh), str(dh), str(dff), str(batch), '1', tables],
                           stdout=f, stderr=subprocess.STDOUT, cwd=ZK, timeout=7300)
    return parse(open(log, 'rb').read().decode('utf8', 'replace'))

REUSE = {'d256s128': '/root/zkrun_i2d_d256s128.log', 'b64': '/root/zkrun_i2d_s128b64.log'}
CFG = [
    # (tag, group, seq, d, nh, dh, dff, batch, tables, reuse)
    ('s64',   'seq',   64,   64, 2, 32, 128,  1, 'tables_ld6.bin', None),
    ('s128',  'seq',   128,  64, 2, 32, 128,  1, 'tables_ld6.bin', None),
    ('s256',  'seq',   256,  64, 2, 32, 128,  1, 'tables_ld6.bin', None),
    ('s512',  'seq',   512,  64, 2, 32, 128,  1, 'tables_ld6.bin', None),
    ('s1024', 'seq',   1024, 64, 2, 32, 128,  1, 'tables_ld6.bin', None),
    ('b4',    'batch', 128,  64, 2, 32, 128,  4, 'tables_ld6.bin', None),
    ('b16',   'batch', 128,  64, 2, 32, 128, 16, 'tables_ld6.bin', None),
    ('b64',   'batch', 128,  64, 2, 32, 128, 64, 'tables_ld6.bin', REUSE['b64']),
    ('d128',  'model', 128, 128, 4, 32, 512,  1, 'p3_rmsnorm_tables_ld7.bin', None),
    ('d256s128', 'model', 128, 256, 4, 64, 1024, 1, 'tables_ld8.bin', REUSE['d256s128']),
]
rows = []
for tag, grp, seq, d, nh, dh, dff, batch, tables, reuse in CFG:
    if reuse:
        pr = parse(open(reuse, 'rb').read().decode('utf8', 'replace'))
    else:
        pr = run_fp8(tag, seq, d, nh, dh, dff, batch, tables)
    if not pr:
        print(f'{tag}: FAILED', flush=True); continue
    prove, nonmm = pr
    f = fwd(seq, d, nh, dh, dff, batch)
    mm = int_mm(seq, d, nh, dh, dff, batch)
    rows.append(dict(tag=tag, grp=grp, seq=seq, d=d, dff=dff, batch=batch,
                     tokens=batch*seq, params=4*d*d + 3*d*dff,
                     fp8_s=prove, int_s=mm + nonmm, nonmm=nonmm, fwd_ms=f))
    print(f'{tag}: fp8={prove:.1f}s int={mm+nonmm:.2f}s fwd={f:.2f}ms '
          f'ovh fp8={prove*1000/f:,.0f}x int={(mm+nonmm)*1000/f:,.0f}x', flush=True)
json.dump(rows, open(f'{ZK}/bench_final.json', 'w'), indent=1)
print('WROTE bench_final.json', flush=True)
