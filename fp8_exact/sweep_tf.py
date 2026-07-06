#!/usr/bin/env python3
"""Overhead sweep driver: runs p3_transformer_bench (one process per point,
zk and non-zk) + tf_fwd_bench.py per config, joins into overhead tables.

  python3 sweep_tf.py            # runs everything, writes sweep_results.json
"""
import json, re, subprocess, sys, os

BENCH = '/root/p3_transformer_bench'
ART = {6: 'p3_rmsnorm_tables.bin', 7: 'p3_rmsnorm_tables_ld7.bin',
       8: 'p3_rmsnorm_tables_ld8.bin'}

def ld(d): return d.bit_length() - 1

# (label, seq, d, nh, dh, dff, batch)
POINTS = []
# 1. params sweep: d & dff together (dff = 4d), fixed tokens = 8
for d in (64, 128, 256):
    POINTS.append((f'params d={d}', 8, d, d // 32, 32, 4 * d, 1))
# 2. tokens sweep at d=64 (dff = 4d = 256), batch=1
for s in (4, 8, 16, 32, 64):
    POINTS.append((f'tokens T={s}', s, 64, 2, 32, 256, 1))
# 3. seq-vs-batch split at fixed tokens = 64 (d=64, dff=256)
for s, b in ((2, 32), (4, 16), (8, 8), (16, 4), (32, 2), (64, 1)):
    POINTS.append((f'split S={s} B={b}', s, 64, 2, 32, 256, b))

def run_bench(seq, d, nh, dh, dff, batch, zk):
    art = ART[ld(d)]
    cmd = [BENCH, *map(str, (seq, d, nh, dh, dff, batch, int(zk))), art]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd='/root/zkllm',
                       timeout=3600)
    out = p.stdout
    m = re.search(r'BENCH .*prove=([\d.]+) verify=([\d.]+) proof_mb=([\d.]+) '
                  r'rss_gb=([\d.]+)', out)
    ok = 'verify_ok=1' in out
    st = re.search(r'STAGES rms=([\d.]+) qnt=([\d.]+) mm=([\d.]+) rope=([\d.]+) '
                   r'smx=([\d.]+) bfa=([\d.]+) swg=([\d.]+) lug=([\d.]+) '
                   r'seam=([\d.]+) batch=([\d.]+)', out)
    if not m or not ok:
        print(out); print(p.stderr[-2000:] if p.stderr else '')
        return None
    r = dict(zip(('prove', 'verify', 'proof_mb', 'rss_gb'), map(float, m.groups())))
    if st:
        r['stages'] = dict(zip(('rms', 'qnt', 'mm', 'rope', 'smx', 'bfa', 'swg',
                                'lug', 'seam', 'batch'), map(float, st.groups())))
    return r

def run_fwd(seq, d, nh, dh, dff, batch):
    cmd = ['python3', 'tf_fwd_bench.py', *map(str, (seq, d, nh, dh, dff, batch))]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd='/root/zkllm',
                       timeout=1200)
    m = re.search(r'fwd_ms=([\d.]+)', p.stdout)
    if not m: print(p.stdout, p.stderr[-1000:]); return None
    return float(m.group(1))

results = []
for label, seq, d, nh, dh, dff, batch in POINTS:
    row = {'label': label, 'seq': seq, 'd': d, 'nh': nh, 'dh': dh,
           'dff': dff, 'batch': batch, 'tokens': seq * batch}
    print(f'== {label}: seq={seq} d={d} nh={nh} dff={dff} batch={batch}', flush=True)
    fwd = run_fwd(seq, d, nh, dh, dff, batch)
    row['fwd_ms'] = fwd
    for zk in (False, True):
        r = run_bench(seq, d, nh, dh, dff, batch, zk)
        key = 'zk' if zk else 'nozk'
        row[key] = r
        if r and fwd:
            row[f'{key}_overhead'] = r['prove'] * 1000.0 / fwd
        print(f'   {key}: {r}', flush=True)
    results.append(row)
    with open('sweep_results.json', 'w') as f:
        json.dump(results, f, indent=1)
print('done ->', os.path.abspath('sweep_results.json'))
