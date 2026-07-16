#!/usr/bin/env python3
"""H100 overhead sweep — replicates the 4090 bench_saturated methodology.

Per config: fp8 composed-layer zk=1 (p3_tb_i2d), integer-layer zk=1
(p3_int_layer_bench), and the saturated-batch NATIVE forward denominator
(tf_fwd_bench batched=1 native=1, batch ladder, min per-seq ms).
Writes /root/bench_h100.json incrementally; status to /root/sweep_status.log.
Failures (OOM/timeout) recorded honestly as None.
"""
import json, re, subprocess, os, resource, datetime

ZK = '/root/zkllm'
FB = '/root/p3_tb_i2d'
IB = '/root/p3_int_layer_bench'
ST = '/root/sweep_status.log'
OUT = '/root/bench_h100.json'

def note(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ST, 'a') as f: f.write(line + '\n')

def sh(cmd, log, to=7200, env_extra=None):
    env = dict(os.environ, P3_MEMLOG='1', **(env_extra or {}))
    with open(log, 'w') as f:
        try:
            subprocess.run(['setsid', 'bash', '-c',
                'echo 1000 > /proc/self/oom_score_adj 2>/dev/null; exec "$@"',
                'x'] + cmd, stdout=f, stderr=subprocess.STDOUT,
                cwd=ZK, env=env, timeout=to)
        except subprocess.TimeoutExpired:
            f.write('\nTIMEOUT\n')

def parse_bench(log):
    t = open(log, 'rb').read().decode('utf8', 'replace')
    m = re.search(r'BENCH .*verify_ok=1 .*prove=([\d.]+) verify=([\d.]+) '
                  r'proof_mb=([\d.]+) rss_gb=([\d.]+)', t)
    if not m: return None
    return dict(prove=float(m.group(1)), verify=float(m.group(2)),
                proof_mb=float(m.group(3)), rss_gb=float(m.group(4)))

def fwd_sat(seq, d, nh, dh, dff):
    """Batch ladder, native bf16 GEMMs; min per-seq ms at saturation."""
    best = None; ladder = []; b = 32
    while True:
        p = subprocess.run(['python3', 'tf_fwd_bench.py', str(seq), str(d),
                            str(nh), str(dh), str(dff), str(b), '20', '1', '1'],
                           capture_output=True, text=True, cwd=ZK, timeout=1800)
        m = re.search(r'fwd_ms=([\d.]+)', p.stdout)
        if not m:
            note(f'  fwd ladder b={b}: failed/OOM, stopping ladder'); break
        per = float(m.group(1)) / b
        ladder.append((b, per))
        if best is None or per < best * 0.97: best = min(best or per, per)
        else: break
        if b >= 4096: break
        b *= 4
    return best, ladder

# tag, grp, seq, d, nh, dh, dff, tables, run_fp8, fp8_env
CFG = [
    ('s64',   'seq',   64,   64,   2, 32, 128,  'tables_ld6.bin',           True,  None),
    ('s128',  'seq',   128,  64,   2, 32, 128,  'tables_ld6.bin',           True,  None),
    ('s256',  'seq',   256,  64,   2, 32, 128,  'tables_ld6.bin',           True,  None),
    ('s512',  'seq',   512,  64,   2, 32, 128,  'tables_ld6.bin',           True,  None),
    ('s1024', 'seq',   1024, 64,   2, 32, 128,  'tables_ld6.bin',           True,  None),
    ('p64',   'model', 64,   64,   2, 32, 128,  'tables_ld6.bin',           True,  None),
    ('p128',  'model', 64,   128,  4, 32, 512,  'p3_rmsnorm_tables_ld7.bin',True,  None),
    ('p256',  'model', 64,   256,  4, 64, 1024, 'tables_ld8.bin',           True,  None),
    ('p512',  'model', 64,   512,  8, 64, 2048, 'tables_ld9.bin',           True,  None),
    # stretch: fp8 attempts that OOM'd on the 4090 (spill lever armed, default gate)
    ('p1024', 'model', 64,   1024, 16, 64, 4096,'tables_ld10.bin',          True,
     {'P3_PK_SPILL': '/root/p3_spill'}),
    ('s2048', 'seq',   2048, 64,   2, 32, 128,  'tables_ld6.bin',           True,
     {'P3_PK_SPILL': '/root/p3_spill'}),
    ('s4096', 'seq',   4096, 64,   2, 32, 128,  'tables_ld6.bin',           True,
     {'P3_PK_SPILL': '/root/p3_spill'}),
]

os.makedirs('/root/p3_spill', exist_ok=True)
rows = []
if os.path.exists(OUT):
    rows = json.load(open(OUT))
done = {r['tag'] for r in rows}
note(f'SWEEP START (resume: {sorted(done)})')

for tag, grp, seq, d, nh, dh, dff, tables, do_fp8, fenv in CFG:
    if tag in done: continue
    args = [str(seq), str(d), str(nh), str(dh), str(dff), '1', '1', tables]
    r = dict(tag=tag, grp=grp, seq=seq, d=d, nh=nh, dh=dh, dff=dff, batch=1,
             tokens=seq, params=4*d*d + 3*d*dff)

    note(f'{tag}: fwd ladder (native)')
    best, ladder = fwd_sat(seq, d, nh, dh, dff)
    r['fwd_native_ms'] = best; r['ladder'] = ladder
    note(f'{tag}: fwd_native_ms={best}')

    note(f'{tag}: INT prove')
    sh([IB] + args, f'/root/zkrun_h100_int_{tag}.log')
    pi = parse_bench(f'/root/zkrun_h100_int_{tag}.log')
    r['int_layer_s'] = pi['prove'] if pi else None
    r['int_meta'] = pi
    note(f'{tag}: int={r["int_layer_s"]}')

    if do_fp8:
        note(f'{tag}: FP8 prove')
        sh([FB] + args, f'/root/zkrun_h100_fp8_{tag}.log', env_extra=fenv)
        pf = parse_bench(f'/root/zkrun_h100_fp8_{tag}.log')
        r['fp8_s'] = pf['prove'] if pf else None
        r['fp8_meta'] = pf
        note(f'{tag}: fp8={r["fp8_s"]}')
    rows.append(r)
    json.dump(rows, open(OUT, 'w'), indent=1)

note('SWEEP DONE')
