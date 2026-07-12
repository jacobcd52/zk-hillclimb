#!/usr/bin/env python3
"""ZK-at-scale overhead sweep (design doc section 22): the section-22
memory-engineered composed prover at LARGE seq / batch / d, zk=1 and zk=0,
against (a) the native fp8 forward (canonical per-op AND batched-attention
well-utilized modes) and (b) the integerized layer (int GEMM proofs +
identical non-matmul gadget stages, as bench2.py).

Writes bench4_results.json.  One config per subprocess (VmHWM per point);
sequential, one GPU job at a time.
"""
import json, re, subprocess, os
ZK = '/root/zkllm'
BENCH = os.environ.get('TB', '/root/p3_tb_c22')
IMM = '/root/p3_matmul_bench2'
ART = {6: 'tables_ld6.bin', 7: 'tables_ld7.bin', 8: 'tables_ld8.bin',
       9: 'tables_ld9.bin', 10: 'tables_ld10.bin'}
ART[6] = 'p3_rmsnorm_tables.bin'
ART[8] = 'p3_rmsnorm_tables_ld8.bin'


def lg(x): return x.bit_length() - 1


def sh(cmd, to=7200):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ZK, timeout=to)


_immcache = {}
def imm(B, IN, OUT):
    k = (B, IN, OUT)
    if k not in _immcache:
        p = sh([IMM, str(lg(B)), str(lg(IN)), str(lg(OUT))], to=1200)
        m = re.search(r'prove_ms=([\d.]+)', p.stdout)
        _immcache[k] = float(m.group(1)) if m else None
    return _immcache[k]


def int_layer_matmul_ms(seq, d, nh, dh, dff, batch):
    T = batch * seq; A = batch * nh; tot = 0.0
    for cnt, B, IN, OUT in ((4, T, d, d), (2, T, d, dff), (1, T, dff, d),
                            (A, seq, dh, seq), (A, seq, seq, dh)):
        v = imm(B, IN, OUT)
        if v is None: return None
        tot += cnt * v
    return tot


def fp8_layer(seq, d, nh, dh, dff, batch, zk):
    art = ART[lg(d)]
    p = sh([BENCH, *map(str, (seq, d, nh, dh, dff, batch, int(zk))), art])
    if 'BENCH' not in p.stdout: return None
    m = re.search(r'prove=([\d.]+) verify=([\d.]+) proof_mb=([\d.]+) rss_gb=([\d.]+)', p.stdout)
    st = re.search(r'STAGES rms=([\d.]+) qnt=([\d.]+) mm=([\d.]+) rope=([\d.]+) smx=([\d.]+) '
                   r'bfa=([\d.]+) swg=([\d.]+) lug=([\d.]+) seam=([\d.]+) batch=([\d.]+)', p.stdout)
    r = dict(zip(('prove', 'verify', 'proof_mb', 'rss_gb'), map(float, m.groups())))
    if st:
        s = dict(zip(('rms', 'qnt', 'mm', 'rope', 'smx', 'bfa', 'swg', 'lug', 'seam', 'batch'),
                     map(float, st.groups())))
        r['mm'] = s['mm']
        r['nonmm'] = s['rms'] + s['qnt'] + s['rope'] + s['smx'] + s['bfa'] + s['swg'] + s['seam']
    return r


def fwd(seq, d, nh, dh, dff, batch, batched):
    p = sh(['python3', 'tf_fwd_bench.py', *map(str, (seq, d, nh, dh, dff, batch, 30)),
            '1' if batched else '0'], to=1200)
    m = re.search(r'fwd_ms=([\d.]+)', p.stdout)
    return float(m.group(1)) if m else None


def cfgs():
    out = []
    # model-width sweep (params x-axis), seq=16 batch=1
    for d in (64, 128, 256):                    out.append(('model', d, 16, 1))
    # seq sweep at batch=1, d=64 -> to 2048
    for s in (16, 64, 128, 256, 512, 1024, 2048): out.append(('seq', 64, s, 1))
    # batch sweep at seq=128, d=64 -> to 64
    for b in (1, 4, 16, 64):                    out.append(('batch', 64, 128, b))
    return out


def params(d, dff): return 4 * d * d + 3 * d * dff


rows = []
for grp, d, seq, batch in cfgs():
    nh = d // 16 if d != 64 else 2
    dh = d // nh
    dff = 4 * d if d != 64 else 128
    T = batch * seq
    r = {'grp': grp, 'd': d, 'seq': seq, 'batch': batch, 'nh': nh, 'dh': dh,
         'dff': dff, 'tokens': T, 'params': params(d, dff)}
    r['fwd_ms'] = fwd(seq, d, nh, dh, dff, batch, False)
    r['fwdb_ms'] = fwd(seq, d, nh, dh, dff, batch, True)
    r['fp8_zk1'] = fp8_layer(seq, d, nh, dh, dff, batch, 1)
    r['fp8_zk0'] = fp8_layer(seq, d, nh, dh, dff, batch, 0)
    r['int_mm_ms'] = int_layer_matmul_ms(seq, d, nh, dh, dff, batch)
    nonmm = (r['fp8_zk0'] or {}).get('nonmm')
    r['int_layer_s'] = ((r['int_mm_ms'] or 0) + (nonmm * 1000 if nonmm else 0)) / 1000.0 \
        if r['int_mm_ms'] is not None and nonmm is not None else None
    rows.append(r)
    z1 = r['fp8_zk1']['prove'] if r['fp8_zk1'] else 'OOM'
    z0 = r['fp8_zk0']['prove'] if r['fp8_zk0'] else 'OOM'
    print(f"{grp:6} d={d:4} seq={seq:5} b={batch:3} T={T:6} fwd={r['fwd_ms']}ms "
          f"fwdB={r['fwdb_ms']}ms zk1={z1} zk0={z0} intL={r['int_layer_s']}", flush=True)
    json.dump(rows, open('/root/zkllm/bench4_results.json', 'w'), indent=1)
print("wrote bench4_results.json")
