#!/usr/bin/env python3
# Final improvement-loop figure: overhead at scale (post-iteration-2) + the
# improvement trajectory across base -> scale-fixes -> iter1 -> iter2.
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

# ---- measured prove seconds (zk=1, verify_ok=1, proof bytes identical) ----
# config key: (label, tokens)
TRAJ = {  # config -> [(stage, prove_s, rss_gb)]
 'd256 s128\n(128 tok)': [('base', 230.3, 25.1), ('iter1', 151.3, 18.4), ('iter2', 114.7, 15.1)],
 'd256 s256\n(256 tok)': [('base', 469.8, 30.7), ('iter1', 352.1, 37.0), ('iter2', 301.7, 27.3)],
 '4096 tok\n(b16 s256)': [('base', 644.2, 34.4), ('iter1', 487.6, 37.3), ('iter2', 399.4, 34.7)],
 '8192 tok\n(b64 s128)': [('iter1', 1176.7, 37.5), ('iter2', 981.8, 37.5)],
}
# 16384-token point appended at render time if the run landed:
S256B64 = '/root/zkrun_i2d_s256b64.log'
T16K = None
if os.path.exists(S256B64):
    for ln in open(S256B64, 'rb').read().decode('utf8','replace').splitlines():
        if ln.startswith('BENCH') and 'verify_ok=1' in ln:
            kv = dict(p.split('=') for p in ln.split()[1:] if '=' in p)
            T16K = (float(kv['prove']), float(kv['rss_gb']))

# batched-attention native forward (ms), the honest baseline
FWD = {'d256 s128\n(128 tok)': 2.178, 'd256 s256\n(256 tok)': 2.426,
       '4096 tok\n(b16 s256)': None, '8192 tok\n(b64 s128)': None}
# filled from fwd_batched_extra.json if present (measured post-run)
if os.path.exists('/root/fwd_batched_extra.json'):
    FWD.update(json.load(open('/root/fwd_batched_extra.json')))

fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
COL = {'base': '#95a5a6', 'iter1': '#e67e22', 'iter2': '#c0392b'}

# Panel A: prove-time trajectory (grouped bars)
a = ax[0]
cfgs = list(TRAJ)
for i, c in enumerate(cfgs):
    stages = TRAJ[c]
    w = 0.8 / 3
    for j, (st, s, r) in enumerate(stages):
        off = (j - (len(stages)-1)/2) * w
        b = a.bar(i + off, s, w*0.92, color=COL[st], label=st if i == 0 else None)
        a.text(i + off, s*1.02, f'{s:.0f}', ha='center', fontsize=7.5)
if T16K:
    a.bar(len(cfgs), T16K[0], 0.27, color=COL['iter2'])
    a.text(len(cfgs), T16K[0]*1.02, f'{T16K[0]:.0f}', ha='center', fontsize=7.5)
    cfgs = cfgs + ['16384 tok\n(b64 s256)\nUNLOCKED']
a.set_xticks(range(len(cfgs))); a.set_xticklabels(cfgs, fontsize=8.5)
a.set_ylabel('prove time (s), full ZK, verified', fontsize=10)
a.set_title('Improvement loop: prove-time trajectory\n(proof bytes identical at every step)', fontsize=10.5)
a.grid(True, axis='y', ls=':', alpha=0.4); a.legend(fontsize=9)

# Panel B: RSS trajectory
b = ax[1]
for i, c in enumerate(list(TRAJ)):
    stages = TRAJ[c]
    w = 0.8 / 3
    for j, (st, s, r) in enumerate(stages):
        off = (j - (len(stages)-1)/2) * w
        b.bar(i + off, r, w*0.92, color=COL[st], label=st if i == 0 else None)
if T16K:
    b.bar(len(TRAJ), T16K[1], 0.27, color=COL['iter2'])
b.axhline(41, color='k', lw=1.4, ls='--'); b.text(0.02, 41.4, '41 GB container cap', fontsize=8.5)
b.set_xticks(range(len(cfgs))); b.set_xticklabels(cfgs, fontsize=8.5)
b.set_ylabel('peak host RSS (GB)', fontsize=10)
b.set_title('Memory trajectory vs the 41 GB cap', fontsize=10.5)
b.grid(True, axis='y', ls=':', alpha=0.4); b.legend(fontsize=9)

# Panel C: overhead multiplier vs tokens (post-iter2, batched fwd baseline)
c = ax[2]
pts = []
POST = {'d256 s128\n(128 tok)': (128, 114.7), 'd256 s256\n(256 tok)': (256, 301.7),
        '4096 tok\n(b16 s256)': (4096, 399.4), '8192 tok\n(b64 s128)': (8192, 981.8)}
for k, (tok, s) in POST.items():
    f = FWD.get(k)
    if f: pts.append((tok, s*1000/f))
pts.sort()
if pts:
    c.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color='#c0392b', lw=2.2, ms=8,
           label='Hawkeye fp8, full ZK (post-iter2)')
    for tok, ov in pts: c.annotate(f'{ov:,.0f}x', (tok, ov), textcoords='offset points',
                                   xytext=(0, 9), ha='center', fontsize=8)
c.set_xscale('log', base=2); c.set_yscale('log')
c.set_xlabel('tokens', fontsize=10); c.set_ylabel('overhead = prove / batched forward (x)', fontsize=10)
c.set_title('Overhead at scale, post-loop\n(batched-attention forward baseline)', fontsize=10.5)
c.grid(True, which='both', ls=':', alpha=0.4); c.legend(fontsize=8.5)

fig.suptitle('P3 full-ZK prover improvement loop (RTX 4090) — every point verify_ok=1, transcripts byte-identical across all levers',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('/root/improvement_loop.png', dpi=130)
print('wrote /root/improvement_loop.png; 16k point:', T16K)
