#!/usr/bin/env python3
# Clean 3-panel overhead figure: float(fp8) vs integerized ZKP overhead,
# vs seq len / batch size / param count. Big text, minimal clutter.
import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
rows = json.load(open('/root/zkllm/bench_final.json'))
def ov(r, k): return r[k]*1000.0/r['fwd_ms']

fig, ax = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
PANELS = [
    ('seq',   'seq',    'sequence length',        'vs sequence length\n(d=64, batch=1)'),
    ('batch', 'batch',  'batch size',             'vs batch size\n(d=64, seq=128)'),
    ('model', 'params', 'parameters per layer (M)','vs model size\n(seq=128, batch=1)'),
]
for a, (grp, xk, xlab, title) in zip(ax, PANELS):
    if grp == 'model':
        rs = [r for r in rows if r['grp'] == 'model' or r['tag'] == 's128']
    elif grp == 'batch':
        rs = [r for r in rows if r['grp'] == 'batch' or r['tag'] == 's128']
    else:
        rs = [r for r in rows if r['grp'] == 'seq']
    rs.sort(key=lambda r: r[xk])
    xs = [r[xk]/1e6 if xk == 'params' else r[xk] for r in rs]
    a.plot(xs, [ov(r, 'fp8_s') for r in rs], 'o-',  color='#c0392b', lw=3, ms=11,
           label='float (exact fp8)')
    a.plot(xs, [ov(r, 'int_s') for r in rs], 's--', color='#2980b9', lw=3, ms=10,
           label='integerized')
    a.set_xscale('log', base=(10 if xk == 'params' else 2))
    a.set_yscale('log')
    a.set_xticks(xs)
    a.set_xticklabels([f'{x:.2f}' if xk == 'params' else str(int(x)) for x in xs])
    a.minorticks_off()
    a.set_xlabel(xlab)
    a.set_title(title)
    a.grid(True, which='major', ls=':', alpha=0.5)
ax[0].set_ylabel('overhead = prove / forward  (×)')
ax[0].legend(loc='upper left')
fig.suptitle('ZK proving overhead: one full transformer layer, full zero-knowledge  (RTX 4090)',
             fontsize=19)
fig.text(0.5, 0.005,
         'Baseline: batched-attention native forward.  Integerized = measured int-GEMM proofs for the same matmul shapes + the same run’s non-matmul gadget stages.',
         ha='center', fontsize=12, color='#555555')
fig.tight_layout(rect=[0.005, 0.03, 1, 0.93])
fig.savefig('/root/overhead_float_vs_int.png', dpi=140)
print('wrote /root/overhead_float_vs_int.png')
