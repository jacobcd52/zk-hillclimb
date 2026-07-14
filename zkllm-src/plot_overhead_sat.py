#!/usr/bin/env python3
# ZKP overhead: float(fp8) vs integerized, vs sequence length and model size.
# Denominator = saturated per-sequence forward (batched ladder / batch).
import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 17,
                     'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 15})
rows = json.load(open('/root/zkllm/bench_sat.json'))
def ov(r, k): return r[k]*1000.0/r['fwd_eff_ms'] if r.get(k) else None

fig, ax = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
PANELS = [('seq',   'seq',    'sequence length',
           'vs sequence length\n(0.04 M params per layer)'),
          ('model', 'params', 'parameters per layer (M)',
           'vs model size\n(sequence length 64)')]
for a, (grp, xk, xlab, title) in zip(ax, PANELS):
    rs = sorted([r for r in rows if r['grp'] == grp], key=lambda r: r[xk])
    xs = [r[xk]/1e6 if xk == 'params' else r[xk] for r in rs]
    a.plot(xs, [ov(r, 'fp8_s') for r in rs], 'o-',  color='#c0392b', lw=3, ms=11,
           label='float (exact fp8)')
    a.plot(xs, [ov(r, 'int_s') for r in rs], 's--', color='#2980b9', lw=3, ms=10,
           label='integerized')
    a.set_xscale('log', base=(10 if xk == 'params' else 2)); a.set_yscale('log')
    a.set_xticks(xs)
    a.set_xticklabels([f'{x:.2f}' if xk == 'params' else str(int(x)) for x in xs])
    a.minorticks_off(); a.set_xlabel(xlab); a.set_title(title)
    a.grid(True, which='major', ls=':', alpha=0.5)
ax[0].set_ylabel('overhead  (×)')
ax[0].legend(loc='lower left')
fig.suptitle('ZKP overhead', fontsize=23)
fig.text(0.5, 0.015,
         'overhead  =  time to prove one sequence  ÷  GPU time that sequence costs in a fully-utilized forward pass\n'
         '(forward time measured with a large batch, then divided by the batch size).  Every proof point is full zero-knowledge and verified.',
         ha='center', fontsize=14, color='#444444')
fig.tight_layout(rect=[0.005, 0.075, 1, 0.94])
fig.savefig('/root/overhead_saturated.png', dpi=140)
print('wrote /root/overhead_saturated.png')
