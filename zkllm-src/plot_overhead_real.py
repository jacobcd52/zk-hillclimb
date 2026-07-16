#!/usr/bin/env python3
# ZKP overhead: exact-fp8 vs REAL integer-layer prover (both measured, same
# Goldilocks/Basefold substrate, same soundness). Denominator = saturated
# native bf16 forward. vs sequence length and model size.
import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 17,
                     'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14})
rows = json.load(open('/root/zkllm/bench_merged.json'))
def nat(r): return r['fwd_native_ms']*r['batch']   # per-config native forward ms
def ovf(r): return r['fp8_s']*1000.0/nat(r)
def ovi(r): return r['int_layer_s']*1000.0/nat(r)

grp = {'seq': ['s64','s128','s256','s512','s1024','s2048','s4096'],
       'model': ['p64','p128','p256','p512','p1024']}
xget = {'seq': lambda r: r['seq'], 'model': lambda r: r['params']/1e6}
byT = {r['tag']: r for r in rows}
fig, ax = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
PAN = [('seq','sequence length','vs sequence length\n(0.04 M params per layer)'),
       ('model','parameters per layer (M)','vs model size\n(sequence length 64)')]
for a,(g,xlab,title) in zip(ax, PAN):
    rs = [byT[t] for t in grp[g] if t in byT]
    xs = [xget[g](r) for r in rs]
    # fp8 line only where it fits (fp8_s not None); integer runs the full range
    fx = [x for x,r in zip(xs,rs) if r.get('fp8_s')]
    fy = [ovf(r) for r in rs if r.get('fp8_s')]
    a.plot(fx, fy, 'o-',  color='#c0392b', lw=3, ms=11, label='exact fp8')
    a.plot(xs, [ovi(r) for r in rs], 's--', color='#2980b9', lw=3, ms=10, label='integer')
    a.annotate('fp8 OOMs\npast here', (fx[-1], fy[-1]), textcoords='offset points',
               xytext=(6, 14), fontsize=12, color='#c0392b')
    a.set_xscale('log', base=(10 if g=='model' else 2)); a.set_yscale('log')
    a.set_xticks(xs); a.set_xticklabels([f'{x:.2f}' if g=='model' else str(int(x)) for x in xs])
    a.minorticks_off(); a.set_xlabel(xlab); a.set_title(title)
    a.grid(True, which='major', ls=':', alpha=0.5)
ax[0].set_ylabel('overhead  (×)'); ax[0].legend(loc='upper left')
fig.suptitle('ZKP overhead', fontsize=23)
fig.text(0.5, 0.015,
  'overhead  =  time to prove one sequence  ÷  GPU time that sequence costs in a fully-utilized forward pass.\n'
  'Both provers measured on the same Goldilocks + Basefold substrate and soundness; every proof point is full zero-knowledge and verified.',
  ha='center', fontsize=13.5, color='#444444')
fig.tight_layout(rect=[0.005, 0.075, 1, 0.94])
fig.savefig('/root/overhead_real.png', dpi=140)
print('wrote /root/overhead_real.png')
