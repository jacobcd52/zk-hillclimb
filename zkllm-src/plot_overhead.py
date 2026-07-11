#!/usr/bin/env python3
# Proving-overhead multiplier (proof construction time / native forward time)
# for the exact-fp8 Hawkeye layer ZKP vs the integerized (plain integer GEMM)
# proof.  Data measured on RTX 4090 (bench_sweep.log + integer MLP-down runs).
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- measured layer prove times (s) and native fwd (ms) ---
# Hawkeye exact-fp8 full layer: zk1 = real ZKP, zk0 = integrity-only. None = OOM on 24GB.
# integerized layer matmul proof (ms) = 4*proj(d,d) + 2*mlpup(d,4d) + 1*mlpdown(4d,d).
# each entry: (fwd_ms, hw_zk1_s, hw_zk0_s, int_layer_ms)
MODEL = {   # seq=64, batch=1, tokens=64; x = d
    64:  (4.39,  32.95,  8.17,  4*9.4  + 2*13.0 + 13.0),
    128: (8.04,  114.3,  21.93, 4*12.5 + 2*21.4 + 21.1),
    256: (14.86, None,   75.62, 4*19.9 + 2*41.4 + 36.9),
    512: (28.08, None,   None,  4*36.2 + 2*99.2 + 109.9),
}
SEQ = {     # d=256, batch=1; x = tokens (=seq)
    16:  (14.50, 108.3,  20.77, 4*16.4 + 2*31.2 + 30.4),
    64:  (14.86, None,   75.62, 4*19.9 + 2*41.4 + 36.9),
    256: (14.47, None,   None,  4*29.8 + 2*51.0 + 55.2),
}
BATCH = {   # d=256, seq=16; x = batch (tokens = 16*batch)
    1:   (14.30, 108.3,  20.77, 4*16.4 + 2*31.2 + 30.4),
    4:   (54.89, 375.0,  81.48, 4*19.9 + 2*41.4 + 36.9),
    16:  (219.85, None,  None,  4*29.8 + 2*51.0 + 55.2),
}

def series(D):
    xs = sorted(D)
    hw1 = [(x, D[x][1]*1e3/D[x][0]) for x in xs if D[x][1] is not None]
    hw0 = [(x, D[x][2]*1e3/D[x][0]) for x in xs if D[x][2] is not None]
    it  = [(x, D[x][3]     /D[x][0]) for x in xs]
    return hw1, hw0, it

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
panels = [(MODEL, 'model width  d  (seq=64, batch=1)', 'd'),
          (SEQ,   'sequence length  (d=256, batch=1)', 'tokens'),
          (BATCH, 'batch size  (d=256, seq=16)',       'batch')]
for ax, (D, title, xlab) in zip(axes, panels):
    hw1, hw0, it = series(D)
    if hw1: ax.plot(*zip(*hw1), 'o-',  color='#c0392b', lw=2.2, ms=8, label='Hawkeye exact-fp8  (ZK)')
    if hw0: ax.plot(*zip(*hw0), 's--', color='#e67e22', lw=2,   ms=7, label='Hawkeye exact-fp8  (no-ZK)')
    ax.plot(*zip(*it), '^-', color='#2980b9', lw=2.2, ms=8, label='Integerized  (int GEMMs)')
    ax.set_yscale('log'); ax.set_xscale('log', base=2)
    ax.set_xticks([x for x in D]); ax.set_xticklabels([str(x) for x in D])
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlab, fontsize=11)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    # annotate the gap on the first panel
    if D is MODEL:
        ax.set_ylabel('proving overhead  =  proof time / forward time  (x)', fontsize=11)
        ax.axhspan(1, 1, color='k')
    ax.legend(fontsize=8.5, loc='upper left')
axes[0].annotate('OOM on 24GB\nbeyond here',(256,5087),(70,300),fontsize=8,color='#c0392b',
                 arrowprops=dict(arrowstyle='->',color='#c0392b',alpha=0.6))
fig.suptitle('ZK proving overhead vs native forward pass  (RTX 4090, one transformer layer)\n'
             'exact-fp8 (Hawkeye) proves the H100 fp8 matmuls bitwise; integerized proves plain int GEMMs',
             fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig('/root/overhead_multiplier.png', dpi=130)
print('wrote /root/overhead_multiplier.png')

# print the numbers for the record
for name, D in [('MODEL',MODEL),('SEQ',SEQ),('BATCH',BATCH)]:
    print(f'\n== {name} ==')
    for x in sorted(D):
        f,h1,h0,im = D[x]
        print(f'  x={x:4}  fwd={f:7.2f}ms  HWzk={"OOM" if h1 is None else f"{h1*1e3/f:8.0f}x":>9}  '
              f'HWnozk={"OOM" if h0 is None else f"{h0*1e3/f:7.0f}x":>8}  INT={im/f:6.1f}x')
