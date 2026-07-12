#!/usr/bin/env python3
"""ZK-at-scale overhead figure (BENCH_ZK_AT_SCALE.md).

Every point is a measured pair: composed-layer BENCH prove time (sweep_run.log /
zk_scale_results*.log / gates_result.log, all zk=1 verify_ok=1) divided by the
batched-attention native forward (fwd_results_batched.log).  Integerized line =
measured p3_matmul_bench2 int-GEMM proofs (int_refs_scale.json) + the same
composed run's measured zk=1 non-matmul STAGES.  Configs lacking either
measurement are omitted -- nothing extrapolated.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RED, ORANGE, BLUE = '#c0392b', '#e67e22', '#2980b9'
INK, MUT = '#333333', '#777777'

# (prove_s, fwd_ms) measured pairs; overhead = prove_s*1e3/fwd_ms
ov = lambda p, f: p * 1e3 / f

# --- panel (a): vs params per layer (b=1; seq per label) ---
PARAMS = [0.10, 1.57]                       # M params/layer: d=64, d=256
A_ZK1 = [ov(63.439, 2.2399), ov(259.518, 2.1779)]   # d64@seq256, d256@seq128
A_ZK1_SEQ = ['seq=256', 'seq=128']
A_ZK0 = [ov(53.208, 2.2403), ov(118.670, 2.1779)]   # d64@seq512, d256@seq128
A_ZK0_SEQ = ['seq=512', 'seq=128']
A_ZK0_X = ov(228.057, 2.4264)               # d256@seq256 zk0 (zk1 = 41GB wall)
A_INT = [ov(0.1758 + 4.044, 2.2399), ov(0.3305 + 5.867, 2.1779)]
# --- panel (b): vs seq, d=64 b=1 ---
B_SEQ = [256, 512, 1024]
B_ZK1 = [ov(63.439, 2.2399), ov(118.323, 2.2403), ov(298.848, 2.4720)]
B_ZK0S = [512, 1024]
B_ZK0 = [ov(53.208, 2.2403), ov(148.182, 2.4720)]
B_INT = [ov(0.1758 + 4.044, 2.2399), ov(0.2743 + 7.250, 2.2403),
         ov(0.5597 + 23.524, 2.4720)]
# --- panel (c): vs tokens, batch rows (seq=128, d=64, b=4/16) ---
C_TOK = [512, 2048]
C_ZK1 = [ov(85.995, 2.1083), ov(269.360, 2.2208)]
C_INT = [ov(0.3205 + 8.388, 2.1083), ov(0.9879 + 31.128, 2.2208)]

def series(a, x, y, color, ls, mk, label):
    a.plot(x, y, ls, color=color, lw=2, ms=8, marker=mk, label=label,
           markeredgecolor='white', markeredgewidth=1.2, zorder=3)

def vlab(a, x, y, dy=1.28, ha='center'):
    a.annotate(f'{y:,.0f}x', (x, y), (x, y * dy), fontsize=8, color=INK,
               ha=ha, zorder=4)

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6), sharey=True)
for a in ax:
    a.set_yscale('log')
    a.set_ylim(1e3, 4e5)
    a.grid(True, which='major', ls=':', lw=0.6, color='#cccccc', zorder=0)
    a.tick_params(labelsize=9)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)

# (a)
a = ax[0]
series(a, PARAMS, A_ZK1, RED, '-', 'o', 'Hawkeye fp8, zk=1')
series(a, PARAMS, A_ZK0, ORANGE, '--', 's', 'Hawkeye fp8, zk=0')
series(a, PARAMS, A_INT, BLUE, '-', '^', 'integerized zk')
a.plot([PARAMS[1]], [A_ZK0_X], 's', color=ORANGE, ms=8, mfc='white',
       markeredgewidth=1.6, zorder=3)
a.annotate(f'zk=0 @ seq=256: {A_ZK0_X:,.0f}x\n(zk=1 exceeds 41 GB cap)',
           (PARAMS[1], A_ZK0_X), (0.24, 2.1e5), fontsize=8, color=MUT,
           arrowprops=dict(arrowstyle='->', color=MUT, lw=0.8,
                           connectionstyle='arc3,rad=-0.15'))
for x, y, s in zip(PARAMS, A_ZK1, A_ZK1_SEQ):
    a.annotate(f'{y:,.0f}x ({s})', (x, y), (x, y * 1.32), fontsize=8,
               color=INK, ha='center', zorder=4)
for x, y, s in zip(PARAMS, A_ZK0, A_ZK0_SEQ):
    a.annotate(f'{y:,.0f}x ({s})', (x, y), (x, y / 1.65), fontsize=8,
               color=INK, ha='center', zorder=4)
for x, y in zip(PARAMS, A_INT):
    vlab(a, x, y)
a.set_xscale('log')
a.set_xticks(PARAMS)
a.set_xticklabels(['0.10\n(d=64)', '1.57\n(d=256)'])
a.set_xlim(0.055, 3.2)
a.set_xlabel('model parameters per layer (M)', fontsize=10)
a.set_ylabel('overhead = prove time / batched forward  (x)', fontsize=10)
a.set_title('(a) vs model width  (batch=1; seq as labeled --\nno matched-seq zk=1 pair fits the 41 GB cap)', fontsize=9.5)
a.legend(fontsize=8.5, loc='lower right', framealpha=0.9)

# (b)
b = ax[1]
series(b, B_SEQ, B_ZK1, RED, '-', 'o', 'Hawkeye fp8, zk=1')
series(b, B_ZK0S, B_ZK0, ORANGE, '--', 's', 'Hawkeye fp8, zk=0')
series(b, B_SEQ, B_INT, BLUE, '-', '^', 'integerized zk')
for x, y in zip(B_SEQ, B_ZK1):
    vlab(b, x, y)
for x, y in zip(B_ZK0S, B_ZK0):
    vlab(b, x, y, dy=0.66)
for x, y in zip(B_SEQ, B_INT):
    vlab(b, x, y)
b.set_xscale('log', base=2)
b.set_xticks(B_SEQ)
b.set_xticklabels([str(s) for s in B_SEQ])
b.set_xlim(200, 1320)
b.set_xlabel('sequence length', fontsize=10)
b.set_title('(b) vs seq  (d=64, batch=1)\nforward is launch-bound flat -> ratio grows ~linearly', fontsize=9.5)
b.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
b.text(0.03, 0.03, 'seq<=128 omitted: no batched-forward baseline\nmeasured (seq=64 zk=1 proves in 10.8 s)',
       transform=b.transAxes, fontsize=7.5, color=MUT, va='bottom')

# (c)
c = ax[2]
series(c, C_TOK, C_ZK1, RED, '-', 'o', 'Hawkeye fp8, zk=1')
series(c, C_TOK, C_INT, BLUE, '-', '^', 'integerized zk')
for x, y in zip(C_TOK, C_ZK1):
    vlab(c, x, y)
for x, y in zip(C_TOK, C_INT):
    vlab(c, x, y)
c.set_xscale('log', base=2)
c.set_xticks(C_TOK)
c.set_xticklabels(['512\n(b=4)', '2048\n(b=16)'])
c.set_xlim(400, 2650)
c.set_xlabel('tokens (batch x seq=128)', fontsize=10)
c.set_title('(c) vs tokens, batched  (d=64, seq=128)\nzk=0 not measured for batch configs', fontsize=9.5)
c.legend(fontsize=8.5, loc='lower right', framealpha=0.9)

fig.suptitle('ZK proving overhead at scale: one full composed transformer LAYER, like-for-like  (RTX 4090, all zk=1 points verify_ok=1)',
             fontsize=12, y=0.99)
fig.text(0.5, 0.005,
         'Denominator = batched-attention native fp8 forward (the honest baseline; the canonical per-op mode is launch-bound, ~90% Python overhead at large batch).   '
         'Integerized = measured int-GEMM proofs\n(p3_matmul_bench2) + the same composed run\'s measured zk=1 non-matmul stages.   '
         'No extrapolated points: configs without both a proof and a batched-forward measurement are omitted.',
         fontsize=8, color=MUT, ha='center', va='bottom')
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.savefig('/root/overhead_zk_at_scale.png', dpi=140)
print('wrote /root/overhead_zk_at_scale.png')
