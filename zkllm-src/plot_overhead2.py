#!/usr/bin/env python3
# Like-for-like ONE-LAYER proving overhead (proof time / native forward time),
# same P3 proof system, from bench2_results.json.  Hawkeye exact-fp8 vs
# integerized -- both prove one full transformer layer, only the matmul differs.
import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=json.load(open('/root/zkllm/bench2_results.json'))
def ov(r,key):
    v=r[key]
    if key=='int_layer_s': p=v
    else: p=v['prove'] if v else None
    return p*1000.0/r['fwd_ms'] if p is not None else None

panels=[('model','d','model width  d   (seq=16, batch=1)'),
        ('seq','seq','sequence length   (d=64, batch=1)'),
        ('batch','batch','batch size   (d=64, seq=16)')]
fig,axes=plt.subplots(1,3,figsize=(15,5.2),sharey=True)
for ax,(grp,xk,title) in zip(axes,panels):
    rs=[r for r in rows if r['grp']==grp]; rs.sort(key=lambda r:r[xk])
    xs=[r[xk] for r in rs]
    hw1=[ov(r,'fp8_zk1') for r in rs]
    hw0=[ov(r,'fp8_zk0') for r in rs]
    it =[ov(r,'int_layer_s') for r in rs]
    ax.plot(xs,hw1,'o-', color='#c0392b',lw=2.4,ms=9,label='Hawkeye exact-fp8  (ZK)')
    ax.plot(xs,hw0,'s--',color='#e67e22',lw=1.8,ms=6,label='Hawkeye exact-fp8  (no-ZK)')
    ax.plot(xs,it ,'^-', color='#2980b9',lw=2.4,ms=9,label='Integerized  (int GEMMs, ZK)')
    ax.set_yscale('log'); ax.set_xscale('log',base=2)
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_title(title,fontsize=11); ax.set_xlabel(xk,fontsize=11)
    ax.grid(True,which='both',ls=':',alpha=0.4); ax.legend(fontsize=8.5,loc='upper left')
axes[0].set_ylabel('proving overhead  =  proof time / forward time   (x)',fontsize=11)
fig.suptitle('ZK proving overhead per transformer LAYER  (RTX 4090, like-for-like, same P3 system, NO OOM)\n'
             'both prove one full layer; only the matmul differs -- Hawkeye = bit-exact H100 fp8, Integerized = plain int GEMMs',
             fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig('/root/overhead_layer.png',dpi=130); print('wrote /root/overhead_layer.png')
for grp,xk,_ in panels:
    print(f'\n== {grp} ==')
    for r in sorted([r for r in rows if r['grp']==grp],key=lambda r:r[xk]):
        print(f"  {xk}={r[xk]:4} T={r['tokens']:4} fwd={r['fwd_ms']:5.1f}ms  "
              f"HWzk={ov(r,'fp8_zk1'):7.0f}x  HWnozk={ov(r,'fp8_zk0'):6.0f}x  INT={ov(r,'int_layer_s'):5.0f}x  "
              f"(HW/INT={ov(r,'fp8_zk1')/ov(r,'int_layer_s'):.0f}x)")
