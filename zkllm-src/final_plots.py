#!/usr/bin/env python3
import json,subprocess,re,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ZK='/root/zkllm'
def lg(x): return max(0,(x-1).bit_length())
def imm(B,IN,OUT):
    p=subprocess.run(['/root/p3_matmul_bench2',str(lg(B)),str(lg(IN)),str(lg(OUT))],
                     capture_output=True,text=True,cwd=ZK,timeout=300)
    m=re.search(r'prove_ms=([\d.]+)',p.stdout); return float(m.group(1))/1000 if m else 0
def int_layer(seq,d,nh,dh,dff,batch,nonmm):
    T=batch*seq; A=batch*nh
    mm=4*imm(T,d,d)+2*imm(T,d,dff)+imm(T,dff,d)+A*imm(seq,dh,seq)+A*imm(seq,seq,dh)
    return mm+nonmm
def params(d): return 4*d*d+3*d*(4*d)
def util(seq,d,nh,dh,dff,batch,fwd_ms):
    T=batch*seq;A=batch*nh; fl=2.0*((4*T*d*d+2*T*d*dff+T*dff*d)+A*(2*seq*seq*dh))
    return fl/((fwd_ms/1e3)*330e12)*100

# ---- ACCURATE monolithic data (measured) ----
# model sweep: seq=16 batch=1.  (d, params, fwd_ms, fp8_zk1, fp8_zk0, nonmm)
MODEL=[(64,4.7,9.904,2.942,0.42),(128,8.2,29.884,7.709,0.63),
       (256,14.1,108.419,20.796,1.156),(512,27.6,None,73.881,2.307)]
# seq sweep: d=64 batch=1.  (seq, fwd_ms, zk1, zk0, nonmm)
SEQ=[(16,4.3,9.645,3.012,0.42),(64,4.8,32.548,8.068,0.55),(128,4.4,74.052,16.723,0.75),
     (256,4.7,179.857,33.279,1.05),(512,4.41,None,91.615,1.6)]
# batch sweep: d=64 seq=16.  (batch, fwd_ms, zk1, zk0)
BATCH=[(1,4.3,9.631,2.861),(4,15.8,35.035,9.503),(8,27.8,72.169,17.005),(16,53.7,139.076,32.284)]
# forward util (measured): seq (batch1) and batch (seq64), d=64
USEQ=[(16,4.34),(64,4.46),(256,4.45),(1024,4.45),(4096,29.21)]
UBAT=[(1,4.34),(4,15.26),(16,56.09),(64,217.93),(256,884.6)]

fig,ax=plt.subplots(1,4,figsize=(20,5),)
# Panel A: overhead vs PARAMS
d64nh=lambda d:d//16
xs=[params(d)/1e6 for d,_,_,_,_ in MODEL]
a=ax[0]
a.plot(xs,[z*1000/f if z else None for d,f,z,_,_ in MODEL],'o-',color='#c0392b',lw=2.4,ms=9,label='Hawkeye fp8 (ZK)')
a.plot(xs,[z*1000/f for d,f,_,z,_ in MODEL],'s--',color='#e67e22',lw=1.8,ms=6,label='Hawkeye fp8 (no-ZK)')
intv=[int_layer(16,d,d//16,16,4*d,1,nm)*1000/f for d,f,_,_,nm in MODEL]
a.plot(xs,intv,'^-',color='#2980b9',lw=2.4,ms=9,label='Integerized (ZK)')
a.set_xscale('log');a.set_yscale('log');a.set_xlabel('model parameters per layer (M)',fontsize=11)
a.set_ylabel('proving overhead = proof / forward  (x)',fontsize=11);a.set_title('vs MODEL PARAMS  (seq=16, batch=1)',fontsize=11)
a.set_xticks(xs);a.set_xticklabels([f'{x:.2f}' for x in xs]);a.grid(True,which='both',ls=':',alpha=0.4);a.legend(fontsize=8.5)
# Panel B: overhead vs SEQ (batch=1)
xs=[s for s,_,_,_,_ in SEQ]; b=ax[1]
b.plot(xs,[z*1000/f if z else None for s,f,z,_,_ in SEQ],'o-',color='#c0392b',lw=2.4,ms=9,label='Hawkeye fp8 (ZK)')
b.plot(xs,[z*1000/f for s,f,_,z,_ in SEQ],'s--',color='#e67e22',lw=1.8,ms=6,label='Hawkeye fp8 (no-ZK)')
b.plot(xs,[int_layer(s,64,4,16,256,1,nm)*1000/f for s,f,_,_,nm in SEQ],'^-',color='#2980b9',lw=2.4,ms=9,label='Integerized (ZK)')
b.set_xscale('log',base=2);b.set_yscale('log');b.set_xlabel('sequence length',fontsize=11);b.set_title('vs SEQ  (d=64, batch=1)  [forward is FLAT -> ratio grows]',fontsize=10)
b.set_xticks(xs);b.set_xticklabels([str(x) for x in xs]);b.grid(True,which='both',ls=':',alpha=0.4);b.legend(fontsize=8.5)
# Panel C: overhead vs BATCH
xs=[bb for bb,_,_,_ in BATCH]; c=ax[2]
c.plot(xs,[z*1000/f for bb,f,z,_ in BATCH],'o-',color='#c0392b',lw=2.4,ms=9,label='Hawkeye fp8 (ZK)')
c.plot(xs,[z*1000/f for bb,f,_,z in BATCH],'s--',color='#e67e22',lw=1.8,ms=6,label='Hawkeye fp8 (no-ZK)')
c.plot(xs,[int_layer(16,64,4,16,256,bb,0.42*bb**0)*1000/f for bb,f,_,_ in BATCH],'^-',color='#2980b9',lw=2.4,ms=9,label='Integerized (ZK)')
c.set_xscale('log',base=2);c.set_yscale('log');c.set_xlabel('batch size',fontsize=11);c.set_title('vs BATCH  (d=64, seq=16)  [forward scales -> ratio flat]',fontsize=10)
c.set_xticks(xs);c.set_xticklabels([str(x) for x in xs]);c.grid(True,which='both',ls=':',alpha=0.4);c.legend(fontsize=8.5)
# Panel D: WHY -- forward time (flat) + GPU utilization (~0)
d=ax[3]
d.plot([s for s,_ in USEQ],[f for _,f in USEQ],'o-',color='#16a085',lw=2.2,ms=8,label='fwd ms vs SEQ (batch=1)')
d.plot([b*64 for b,_ in UBAT],[f for _,f in UBAT],'D--',color='#8e44ad',lw=2,ms=7,label='fwd ms vs TOKENS (batch sweep)')
d.set_xscale('log',base=2);d.set_yscale('log');d.set_xlabel('sequence length (batch=1)  /  tokens (batch sweep)',fontsize=10)
d.set_ylabel('native forward time (ms)',fontsize=11)
d.set_title('WHY: forward is LAUNCH-BOUND\nflat vs seq (util 0.001-0.05%)',fontsize=10)
d.grid(True,which='both',ls=':',alpha=0.4);d.legend(fontsize=8.5,loc='upper left')
d.annotate('flat 4.4ms\nseq16->1024\n(GPU idle)',(256,4.45),(24,9),fontsize=8,color='#16a085',
           arrowprops=dict(arrowstyle='->',color='#16a085'))
fig.suptitle('ZK proving overhead per transformer LAYER, and WHY it grows with seq  (RTX 4090, d=64 unless noted)\n'
             'Hawkeye = bit-exact H100 fp8 matmuls;  Integerized = plain int GEMMs;  same P3 system, like-for-like one layer',fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.92]); fig.savefig('/root/overhead_final.png',dpi=125)
print('wrote /root/overhead_final.png')
