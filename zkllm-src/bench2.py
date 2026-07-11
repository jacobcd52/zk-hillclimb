#!/usr/bin/env python3
"""Like-for-like ONE-LAYER overhead: exact-fp8 (Hawkeye) vs integerized, SAME
P3 proof system.  Both prove one full transformer layer; the ONLY difference is
the matmul gadget (Hawkeye fp8 vs plain integer GEMM) -- every non-matmul op
(rmsnorm/rope/softmax/swiglu/residual/quantize) is the identical P3 gadget.

  fp8 layer proof     = p3_transformer_bench (measured, full layer).
  integer layer proof = SUM of integer GEMM proofs for the SAME layer's matmuls
                        (4 proj + 2 mlp-up + 1 mlp-down + 2*batch*nh attention)
                        + the layer's non-matmul stages (taken identical to the
                        fp8 run: rms+qnt+rope+smx+bfa+swg+seam).
  forward             = tf_fwd_bench.py (native fp8, per layer).

All configs kept under the 41 GB cgroup cap (products <= ~1M) so NO OOMs.
Writes bench2_results.json.
"""
import json, re, subprocess, math, os
ZK='/root/zkllm'; BENCH='/root/p3_transformer_bench'; IMM='/root/p3_matmul_bench2'
ART={6:'p3_rmsnorm_tables.bin',7:'p3_rmsnorm_tables_ld7.bin',8:'p3_rmsnorm_tables_ld8.bin'}
def lg(x): return x.bit_length()-1
def sh(cmd,to=2400):
    return subprocess.run(cmd,capture_output=True,text=True,cwd=ZK,timeout=to)

_immcache={}
def imm(B,IN,OUT):                       # integer GEMM prove ms (cached)
    k=(B,IN,OUT)
    if k not in _immcache:
        p=sh([IMM,str(lg(B)),str(lg(IN)),str(lg(OUT))],to=600)
        m=re.search(r'prove_ms=([\d.]+)',p.stdout)
        _immcache[k]=float(m.group(1)) if m else None
    return _immcache[k]

def int_layer_matmul_ms(seq,d,nh,dh,dff,batch):
    T=batch*seq; A=batch*nh; tot=0.0
    tot+=4*imm(T,d,d)                     # Wq,Wk,Wv,Wo
    tot+=2*imm(T,d,dff)                   # Wg,Wu
    tot+=1*imm(T,dff,d)                   # Wd
    tot+=A*imm(seq,dh,seq)               # QK^T per (batch,head)
    tot+=A*imm(seq,seq,dh)              # P.V  per (batch,head)
    return tot

def fp8_layer(seq,d,nh,dh,dff,batch,zk):
    art=ART[lg(d)]
    p=sh([BENCH,*map(str,(seq,d,nh,dh,dff,batch,int(zk))),art])
    if p.returncode==137 or 'BENCH' not in p.stdout: return None
    m=re.search(r'prove=([\d.]+) verify=([\d.]+) proof_mb=([\d.]+) rss_gb=([\d.]+)',p.stdout)
    st=re.search(r'STAGES rms=([\d.]+) qnt=([\d.]+) mm=([\d.]+) rope=([\d.]+) smx=([\d.]+) '
                 r'bfa=([\d.]+) swg=([\d.]+) lug=([\d.]+) seam=([\d.]+) batch=([\d.]+)',p.stdout)
    r=dict(zip(('prove','verify','proof_mb','rss_gb'),map(float,m.groups())))
    if st:
        s=dict(zip(('rms','qnt','mm','rope','smx','bfa','swg','lug','seam','batch'),map(float,st.groups())))
        r['mm']=s['mm']; r['nonmm']=s['rms']+s['qnt']+s['rope']+s['smx']+s['bfa']+s['swg']+s['seam']
    return r

def fwd(seq,d,nh,dh,dff,batch):
    p=sh(['python3','tf_fwd_bench.py',*map(str,(seq,d,nh,dh,dff,batch,30))],to=600)
    m=re.search(r'fwd_ms=([\d.]+)',p.stdout); return float(m.group(1)) if m else None

# config grid -- ALL under ~1M products so fp8 fits in 41GB (no OOM).
# dh=16 fixed, nh=d/16, dff=4d.
def cfgs():
    out=[]
    for d in (64,128,256):                       out.append(('model',d, 16,1))  # seq16 batch1
    for s in (16,64,128,256):                    out.append(('seq',  64, s, 1))  # d64 batch1
    for b in (1,4,8,16):                         out.append(('batch',64, 16,b))  # d64 seq16
    return out

rows=[]
for grp,d,seq,batch in cfgs():
    nh=d//16; dh=16; dff=4*d; T=batch*seq
    r={'grp':grp,'d':d,'seq':seq,'batch':batch,'nh':nh,'dh':dh,'dff':dff,'tokens':T,
       'products':T*d*d}
    r['fwd_ms']=fwd(seq,d,nh,dh,dff,batch)
    r['fp8_zk1']=fp8_layer(seq,d,nh,dh,dff,batch,1)
    r['fp8_zk0']=fp8_layer(seq,d,nh,dh,dff,batch,0)
    r['int_mm_ms']=int_layer_matmul_ms(seq,d,nh,dh,dff,batch)
    # integer full layer = int matmuls + the SAME non-matmul gadget stages (s)
    nonmm = (r['fp8_zk0'] or {}).get('nonmm')
    r['int_layer_s'] = (r['int_mm_ms'] + (nonmm*1000 if nonmm else 0))/1000.0
    rows.append(r)
    print(f"{grp:6} d={d:4} seq={seq:4} b={batch:3} T={T:5} prod={r['products']:>8} "
          f"fwd={r['fwd_ms']:.1f}ms  fp8zk1={r['fp8_zk1']['prove'] if r['fp8_zk1'] else 'OOM':>7} "
          f"fp8zk0={r['fp8_zk0']['prove'] if r['fp8_zk0'] else 'OOM':>7} intL={r['int_layer_s']:.2f}s",flush=True)

json.dump(rows,open('/root/zkllm/bench2_results.json','w'),indent=1)
print("wrote bench2_results.json")
