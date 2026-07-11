#!/usr/bin/env python3
"""SLICED one-layer overhead: prove the layer OP-BY-OP (each matmul a separate
p3_hawkeye_bench / p3_matmul_bench2 call, memory reused) so we reach high seq
with NO OOM.  Attention's O(S^2) matmuls are tiled over query blocks (Cq) so
each sub-GEMM fits.  Also measures native forward + GPU utilization to explain
the overhead-vs-seq shape.

  fp8 layer proof  = sum of p3_hawkeye_bench PROVE over the layer's matmuls
  int layer proof  = sum of p3_matmul_bench2 PROVE  over the same matmuls
  (both sliced identically; a merged-opening monolithic proof is ~2x cheaper --
   we anchor that factor against bench2_results.json below.)
Writes bench3_results.json.
"""
import json, re, subprocess, math
ZK='/root/zkllm'; HB='/root/p3_hawkeye_bench'; IMM='/root/p3_matmul_bench2'
CQ=128                                   # attention query-block tile
def lg(x): return max(0,(x-1).bit_length())
def sh(c,to=1200): return subprocess.run(c,capture_output=True,text=True,cwd=ZK,timeout=to)

_hb={}; _im={}
def hb(B,K,N):                           # fp8 GEMM prove seconds (cached)
    k=(B,K,N)
    if k not in _hb:
        p=sh([HB,str(B),str(K),str(N)],to=1200)
        m=re.search(r'PROVE\s+([\d.]+)\s*s',p.stdout)
        _hb[k]=float(m.group(1)) if m else None
        if _hb[k] is None: print('  hb FAIL',k,p.stdout[-200:],p.stderr[-200:])
    return _hb[k]
def im(B,IN,OUT):                        # int GEMM prove seconds (cached)
    k=(B,IN,OUT)
    if k not in _im:
        p=sh([IMM,str(lg(B)),str(lg(IN)),str(lg(OUT))],to=600)
        m=re.search(r'prove_ms=([\d.]+)',p.stdout); _im[k]=float(m.group(1))/1000 if m else None
    return _im[k]

def layer_matmuls(seq,d,nh,dh,dff,batch):
    """yield (count, B,K,N) for every matmul in the layer, attention query-tiled."""
    T=batch*seq; A=batch*nh
    yield (4, T,d,d)                      # Wq,Wk,Wv,Wo
    yield (2, T,d,dff)                    # Wg,Wu
    yield (1, T,dff,d)                    # Wd
    nqb=max(1,seq//CQ); cq=min(CQ,seq)
    yield (A*nqb, cq,dh,seq)             # QK^T, query-tiled
    yield (A*nqb, cq,seq,dh)             # P.V , query-tiled

def sliced(seq,d,nh,dh,dff,batch,fn):
    return sum(cnt*fn(B,K,N) for (cnt,B,K,N) in layer_matmuls(seq,d,nh,dh,dff,batch))

def fwd(seq,d,nh,dh,dff,batch):
    p=sh(['python3','tf_fwd_bench.py',str(seq),str(d),str(nh),str(dh),str(dff),str(batch),'30'],to=600)
    m=re.search(r'fwd_ms=([\d.]+)',p.stdout); return float(m.group(1)) if m else None

def flops(seq,d,nh,dh,dff,batch):        # layer forward MACs*2
    T=batch*seq; A=batch*nh
    proj=(4*T*d*d + 2*T*d*dff + 1*T*dff*d)
    attn=A*(seq*seq*dh + seq*seq*dh)
    return 2.0*(proj+attn)

PEAK=330e12                              # 4090 bf16 tensor TFLOP/s (fp8 ~2x; util is relative)
def params(d,dff): return 4*d*d + 3*d*dff   # per-layer weights (QKVO + gate/up/down)

CFG=[]
# model sweep (seq=16,batch=1)
for d in (64,128,256,512): CFG.append(('model',d,16,1))
# seq sweep at batch=1 (UNDER-utilized fwd) -> to 1024
for s in (16,64,128,256,512,1024): CFG.append(('seq_b1',64,s,1))
# seq sweep at batch=16 (well-utilized fwd) -> to 1024
for s in (16,64,128,256,512,1024): CFG.append(('seq_b16',64,s,16))
# batch sweep at seq=64 -> to 64
for b in (1,4,16,64): CFG.append(('batch',64,64,b))

rows=[]
for grp,d,seq,batch in CFG:
    nh=d//16; dh=16; dff=4*d; T=batch*seq
    fw=fwd(seq,d,nh,dh,dff,batch)
    fl=flops(seq,d,nh,dh,dff,batch)
    util=fl/((fw/1000)*PEAK) if fw else None
    r=dict(grp=grp,d=d,seq=seq,batch=batch,nh=nh,dh=dh,dff=dff,tokens=T,
           params=params(d,dff),fwd_ms=fw,gflop=fl/1e9,util=util,
           fp8_s=sliced(seq,d,nh,dh,dff,batch,hb),
           int_s=sliced(seq,d,nh,dh,dff,batch,im))
    rows.append(r)
    print(f"{grp:8} d={d:4} seq={seq:5} b={batch:3} params={r['params']/1e6:5.2f}M "
          f"fwd={fw:6.1f}ms util={util*100:5.2f}% fp8={r['fp8_s']:7.1f}s int={r['int_s']:6.2f}s "
          f"| ovh fp8={r['fp8_s']*1000/fw:8.0f}x int={r['int_s']*1000/fw:6.0f}x",flush=True)
json.dump(rows,open(f'{ZK}/bench3_results.json','w'),indent=1)
print('wrote bench3_results.json')
