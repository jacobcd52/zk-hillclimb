"""Does the accumulation method (K0) actually matter in the OUTPUT dtype (bf16/fp8)?
If K0-variants agree in bf16, proving fp32-lossy accumulation is pointless vs proving exact-int."""
import torch, numpy as np
d='cuda'; torch.manual_seed(0)
def q(t):
    s=(t.abs().amax(-1,keepdim=True)/448).clamp_min(1e-12)
    return (t/s).to(torch.float8_e4m3fn), s.float()
def canon(xd,wd,K0):
    # fp8 values xd[B,K], wd[N,K]; block-wise fp32 accumulate with K0
    B,K=xd.shape; N=wd.shape[0]
    acc=torch.zeros(B,N,device=d,dtype=torch.float32)
    for b in range(0,K,K0):
        Sb=(xd[:,b:b+K0]@wd[:,b:b+K0].t())   # exact-ish block partial (fp32)
        acc=acc+Sb                            # fp32 add
    return acc
for K in [256,768,4096]:
    B,N=16,64
    x=torch.randn(B,K,device=d); w=torch.randn(N,K,device=d)
    xf,xs=q(x); wf,ws=q(w); xd=xf.float(); wd=wf.float()
    scale=xs*ws.reshape(1,-1)
    outs={}
    for K0 in [1,32,128,K]:
        outs[K0]=(canon(xd,wd,K0)*scale)
    kul=outs[K]  # exact-int-ish (single accumulate; = Kulisch in fp32)
    def cmp(a,b,dt): 
        A=a.to(dt).float(); Bb=b.to(dt).float(); return (A==Bb).float().mean().item()
    print(f"--- K={K} (output-dtype agreement across accumulation methods) ---")
    for K0 in [1,32,128]:
        print(f"  K0={K0:<4} vs Kulisch:  bf16-equal={cmp(outs[K0],kul,torch.bfloat16):.4f}   fp8-equal={cmp(outs[K0],kul,torch.float8_e4m3fn):.4f}   fp32-equal={cmp(outs[K0],kul,torch.float32):.4f}")
