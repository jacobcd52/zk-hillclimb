import torch
d='cuda'; torch.manual_seed(1)
def q(t):
    s=(t.abs().amax(-1,keepdim=True)/448).clamp_min(1e-12)
    return (t/s).to(torch.float8_e4m3fn), s.float()
def canon(xd,wd,K0,scale):
    B,K=xd.shape; N=wd.shape[0]; acc=torch.zeros(B,N,device=d)
    for b in range(0,K,K0): acc=acc+(xd[:,b:b+K0]@wd[:,b:b+K0].t())
    return acc*scale
for K in [768,4096]:
    B,N=16,64; x=torch.randn(B,K,device=d); w=torch.randn(N,K,device=d)
    xf,xs=q(x); wf,ws=q(w); xd=xf.float(); wd=wf.float(); sc=xs*ws.reshape(1,-1)
    stock=torch._scaled_mm(xf,wf.t(),scale_a=xs,scale_b=ws.reshape(1,-1),out_dtype=torch.bfloat16).float()
    k32=canon(xd,wd,32,sc); kul=canon(xd,wd,K,sc)
    e=lambda a,b:(a.to(torch.bfloat16).float()==b.to(torch.bfloat16).float()).float().mean().item()
    print(f"K={K}: stock vs K0=32 (bf16)={e(stock,k32):.4f}   stock vs Kulisch (bf16)={e(stock,kul):.4f}   K0=32 vs Kulisch={e(k32,kul):.4f}")
