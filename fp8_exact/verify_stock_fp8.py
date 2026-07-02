"""Independent check of Fable's load-bearing claim: is stock Ada _scaled_mm == a clean fp32
sequential-RNE accumulation of the same fp8 products? And how close is it to exact-integer?"""
import torch, numpy as np
d='cuda'; torch.manual_seed(0)
B,K,N=16,256,64
x=torch.randn(B,K,device=d); w=torch.randn(N,K,device=d)
def q(t):
    s=(t.abs().amax(-1,keepdim=True)/448).clamp_min(1e-12)
    return (t/s).to(torch.float8_e4m3fn), s.float()
xf,xs=q(x); wf,ws=q(w)
# stock kernel (bf16 out is the only row-wise option; also try fp32 accEmul via upcast path)
y_stock=torch._scaled_mm(xf,wf.t(),scale_a=xs,scale_b=ws.reshape(1,-1),out_dtype=torch.bfloat16).float()
# reference computations on the SAME fp8 operands:
xd=xf.float(); wd=wf.float()                       # decoded fp8 values (exact)
# (a) exact fp32 sequential RNE accumulation (torch matmul in fp32 = close to fp32 accumulate)
y_fp32=(xd@wd.t())*xs*ws.reshape(1,-1)
# (b) exact integer (fp8 values are int*2^-9; exact accumulation) then scale
xi=torch.round(xd*512).to(torch.int64); wi=torch.round(wd*512).to(torch.int64)
y_int=(((xi.double()@wi.t().double()))/(512.0*512.0))*xs.double()*ws.reshape(1,-1).double()
# compare in bf16-rounded space (stock is bf16)
def bf16(t): return t.to(torch.bfloat16).float()
sm=y_stock; a=bf16(y_fp32); b=bf16(y_int)
print(f"stock vs fp32-accum : bitwise-equal(bf16)={(sm==a).float().mean().item():.4f}  mean|rel|={((sm-a).abs()/(a.abs()+1e-6)).mean().item():.2e}")
print(f"stock vs exact-int  : bitwise-equal(bf16)={(sm==b).float().mean().item():.4f}  mean|rel|={((sm-b).abs()/(b.abs()+1e-6)).mean().item():.2e}")
print(f"fp32 vs exact-int   : bitwise-equal(bf16)={(a==b).float().mean().item():.4f}")
# how many bits does stock differ from fp32 in the mantissa (ulp-ish)?
diff=(sm-a).abs(); print(f"stock-vs-fp32 max abs diff={diff.max().item():.3e}, frac within 1e-3 rel={((sm-a).abs()/(a.abs()+1e-6)<1e-3).float().mean().item():.3f}")
