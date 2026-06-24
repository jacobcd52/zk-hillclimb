"""Is one matmul's fp32 accumulation-noise predictable from its operands?
Real layer (down_proj) + real activations. delta = _scaled_mm(fp32 accum) - exact(fp64).
Predict (a) from summary features, (b) from the FULL per-term product vector (max info).
Report sign-accuracy (the decisive number for fixing rank flips) + R^2 of delta and |delta|."""
import os, sys, gc, math
HERE="/workspace/projects/zk-hillclimb/capacity"; sys.path.insert(0, HERE)
os.environ.setdefault("IMA_TEACHER_KERNEL","fp8_scaled_mm")
import numpy as np, torch, torch.nn as nn
import rank_entropy_sweep as S
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
BASE="Qwen/Qwen2.5-0.5B"; FP8MAX=448.0

def per_tok_fp8(x):
    s=(x.abs().amax(-1,keepdim=True)/FP8MAX).clamp_min(1e-12)
    return (x/s).to(torch.float8_e4m3fn), s.float()
def per_row_fp8(w):
    s=(w.abs().amax(-1,keepdim=True)/FP8MAX).clamp_min(1e-12)
    return (w/s).to(torch.float8_e4m3fn), s.float()

def r2(pred,y):
    pred=pred.double(); y=y.double(); ss=((y-y.mean())**2).sum()
    return float(1-((y-pred)**2).sum()/ss)

def main():
    with S.GpuLock():
        tok=AutoTokenizer.from_pretrained(BASE, token=S.HF_TOKEN)
        m=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, token=S.HF_TOKEN).to("cuda").eval()
        # capture real activations into a chosen down_proj (large K)
        layer=m.model.layers[8].mlp.down_proj   # in=4864 (intermediate), out=896
        cap={}
        h=layer.register_forward_pre_hook(lambda mod,inp: cap.__setitem__('x', inp[0].detach()))
        C=np.load(os.path.join(HERE,"corpus","mq_corpus.npz"),allow_pickle=True); ids=C["ids"]
        xs_all=[]
        with torch.no_grad():
            for i in range(6):
                idr=torch.from_numpy(ids[i].astype(np.int64))[None].cuda(); m(input_ids=idr)
                xs_all.append(cap['x'].reshape(-1, cap['x'].shape[-1]))
        h.remove()
        x=torch.cat(xs_all,0).float()        # [Ntok, K]
        w=layer.weight.detach().float()      # [out, K]
        K=x.shape[1]; out=w.shape[0]
        print(f"layer down_proj: K={K} out={out} Ntok={x.shape[0]}",flush=True)
        xq,sa=per_tok_fp8(x); wq,sb=per_row_fp8(w)
        # kernel (fp32 accumulation) vs exact (fp64)
        yk=torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=sb.reshape(1,-1), out_dtype=torch.bfloat16)  # [Ntok,out] (M_q real output)
        A=xq.float().double(); W=wq.float().double()
        ye=(A@W.t())*sa.double()*sb.reshape(1,-1).double()
        delta=(yk.double()-ye)               # [Ntok,out] accumulation noise (post-scale)
        absd=delta.abs()
        print(f"delta: sign-balance P(delta>0)={ (delta>0).float().mean().item():.4f}  "
              f"mean|delta|={absd.mean().item():.3e}  median|delta|={absd.median().item():.3e}",flush=True)
        # ---- features (summary) ----
        Su=(A@W.t())                          # unscaled exact sum
        L1=(A.abs()@W.abs().t())              # sum |products|
        L2=((A*A)@(W*W).t()).sqrt()
        F=torch.stack([Su, L1, L2, sa.double().expand(-1,out), sb.reshape(1,-1).double().expand(x.shape[0],-1)],-1)  # [Ntok,out,5]
        F=F.reshape(-1,5); d=delta.reshape(-1); ad=absd.reshape(-1)
        # normalize features
        Fm=F.mean(0,keepdim=True); Fs=F.std(0,keepdim=True)+1e-9; Fn=((F-Fm)/Fs).float()
        n=Fn.shape[0]; idx=torch.randperm(n); tr=idx[:int(.8*n)]; te=idx[int(.8*n):]
        def train_mlp(X,ytarget,classify,tr,te,epochs=40,hid=128):
            din=X.shape[1]; net=nn.Sequential(nn.Linear(din,hid),nn.GELU(),nn.Linear(hid,hid),nn.GELU(),nn.Linear(hid,1)).cuda()
            opt=torch.optim.Adam(net.parameters(),1e-3)
            Xtr,Xte=X[tr],X[te]; ytr=ytarget[tr]; yte=ytarget[te]
            for e in range(epochs):
                for b in range(0,len(Xtr),65536):
                    xb=Xtr[b:b+65536]; yb=ytr[b:b+65536]; p=net(xb).squeeze(-1)
                    loss=(nn.functional.binary_cross_entropy_with_logits(p,yb) if classify else ((p-yb)**2).mean())
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                p=net(Xte).squeeze(-1)
                if classify: return ((p>0).float()==yte).float().mean().item()
                else: return r2(p, yte)
        sign=(d>0).float()
        accF=train_mlp(Fn, sign, True, tr, te)
        r2dF=train_mlp(Fn, ((d-d.mean())/(d.std()+1e-9)).float(), False, tr, te)
        r2adF=train_mlp(Fn, ((ad-ad.mean())/(ad.std()+1e-9)).float(), False, tr, te)
        print(f"[FEATURES->] sign-acc={accF:.4f}  R2(delta)={r2dF:.3f}  R2(|delta|)={r2adF:.3f}",flush=True)
        # ---- full per-term products (max info), subsample ----
        nsub=60000; ii=torch.randint(0,x.shape[0],(nsub,)); jj=torch.randint(0,out,(nsub,))
        P=(A[ii]*W[jj]).float()               # [nsub,K] exact products
        dsub=delta[ii,jj].float(); signsub=(dsub>0).float()
        Pm=P.mean(0,keepdim=True); Ps=P.std(0,keepdim=True)+1e-9; Pn=((P-Pm)/Ps)
        idx2=torch.randperm(nsub); tr2=idx2[:int(.8*nsub)]; te2=idx2[int(.8*nsub):]
        accP=train_mlp(Pn, signsub, True, tr2, te2, epochs=60, hid=256)
        print(f"[FULL-PRODUCTS->] sign-acc={accP:.4f}  (chance=0.5; ~0.5 => sign is NOT predictable from operands)",flush=True)
        print("RESULT_DONE",flush=True)

if __name__=="__main__": main()
