"""Verify the harness R_rank (~0.64 for the codebook on this base model) is real, not a
top-64-served artifact. M_q=FP8Linear, M_int=CodebookLinear (exact int, = the original
optionb comparison). Compute, on held-out seqs with FULL logits:
  - R_rank with served = argmax(z_q+g) over FULL vocab  (clean, no top-64 approx)
  - R_rank with served = argmax over M_q top-64 (the harness approximation)
  - R_topK (rank within M_q top-64)
  - competitor decomposition: of tokens outranking served under M_int, how many are
    inside vs outside M_q's top-64.
"""
import os, sys, gc
HERE="/workspace/projects/zk-hillclimb/capacity"; sys.path.insert(0, HERE)
os.environ.setdefault("IMA_TEACHER_KERNEL","fp8_scaled_mm")
import numpy as np, torch
import rank_entropy_sweep as S
from int_model_approximation.__main__ import FP8Linear, CodebookLinear
from transformers import AutoModelForCausalLM
BASE="Qwen/Qwen2.5-0.5B"; EVAL_SEED=20260623; NSEQ=80

def build(factory):
    m=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, token=S.HF_TOKEN).to("cuda").eval()
    S.replace_linears_robust(m, factory); return m

def rank_of(zr, tok):  # rank of token tok under zr [L,V], token-id tiebreak
    s=zr.gather(1,tok[:,None]); ids=torch.arange(zr.shape[-1],device=zr.device)[None,:]
    return (zr>s).sum(-1) + ((zr==s)&(ids<tok[:,None])).sum(-1)

def entropy(r):
    v,c=np.unique(r,return_counts=True); p=c/c.sum(); return float(-(p*np.log2(p)).sum())

def main():
    C=np.load(os.path.join(HERE,"corpus","mq_corpus.npz"),allow_pickle=True)
    ids=C["ids"]; mask=C["mask"]
    perm=np.random.default_rng(0).permutation(len(ids)); held=perm[:len(ids)//10][:NSEQ]
    with S.GpuLock():
        mq=build(lambda w,s,b: FP8Linear(w,s,b))
        mint=build(lambda w,s,b: CodebookLinear(w,s,b))
        rf=[]; rt=[]; rk=[]; cin=[]; cout=[]
        with torch.no_grad():
            for i in held:
                idr=torch.from_numpy(ids[i].astype(np.int64))[None].cuda()
                zq=mq(input_ids=idr).logits.float()[0]      # [L,V]
                zi=mint(input_ids=idr).logits.float()[0]
                L=zq.shape[0]
                gen=torch.Generator(device='cuda'); gen.manual_seed(EVAL_SEED+int(i))
                g=torch.empty_like(zq); g.exponential_(generator=gen).log_().neg_()
                zqg=zq+g; zig=zi+g
                served_full=zqg.argmax(-1)                  # full-vocab gumbel argmax of M_q
                t64=zq.topk(64,dim=-1).indices               # M_q top-64 [L,64]
                gat=g.gather(1,t64); served_t64=t64.gather(1,(zq.gather(1,t64)+gat).argmax(1,keepdim=True)).squeeze(1)
                rkfull=rank_of(zig, served_full)             # rank under M_int (full)
                rkt64 =rank_of(zig, served_t64)
                # R_topK: rank of served_full within M_q top-64 set, under M_int
                zik=zig.gather(1,t64); s=zig.gather(1,served_full[:,None])
                gtk=(zik>s).sum(-1) + ((zik==s)&(t64<served_full[:,None])).sum(-1)
                # competitor decomposition for served_full
                comp=(zig > s)                               # [L,V] bool, who outranks served
                inmask=torch.zeros_like(comp); inmask.scatter_(1,t64,True)
                c_in=(comp & inmask).sum(-1); c_out=(comp & ~inmask).sum(-1)
                m=torch.from_numpy(mask[i]).cuda()
                rf.append(rkfull[m].cpu().numpy()); rt.append(rkt64[m].cpu().numpy()); rk.append(gtk[m].cpu().numpy())
                cin.append(c_in[m].cpu().numpy()); cout.append(c_out[m].cpu().numpy())
                del zq,zi,g,zqg,zig,comp,inmask; torch.cuda.empty_cache()
    rf=np.concatenate(rf); rt=np.concatenate(rt); rk=np.concatenate(rk); cin=np.concatenate(cin); cout=np.concatenate(cout)
    print(f"n_completion_tokens={rf.size}")
    print(f"R_rank (served=FULL-vocab argmax, clean)   = {entropy(rf):.4f}   frac_rank0={ (rf==0).mean():.4f}  mean_rank={rf.mean():.3f}  max={rf.max()}")
    print(f"R_rank (served=top-64 argmax, harness way) = {entropy(rt):.4f}   frac_rank0={ (rt==0).mean():.4f}")
    print(f"R_topK (within M_q top-64)                 = {entropy(rk):.4f}   frac0={ (rk==0).mean():.4f}")
    print(f"competitors per token: inside M_q top-64 = {cin.mean():.3f}   OUTSIDE = {cout.mean():.3f}")
    print(f"of non-rank0 tokens, mean outside competitors = {cout[rf>0].mean():.3f}  inside = {cin[rf>0].mean():.3f}")

if __name__=="__main__": main()
