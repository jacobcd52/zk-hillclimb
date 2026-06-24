"""Option A: FIX M_int (exact codebook), TRAIN M_q (fp8) to approximate it -> lower R_rank.
trainable = QATFp8Linear (M_q, fp8 with fp32 accum, STE); frozen ref = CodebookLinear (M_int).
Loss = MSE(M_q logits at M_int's top-m indices, M_int's top-m values)  [distill M_int -> M_q].
R_rank: served = argmax(z_q + g) [trained M_q], rank under (z_int + g) [frozen codebook]."""
import os, sys, gc, time, argparse
HERE="/workspace/projects/zk-hillclimb/capacity"; sys.path.insert(0, HERE)
os.environ.setdefault("IMA_TEACHER_KERNEL","fp8_scaled_mm")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import rank_entropy_sweep as S
from int_model_approximation.__main__ import CodebookLinear
import train_agree as TA   # QATFp8Linear, build pieces
from transformers import AutoModelForCausalLM
BASE="Qwen/Qwen2.5-0.5B"; EVAL_SEED=20260623; TARGETS=TA.TARGETS

def build(kind):  # 'mq'=trainable QATFp8 ; 'int'=frozen codebook
    m=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, token=S.HF_TOKEN).to("cuda")
    from llama_difr import quantize_fp8_per_row
    for layer in S._iter_decoder_layers(m):
        for parent in [getattr(layer,'self_attn',None), getattr(layer,'mlp',None)]:
            if parent is None: continue
            for name,mod in list(parent.named_children()):
                if isinstance(mod,nn.Linear) and name in TARGETS:
                    if kind=='mq':
                        new=TA.QATFp8Linear(S._true_weight(mod), mod.bias, train_w=True, train_bias=True)
                    else:
                        wf,ws=quantize_fp8_per_row(S._true_weight(mod).cuda()); new=CodebookLinear(wf,ws,mod.bias)
                    setattr(parent,name,new.to("cuda"))
    if kind=='int':
        for p in m.parameters(): p.requires_grad_(False)
        m.eval()
    return m

def lf(z): return z.logits.float()

@torch.no_grad()
def eval_rrank(mq, mint, idx, ids, mask):
    mq.eval(); ranks=[]
    for i in idx:
        idr=torch.from_numpy(ids[i].astype(np.int64))[None].cuda()
        zq=lf(mq(input_ids=idr))[0]; zi=lf(mint(input_ids=idr))[0]; n=zq.shape[0]
        gen=torch.Generator(device='cuda'); gen.manual_seed(EVAL_SEED+int(i))
        g=torch.empty_like(zq); g.exponential_(generator=gen).log_().neg_()
        served=(zq+g).argmax(-1)                    # M_q (trained) gumbel-argmax, full vocab
        zr=zi+g; s=zr.gather(1,served[:,None])
        idr2=torch.arange(zr.shape[-1],device=zr.device)[None,:]
        rank=(zr>s).sum(-1)+((zr==s)&(idr2<served[:,None])).sum(-1)
        m=torch.from_numpy(mask[i]).cuda(); ranks.append(rank[m].cpu().numpy())
    mq.train(); r=np.concatenate(ranks); v,c=np.unique(r,return_counts=True); p=c/c.sum()
    return float(-(p*np.log2(p)).sum()), float((r==0).mean())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--lr",type=float,default=3e-7); ap.add_argument("--topm",type=int,default=16)
    ap.add_argument("--bs",type=int,default=2); ap.add_argument("--max_steps",type=int,default=1500); ap.add_argument("--eval_every",type=int,default=200)
    ap.add_argument("--neval",type=int,default=200); ap.add_argument("--sharpen",type=float,default=0.0); ap.add_argument("--tag",default="mq")
    a=ap.parse_args()
    C=np.load(os.path.join(HERE,"corpus","mq_corpus.npz"),allow_pickle=True); ids=C["ids"]; mask=C["mask"]
    perm=np.random.default_rng(0).permutation(len(ids)); held=perm[:len(ids)//10]; train=perm[len(ids)//10:]
    with S.GpuLock():
        mq=build('mq'); mint=build('int')
        try: mq.config.use_cache=False; mq.gradient_checkpointing_enable()
        except: pass
        opt=torch.optim.Adam([p for p in mq.parameters() if p.requires_grad], a.lr)
        r0,f0=eval_rrank(mq,mint,held[:a.neval],ids,mask); print(f"[init] R_rank={r0:.4f} frac0={f0:.4f}",flush=True)
        best=r0; step=0; t0=time.time()
        for bstart in range(0,len(train),a.bs):
            if step>=a.max_steps: break
            sel=list(train[bstart:bstart+a.bs])
            L=max(len(ids[i]) for i in sel); bi=torch.zeros(len(sel),L,dtype=torch.long); am=torch.zeros(len(sel),L,dtype=torch.long)
            for r,i in enumerate(sel): nn_=len(ids[i]); bi[r,:nn_]=torch.from_numpy(ids[i].astype(np.int64)); am[r,:nn_]=1
            bi=bi.cuda(); am=am.cuda()
            zq=lf(mq(input_ids=bi,attention_mask=am))
            with torch.no_grad(): zi=lf(mint(input_ids=bi,attention_mask=am))
            tv,ti=zi.topk(a.topm,dim=-1)
            if a.sharpen>0: tv=tv*(1.0+a.sharpen*torch.arange(a.topm,0,-1,device=tv.device)/a.topm)  # widen M_int margins as target
            zk=torch.gather(zq,2,ti)
            # completion mask
            cm=torch.zeros_like(bi,dtype=torch.bool)
            for r,i in enumerate(sel): nn_=len(ids[i]); cm[r,:nn_]=torch.from_numpy(mask[i])
            loss=(((zk-tv)**2).mean(-1)[cm]).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in mq.parameters() if p.requires_grad],1.0); opt.step()
            step+=1
            if step%a.eval_every==0:
                r,f=eval_rrank(mq,mint,held[:a.neval],ids,mask)
                print(f"[{step}] loss={float(loss):.4f} R_rank={r:.4f} frac0={f:.4f} ({time.time()-t0:.0f}s)",flush=True); best=min(best,r)
        print(f"[final] best_R_rank={best:.4f}",flush=True)

if __name__=="__main__": main()
