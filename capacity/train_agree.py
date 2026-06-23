"""Train an integer M_int to agree with the fp8 M_q (lower R_rank). Reads the cached
M_q on-policy corpus (gen_corpus.py). One epoch, select on held-out R_rank.

Modes (--mode): full_qat (train all int8 weights via STE) | bias (train per-out bias only)
               | frozen (baseline int8, no training)
Loss (--loss): topk_kl | hard_ce | gumbel   (+ --neartie to upweight small-margin positions)
"""
import os, sys, json, gc, time, argparse, math
HERE="/workspace/projects/zk-hillclimb/capacity"; sys.path.insert(0, HERE)
os.environ.setdefault("IMA_TEACHER_KERNEL","fp8_scaled_mm")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import rank_entropy_sweep as S
from transformers import AutoModelForCausalLM
BASE="Qwen/Qwen2.5-0.5B"
TARGETS=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
EVAL_SEED=20260623

def ste_round(x): return (torch.round(x)-x).detach()+x

class QATInt8Linear(nn.Module):
    def __init__(self, w, bias, train_w=False, train_bias=False):
        super().__init__()
        self.W=nn.Parameter(w.detach().clone().float(), requires_grad=train_w)
        b = (bias.detach().clone().float() if bias is not None else torch.zeros(w.shape[0]))
        self.bias=nn.Parameter(b, requires_grad=(train_bias or bias is not None))
        self.bias.requires_grad_(train_w or train_bias)
    def forward(self,x):
        w=self.W
        ws=(w.abs().amax(-1,keepdim=True)/127.0).clamp_min(1e-12)
        wq=(ste_round(w/ws).clamp(-127,127)*ws).float()
        xs=(x.abs().amax(-1,keepdim=True)/127.0).clamp_min(1e-12).float()
        xf=x.float(); xq=ste_round(xf/xs).clamp(-127,127)*xs
        y=torch.matmul(xq, wq.t())
        y=y+self.bias.float()
        return y.to(x.dtype)

def fq_fp8(x):  # per-last-dim E4M3 fake-quant with straight-through estimator (matches FP8Linear)
    s=(x.abs().amax(-1,keepdim=True)/448.0).clamp_min(1e-12)
    q=(x/s).to(torch.float8_e4m3fn).float()*s
    return x + (q - x).detach()

class QATFp8Linear(nn.Module):  # codebook/fp8 base: fp8 operands, fp32 accumulation (~exact-codebook starting pt)
    def __init__(self, w, bias, train_w=False, train_bias=False):
        super().__init__()
        self.W=nn.Parameter(w.detach().clone().float(), requires_grad=train_w)
        b=(bias.detach().clone().float() if bias is not None else torch.zeros(w.shape[0]))
        self.bias=nn.Parameter(b, requires_grad=(train_w or train_bias))
    def forward(self,x):
        wq=fq_fp8(self.W); xq=fq_fp8(x.float())
        y=torch.matmul(xq, wq.t())+self.bias.float()
        return y.to(x.dtype)

def build_mint(mode, base="int8", dtype="bf16"):
    dt = torch.float32 if dtype=="fp32" else torch.bfloat16
    m=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dt, token=S.HF_TOKEN).to("cuda")
    tw = mode=="full_qat"; tb = mode in ("full_qat","bias")
    Lin = QATInt8Linear if base=="int8" else QATFp8Linear
    for layer in S._iter_decoder_layers(m):
        for parent in [getattr(layer,'self_attn',None), getattr(layer,'mlp',None)]:
            if parent is None: continue
            for name,mod in list(parent.named_children()):
                if isinstance(mod,nn.Linear) and name in TARGETS:
                    setattr(parent,name, Lin(S._true_weight(mod), mod.bias, tw, tb).to("cuda"))
    return m

def trainable(m):
    return [p for p in m.parameters() if p.requires_grad]

def batches(idx, ids, mask, bs):
    for i in range(0,len(idx),bs):
        yield idx[i:i+bs]

def collate(sel, ids, mask):
    L=max(len(ids[i]) for i in sel)
    B=len(sel); pad=0
    bi=torch.full((B,L),pad,dtype=torch.long); am=torch.zeros(B,L,dtype=torch.long); cm=torch.zeros(B,L,dtype=torch.bool)
    for r,i in enumerate(sel):
        n=len(ids[i]); bi[r,:n]=torch.from_numpy(ids[i].astype(np.int64)); am[r,:n]=1; cm[r,:n]=torch.from_numpy(mask[i])
    return bi.cuda(), am.cuda(), cm.cuda()

def mint_logits(m, bi, am):
    return m(input_ids=bi, attention_mask=am).logits.float()  # [B,L,V]

class Head(nn.Module):  # frozen-base logit correction: per-vocab bias + low-rank(h)->vocab
    def __init__(self, V, d, r=64):
        super().__init__()
        self.bias=nn.Parameter(torch.zeros(V))
        self.Vm=nn.Parameter(torch.randn(r,d)*0.01)
        self.Um=nn.Parameter(torch.zeros(V,r))   # init 0 => correction starts at 0
    def forward(self, base_logits, h):
        return base_logits + self.bias[None,None,:] + (h.float() @ self.Vm.t()) @ self.Um.t()

def logits_of(m, head, bi, am):
    if head is None: return mint_logits(m, bi, am)
    out=m(input_ids=bi, attention_mask=am, output_hidden_states=True)
    return head(out.logits.float(), out.hidden_states[-1])

def loss_fn(z, sel, cm, tv, ti, argm, kind, neartie):
    # z:[B,L,V]; gather cached targets per (b,pos)
    B,L,V=z.shape
    tot=0.0; ntok=0
    for r,i in enumerate(sel):
        n=len(argm[i]); m=cm[r,:n]
        if m.sum()==0: continue
        zr=z[r,:n][m]                      # [t,V]
        tvi=torch.from_numpy(tv[i]).cuda().float()[m]   # [t,K]
        tii=torch.from_numpy(ti[i]).cuda().long()[m]    # [t,K]
        ai =torch.from_numpy(argm[i]).cuda().long()[m]  # [t]
        if kind=="hard_ce":
            l=F.cross_entropy(zr, ai, reduction='none')
        elif kind=="topk_kl":
            p=F.softmax(tvi,dim=-1)
            zk=torch.gather(zr,1,tii)
            logq=F.log_softmax(zk,dim=-1)
            l=(p*(torch.log(p+1e-12)-logq)).sum(-1)
        elif kind=="gumbel":
            g=torch.empty_like(tvi); gen=torch.Generator(device='cuda'); gen.manual_seed(int(1234+i))
            g.exponential_(generator=gen).log_().neg_()
            served=tii.gather(1,(tvi+g).argmax(1,keepdim=True)).squeeze(1)  # [t]
            l=F.cross_entropy(zr, served, reduction='none')
        elif kind=="mse5":
            # MSE between M_int's logits and M_q's VALUES at M_q's top-5 tokens, in fp32
            zk=torch.gather(zr.float(),1,tii[:,:5])           # [t,5] M_int logits (fp32)
            l=((zk - tvi[:,:5].float())**2).mean(-1)          # per-position MSE over top-5
        if neartie:
            w=(tvi[:,0]-tvi[:,1]); wt=torch.exp(-w).clamp(max=10.0)  # small margin -> high weight
            l=l*wt
        tot=tot+l.sum(); ntok+=int(m.sum())
    return tot/max(ntok,1), ntok

@torch.no_grad()
def eval_rrank(m, idx, ids, mask, tv, ti, head=None):
    m.eval(); ranks=[]; ranksk=[]
    for i in idx:
        bi,am,cm=collate([i],ids,mask)
        z=logits_of(m,head,bi,am)[0]  # [L,V]
        n=len(ids[i]); z=z[:n]
        tvi=torch.from_numpy(tv[i]).cuda().float(); tii=torch.from_numpy(ti[i]).cuda().long()
        # ONE shared full-vocab Gumbel added to BOTH M_q and M_int (correct coupling)
        gen=torch.Generator(device='cuda'); gen.manual_seed(EVAL_SEED+int(i))
        g=torch.empty_like(z); g.exponential_(generator=gen).log_().neg_()   # [L,V]
        g_at=torch.gather(g,1,tii)                                           # gumbel at M_q's top-K
        served=tii.gather(1,(tvi+g_at).argmax(1,keepdim=True)).squeeze(1)    # [L] M_q served token (top-K approx)
        zr=z+g                                                              # M_int + same gumbel
        s=zr.gather(1,served[:,None])
        ids_row=torch.arange(z.shape[-1],device=z.device)[None,:]
        gt=(zr>s).sum(-1); eq=((zr==s)&(ids_row<served[:,None])).sum(-1)
        rank=(gt+eq)
        # diagnostic: rank restricted to M_q's top-K candidate set (ignores tail)
        zk=zr.gather(1,tii); gtk=(zk>s).sum(-1); eqk=((zk==s)&(tii<served[:,None])).sum(-1)
        rankk=(gtk+eqk)
        mcm=torch.from_numpy(mask[i]).cuda()
        ranks.append(rank[mcm].cpu().numpy()); ranksk.append(rankk[mcm].cpu().numpy())
    r=np.concatenate(ranks); rk=np.concatenate(ranksk)
    vals,cnt=np.unique(r,return_counts=True); pp=cnt/cnt.sum()
    vk,ck=np.unique(rk,return_counts=True); pk=ck/ck.sum()
    Rk=float(-(pk*np.log2(pk)).sum())
    return float(-(pp*np.log2(pp)).sum()), float((r==0).mean()), int(r.size), Rk

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",default="full_qat"); ap.add_argument("--base",default="int8"); ap.add_argument("--loss",default="topk_kl")
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--bs",type=int,default=8)
    ap.add_argument("--neartie",action="store_true"); ap.add_argument("--max_steps",type=int,default=100000)
    ap.add_argument("--eval_every",type=int,default=200); ap.add_argument("--neval",type=int,default=200)
    ap.add_argument("--rank",type=int,default=64); ap.add_argument("--dtype",default="bf16"); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--tag",default="run")
    a=ap.parse_args()
    if a.smoke:
        with S.GpuLock():
            m=build_mint(a.mode,a.base,a.dtype); bi=torch.randint(0,1000,(2,16)).cuda(); am=torch.ones(2,16,dtype=torch.long).cuda()
            z=mint_logits(m,bi,am); print("smoke logits",z.shape, "trainable params", sum(p.numel() for p in trainable(m))/1e6,"M")
            opt=torch.optim.Adam(trainable(m),lr=a.lr)
            # fake target
            tv=[np.random.randn(16,64).astype(np.float32) for _ in range(2)]; ti=[np.random.randint(0,1000,(16,64)) for _ in range(2)]
            argm=[np.random.randint(0,1000,16) for _ in range(2)]; cm=torch.ones(2,16,dtype=torch.bool).cuda()
            l,nt=loss_fn(z,[0,1],cm,tv,ti,argm,a.loss,a.neartie); print("smoke loss",float(l)); l.backward(); opt.step()
            print("SMOKE OK"); return
    C=np.load(os.path.join(HERE,"corpus","mq_corpus.npz"),allow_pickle=True)
    ids=C["ids"]; mask=C["mask"]; tv=C["topk_val"]; ti=C["topk_idx"]; argm=C["argmax"]
    N=len(ids); rng=np.random.default_rng(0); perm=rng.permutation(N)
    held=perm[:max(a.neval,N//10)]; train=perm[len(held):] if False else perm[N//10:]
    held=perm[:N//10]; train=perm[N//10:]
    print(f"corpus {N} seqs; train {len(train)} held {len(held)}",flush=True)
    with S.GpuLock():
        m=build_mint(a.mode,a.base,a.dtype)
        head=None
        if a.mode=="head":
            V=m.get_output_embeddings().weight.shape[0]; d=m.config.hidden_size
            head=Head(V,d,r=a.rank).to("cuda")
        if a.mode!="frozen":
            m.config.use_cache=False
            try: m.gradient_checkpointing_enable()
            except Exception as e: print("grad-ckpt unavailable:",e,flush=True)
        tp = list(head.parameters()) if head is not None else trainable(m)
        opt=torch.optim.Adam(tp,lr=a.lr)
        r0,f0,nt,rk0=eval_rrank(m, held[:a.neval], ids,mask,tv,ti,head); print(f"[init] R_rank={r0:.4f} R_topK={rk0:.4f} frac0={f0:.4f} (n={nt})",flush=True)
        best=r0; hist=[{"step":0,"rrank":r0,"frac0":f0}]; step=0; t0=time.time()
        if a.mode!="frozen":
            for bstart in range(0,len(train),a.bs):
                sel=list(train[bstart:bstart+a.bs])
                bi,am,cm=collate(sel,ids,mask); z=logits_of(m,head,bi,am)
                l,ntok=loss_fn(z,sel,cm,tv,ti,argm,a.loss,a.neartie)
                opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(tp,1.0); opt.step()
                step+=1
                if step%a.eval_every==0:
                    r,f,_,rk=eval_rrank(m,held[:a.neval],ids,mask,tv,ti,head); m.train()
                    hist.append({"step":step,"loss":float(l),"rrank":r,"frac0":f})
                    print(f"[{step}] loss={float(l):.4f} R_rank={r:.4f} R_topK={rk:.4f} frac0={f:.4f} ({time.time()-t0:.0f}s)",flush=True)
                    best=min(best,r)
                if step>=a.max_steps: break
        rF,fF,_,rkF=eval_rrank(m,held[:a.neval],ids,mask,tv,ti,head)
        print(f"[final] R_rank={rF:.4f} frac0={fF:.4f} best={min(best,rF):.4f}",flush=True)
    out={"tag":a.tag,"mode":a.mode,"base":a.base,"dtype":a.dtype,"loss":a.loss,"lr":a.lr,"neartie":a.neartie,"init_rrank":r0,"final_rrank":rF,"best_rrank":min(best,rF),"hist":hist}
    os.makedirs(os.path.join(HERE,"agree_runs"),exist_ok=True)
    json.dump(out,open(os.path.join(HERE,"agree_runs",f"{a.tag}.json"),"w"),indent=2)
    print("RESULT "+json.dumps({k:out[k] for k in ['tag','mode','base','loss','lr','neartie','init_rrank','final_rrank','best_rrank']}),flush=True)

if __name__=="__main__": main()
