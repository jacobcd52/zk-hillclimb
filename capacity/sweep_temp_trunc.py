"""R_rank vs sampling temperature and top-k truncation.
served M_q=FP8Linear, ref M_int=CodebookLinear. Shared Gumbel g.
 - temperature T: served t*=argmax(z_q + T*g); rank of t* under (z_int + T*g). (T->0 = greedy)
 - top-k (T=1): restrict served sampling to M_q top-k; rank of t* under M_int (full vocab).
Reports R_rank (entropy of full-vocab rank, token-id tiebreak) + frac0."""
import os, sys
HERE="/workspace/projects/zk-hillclimb/capacity"; sys.path.insert(0, HERE)
os.environ.setdefault("IMA_TEACHER_KERNEL","fp8_scaled_mm")
import numpy as np, torch, torch.nn as nn
import rank_entropy_sweep as S
from int_model_approximation.__main__ import FP8Linear, CodebookLinear
from transformers import AutoModelForCausalLM
from llama_difr import quantize_fp8_per_row
BASE="Qwen/Qwen2.5-0.5B"; EVAL_SEED=20260623; NEVAL=140
TARGETS=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def build(fp8):
    m=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, token=S.HF_TOKEN).to("cuda").eval()
    for layer in S._iter_decoder_layers(m):
        for parent in [getattr(layer,'self_attn',None), getattr(layer,'mlp',None)]:
            if parent is None: continue
            for name,mod in list(parent.named_children()):
                if isinstance(mod,nn.Linear) and name in TARGETS:
                    wf,ws=quantize_fp8_per_row(S._true_weight(mod).cuda())
                    setattr(parent,name,(FP8Linear(wf,ws,mod.bias) if fp8 else CodebookLinear(wf,ws,mod.bias)).to("cuda"))
    for p in m.parameters(): p.requires_grad_(False)
    return m
def ent(r):
    v,c=np.unique(r,return_counts=True); p=c/c.sum(); return float(-(p*np.log2(p)).sum())
def rank_under(zr, served):
    s=zr.gather(1,served[:,None]); ids=torch.arange(zr.shape[-1],device=zr.device)[None,:]
    return (zr>s).sum(-1)+((zr==s)&(ids<served[:,None])).sum(-1)
def main():
    C=np.load(os.path.join(HERE,"corpus","mq_corpus.npz"),allow_pickle=True); ids=C["ids"]; mask=C["mask"]
    held=np.random.default_rng(0).permutation(len(ids))[:len(ids)//10][:NEVAL]
    with S.GpuLock():
        mq=build(True); mint=build(False)
        Ts=[0.0,0.25,0.5,0.7,1.0]; Ks=[1,5,10,40,100,1000,1000000]
        Tres={T:[] for T in Ts}; Kres={k:[] for k in Ks}; Kacc={k:[] for k in Ks}
        with torch.no_grad():
            for i in held:
                idr=torch.from_numpy(ids[i].astype(np.int64))[None].cuda()
                zq=mq(input_ids=idr).logits.float()[0]; zi=mint(input_ids=idr).logits.float()[0]
                n=zq.shape[0]; m=torch.from_numpy(mask[i]).cuda()
                gen=torch.Generator(device='cuda'); gen.manual_seed(EVAL_SEED+int(i))
                g=torch.empty_like(zq); g.exponential_(generator=gen).log_().neg_()
                # temperature sweep
                for T in Ts:
                    served=(zq+T*g).argmax(-1)
                    r=rank_under(zi+T*g, served)
                    Tres[T].append(r[m].cpu().numpy())
                # top-k sweep (T=1): mask served logits to M_q top-k
                for k in Ks:
                    if k>=zq.shape[-1]: zqk=zq
                    else:
                        thr=zq.topk(k,dim=-1).values[:,-1:]; zqk=torch.where(zq>=thr, zq, torch.full_like(zq,-1e30))
                    served=(zqk+g).argmax(-1)
                    r=rank_under(zi+g, served)
                    Kres[k].append(r[m].cpu().numpy())
                    # is served in M_int's top-k? (acceptance under symmetric truncation)
                    if k<zq.shape[-1]:
                        tk=zi.topk(k,dim=-1).indices
                        inset=(tk==served[:,None]).any(-1)
                        Kacc[k].append(inset[m].cpu().numpy())
        print("== TEMPERATURE (no truncation) ==")
        for T in Ts:
            r=np.concatenate(Tres[T]); print(f"  T={T:<4}: R_rank={ent(r):.4f}  frac0={(r==0).mean():.4f}")
        print("== TOP-K TRUNCATION (T=1) ==")
        for k in Ks:
            r=np.concatenate(Kres[k]); lbl=("full" if k>=1000000 else str(k))
            acc = "" if not Kacc.get(k) else f"  served_in_Mint_top{lbl}={np.concatenate(Kacc[k]).mean():.4f}"
            print(f"  k={lbl:<8}: R_rank={ent(r):.4f}  frac0={(r==0).mean():.4f}{acc}")
        print("DONE_TT")
if __name__=="__main__": main()
