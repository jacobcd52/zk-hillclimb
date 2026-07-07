"""For flip positions (served t* not rank-0 under M_int+Gumbel, beaten by t=argmax(z_int+g)):
is it a RAW near-tie (z_int[t*]~z_int[t]) or GUMBEL-driven (z_int[t] << z_int[t*], big noise)?
Report distribution of raw-logit gap dz = z_int[t*]-z_int[t], raw ranks, and the noise gap."""
import os, sys
HERE="/workspace/projects/zk-hillclimb/capacity"; sys.path.insert(0, HERE)
os.environ.setdefault("IMA_TEACHER_KERNEL","fp8_scaled_mm")
import numpy as np, torch, torch.nn as nn
import rank_entropy_sweep as S
from int_model_approximation.__main__ import CodebookLinear
from transformers import AutoModelForCausalLM
from llama_difr import quantize_fp8_per_row
BASE="Qwen/Qwen2.5-0.5B"; EVAL_SEED=20260623; NEVAL=150
def main():
    C=np.load(os.path.join(HERE,"corpus","mq_corpus.npz"),allow_pickle=True); ids=C["ids"]; mask=C["mask"]
    SV=np.load(os.path.join(HERE,"corpus","served_full.npz"),allow_pickle=True)["served"]
    held=np.random.default_rng(0).permutation(len(ids))[:len(ids)//10][:NEVAL]
    with S.GpuLock():
        m=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, token=S.HF_TOKEN).to("cuda").eval()
        for layer in S._iter_decoder_layers(m):
            for parent in [getattr(layer,'self_attn',None), getattr(layer,'mlp',None)]:
                if parent is None: continue
                for name,mod in list(parent.named_children()):
                    if isinstance(mod,nn.Linear) and name in S.TARGETS if hasattr(S,'TARGETS') else name in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"):
                        wf,ws=quantize_fp8_per_row(S._true_weight(mod).cuda()); setattr(parent,name,CodebookLinear(wf,ws,mod.bias).to("cuda"))
        DZ=[]; rrk_tstar=[]; rrk_t=[]; DG=[]; pserved=[]
        with torch.no_grad():
            for i in held:
                idr=torch.from_numpy(ids[i].astype(np.int64))[None].cuda()
                zi=m(input_ids=idr).logits.float()[0]; n=zi.shape[0]
                gen=torch.Generator(device='cuda'); gen.manual_seed(EVAL_SEED+int(i))
                g=torch.empty_like(zi); g.exponential_(generator=gen).log_().neg_()
                served=torch.from_numpy(SV[i].astype(np.int64)).cuda()[:n]
                t=(zi+g).argmax(-1)                       # M_int's gumbel winner
                cm=torch.from_numpy(mask[i]).cuda()
                flip=cm & (served!=t)
                if flip.sum()==0: continue
                fi=flip.nonzero().squeeze(-1)
                zs=zi[fi].gather(1,served[fi,None]).squeeze(1)   # raw z_int of served (t*)
                zt=zi[fi].gather(1,t[fi,None]).squeeze(1)        # raw z_int of beater (t)
                DZ.append((zs-zt).cpu().numpy())                  # >0 => t* rawly higher (t gumbel-lifted)
                rrk_tstar.append((zi[fi] > zs[:,None]).sum(-1).cpu().numpy())   # raw rank of t* (0=raw top)
                rrk_t.append((zi[fi] > zt[:,None]).sum(-1).cpu().numpy())
                DG.append((g[fi].gather(1,t[fi,None]).squeeze(1) - g[fi].gather(1,served[fi,None]).squeeze(1)).cpu().numpy())
                pserved.append(torch.softmax(zi[fi],-1).gather(1,served[fi,None]).squeeze(1).cpu().numpy())
        DZ=np.concatenate(DZ); rt=np.concatenate(rrk_tstar); rT=np.concatenate(rrk_t); DG=np.concatenate(DG); ps=np.concatenate(pserved)
        print(f"n_flip_tokens={DZ.size}")
        print(f"dz = z_int[t*]-z_int[t] (raw logit gap, >0 => t* rawly higher, t lifted by noise):")
        for q in [1,5,25,50,75,95,99]: print(f"   p{q}: {np.percentile(DZ,q):+.3f}")
        print(f"   mean={DZ.mean():+.3f}  frac dz>0 (t* rawly higher)={ (DZ>0).mean():.3f}")
        print(f"   frac |dz|<0.5 (raw near-tie)            = {(np.abs(DZ)<0.5).mean():.3f}")
        print(f"   frac dz>2  (t* rawly >2 above, gumbel)  = {(DZ>2).mean():.3f}")
        print(f"   frac dz>5                               = {(DZ>5).mean():.3f}")
        print(f"raw rank of served t* in z_int (0=raw top): median={np.median(rt):.0f} p90={np.percentile(rt,90):.0f} p99={np.percentile(rt,99):.0f} frac rank0={ (rt==0).mean():.3f} frac<=2={(rt<=2).mean():.3f}")
        print(f"raw rank of beater t  in z_int:            median={np.median(rT):.0f} frac rank0={(rT==0).mean():.3f}")
        print(f"noise gap dg=g[t]-g[t*]: median={np.median(DG):+.3f} p90={np.percentile(DG,90):.2f}  frac dg>0={ (DG>0).mean():.3f}")
        print(f"M_int prob of served token at flips: median={np.median(ps):.3f}")
        print("FLIPDONE")
if __name__=="__main__": main()
