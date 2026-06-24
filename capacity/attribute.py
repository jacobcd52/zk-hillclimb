"""Attribute R_rank to model components. reference = all-exact (codebook). For each config,
test = all-exact EXCEPT the chosen linears are fp8 (accumulation noise). R_rank(test served,
reference) = isolated contribution of those components. (layernorm/softmax/rope are shared
=> contribute 0.)"""
import os, sys, gc, time
HERE="/workspace/projects/zk-hillclimb/capacity"; sys.path.insert(0, HERE)
os.environ.setdefault("IMA_TEACHER_KERNEL","fp8_scaled_mm")
import numpy as np, torch, torch.nn as nn
import rank_entropy_sweep as S
from int_model_approximation.__main__ import FP8Linear, CodebookLinear
from transformers import AutoModelForCausalLM
from llama_difr import quantize_fp8_per_row
BASE="Qwen/Qwen2.5-0.5B"; EVAL_SEED=20260623; NEVAL=120
TYPES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")

def build(fp8_pred):
    m=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, token=S.HF_TOKEN).to("cuda").eval()
    layers=S._iter_decoder_layers(m)
    for li,layer in enumerate(layers):
        for parent in [getattr(layer,'self_attn',None), getattr(layer,'mlp',None)]:
            if parent is None: continue
            for name,mod in list(parent.named_children()):
                if isinstance(mod,nn.Linear) and name in TYPES:
                    wf,ws=quantize_fp8_per_row(S._true_weight(mod).cuda())
                    new=FP8Linear(wf,ws,mod.bias) if fp8_pred(li,name) else CodebookLinear(wf,ws,mod.bias)
                    setattr(parent,name,new.to("cuda"))
    for p in m.parameters(): p.requires_grad_(False)
    return m

@torch.no_grad()
def rrank(test, ref, idx, ids, mask):
    ranks=[]
    for i in idx:
        idr=torch.from_numpy(ids[i].astype(np.int64))[None].cuda()
        zt=test(input_ids=idr).logits.float()[0]; zi=ref(input_ids=idr).logits.float()[0]
        gen=torch.Generator(device='cuda'); gen.manual_seed(EVAL_SEED+int(i))
        g=torch.empty_like(zt); g.exponential_(generator=gen).log_().neg_()
        served=(zt+g).argmax(-1); zr=zi+g; s=zr.gather(1,served[:,None])
        idr2=torch.arange(zr.shape[-1],device=zr.device)[None,:]
        rank=(zr>s).sum(-1)+((zr==s)&(idr2<served[:,None])).sum(-1)
        m=torch.from_numpy(mask[i]).cuda(); ranks.append(rank[m].cpu().numpy())
    r=np.concatenate(ranks); v,c=np.unique(r,return_counts=True); p=c/c.sum()
    return float(-(p*np.log2(p)).sum()), float((r==0).mean())

def main():
    C=np.load(os.path.join(HERE,"corpus","mq_corpus.npz"),allow_pickle=True); ids=C["ids"]; mask=C["mask"]
    held=np.random.default_rng(0).permutation(len(ids))[:len(ids)//10][:NEVAL]
    nl=len(S._iter_decoder_layers(build.__wrapped__ if hasattr(build,'__wrapped__') else None)) if False else None
    with S.GpuLock():
        ref=build(lambda li,n: False)   # all-exact reference (resident)
        nl=len(S._iter_decoder_layers(ref))
        print(f"n_layers={nl}",flush=True)
        configs=[("ALL_fp8(=M_q)", lambda li,n: True),
                 ("ALL_exact(sanity)", lambda li,n: False),
                 ("attn(q,k,v,o)", lambda li,n: n in ("q_proj","k_proj","v_proj","o_proj")),
                 ("mlp(gate,up,down)", lambda li,n: n in ("gate_proj","up_proj","down_proj"))]
        for t in TYPES: configs.append((f"type:{t}", (lambda tt: (lambda li,n: n==tt))(t)))
        for L in range(nl): configs.append((f"layer:{L}", (lambda LL: (lambda li,n: li==LL))(L)))
        res={}
        for name,pred in configs:
            test=build(pred); R,f=rrank(test,ref,held,ids,mask); res[name]=R
            print(f"{name:20s} R_rank={R:.4f} frac0={f:.4f}",flush=True)
            del test; gc.collect(); torch.cuda.empty_cache()
    import json; json.dump(res, open(os.path.join(HERE,"attribute.json"),"w"), indent=2)
    print("ATTR_DONE",flush=True)

if __name__=="__main__": main()
