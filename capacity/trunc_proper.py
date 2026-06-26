"""Protocol-faithful truncation: verifier accepts served iff served in M_int's top-k.
Capacity = rank of served WITHIN M_int top-k (over accepted positions). Also report
rejection rate (honest false-reject cost) and top-p variant."""
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
    if len(r)==0: return 0.0
    v,c=np.unique(r,return_counts=True); p=c/c.sum(); return float(-(p*np.log2(p)).sum())
def main():
    C=np.load(os.path.join(HERE,"corpus","mq_corpus.npz"),allow_pickle=True); ids=C["ids"]; mask=C["mask"]
    held=np.random.default_rng(0).permutation(len(ids))[:len(ids)//10][:NEVAL]
    Ks=[1,5,10,40,100]
    with S.GpuLock():
        mq=build(True); mint=build(False)
        ranks={k:[] for k in Ks}; rej={k:[] for k in Ks}
        with torch.no_grad():
            for i in held:
                idr=torch.from_numpy(ids[i].astype(np.int64))[None].cuda()
                zq=mq(input_ids=idr).logits.float()[0]; zi=mint(input_ids=idr).logits.float()[0]
                m=torch.from_numpy(mask[i]).cuda()
                gen=torch.Generator(device='cuda'); gen.manual_seed(EVAL_SEED+int(i))
                g=torch.empty_like(zq); g.exponential_(generator=gen).log_().neg_()
                for k in Ks:
                    thr=zq.topk(k,dim=-1).values[:,-1:]; zqk=torch.where(zq>=thr,zq+g,torch.full_like(zq,-1e30))
                    served=zqk.argmax(-1)                                # served in M_q top-k
                    tk=zi.topk(k,dim=-1).indices                          # M_int top-k
                    inset=(tk==served[:,None]).any(-1)                    # accept?
                    # rank within M_int top-k under (z_int+g): count M_int-topk tokens beating served
                    zik=zi.gather(1,tk)+g.gather(1,tk)                    # [L,k] noised scores of M_int topk
                    s=(zi.gather(1,served[:,None])+g.gather(1,served[:,None]))
                    rin=(zik>s).sum(-1)                                   # rank within set (only valid if inset)
                    mm=(m&inset); ranks[k].append(rin[mm].cpu().numpy()); rej[k].append((m&~inset).float()[m].cpu().numpy())
        print("== SYMMETRIC TOP-K (accept iff served in M_int top-k) ==")
        print(f"  baseline full-vocab R_rank (T=1) = 0.385")
        for k in Ks:
            r=np.concatenate(ranks[k]); rr=np.concatenate(rej[k])
            print(f"  k={k:<4}: R_rank(in-set)={ent(r):.4f}  capacity_bound=log2({k})={np.log2(k):.2f}  honest_reject_rate={rr.mean():.4f}  frac0={(r==0).mean():.4f}")
        print("DONE_TR")
if __name__=="__main__": main()
