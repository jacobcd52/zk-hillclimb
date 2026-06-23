"""Cache the FULL-vocab served token per position: served[i] = argmax(z_q + g) over all
152k tokens, with g the SAME per-position Gumbel the eval uses (seed EVAL_SEED+i). Fixes
the top-64 served-token approximation that inflated R_rank."""
import os, sys, gc
HERE="/workspace/projects/zk-hillclimb/capacity"; sys.path.insert(0, HERE)
os.environ.setdefault("IMA_TEACHER_KERNEL","fp8_scaled_mm")
import numpy as np, torch
import rank_entropy_sweep as S
from int_model_approximation.__main__ import FP8Linear
from transformers import AutoModelForCausalLM
BASE="Qwen/Qwen2.5-0.5B"; EVAL_SEED=20260623

def main():
    C=np.load(os.path.join(HERE,"corpus","mq_corpus.npz"),allow_pickle=True)
    ids=C["ids"]
    with S.GpuLock():
        m=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, token=S.HF_TOKEN).to("cuda").eval()
        S.replace_linears_robust(m, lambda w,s,b: FP8Linear(w,s,b))
        served=[]
        with torch.no_grad():
            for i in range(len(ids)):
                idr=torch.from_numpy(ids[i].astype(np.int64))[None].cuda()
                zq=m(input_ids=idr).logits.float()[0]   # [L,V]
                gen=torch.Generator(device='cuda'); gen.manual_seed(EVAL_SEED+int(i))
                g=torch.empty_like(zq); g.exponential_(generator=gen).log_().neg_()
                served.append((zq+g).argmax(-1).int().cpu().numpy())
                del zq,g; 
                if i%500==0: print(f"  {i}/{len(ids)}",flush=True); torch.cuda.empty_cache()
    np.savez(os.path.join(HERE,"corpus","served_full.npz"), served=np.array(served,dtype=object))
    print(f"DONE cached served for {len(served)} seqs",flush=True)

if __name__=="__main__": main()
