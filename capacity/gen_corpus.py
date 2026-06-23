"""Pre-generate a reusable M_q on-policy corpus for the int-model agreement training.
Policy = bf16 base (our standard on-policy setup, ~=M_q policy, fast). Targets = M_q
(fp8 FP8Linear) teacher-forced logits: cache top-64 logits (val+idx) + argmax per
completion position. Reused across all training/eval runs (no teacher reruns)."""
import os, sys, gc, time
HERE="/workspace/projects/zk-hillclimb/capacity"; sys.path.insert(0, HERE)
os.environ.setdefault("IMA_TEACHER_KERNEL","fp8_scaled_mm")
import numpy as np, torch
import rank_entropy_sweep as S
from int_model_approximation.__main__ import FP8Linear
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE="Qwen/Qwen2.5-0.5B"; TOPK=64
NPROMPTS=int(os.environ.get("NPROMPTS","4000"))
OUTDIR=os.path.join(HERE,"corpus"); os.makedirs(OUTDIR,exist_ok=True)

def main():
    tok=AutoTokenizer.from_pretrained(BASE, token=S.HF_TOKEN)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    prompts=S.load_prompts(NPROMPTS, S.ROWSEED)
    print(f"{len(prompts)} prompts", flush=True)
    with S.GpuLock():
        pol=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, token=S.HF_TOKEN).to("cuda").eval()
        t0=time.time(); seqs,total=S.generate_onpolicy(pol, tok, prompts, instruct=False)
        del pol; gc.collect(); torch.cuda.empty_cache()
        print(f"generated {len(seqs)} seqs, {total} comp tokens in {time.time()-t0:.0f}s", flush=True)
        mq=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, token=S.HF_TOKEN).to("cuda").eval()
        n,_=S.replace_linears_robust(mq, lambda w,s,b: FP8Linear(w,s,b)); print(f"M_q: {n} fp8 linears", flush=True)
        ids_l=[]; mask_l=[]; tv_l=[]; ti_l=[]; am_l=[]
        t1=time.time()
        with torch.no_grad():
            for j,(full,m) in enumerate(seqs):
                lg=mq(input_ids=full.to("cuda")).logits.float()[0]   # [L,V]
                tv,ti=lg.topk(TOPK,dim=-1)
                ids_l.append(full[0].cpu().numpy().astype(np.int32)); mask_l.append(m.astype(bool))
                tv_l.append(tv.half().cpu().numpy()); ti_l.append(ti.int().cpu().numpy()); am_l.append(lg.argmax(-1).int().cpu().numpy())
                if j%300==0: print(f"  teacher-force {j}/{len(seqs)} ({time.time()-t1:.0f}s)", flush=True)
    np.savez(os.path.join(OUTDIR,"mq_corpus.npz"),
             ids=np.array(ids_l,dtype=object), mask=np.array(mask_l,dtype=object),
             topk_val=np.array(tv_l,dtype=object), topk_idx=np.array(ti_l,dtype=object),
             argmax=np.array(am_l,dtype=object), topk=TOPK, base=BASE)
    print(f"DONE saved {len(ids_l)} seqs / {total} comp tokens -> {OUTDIR}/mq_corpus.npz", flush=True)

if __name__=="__main__": main()
