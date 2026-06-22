"""Capability gap: bf16 vs fp8 vs int8 on Qwen2.5-0.5B. Perplexity on held-out
natural text (dolly) + top-1 next-token agreement with bf16. fp8 and int8 are
both derived from the SAME bf16 weights (fair). int8 = per-row weight / per-token
activation int8 (qmax 127), exact int32 accumulation = the deterministic kernel we
would serve AND prove."""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("IMA_TEACHER_KERNEL", "fp8_scaled_mm")
import numpy as np, torch, torch.nn as nn
import rank_entropy_sweep as S
from llama_difr import quantize_fp8_per_row
from int_model_approximation.__main__ import FP8Linear, _per_token_int32, _per_row_int32, _int32_matmul
from transformers import AutoModelForCausalLM, AutoTokenizer

BF16_ID = "Qwen/Qwen2.5-0.5B"
TARGETS = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
L = 512; NWIN = 60

class Int8Linear(nn.Module):
    def __init__(self, w_i8, w_s, bias):
        super().__init__()
        self.register_buffer("w_t", w_i8.t().contiguous(), persistent=False)
        self.register_buffer("w_scale", w_s.reshape(1,-1), persistent=False)
        self.bias = None if bias is None else bias.detach()
        self.in_features=w_i8.shape[1]; self.out_features=w_i8.shape[0]
    def forward(self,x):
        sh=x.shape
        xi,xs=_per_token_int32(x,127.0)
        y=_int32_matmul(xi,self.w_t,xs,self.w_scale)
        if self.bias is not None: y=y+self.bias.float()
        return y.to(x.dtype).reshape(*sh[:-1],self.out_features)

def replace(model, kind):
    layers=S._iter_decoder_layers(model); n=0
    for layer in layers:
        for parent in [getattr(layer,'self_attn',None), getattr(layer,'mlp',None)]:
            if parent is None: continue
            for name,mod in list(parent.named_children()):
                if isinstance(mod,nn.Linear) and name in TARGETS:
                    w=S._true_weight(mod).cuda()
                    if kind=="fp8":
                        wf,ws=quantize_fp8_per_row(w); new=FP8Linear(wf,ws,mod.bias)
                    else:
                        wi,ws=_per_row_int32(w,127.0); new=Int8Linear(wi,ws,mod.bias)
                    setattr(parent,name,new.to("cuda")); n+=1
    return n

def windows(tok):
    lines=open(S.DOLLY).read().splitlines()
    buf=[]
    for ln in lines[:400]:
        r=json.loads(ln); t=((r.get("instruction") or "")+" "+(r.get("context") or "")+" "+(r.get("response") or "")).strip()
        if t: buf.append(t)
    big=tok("\n\n".join(buf), return_tensors="pt").input_ids[0]
    ws=[big[i*L:(i+1)*L][None,:] for i in range(min(NWIN, big.numel()//L))]
    return ws

@torch.no_grad()
def evalppl(model, ws, ref_argmax=None):
    nll=0.0; ntok=0; agree=0; aN=0; my_argmax=[]
    for j,ids in enumerate(ws):
        ids=ids.to("cuda")
        out=model(input_ids=ids).logits.float()[0]   # [L,V]
        lp=torch.log_softmax(out[:-1],-1); tgt=ids[0,1:]
        nll+=-lp.gather(-1,tgt[:,None]).squeeze(-1).sum().item(); ntok+=tgt.numel()
        am=out[:-1].argmax(-1)
        if ref_argmax is not None:
            agree+=(am==ref_argmax[j].to("cuda")).sum().item(); aN+=am.numel()
        my_argmax.append(am.cpu())
    return float(np.exp(nll/ntok)), (agree/aN if aN else None), my_argmax

def fresh():
    return AutoModelForCausalLM.from_pretrained(BF16_ID, torch_dtype=torch.bfloat16, token=S.HF_TOKEN).to("cuda").eval()

def main():
    tok=AutoTokenizer.from_pretrained(BF16_ID, token=S.HF_TOKEN)
    ws=windows(tok); print(f"{len(ws)} windows x {L} tokens", flush=True)
    out={}
    with S.GpuLock():
        m=fresh(); ppl_bf,_,ref_am=evalppl(m,ws); out["bf16"]={"ppl":ppl_bf,"top1_vs_bf16":1.0}; del m; gc.collect(); torch.cuda.empty_cache()
        print("bf16 ppl",ppl_bf,flush=True)
        m=fresh(); n=replace(m,"fp8"); ppl,ag,_=evalppl(m,ws,ref_am); out["fp8"]={"ppl":ppl,"top1_vs_bf16":ag,"n_swapped":n}; del m; gc.collect(); torch.cuda.empty_cache()
        print("fp8 ppl",ppl,"top1",ag,flush=True)
        m=fresh(); n=replace(m,"int8"); ppl,ag,_=evalppl(m,ws,ref_am); out["int8"]={"ppl":ppl,"top1_vs_bf16":ag,"n_swapped":n}; del m; gc.collect(); torch.cuda.empty_cache()
        print("int8 ppl",ppl,"top1",ag,flush=True)
    print("RESULT "+json.dumps(out),flush=True)
    json.dump(out, open("cap_int8_vs_fp8.json","w"), indent=2)

if __name__=="__main__": main()
