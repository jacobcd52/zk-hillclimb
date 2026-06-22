"""Option B decomposition on an fp8-native model (Qwen2.5-0.5B-FP8-dynamic, base).

Served M_q = FP8Linear (_scaled_mm, fp32 accumulation).  Both references use the
IDENTICAL per-token fp8 activations and per-row fp8 weights as M_q, so the only
difference is HOW the matmul accumulates:
  - codebook : exact int64 accumulation (current provable reference)
  - fp32     : fp32 accumulation via torch matmul (TF32 OFF) = the served kernel's
               accumulation PRECISION, different summation ORDER

R_rank(served=q, ref=R) = covert capacity if we prove R.  Comparing the two
references decomposes the gap:
  R_rank(q,codebook)              = full gap (precision + order)
  R_rank(q,fp32)                  = order-only gap (precision matched)
  R_rank(codebook,fp32)           = precision-only gap
"""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("IMA_TEACHER_KERNEL", "fp8_scaled_mm")
import numpy as np, torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import rank_entropy_sweep as S
from int_model_approximation.__main__ import FP8Linear, CodebookLinear, _per_token_fp8
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "RedHatAI/Qwen2.5-0.5B-FP8-dynamic"
NPROMPTS = int(os.environ.get("NPROMPTS", "48"))

class Fp32AccumLinear(nn.Module):
    """Same fp8 weights+activations as FP8Linear; fp32 accumulation instead of exact int."""
    def __init__(self, w_fp8, w_scale, bias):
        super().__init__()
        self.register_buffer("w", w_fp8.detach().float().contiguous(), persistent=False)
        self.register_buffer("w_scale", w_scale.detach().float().reshape(1, -1), persistent=False)
        self.bias = None if bias is None else bias.detach()
        self.in_features = w_fp8.shape[1]; self.out_features = w_fp8.shape[0]
    def forward(self, x):
        in_shape = x.shape
        x_fp8, x_scale = _per_token_fp8(x)
        y = (x_fp8.float() @ self.w.t())          # fp32 accumulation (TF32 off)
        y = y * x_scale * self.w_scale
        if self.bias is not None: y = y + self.bias.float()
        return y.to(x.dtype).reshape(*in_shape[:-1], self.out_features)

def flogits(model, ids):
    return model(input_ids=ids).logits.float().squeeze(0)   # [L,V] gpu

def ranks(z_served, z_ref, gen):
    g = torch.empty(z_served.shape, device="cuda", dtype=torch.float32)
    g.exponential_(generator=gen).log_().neg_()
    zrf = z_ref + g; zsf = z_served + g
    ref_pref = zrf.max(dim=-1).values
    srv = zsf.argmax(dim=-1)
    delta = ref_pref[:, None] - zrf
    margin = delta.gather(-1, srv[:, None]).squeeze(-1)
    ds, _ = torch.sort(delta, dim=-1)
    r = torch.searchsorted(ds, margin[:, None], right=False).squeeze(-1)
    return r.cpu().numpy().astype(np.int64)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=S.HF_TOKEN)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    prompts = S.load_prompts(NPROMPTS, S.ROWSEED)
    def fresh():
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, token=S.HF_TOKEN)
        return m.to("cuda").eval()
    with S.GpuLock():
        gen_model = fresh()
        seqs, total = S.generate_onpolicy(gen_model, tok, prompts, instruct=False)
        del gen_model; gc.collect(); torch.cuda.empty_cache()
        print(f"generated {len(seqs)} seqs, {total} completion tokens", flush=True)
        ids_list = [full.to("cuda") for full, _ in seqs]
        masks = [m for _, m in seqs]

        mq = fresh(); n,_ = S.replace_linears_robust(mq, lambda w,s,b: FP8Linear(w,s,b))
        mcb = fresh(); S.replace_linears_robust(mcb, lambda w,s,b: CodebookLinear(w,s,b))
        mf = fresh(); S.replace_linears_robust(mf, lambda w,s,b: Fp32AccumLinear(w,s,b))
        print(f"swapped {n} linears per model; 3 models resident", flush=True)

        r_qcb, r_qf, r_cbf = [], [], []
        l2_qcb=l2_qf=l2_cbf=0.0; npos=0
        for i,(ids,m) in enumerate(zip(ids_list,masks)):
            with torch.no_grad():
                zq=flogits(mq,ids); zcb=flogits(mcb,ids); zf=flogits(mf,ids)
            gen=torch.Generator(device="cuda"); gen.manual_seed(S.GSEED_BASE+1+i)
            mm=torch.from_numpy(m).to("cuda")
            # reuse same gumbel draw across the 3 comparisons for a fair split
            for (zs,zr,acc) in [(zq,zcb,r_qcb),(zq,zf,r_qf),(zcb,zf,r_cbf)]:
                gen2=torch.Generator(device="cuda"); gen2.manual_seed(S.GSEED_BASE+1+i)
                acc.append(ranks(zs,zr,gen2)[m])
            l2_qcb+=float(((zq-zcb)[mm].pow(2).sum(-1).sqrt()).sum()); 
            l2_qf +=float(((zq-zf )[mm].pow(2).sum(-1).sqrt()).sum())
            l2_cbf+=float(((zcb-zf)[mm].pow(2).sum(-1).sqrt()).sum())
            npos+=int(m.sum())
            del zq,zcb,zf; torch.cuda.empty_cache()
            if i%10==0: print(f"  seq {i}/{len(ids_list)}",flush=True)
    def summ(rs):
        a=np.concatenate(rs); return dict(R_rank=S.rank_entropy_bits(a),
            frac_rank0=float((a==0).mean()), mean_rank=float(a.mean()), max_rank=int(a.max()))
    out={"model":MODEL_ID,"n_completion_tokens":npos,
         "q_vs_codebook":summ(r_qcb),"q_vs_fp32":summ(r_qf),"codebook_vs_fp32":summ(r_cbf),
         "logit_l2_mean":{"q_codebook":l2_qcb/npos,"q_fp32":l2_qf/npos,"codebook_fp32":l2_cbf/npos}}
    print("RESULT "+json.dumps(out), flush=True)
    json.dump(out, open("optionb_decompose.json","w"), indent=2)

if __name__=="__main__": main()
