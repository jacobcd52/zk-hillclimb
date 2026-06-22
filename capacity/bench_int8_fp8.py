"""Throughput: bf16 vs fp8 (_scaled_mm) vs int8 (_int_mm) GEMMs at Qwen2.5-0.5B
shapes, on this GPU. Also a run-to-run determinism check for each kernel."""
import json, torch, time
d="cuda"
def fp8q(x):
    s=(x.abs().amax(-1,keepdim=True)/448).clamp_min(1e-12)
    return (x/s).to(torch.float8_e4m3fn), s.float()
def int8q(x):
    s=(x.abs().amax(-1,keepdim=True)/127).clamp_min(1e-12)
    return (x/s).round().clamp(-127,127).to(torch.int8), s.float()
def bench(M,K,N,iters=100):
    x=torch.randn(M,K,device=d,dtype=torch.bfloat16); w=torch.randn(N,K,device=d,dtype=torch.bfloat16)
    xf,xs=fp8q(x); wf,ws=fp8q(w); xi,_=int8q(x); wi,_=int8q(w)
    fns={
      "bf16": lambda: x@w.t(),
      "fp8":  lambda: torch._scaled_mm(xf, wf.t(), scale_a=xs, scale_b=ws.reshape(1,-1), out_dtype=torch.bfloat16),
      "int8": lambda: torch._int_mm(xi, wi.t()),
    }
    out={}
    for name,fn in fns.items():
        try:
            for _ in range(15): fn()
            torch.cuda.synchronize(); t=time.time()
            for _ in range(iters): fn()
            torch.cuda.synchronize(); ms=(time.time()-t)/iters*1000
            a=fn(); b=fn(); det=bool((a==b).all().item())
            out[name]={"ms":round(ms,4),"tops":round(2*M*K*N/(ms/1000)/1e12,1),"deterministic_run2run":det}
        except Exception as e:
            out[name]={"error":repr(e)[:100]}
    return out
shapes=[("o_proj/qkv 896x896",4096,896,896),
        ("up/gate 896x4864",4096,896,4864),
        ("down 4864x896",4096,4864,896),
        ("lm_head 896x152k",4096,896,151936)]
res={}
for lbl,M,K,N in shapes:
    res[lbl]=bench(M,K,N); print(lbl,res[lbl],flush=True)
json.dump(res, open("bench_int8_fp8.json","w"), indent=2)
print("RESULT "+json.dumps(res),flush=True)
