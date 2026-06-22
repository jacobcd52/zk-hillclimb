import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=list(csv.DictReader(open("sweep_results.csv")))
batch=[r for r in rows if r["label"].startswith("llama68m_B")]
models=[r for r in rows if not r["label"].startswith("llama68m_B")]
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
if batch:
    B=[int(r["B"]) for r in batch]
    pr=[float(r["prove_ms"]) for r in batch]
    gen=[float(r["gen_ms"]) for r in batch]
    fwd=[float(r["fwd_us"])/1000 for r in batch]
    ax[0].plot(B,pr,'o-',color='C0',label="prove (ZK proof)")
    ax[0].plot(B,gen,'^-',color='C3',label="generate B tokens (decode ~50x prefill, est.)")
    ax[0].plot(B,fwd,'s--',color='C2',label="prefill / forward (B tokens)")
    ax[0].set_xscale('log',base=2); ax[0].set_yscale('log'); ax[0].set_xlabel("B = #tokens (llama-68m up_proj 1024x4096)")
    ax[0].set_ylabel("ms"); ax[0].set_title("Token-count scaling (one proof covers all B tokens)"); ax[0].legend(fontsize=8); ax[0].grid(True,alpha=.3)
if models:
    lbl=[r["label"].split("(")[0] for r in models]
    pr=[float(r["prove_ms"]) for r in models]
    gen=[float(r["gen_ms"]) for r in models]; fwd=[float(r["fwd_us"])/1000 for r in models]
    x=range(len(models)); w=.26
    ax[1].bar([i-w for i in x],pr,w,label="prove (ZK proof)",color='C0')
    ax[1].bar([i for i in x],gen,w,label="generate 1024 tokens (decode ~50x prefill, est.)",color='C3')
    ax[1].bar([i+w for i in x],fwd,w,label="prefill / forward (1024 tokens)",color='C2')
    ax[1].set_yscale('log'); ax[1].set_xticks(list(x)); ax[1].set_xticklabels(lbl,rotation=20,ha='right',fontsize=8)
    ax[1].set_title("Model-size scaling (context = 1024 tokens)"); ax[1].set_ylabel("ms"); ax[1].legend(fontsize=8); ax[1].grid(True,alpha=.3,axis='y')
fig.suptitle("ZK private-FC prover vs inference  |  generate (decode) ESTIMATE = 50x prefill per token (vLLM single-stream decode:prefill ratio; typical range ~10-100x)",fontsize=8.5,y=1.005)
plt.tight_layout(); plt.savefig("sweep_plot.png",dpi=130,bbox_inches='tight'); print("WROTE sweep_plot.png")
