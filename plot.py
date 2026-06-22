import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=list(csv.DictReader(open("sweep_results.csv")))
batch=[r for r in rows if r["label"].startswith("llama68m_B")]
models=[r for r in rows if not r["label"].startswith("llama68m_B")]
fig,ax=plt.subplots(1,2,figsize=(13,5))
if batch:
    B=[int(r["B"]) for r in batch]; fwd=[float(r["fwd_us"])/1000 for r in batch]
    pr=[float(r["prove_ms"]) for r in batch]; prold=[float(r["prove_old_ms"]) for r in batch]
    ax[0].plot(B,prold,'x--',color='gray',label="prove (before)")
    ax[0].plot(B,pr,'o-',label="prove (after)"); ax[0].plot(B,fwd,'s--',label="forward pass")
    ax[0].set_xscale('log',base=2); ax[0].set_yscale('log'); ax[0].set_xlabel("batch size B (llama-68m up_proj, 1024x4096)")
    ax[0].set_ylabel("ms"); ax[0].set_title("Batch scaling (single proof covers the whole batch)"); ax[0].legend(); ax[0].grid(True,alpha=.3)
if models:
    lbl=[r["label"].split("(")[0] for r in models]
    pr=[float(r["prove_ms"]) for r in models]; prold=[float(r["prove_old_ms"]) for r in models]
    fwd=[float(r["fwd_us"])/1000 for r in models]
    x=range(len(models)); w=.27
    ax[1].bar([i-w for i in x],prold,w,label="prove (before)",color='lightgray')
    ax[1].bar([i for i in x],pr,w,label="prove (after)")
    ax[1].bar([i+w for i in x],fwd,w,label="forward")
    ax[1].set_yscale('log'); ax[1].set_xticks(list(x)); ax[1].set_xticklabels(lbl,rotation=20,ha='right',fontsize=8)
    ax[1].set_title("Model-size scaling (B=16): prover speedup"); ax[1].set_ylabel("ms"); ax[1].legend(); ax[1].grid(True,alpha=.3,axis='y')
plt.tight_layout(); plt.savefig("sweep_plot.png",dpi=130); print("WROTE sweep_plot.png")
