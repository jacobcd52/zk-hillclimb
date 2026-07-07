import subprocess, time, csv, sys
PY="/workspace/envs/aqlm/bin/python"
def fwd_us(B,IN,OUT):
    code=f"""
import torch,time
d='cuda'
X=torch.randint(-8,9,({B},{IN}),device=d,dtype=torch.float32)
W=torch.randint(-100,128,({IN},{OUT}),device=d,dtype=torch.float32)
for _ in range(30): Y=X@W
torch.cuda.synchronize(); t=time.time(); N=500
for _ in range(N): Y=X@W
torch.cuda.synchronize(); print((time.time()-t)/N*1e6)
"""
    return float(subprocess.check_output([PY,"-c",code]).decode().strip())
def prove(bb,ii,oo,R,Q):
    out=subprocess.check_output(["./p3_sweep",str(bb),str(ii),str(oo),str(R),str(Q)],stderr=subprocess.DEVNULL).decode().strip()
    return out.split(",")  # B,IN,OUT,prove_ms,verify_ms,proof_kb,ok
R,Q=1,64
rows=[]
# label, bb, ii, oo
configs=[]
# batch sweep on llama-68m/gpt2 padded (IN=1024,OUT=4096)
for bb in [0,2,4,6,8]: configs.append((f"llama68m_B{1<<bb}",bb,10,12))
# model-size sweep at B=16 (bb=4): (ii,oo) padded power-of-2 of real (hidden,intermediate)
for lbl,ii,oo in [("gpt2-medium(1024x4096)",10,12),("gpt2-large(1280x5120)",11,13),
                  ("3B-class(3200x8640)",12,14),("7B-class(4096x11008)",12,14)]:
    configs.append((lbl,4,ii,oo))
seen=set()
for lbl,bb,ii,oo in configs:
    key=(bb,ii,oo)
    try:
        p=prove(bb,ii,oo,R,Q)
        B,IN,OUT=int(p[0]),int(p[1]),int(p[2]); pr_ms,vf_ms,pk,ok=float(p[3]),float(p[4]),float(p[5]),int(p[6])
        f=fwd_us(B,IN,OUT)
        rows.append([lbl,B,IN,OUT,round(f,1),round(pr_ms,1),round(vf_ms,1),round(pk,1),round(pr_ms*1000/f,0),ok])
        print(rows[-1]); sys.stdout.flush()
    except Exception as e:
        print("FAIL",lbl,e); sys.stdout.flush()
with open("sweep_results.csv","w",newline="") as fcsv:
    w=csv.writer(fcsv); w.writerow(["label","B","IN","OUT","fwd_us","prove_ms","verify_ms","proof_kb","overhead_x","ok"]); w.writerows(rows)
print("WROTE sweep_results.csv")
