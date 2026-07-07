import subprocess, csv, sys
PY="/workspace/envs/aqlm/bin/python"
def fwd_us(B,IN,OUT):
    code=f"import torch,time\nd='cuda'\nX=torch.randint(-8,9,({B},{IN}),device=d,dtype=torch.float32)\nW=torch.randint(-100,128,({IN},{OUT}),device=d,dtype=torch.float32)\nfor _ in range(30):Y=X@W\ntorch.cuda.synchronize();t=time.time();N=300\nfor _ in range(N):Y=X@W\ntorch.cuda.synchronize();print((time.time()-t)/N*1e6)"
    return float(subprocess.check_output([PY,"-c",code]).decode().strip())
def prove(bb,ii,oo):
    out=subprocess.check_output(["./p3_sweep",str(bb),str(ii),str(oo),"1","64"],stderr=subprocess.STDOUT,timeout=600).decode().strip().splitlines()[-1]
    p=out.split(","); return int(p[0]),int(p[1]),int(p[2]),float(p[3]),float(p[4]),float(p[5]),int(p[6])
rows=[]
# batch panel: llama up_proj 1024x4096, B = 2^bb up to 1024
for bb in [2,4,6,8,10]:
    B,IN,OUT,pr,vf,pk,ok=prove(bb,10,12)
    fwdB=fwd_us(B,IN,OUT); tdec=fwd_us(1,IN,OUT); gen=B*tdec/1000.0
    rows.append([f"llama68m_B{B}",B,IN,OUT,round(fwdB,1),round(pr,1),round(vf,1),round(pk,1),round(pr*1000/fwdB,0),ok,round(gen,3),round(tdec,2)])
    print("batch",rows[-1]); sys.stdout.flush()
# model panel at context length B=1024
for name,ii,oo in [("gpt2-medium(1024x4096)",10,12),("gpt2-large(1280x5120)",11,13),("3B-class(4096x8192)",12,13),("wide(2048x16384)",11,14)]:
    B,IN,OUT,pr,vf,pk,ok=prove(10,ii,oo)
    fwdB=fwd_us(B,IN,OUT); tdec=fwd_us(1,IN,OUT); gen=B*tdec/1000.0
    rows.append([name,B,IN,OUT,round(fwdB,1),round(pr,1),round(vf,1),round(pk,1),round(pr*1000/fwdB,0),ok,round(gen,3),round(tdec,2)])
    print("model",rows[-1]); sys.stdout.flush()
with open("sweep_results.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["label","B","IN","OUT","fwd_us","prove_ms","verify_ms","proof_kb","overhead_x","ok","gen_ms","tdec_us"]); w.writerows(rows)
print("DONE",len(rows),"rows")
