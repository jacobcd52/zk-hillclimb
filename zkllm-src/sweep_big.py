import subprocess, csv, sys
PY="/workspace/envs/aqlm/bin/python"
def fwd_us(B,IN,OUT):
    code=f"import torch,time\nd='cuda'\nX=torch.randint(-8,9,({B},{IN}),device=d,dtype=torch.float32)\nW=torch.randint(-100,128,({IN},{OUT}),device=d,dtype=torch.float32)\nfor _ in range(20):Y=X@W\ntorch.cuda.synchronize();t=time.time();N=200\nfor _ in range(N):Y=X@W\ntorch.cuda.synchronize();print((time.time()-t)/N*1e6)"
    return float(subprocess.check_output([PY,"-c",code]).decode().strip())
R,Q=1,64
new=[("3B-class(4096x8192)",4,12,13),("wide(2048x16384)",4,11,14),("7B-class(4096x16384)",4,12,14)]
rows=[]
for lbl,bb,ii,oo in new:
    try:
        out=subprocess.check_output(["./p3_sweep",str(bb),str(ii),str(oo),str(R),str(Q)],stderr=subprocess.STDOUT,timeout=600).decode().strip().splitlines()[-1]
        p=out.split(",")
        B,IN,OUT=int(p[0]),int(p[1]),int(p[2]); pr,vf,pk,ok=float(p[3]),float(p[4]),float(p[5]),int(p[6])
        f=fwd_us(B,IN,OUT)
        row=[lbl,B,IN,OUT,round(f,1),round(pr,1),round(vf,1),round(pk,1),round(pr*1000/f,0),ok]
        rows.append(row); print("OK",row); sys.stdout.flush()
    except subprocess.CalledProcessError as e:
        print("FAIL(OOM/err)",lbl,(e.output or b'').decode()[:120]); sys.stdout.flush()
    except Exception as e:
        print("FAIL",lbl,repr(e)[:120]); sys.stdout.flush()
# append survivors to CSV
if rows:
    with open("sweep_results.csv","a",newline="") as f:
        w=csv.writer(f); w.writerows(rows)
print("appended",len(rows),"rows")
