// Privacy attack v2: only the RANDOM slice RX (N coeffs) is unknown; the real X
// is the target. Each revealed cw position gives one linear equation in (X|RX).
// Treat X (N unknowns) and RX (N unknowns) but we only need to recover X. With m
// revealed positions and m >= N we can solve for X if the columns for X are
// identifiable. Equivalent: solve full 2N system needs 2N eqs. BUT the attacker
// can instead use the KNOWN soundness Q. Real systems use Q>=64-100. Test Q sweep.
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_private_fc.cuh"
using std::vector; using namespace p3pfc;
static uint64_t S=3; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }

static int attack(uint32_t Q){
  uint32_t bb=2,ii=3,oo=2, B=1u<<bb,IN=1u<<ii,OUT=1u<<oo, R=2;
  vector<gl_t> X((size_t)B*IN),W((size_t)IN*OUT); for(auto&x:X)x=rng()%257; for(auto&x:W)x=rng()%257;
  vector<gl_t> Y((size_t)B*OUT,0);
  for(uint32_t i=0;i<B;i++)for(uint32_t k=0;k<OUT;k++){gl_t a=0;for(uint32_t j=0;j<IN;j++)a=gl_add(a,gl_mul(X[i*IN+j],W[j*OUT+k]));Y[i*OUT+k]=a;}
  vector<gl_t> RX((size_t)B*IN),RW((size_t)IN*OUT),RY((size_t)B*OUT);
  for(auto&x:RX)x=rng();for(auto&x:RW)x=rng();for(auto&x:RY)x=rng();
  auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,111,false);
  uint32_t Naug=2*B*IN, v=p3bf::ilog2(Naug), logM0=v+R, M0=1u<<logM0; gl_t w=gl_root_of_unity(logM0);
  fs::Transcript tr("pfc-X");
  tr.absorb("z",pf.openX.z.data(),pf.openX.z.size()*sizeof(gl_t)); tr.absorb("y",&pf.openX.y,sizeof(gl_t));
  for(uint32_t r=0;r<v;r++){ tr.absorb("root",pf.openX.roots[r].data(),32); tr.absorb("sc",&pf.openX.msgs[r],sizeof(p3bf::SumMsg)); uint8_t b[32]; tr.challenge_bytes(b);} 
  tr.absorb("final",pf.openX.final_word.data(),pf.openX.final_word.size()*sizeof(gl_t));
  vector<uint32_t> c0s(Q); for(uint32_t q=0;q<Q;q++) c0s[q]=(uint32_t)p3fri::idx_from(tr,M0/2);
  uint32_t half0=M0/2; vector<uint32_t> P; vector<gl_t> Vv;
  auto add=[&](uint32_t pos,gl_t val){ for(auto pp:P) if(pp==pos) return; P.push_back(pos); Vv.push_back(val); };
  for(uint32_t q=0;q<Q;q++){ uint32_t c=c0s[q]%half0; add(c,pf.openX.queries[q].rounds[0].a); add(c+half0,pf.openX.queries[q].rounds[0].b); }
  uint32_t n=Naug, m=(uint32_t)P.size();
  vector<vector<gl_t>> A(m, vector<gl_t>(n+1));
  for(uint32_t r=0;r<m;r++){ gl_t xj=gl_pow(w,P[r]); gl_t p=1; for(uint32_t i=0;i<n;i++){A[r][i]=p;p=gl_mul(p,xj);} A[r][n]=Vv[r]; }
  uint32_t row=0; vector<int> pivcol(n,-1);
  for(uint32_t col=0; col<n && row<m; col++){ uint32_t sel=row; while(sel<m && A[sel][col]==0) sel++; if(sel==m) continue;
    std::swap(A[sel],A[row]); gl_t inv=gl_inv(A[row][col]); for(uint32_t c=col;c<=n;c++) A[row][c]=gl_mul(A[row][c],inv);
    for(uint32_t r2=0;r2<m;r2++) if(r2!=row&&A[r2][col]!=0){gl_t f=A[r2][col]; for(uint32_t c=col;c<=n;c++) A[r2][c]=gl_sub(A[r2][c],gl_mul(f,A[row][c]));} pivcol[col]=row; row++; }
  vector<gl_t> sol(n,0); bool full=true; for(uint32_t col=0;col<n;col++){ if(pivcol[col]>=0) sol[col]=A[pivcol[col]][n]; else full=false; }
  uint32_t matchX=0; for(uint32_t i=0;i<B*IN;i++) if(sol[i]==X[i]) matchX++;
  printf("Q=%2u: distinct positions=%2u  Naug=%u  recovered X=%u/%u  %s\n",Q,m,Naug,matchX,B*IN, (full&&matchX==B*IN)?"<<< FULL BREAK":(matchX>0?"partial":""));
  return (full&&matchX==B*IN)?1:0;
}
int main(){ for(uint32_t Q:{24u,32u,48u,64u,96u}) attack(Q); return 0; }
