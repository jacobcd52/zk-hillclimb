// Concrete privacy attack on the capstone openings: recover the REAL witness X
// from the revealed FRI query codeword values, exploiting that the SAME random
// slice RX appears (as a fixed linear map) in every revealed codeword position.
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_private_fc.cuh"
using std::vector; using namespace p3pfc;
static uint64_t S=3; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }

int main(){
  // Use HOST path (no GPU dependence) so the attack math is transparent.
  uint32_t bb=2,ii=3,oo=2, B=1u<<bb,IN=1u<<ii,OUT=1u<<oo, R=2,Q=24;
  vector<gl_t> X((size_t)B*IN),W((size_t)IN*OUT); for(auto&x:X)x=rng()%257; for(auto&x:W)x=rng()%257;
  vector<gl_t> Y((size_t)B*OUT,0);
  for(uint32_t i=0;i<B;i++)for(uint32_t k=0;k<OUT;k++){gl_t a=0;for(uint32_t j=0;j<IN;j++)a=gl_add(a,gl_mul(X[i*IN+j],W[j*OUT+k]));Y[i*OUT+k]=a;}
  vector<gl_t> RX((size_t)B*IN),RW((size_t)IN*OUT),RY((size_t)B*OUT);
  for(auto&x:RX)x=rng();for(auto&x:RW)x=rng();for(auto&x:RY)x=rng();

  auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,111,false); // host

  // Augmented X coeff length and codeword domain
  uint32_t Naug=2*B*IN, v=p3bf::ilog2(Naug), logM0=v+R, M0=1u<<logM0;
  gl_t w=gl_root_of_unity(logM0);
  // Collect revealed round-0 codeword positions from openX queries.
  // Reconstruct the coset index c0 per query exactly as the prover did (transcript).
  // Easier: openX.queries[q].rounds[0].{a,b} are cw[c] and cw[c+half]; we need their indices.
  // Recompute c0s from the proof transcript.
  fs::Transcript tr("pfc-X");
  tr.absorb("z",pf.openX.z.data(),pf.openX.z.size()*sizeof(gl_t)); tr.absorb("y",&pf.openX.y,sizeof(gl_t));
  for(uint32_t r=0;r<v;r++){ tr.absorb("root",pf.openX.roots[r].data(),32);
    tr.absorb("sc",&pf.openX.msgs[r],sizeof(p3bf::SumMsg)); uint8_t b[32]; tr.challenge_bytes(b); }
  tr.absorb("final",pf.openX.final_word.data(),pf.openX.final_word.size()*sizeof(gl_t));
  vector<uint32_t> c0s(Q); for(uint32_t q=0;q<Q;q++) c0s[q]=(uint32_t)p3fri::idx_from(tr,M0/2);

  // Build linear system: each revealed round-0 value V = sum_i coeff[i]*w^(pos*i).
  // Unknowns: all Naug coeffs (X|RX). But we KNOW the structure; treat ALL Naug as unknown
  // and solve from revealed positions -> if #eqs >= Naug we recover EVERYTHING incl. X.
  uint32_t half0=M0/2;
  vector<uint32_t> positions; vector<gl_t> values;
  for(uint32_t q=0;q<Q;q++){ uint32_t c=c0s[q]%half0;
    positions.push_back(c); values.push_back(pf.openX.queries[q].rounds[0].a);
    positions.push_back(c+half0); values.push_back(pf.openX.queries[q].rounds[0].b);
  }
  // dedup positions
  vector<uint32_t> P; vector<gl_t> Vv;
  for(size_t i=0;i<positions.size();i++){ bool seen=false; for(size_t j=0;j<P.size();j++) if(P[j]==positions[i]){seen=true;break;}
    if(!seen){P.push_back(positions[i]);Vv.push_back(values[i]);} }
  printf("Naug coeffs=%u, distinct revealed cw positions=%zu\n",Naug,P.size());
  if(P.size()<Naug){ printf("Not enough equations to fully solve (%zu < %u). Attack needs >= Naug eqs.\n",P.size(),Naug); }
  // Build Vandermonde A[r][i] = w^(P[r]*i), solve A x = Vv via Gaussian elim over GL.
  uint32_t n=Naug, m=(uint32_t)P.size();
  vector<vector<gl_t>> A(m, vector<gl_t>(n+1));
  for(uint32_t r=0;r<m;r++){ gl_t xj=gl_pow(w,P[r]); gl_t p=1; for(uint32_t i=0;i<n;i++){A[r][i]=p; p=gl_mul(p,xj);} A[r][n]=Vv[r]; }
  // Gaussian elimination (m x n). Use first n independent rows.
  uint32_t row=0; vector<int> pivcol(n,-1);
  for(uint32_t col=0; col<n && row<m; col++){
    uint32_t sel=row; while(sel<m && A[sel][col]==0) sel++;
    if(sel==m) continue; std::swap(A[sel],A[row]);
    gl_t inv=gl_inv(A[row][col]); for(uint32_t c=col;c<=n;c++) A[row][c]=gl_mul(A[row][c],inv);
    for(uint32_t r2=0;r2<m;r2++) if(r2!=row && A[r2][col]!=0){ gl_t f=A[r2][col]; for(uint32_t c=col;c<=n;c++) A[r2][c]=gl_sub(A[r2][c],gl_mul(f,A[row][c])); }
    pivcol[col]=row; row++;
  }
  // Read out solution where pivot exists
  vector<gl_t> sol(n,0); bool full=true;
  for(uint32_t col=0;col<n;col++){ if(pivcol[col]>=0) sol[col]=A[pivcol[col]][n]; else full=false; }
  printf("Solved rank rows=%u (need %u for unique)\n",row,n);
  uint32_t matchX=0; for(uint32_t i=0;i<B*IN;i++) if(sol[i]==X[i]) matchX++;
  printf("Recovered REAL X entries matching true X: %u / %u\n", matchX, B*IN);
  if(full && matchX==B*IN) printf(">>> PRIVACY BROKEN: full real witness X recovered from openX query values.\n");
  else if(matchX>0) printf(">>> PARTIAL leak: %u/%u real X entries recovered.\n",matchX,B*IN);
  else printf("No recovery at these params.\n");
  return 0;
}
