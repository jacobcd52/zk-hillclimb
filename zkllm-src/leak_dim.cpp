#include <cstdio>
int main(){
  // capstone test shapes
  int bb=2,ii=3,oo=2; int B=1<<bb, IN=1<<ii, OUT=1<<oo, Q=24;
  // operand X: augmented length 2*B*IN, random slice size B*IN
  int Nx_real=B*IN, Nx_aug=2*B*IN, vx=0; { int t=Nx_aug; while(t>1){t>>=1;vx++;} }
  // openX has logN = bb+ii+1 variables -> v rounds
  printf("X: real=%d randslice=%d aug=%d  logN(v)=%d  round0 positions revealed=2*Q=%d\n", Nx_real, Nx_real, Nx_aug, vx, 2*Q);
  printf("   => revealed round-0 cw positions (%d) vs random unknowns (%d): %s\n",
         2*Q, Nx_real, 2*Q>=Nx_real?"OVER-DETERMINED (leak possible)":"under-determined");
  // llama-scale: check a realistic shape too
  int LB=7, LI=12, LO=12; // 128 x 4096 x 4096-ish
  long lr=1L<<(LB+LI); long la=lr*2;
  printf("llama-ish X: randslice=%ld  2*Q(Q=24)=%d  -> %s\n", lr, 2*24, (2L*24>=lr)?"over":"under (random slice >> queries)");
  return 0;
}
