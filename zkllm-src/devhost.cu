// Check prove_eval_dev (GPU) and host prove_eval produce IDENTICAL EvalProofs / verify both.
#include <cstdio>
#include <vector>
#include <cstring>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
using std::vector;
static uint64_t S=5; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }
int main(){
  uint32_t v=6,R=2,Q=24; uint32_t N=1u<<v;
  vector<gl_t> c(N); for(auto&x:c)x=rng()%257;
  vector<gl_t> z(v); for(auto&x:z)x=rng();
  // host
  p3fri::g_gpu_merkle=false;
  vector<gl_t> cw; auto root=p3bf::commit(c,R,cw);
  gl_t y=p3bf::eval_h(c,p3bf::build_eq(z));
  auto ph=p3bf::prove_eval(c,z,y,R,Q,cw,"x");
  // dev
  p3fri::g_gpu_merkle=true; p3bf::p3_enable_mempool();
  vector<gl_t> cwg; auto rootg=p3bf::commit_gpu(c,R,cwg);
  auto pd=p3bf::prove_eval(c,z,y,R,Q,cwg,"x"); // routes to prove_eval_dev
  printf("roots match: %d\n", root==rootg);
  printf("nmsgs h=%zu d=%zu\n",ph.msgs.size(),pd.msgs.size());
  int msgdiff=0; for(size_t i=0;i<ph.msgs.size();i++) if(memcmp(&ph.msgs[i],&pd.msgs[i],sizeof(p3bf::SumMsg))) msgdiff++;
  printf("msg diffs: %d\n",msgdiff);
  int finaldiff=0; for(size_t i=0;i<ph.final_word.size();i++) if(ph.final_word[i]!=pd.final_word[i]) finaldiff++;
  printf("final_word diffs: %d\n",finaldiff);
  // query values
  int qdiff=0; for(uint32_t q=0;q<Q;q++) for(size_t r=0;r<ph.queries[q].rounds.size();r++){
    if(ph.queries[q].rounds[r].a!=pd.queries[q].rounds[r].a) qdiff++;
    if(ph.queries[q].rounds[r].b!=pd.queries[q].rounds[r].b) qdiff++; }
  printf("query value diffs: %d\n",qdiff);
  const char* why=nullptr;
  printf("host verify: %d\n",p3bf::verify_eval(ph,"x",&why)); 
  printf("dev verify:  %d\n",p3bf::verify_eval(pd,"x",&why));
  return 0;
}
