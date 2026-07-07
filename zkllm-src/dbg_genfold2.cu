#include "fr-tensor.cuh"
#include "commitment.cuh"
#include "g1-tensor.cuh"
#include "zkserial.cuh"
#include "zkhelpers.cuh"
#include <iostream>
#include <algorithm>
using namespace std;
static bool eqp(const G1Jacobian_t&a,const G1Jacobian_t&b){for(int i=0;i<12;i++){if(a.x.val[i]!=b.x.val[i]||a.y.val[i]!=b.y.val[i]||a.z.val[i]!=b.z.val[i])return false;}return true;}
G1Jacobian_t fold(const string&dir, vector<Fr_t> u_in){
    Commitment* g=new Commitment(dir+"/pp.bin");
    for(auto it=u_in.begin();it!=u_in.end();++it){uint ns=(g->size+1)/2;Commitment* ng=new Commitment(ns);me_gen_fold<<<(ns+255)/256,256>>>(g->gpu_data,ng->gpu_data,*it,g->size,ns);cudaDeviceSynchronize();delete g;g=ng;}
    auto r=(*g)(0);delete g;return r;
}
int main(int argc,char**argv){
    string dir=argv[1];
    auto open_proof=load_g1_vec(dir+"/open_proof.bin");
    auto u_in=load_fr_vec(dir+"/point_uin.bin");
    auto bg=open_proof.back();
    auto f1=fold(dir,u_in); cout<<"forward match="<<eqp(f1,bg)<<endl;
    auto ur=u_in; reverse(ur.begin(),ur.end());
    auto f2=fold(dir,ur); cout<<"reverse match="<<eqp(f2,bg)<<endl;
    return 0;
}
