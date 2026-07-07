#include "fr-tensor.cuh"
#include "commitment.cuh"
#include "g1-tensor.cuh"
#include "zkserial.cuh"
#include "zkhelpers.cuh"
#include <iostream>
using namespace std;
int main(int argc, char**argv){
    string dir=argv[1];
    auto open_proof=load_g1_vec(dir+"/open_proof.bin");
    auto u_in=load_fr_vec(dir+"/point_uin.bin");
    cout<<"open_proof steps="<<open_proof.size()<<" u_in="<<u_in.size()<<endl;
    Commitment* g=new Commitment(dir+"/pp.bin");
    cout<<"pp size="<<g->size<<endl;
    for(auto it=u_in.begin();it!=u_in.end();++it){
        uint ns=(g->size+1)/2; Commitment* ng=new Commitment(ns);
        me_gen_fold<<<(ns+255)/256,256>>>(g->gpu_data,ng->gpu_data,*it,g->size,ns);
        cudaDeviceSynchronize(); delete g; g=ng;
    }
    cout<<"folded size="<<g->size<<endl;
    auto fg=(*g)(0); auto bg=open_proof.back();
    cout<<"final_gen.x[0..2]="<<fg.x.val[0]<<","<<fg.x.val[1]<<","<<fg.x.val[2]<<endl;
    cout<<"proof_back.x[0..2]="<<bg.x.val[0]<<","<<bg.x.val[1]<<","<<bg.x.val[2]<<endl;
    cout<<"final_gen.z[0..2]="<<fg.z.val[0]<<","<<fg.z.val[1]<<","<<fg.z.val[2]<<endl;
    cout<<"proof_back.z[0..2]="<<bg.z.val[0]<<","<<bg.z.val[1]<<","<<bg.z.val[2]<<endl;
    return 0;
}
