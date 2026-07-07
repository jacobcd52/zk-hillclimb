#include "fr-tensor.cuh"
#include "commitment.cuh"
#include "g1-tensor.cuh"
#include "zkserial.cuh"
#include "zkhelpers.cuh"
#include <iostream>
using namespace std;
static bool eqp(const G1Jacobian_t&a,const G1Jacobian_t&b){for(int i=0;i<12;i++){if(a.x.val[i]!=b.x.val[i]||a.y.val[i]!=b.y.val[i]||a.z.val[i]!=b.z.val[i])return false;}return true;}
int main(int argc,char**argv){
    string dir=argv[1];
    auto open_proof=load_g1_vec(dir+"/open_proof.bin");
    auto u_in=load_fr_vec(dir+"/point_uin.bin");
    Commitment generator(dir+"/pp.bin");
    // run the REAL me_open with random scalars, capture its base generators(0)
    FrTensor t=FrTensor::random(generator.size);
    vector<G1Jacobian_t> proof;
    Commitment::me_open(t, generator, u_in.begin(), u_in.end(), proof);
    cout<<"real me_open back matches saved open_proof back="<<eqp(proof.back(),open_proof.back())<<endl;
    // now my fold
    Commitment* g=new Commitment(dir+"/pp.bin");
    for(auto it=u_in.begin();it!=u_in.end();++it){uint ns=(g->size+1)/2;Commitment* ng=new Commitment(ns);me_gen_fold<<<(ns+255)/256,256>>>(g->gpu_data,ng->gpu_data,*it,g->size,ns);cudaDeviceSynchronize();delete g;g=ng;}
    cout<<"my fold matches real me_open back="<<eqp((*g)(0),proof.back())<<endl;
    return 0;
}
