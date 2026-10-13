// Shared Mellum scalar arithmetic; extracted unchanged from inference.cu.
#pragma once
__global__ void Add(float* x,const float* y,int n,int* bad){const int i=blockIdx.x*128+threadIdx.x;if(i<n){x[i]=B(x[i]+y[i]);Finite(x[i],bad);}}
__global__ void Route(const float* logits,int* ids,float* weights,int* bad){
  // One deterministic route: softmax FP32, top eight, normalize, then ascending expert order.
  if(threadIdx.x!=0)return;logits+=blockIdx.x*64;ids+=blockIdx.x*8;weights+=blockIdx.x*8;
  // Preserve safe descriptor indices after a sticky numerical failure. The host
  // rejects the token; dummy routes never become a successful continuation.
  if(*bad){for(int s=0;s<8;++s){ids[s]=0;weights[s]=0;}return;}
  float probabilities[64],mx=-INFINITY,total=0,total_error=0;
  for(int i=0;i<64;++i)mx=fmaxf(mx,logits[i]);
  for(int i=0;i<64;++i){probabilities[i]=ExpRN(__fsub_rn(logits[i],mx));Accumulate(probabilities[i],total,total_error);}
  total=__fadd_rn(total,total_error);
  for(int i=0;i<64;++i)probabilities[i]=__fdiv_rn(probabilities[i],total);
  float selected=0,selected_error=0;
  for(int s=0;s<8;++s){int best=-1;float p=-1;for(int i=0;i<64;++i)if(probabilities[i]>p){p=probabilities[i];best=i;}
    ids[s]=best;weights[s]=p;Accumulate(p,selected,selected_error);probabilities[best]=-1;}
  selected=__fadd_rn(selected,selected_error);
  for(int s=0;s<8;++s)weights[s]=B(__fdiv_rn(weights[s],selected));
  for(int a=0;a<8;++a)for(int b=a+1;b<8;++b)if(ids[b]<ids[a]){int id=ids[a];ids[a]=ids[b];ids[b]=id;float w=weights[a];weights[a]=weights[b];weights[b]=w;}
  Finite(selected,bad);
}
