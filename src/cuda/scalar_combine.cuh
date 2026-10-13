// Shared Mellum scalar arithmetic; extracted unchanged from inference.cu.
#pragma once
__global__ void Activate(float* gate,const float* up,int tokens=1){const int i=blockIdx.x*128+threadIdx.x;if(i<tokens*8*896)gate[i]=B(B(__fdiv_rn(gate[i],__fadd_rn(1.f,ExpRN(-gate[i]))))*up[i]);}
__global__ void Combine(float* x,const float* down,const float* weights,int* bad){
  x+=blockIdx.y*2304;down+=blockIdx.y*8*2304;weights+=blockIdx.y*8;
  const int i=blockIdx.x*128+threadIdx.x;if(i>=2304)return;float sum=0;
  for(int s=0;s<8;++s)sum=B(sum+B(down[s*2304+i]*weights[s]));x[i]=B(x[i]+sum);Finite(x[i],bad);
}
