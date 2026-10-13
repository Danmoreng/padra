#include "runtime/inference.h"
#include "cuda/device_matrix.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace mellum16 {namespace {
#include "cuda/simt_kernels.cuh"
#include "cuda/simt_prefill.cuh"
#include "cuda/fp8_prefill_simt.cuh"
void Blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS failed: "+std::to_string(int(status)));}
struct Handle{cublasHandle_t value;Handle(){Blas(cublasCreate(&value));Blas(cublasSetMathMode(value,CUBLAS_DEFAULT_MATH));}~Handle(){cublasDestroy(value);}};
__global__ void Decode(const unsigned char* packed,float* out,int n){const int i=blockIdx.x*256+threadIdx.x;if(i<n)out[i]=DecodeFP8Pascal(packed[i]);}
__global__ void Scale(DeviceMatrix m,float* output,const float* scales,int tokens,int* bad){const int i=blockIdx.x*256+threadIdx.x;if(i>=tokens*m.rows)return;const float value=B(__fmul_rn(__fmul_rn(output[i],scales[i/m.rows]),BF(m.scales,i%m.rows)));output[i]=value;Finite(value,bad);}
template<class F> float Time(F launch){for(int i=0;i<5;++i)launch();Check(cudaDeviceSynchronize());cudaEvent_t a,b;Check(cudaEventCreate(&a));Check(cudaEventCreate(&b));Check(cudaEventRecord(a));for(int i=0;i<20;++i)launch();Check(cudaEventRecord(b));Check(cudaEventSynchronize(b));float ms=0;Check(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms/20;}
void Bench(int rows,int cols,int tokens){
  const std::size_t size=std::size_t(rows)*cols;std::vector<unsigned char> weights(size),input(tokens*cols);unsigned seed=19;for(auto* v:{&weights,&input})for(auto& byte:*v){seed=1664525U*seed+1013904223U;byte=static_cast<unsigned char>(seed>>24);if((byte&127)==127)--byte;}
  std::vector<unsigned short> scales(rows,0x3c00);std::vector<float> activation_scales(tokens,.00390625f);
  Buffer w(size),a(256ULL*cols),ws(rows*2),as(256*4),wf(size*4),af(256ULL*cols*4),reference(256ULL*rows*4),output(256ULL*rows*4),state(sizeof(SimtBatch)),workspace(16ULL*1024*1024),bad(4);Handle handle;
  Check(cudaMemcpy(w.p,weights.data(),w.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(a.p,input.data(),input.size(),cudaMemcpyHostToDevice));Check(cudaMemcpy(ws.p,scales.data(),ws.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(as.p,activation_scales.data(),activation_scales.size()*4,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));SetSimtBatch<<<1,1>>>(state.As<SimtBatch>(),0,tokens,8192,bad.As<int>());Blas(cublasSetWorkspace(handle.value,workspace.p,workspace.bytes));
  DeviceMatrix matrix{w.As<unsigned char>(),ws.As<unsigned char>(),nullptr,rows,cols,1};const float alpha=1,beta=0;
  auto own=[&]{if(tokens==1)fast_projection::Vector<<<rows/4,128>>>(matrix,a.As<unsigned char>(),as.As<float>(),reference.As<float>(),bad.As<int>());else simt_batch::FP8ExactTile<<<dim3(rows/32,32),256>>>(matrix,a.As<unsigned char>(),as.As<float>(),reference.As<float>(),bad.As<int>(),state.As<SimtBatch>());};
  auto blas=[&]{Decode<<<(size+255)/256,256>>>(w.As<unsigned char>(),wf.As<float>(),size);Decode<<<(tokens*cols+255)/256,256>>>(a.As<unsigned char>(),af.As<float>(),tokens*cols);Blas(cublasSgemm(handle.value,CUBLAS_OP_T,CUBLAS_OP_N,rows,tokens,cols,&alpha,wf.As<float>(),cols,af.As<float>(),cols,&beta,output.As<float>(),rows));Scale<<<(tokens*rows+255)/256,256>>>(matrix,output.As<float>(),as.As<float>(),tokens,bad.As<int>());};
  own();blas();Check(cudaDeviceSynchronize());std::vector<float> x(tokens*rows),y(x);Check(cudaMemcpy(x.data(),reference.p,x.size()*4,cudaMemcpyDeviceToHost));Check(cudaMemcpy(y.data(),output.p,y.size()*4,cudaMemcpyDeviceToHost));int differences=0;float max_error=0;for(std::size_t i=0;i<x.size();++i){differences+=memcmp(&x[i],&y[i],4)!=0;max_error=fmaxf(max_error,fabsf(x[i]-y[i]));}
  int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));if(flag)throw std::runtime_error("nonfinite SGEMM comparison");
  for(int rep=0;rep<5;++rep){float tiled,sgemm;if(rep%2){sgemm=Time(blas);tiled=Time(own);}else{tiled=Time(own);sgemm=Time(blas);}std::cout<<"rows="<<rows<<" cols="<<cols<<" M="<<tokens<<" rep="<<rep<<" simt_ms="<<tiled<<" sgemm_including_decode_ms="<<sgemm<<" differing_outputs="<<differences<<" max_absolute_error="<<max_error<<" extra_scratch_bytes="<<wf.bytes+af.bytes+workspace.bytes<<'\n';}
}
}}
int main(){try{for(int m:{1,16,32,64,128,256})for(auto shape:std::array<std::pair<int,int>,3>{{{4096,2304},{512,2304},{2304,4096}}})mellum16::Bench(shape.first,shape.second,m);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
