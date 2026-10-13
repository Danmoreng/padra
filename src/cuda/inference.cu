// Experimental fully resident Mellum forward with explicit activation/dispatch profiles.
#include "runtime/inference.h"
#include "runtime/server_memory.h"
#include "runtime/chunk_graph_contract.h"
#include "compiler/sha256.h"
#include "cuda/model_runtime_worker.h"
#include "cuda/kv_checkpoint.cuh"
#include "runtime/size_checked.h"
#include "runtime/exl3_masked_schedule.h"
#include "cuda/tensor_cores.h"
#include "expert_scaled_bf16_down.cuh"
#include "expert_schedule_hybrid.cuh"
#include "mellum16/nvfp4.h"
#include <ostream>
#include <fstream>
#include <iostream>
#include "model/compiled.h"
#include "device_matrix.cuh"
#include "util/json.h"
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <limits>
#include <string_view>
namespace mellum16 {
namespace {
using internal::WeightFormat;
namespace scaled_prefill=::mellum16::expert_scaled_bf16_layer;
__constant__ float e4m3[256];
__constant__ float e2m1[16];
__device__ float B(float x){return __bfloat162float(__float2bfloat16_rn(x));}
// Accuracy baseline: evaluate this scalar transcendental in FP64 and immediately
// round to FP32. No FP64 model tensors or accumulation. Explicit rnexp profile.
__device__ float ExpRN(float x){return __double2float_rn(exp(static_cast<double>(x)));}
__device__ float BF(const unsigned char* p,int i){return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);}
__device__ float W(DeviceMatrix m,int r,int c){
  const int i=r*m.cols+c;
  if(m.format==0)return BF(m.data,i);
  if(m.format==1)return B(e4m3[m.data[i]]*BF(m.scales,r));
  if(m.format!=2)return NAN;
  const auto packed=m.data[i/2];const int code=(i&1)?packed>>4:packed&15;
  return B(e2m1[code]*e4m3[m.scales[i/16]]/(*m.divisor));
}
__device__ float DecodeFP8(unsigned char code){__nv_fp8_e4m3 value;value.__x=code;return static_cast<float>(value);}
// E2M1 decode adapted from gem16's NVFP4 GEMV helper (Apache-2.0).
// See NOTICE. Mellum retains its own BF16 dequantization boundary.
__device__ float DecodeFP4(unsigned char code){const int magnitude=code&7;
  const float value=magnitude<4?magnitude*.5f:static_cast<float>(magnitude-2+(magnitude==7));return code&8?-value:value;}
__device__ float WWarp(DeviceMatrix m,int r,int c){const int i=r*m.cols+c;
  if(m.format==0)return BF(m.data,i);
  if(m.format==1)return B(DecodeFP8(m.data[i])*BF(m.scales,r));
  if(m.format!=2)return NAN;
  const auto packed=m.data[i/2];const int code=(i&1)?packed>>4:packed&15;
  return B(DecodeFP4(code)*DecodeFP8(m.scales[i/16])/(*m.divisor));
}
// BF16 products are exactly representable in FP32. Retain FP32 roundoff from
// both the per-thread sum and cross-warp merge before the BF16 output cast.
// Explicit intrinsics prevent compiler reassociation from erasing compensation.
__device__ void Accumulate(float value,float& sum,float& error){
  const float next=__fadd_rn(sum,value);
  const float residual=fabsf(sum)>=fabsf(value)?__fadd_rn(__fsub_rn(sum,next),value):__fadd_rn(__fsub_rn(value,next),sum);
  error=__fadd_rn(error,residual);sum=next;
}
__device__ float AccurateReduce(float sum,float error,float* scratch){
  for(int d=16;d;d/=2){const float other=__shfl_down_sync(0xffffffff,sum,d),low=__shfl_down_sync(0xffffffff,error,d);if((threadIdx.x&31)+d<32){Accumulate(other,sum,error);Accumulate(low,sum,error);}}
  if((threadIdx.x&31)==0){scratch[threadIdx.x/32]=sum;scratch[4+threadIdx.x/32]=error;}
  __syncthreads();sum=threadIdx.x<4?scratch[threadIdx.x]:0;error=threadIdx.x<4?scratch[4+threadIdx.x]:0;
  if(threadIdx.x<32)for(int d=16;d;d/=2){const float other=__shfl_down_sync(0xffffffff,sum,d),low=__shfl_down_sync(0xffffffff,error,d);if(threadIdx.x+d<32){Accumulate(other,sum,error);Accumulate(low,sum,error);}}
  return __fadd_rn(sum,error);
}
__device__ void Finite(float x,int* bad){if(!isfinite(x))atomicExch(bad,1);}
__global__ void Gemv(DeviceMatrix m,const float* x,float* y,int* bad){
  __shared__ float sums[8];float sum=0,error=0;const int r=blockIdx.x;
  for(int c=threadIdx.x;c<m.cols;c+=128)Accumulate(__fmul_rn(W(m,r,c),x[c]),sum,error);
  sum=AccurateReduce(sum,error,sums);if(threadIdx.x==0){y[r]=B(sum);Finite(y[r],bad);}
}
// Four independent rows per CTA. The compensation/cast contract is preserved;
// only the work partition changes. Full warps (including tail CTAs) are required.
__device__ float WarpAccurateReduce(float sum,float error){
  for(int d=16;d;d/=2){const float other=__shfl_down_sync(0xffffffff,sum,d),low=__shfl_down_sync(0xffffffff,error,d);
    if((threadIdx.x&31)+d<32){Accumulate(other,sum,error);Accumulate(low,sum,error);}}
  return __fadd_rn(sum,error);
}
__global__ void GemvWarp(DeviceMatrix m,const float* x,float* y,int* bad){
  const int lane=threadIdx.x&31,r=blockIdx.x*4+threadIdx.x/32;if(r>=m.rows)return;
  float sum=0,error=0;for(int c=lane;c<m.cols;c+=32)Accumulate(__fmul_rn(WWarp(m,r,c),x[c]),sum,error);
  sum=WarpAccurateReduce(sum,error);if(lane==0){y[r]=B(sum);Finite(y[r],bad);}
}
#include "head_bf16_prefetch.cuh"
#include "head_bf16_mma.cuh"
__global__ void Embed(DeviceMatrix m,int token,float* x,const int* tokens=nullptr){if(tokens)token=tokens[token+blockIdx.x];x+=blockIdx.x*2304;for(int i=threadIdx.x;i<2304;i+=128)x[i]=BF(m.data,token*2304+i);}
__global__ void Norm(const float* x,float* out,DeviceMatrix weights,int dim,int heads,float* statistics=nullptr){
  __shared__ float sums[8];__shared__ float inv;const int h=blockIdx.x; x+=blockIdx.y*dim*heads;out+=blockIdx.y*dim*heads;
  if(h>=heads)return;
  float sum=0,error=0;for(int i=threadIdx.x;i<dim;i+=128)Accumulate(__fmul_rn(x[h*dim+i],x[h*dim+i]),sum,error);
  // Upstream CPU rsqrt uses rounded sqrt followed by rounded division. CUDA's
  // approximate reciprocal-root can move the subsequent BF16 cast across a tie.
  sum=AccurateReduce(sum,error,sums);if(threadIdx.x==0){const float variance=__fdiv_rn(sum,static_cast<float>(dim));inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(variance,1e-6f)));if(statistics){statistics[2*h]=variance;statistics[2*h+1]=inv;}}__syncthreads();
  for(int i=threadIdx.x;i<dim;i+=128)out[h*dim+i]=B(B(x[h*dim+i]*inv)*BF(weights.data,i));
}
__global__ void Rope(float* q,float* k,const float* table,int position){
  position+=blockIdx.y;q+=blockIdx.y*4096;k+=blockIdx.y*512;
  const int h=blockIdx.x,i=threadIdx.x;float* x=h<32?q+h*128:k+(h-32)*128;
  __shared__ float old[128];old[i]=x[i];__syncthreads();
  const float c=table[position*128+(i%64)],s=table[position*128+64+(i%64)];
  const float rotated=i<64?-old[i+64]:old[i-64];x[i]=B(B(old[i]*c)+B(rotated*s));
}
#include "expert_exl3.cuh"
#include "expert_exl3_prefill_fp16.cuh"
#include "expert_exl3_prefill_masked.cuh"
#include "expert_exl3_prefill_register.cuh"
#include "expert_exl3_prefill_dot.cuh"
#include "expert_exl3_prefill_warp.cuh"
#include "expert_exl3_prefill_packed_activations.cuh"
#if PADRA_EXL3_PREFILL_ASYNC64
#include "expert_exl3_prefill_transport_async64.cuh"
#endif
#include "expert_exl3_selected.cuh"
#include "expert_exl3_decode_n32.cuh"
#include "expert_exl3_decode_mma.cuh"
#include "expert_exl3_decode_dot.cuh"
#include "expert_exl3_decode_fused.cuh"
#include "expert_exl3_decode_lane.cuh"
#include "expert_exl3_decode_dual.cuh"
#include "prefill_control.cuh"
#include "qknorm_rope.cuh"
#include "qknorm_rope_pair64.cuh"
__global__ void Store(const float* k,const float* v,__nv_bfloat16* kc,__nv_bfloat16* vc,int position,int cap){
  position+=blockIdx.y;k+=blockIdx.y*512;v+=blockIdx.y*512;
  const int i=blockIdx.x*128+threadIdx.x;kc[(position%cap)*512+i]=__float2bfloat16_rn(k[i]);vc[(position%cap)*512+i]=__float2bfloat16_rn(v[i]);
}
__device__ float MaxReduce(float x,float* scratch){
  for(int d=16;d;d/=2)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,d));
  if((threadIdx.x&31)==0)scratch[threadIdx.x/32]=x;__syncthreads();
  x=threadIdx.x<4?scratch[threadIdx.x]:-INFINITY;
  if(threadIdx.x<32)for(int d=16;d;d/=2)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,d));return x;
}
#include "norm_fp8.cuh"
#include "norm_fp8_exact.cuh"
#include "norm_fp8_positive_exact.cuh"
#include "norm_fp8_wide_exact.cuh"
#include "residual_norm_wide_exact.cuh"
#include "norm_fp8_rms_tail_exact.cuh"
#include "residual_norm_rms_tail_exact.cuh"
#include "residual_norm_positive_exact.cuh"
#include "row_quantization.cuh"
#include "residual_norm_quantize.cuh"
#include "residual_norm_bf16.cuh"
__device__ float ReadKV(const __nv_bfloat16* cache,const float* chunk,int position,int base,int cap,int feature){return chunk && position>=base?chunk[(position-base)*512+feature]:__bfloat162float(cache[(position%cap)*512+feature]);}
__global__ void Attend(const float* q,const __nv_bfloat16* kc,const __nv_bfloat16* vc,float* out,int pos,int cap,bool local,int* bad,float* captured_scores=nullptr,float* captured_probabilities=nullptr,const float* chunkK=nullptr,const float* chunkV=nullptr,int chunkBase=0){
  pos+=blockIdx.y;q+=blockIdx.y*4096;out+=blockIdx.y*4096;
  extern __shared__ float scores[];__shared__ float scratch[8];__shared__ float maxv,total;
  const int h=blockIdx.x,kvh=h/8,start=local?max(0,pos-1023):0,n=pos-start+1;
  float mx=-INFINITY;
  for(int j=threadIdx.x;j<n;j+=128){float dot=0,error=0;const int feature=kvh*128;
    for(int d=0;d<128;++d)Accumulate(__fmul_rn(q[h*128+d],ReadKV(kc,chunkK,start+j,chunkBase,cap,feature+d)),dot,error);
    const float score=B(B(__fadd_rn(dot,error))*0.08838834764831845f);scores[j]=score;if(captured_scores)captured_scores[h*n+j]=score;mx=fmaxf(mx,score);
  }
  mx=MaxReduce(mx,scratch);if(threadIdx.x==0)maxv=mx;__syncthreads();
  float sum=0,sum_error=0;for(int j=threadIdx.x;j<n;j+=128){scores[j]=ExpRN(__fsub_rn(scores[j],maxv));Accumulate(scores[j],sum,sum_error);}
  sum=AccurateReduce(sum,sum_error,scratch);if(threadIdx.x==0)total=sum;__syncthreads();
  for(int j=threadIdx.x;j<n;j+=128){scores[j]=B(__fdiv_rn(scores[j],total));if(captured_probabilities)captured_probabilities[h*n+j]=scores[j];}__syncthreads();
  float value=0,error=0;for(int j=0;j<n;++j)Accumulate(__fmul_rn(scores[j],ReadKV(vc,chunkV,start+j,chunkBase,cap,kvh*128+threadIdx.x)),value,error);
  out[h*128+threadIdx.x]=B(__fadd_rn(value,error));Finite(value,bad);
}
// R06 source/window scheduling inspired by pinned Gem16 chunk staging;
// arithmetic below is the unchanged Mellum Attend body, never Gemma softmax.
template<bool ChunkOnly>
__device__ __forceinline__ float ReadPrefillKV(const __nv_bfloat16* cache,const float* chunk,int position,int base,int cap,int feature){
  if constexpr(ChunkOnly)return chunk[position*512+feature];
  return position>=base?chunk[(position-base)*512+feature]:__bfloat162float(cache[(position%cap)*512+feature]);
}
template<bool Local,bool ChunkOnly>
__global__ void AttendPrefill(const float* q,const __nv_bfloat16* kc,const __nv_bfloat16* vc,float* out,int pos,int cap,int* bad,float* captured_scores=nullptr,float* captured_probabilities=nullptr,const float* chunkK=nullptr,const float* chunkV=nullptr,int chunkBase=0){
  if(!chunkK || !chunkV || cap<1 || pos<0 || chunkBase<0 || chunkBase>pos || (ChunkOnly && chunkBase!=0)){if(threadIdx.x==0)atomicExch(bad,1);return;}
  pos+=blockIdx.y;q+=blockIdx.y*4096;out+=blockIdx.y*4096;
  extern __shared__ float scores[];__shared__ float scratch[8];__shared__ float maxv,total;
  const int h=blockIdx.x,kvh=h/8,start=Local?max(0,pos-1023):0,n=pos-start+1;
  float mx=-INFINITY;
  for(int j=threadIdx.x;j<n;j+=128){float dot=0,error=0;const int feature=kvh*128;
    for(int d=0;d<128;++d)Accumulate(__fmul_rn(q[h*128+d],ReadPrefillKV<ChunkOnly>(kc,chunkK,start+j,chunkBase,cap,feature+d)),dot,error);
    const float score=B(B(__fadd_rn(dot,error))*0.08838834764831845f);scores[j]=score;if(captured_scores)captured_scores[h*n+j]=score;mx=fmaxf(mx,score);
  }
  mx=MaxReduce(mx,scratch);if(threadIdx.x==0)maxv=mx;__syncthreads();
  float sum=0,sum_error=0;for(int j=threadIdx.x;j<n;j+=128){scores[j]=ExpRN(__fsub_rn(scores[j],maxv));Accumulate(scores[j],sum,sum_error);}
  sum=AccurateReduce(sum,sum_error,scratch);if(threadIdx.x==0)total=sum;__syncthreads();
  for(int j=threadIdx.x;j<n;j+=128){scores[j]=B(__fdiv_rn(scores[j],total));if(captured_probabilities)captured_probabilities[h*n+j]=scores[j];}__syncthreads();
  float value=0,error=0;for(int j=0;j<n;++j)Accumulate(__fmul_rn(scores[j],ReadPrefillKV<ChunkOnly>(vc,chunkV,start+j,chunkBase,cap,kvh*128+threadIdx.x)),value,error);
  out[h*128+threadIdx.x]=B(__fadd_rn(value,error));Finite(value,bad);
}
template<bool Local>
__global__ void AttendDecode(const float* q,const __nv_bfloat16* kc,const __nv_bfloat16* vc,float* out,int pos,int cap,int* bad,float* captured_scores=nullptr,float* captured_probabilities=nullptr){
  extern __shared__ float scores[];__shared__ float scratch[8];__shared__ float maxv,total;
  const int h=blockIdx.x,kvh=h/8,start=Local?max(0,pos-1023):0,n=pos-start+1;
  float mx=-INFINITY;
  for(int j=threadIdx.x;j<n;j+=128){float dot=0,error=0;const int offset=((start+j)%cap)*512+kvh*128;
    for(int d=0;d<128;++d)Accumulate(__fmul_rn(q[h*128+d],__bfloat162float(kc[offset+d])),dot,error);
    const float score=B(B(__fadd_rn(dot,error))*0.08838834764831845f);scores[j]=score;if(captured_scores)captured_scores[h*n+j]=score;mx=fmaxf(mx,score);
  }
  mx=MaxReduce(mx,scratch);if(threadIdx.x==0)maxv=mx;__syncthreads();
  float sum=0,sum_error=0;for(int j=threadIdx.x;j<n;j+=128){scores[j]=ExpRN(__fsub_rn(scores[j],maxv));Accumulate(scores[j],sum,sum_error);}
  sum=AccurateReduce(sum,sum_error,scratch);if(threadIdx.x==0)total=sum;__syncthreads();
  for(int j=threadIdx.x;j<n;j+=128){scores[j]=B(__fdiv_rn(scores[j],total));if(captured_probabilities)captured_probabilities[h*n+j]=scores[j];}__syncthreads();
  float value=0,error=0;for(int j=0;j<n;++j)Accumulate(__fmul_rn(scores[j],__bfloat162float(vc[((start+j)%cap)*512+kvh*128+threadIdx.x])),value,error);
  out[h*128+threadIdx.x]=B(__fadd_rn(value,error));Finite(value,bad);
}
#include "attention_cta.cuh"
#include "attention_tc.cuh"
#include "attention_prefill_fast.cuh"
#include "attention_prefill_gqa8.cuh"
#include "attention_prefill_packed_bf16.cuh"
#include "attention_prefill_padded_single.cuh"
// All launches retain the absolute RoPE position and the admitted cache capacity.
void DecodeAttention(const float* q,const __nv_bfloat16* kc,const __nv_bfloat16* vc,float* out,int pos,int cap,bool local,int* bad,cudaStream_t stream=nullptr,float* scores=nullptr,float* probabilities=nullptr){
  if(local)AttendDecodeCTA256<true><<<32,256,std::min(pos+1,1024)*sizeof(float),stream>>>(q,kc,vc,out,pos,cap,bad,scores,probabilities);
  else AttendDecodeCTA256<false><<<32,256,(pos+1)*sizeof(float),stream>>>(q,kc,vc,out,pos,cap,bad,scores,probabilities);
}
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
#include "router_parallel.cuh"
#include "router_bf16_warp32_exact.cuh"
#include "router_rank_exact.cuh"
__global__ void Expert(const DeviceMatrix* experts,int layer,int part,const int* ids,const float* x,float* y,int* bad){
  const int slot=blockIdx.y;const DeviceMatrix m=experts[(layer*64+ids[slot])*3+part];const int r=blockIdx.x;
  const float* input=part==2?x+slot*896:x;__shared__ float sums[8];float sum=0,error=0;
  for(int c=threadIdx.x;c<m.cols;c+=128)Accumulate(__fmul_rn(W(m,r,c),input[c]),sum,error);
  sum=AccurateReduce(sum,error,sums);if(threadIdx.x==0){y[slot*m.rows+r]=B(sum);Finite(y[slot*m.rows+r],bad);}
}
__global__ void ExpertWarp(const DeviceMatrix* experts,int layer,int part,const int* ids,const float* x,float* y,int* bad){
  const int slot=blockIdx.y,lane=threadIdx.x&31;const DeviceMatrix m=experts[(layer*64+ids[slot])*3+part];
  const int r=blockIdx.x*4+threadIdx.x/32;if(r>=m.rows)return;
  const float* input=part==2?x+slot*896:x;float sum=0,error=0;
  for(int c=lane;c<m.cols;c+=32)Accumulate(__fmul_rn(WWarp(m,r,c),input[c]),sum,error);
  sum=WarpAccurateReduce(sum,error);if(lane==0){y[slot*m.rows+r]=B(sum);Finite(y[slot*m.rows+r],bad);}
}
#include "expert_vector.cuh"
#include "exl3_silu_lut.cuh"
__global__ void Activate(float* gate,const float* up,int tokens=1){const int i=blockIdx.x*128+threadIdx.x;if(i<tokens*8*896)gate[i]=B(B(__fdiv_rn(gate[i],__fadd_rn(1.f,ExpRN(-gate[i]))))*up[i]);}
__global__ void Combine(float* x,const float* down,const float* weights,int* bad){
  x+=blockIdx.y*2304;down+=blockIdx.y*8*2304;weights+=blockIdx.y*8;
  const int i=blockIdx.x*128+threadIdx.x;if(i>=2304)return;float sum=0;
  for(int s=0;s<8;++s)sum=B(sum+B(down[s*2304+i]*weights[s]));x[i]=B(x[i]+sum);Finite(x[i],bad);
}
#include "combine_norm_exact.cuh"
// X1 bounded record: actual frozen K4 postnorm and free routes, one layer/chunk.
__global__ void CaptureExl3Fixture(unsigned char* dst,const float* input,const int* ids,const float* routing,int tokens){
  const int i=blockIdx.x*128+threadIdx.x;
  auto* x=reinterpret_cast<unsigned short*>(dst);auto* e=reinterpret_cast<int*>(dst+tokens*2304*2);auto* w=reinterpret_cast<unsigned short*>(dst+tokens*(2304*2+8*4));
  if(i<tokens*2304)x[i]=__bfloat16_as_ushort(__float2bfloat16_rn(input[i]));
  if(i<tokens*8){e[i]=ids[i];w[i]=__bfloat16_as_ushort(__float2bfloat16_rn(routing[i]));}
}
__global__ void Trace(float* out,const float* x,const float* router,const int* ids,const float* weights){
  out+=blockIdx.y*28*2384;x+=blockIdx.y*2304;router+=blockIdx.y*64;ids+=blockIdx.y*8;weights+=blockIdx.y*8;
  const int i=blockIdx.x*128+threadIdx.x;if(i<2304)out[i]=x[i];else if(i<2368)out[i]=router[i-2304];else if(i<2376)out[i]=static_cast<float>(ids[i-2368]);else if(i<2384)out[i]=weights[i-2376];
}
__global__ void CaptureKV(float* out,const __nv_bfloat16* keys,const __nv_bfloat16* values,int position,int cap,bool local){
  const int start=local?max(0,position-1023):0,n=position-start+1,i=blockIdx.x*128+threadIdx.x;
  if(i<n*512){const int index=((start+i/512)%cap)*512+i%512;out[i]=__bfloat162float(keys[index]);out[n*512+i]=__bfloat162float(values[index]);}
}
__global__ void CaptureRouting(float* out,const int* ids,const float* weights){const int i=threadIdx.x;if(i<8){out[i]=static_cast<float>(ids[i]);out[8+i]=weights[i];}}
#include "prefill.cuh"
#include "router_bf16_batch_quad.cuh"
#include "fp8_projection_fast.cuh"
#include "fp8_projection_pair_exact.cuh"
#include "norm_fast.cuh"
#include "expert_schedule.cuh"
static_assert(sizeof(scaled_prefill::Tail)==sizeof(fast_expert::Tile));
static_assert(alignof(scaled_prefill::Tail)==alignof(fast_expert::Tile));
#include "fp8_partial.cuh"
#include "qkv_grouped.cuh"
#include "fp8_tiles.cuh"
#include "gem16_grouped.cuh"
#include "prefill_m16.cuh"
#include "expert_compact.cuh"
#include "expert_selected_block.cuh"
#include "expert_staged.cuh"
#include "expert_fast_vector.cuh"
#include "expert_packed_vector.cuh"
#include "expert_q8_block.cuh"
#include "split_gateup.cuh"
#include "w4a4_simt.cuh"
#include "w4a8_simt.cuh"
#include "w4a8_planes.cuh"
__global__ void ExpandBF16(const std::uint16_t* input,float* out,int n){const int i=blockIdx.x*128+threadIdx.x;if(i<n)out[i]=__bfloat162float(__ushort_as_bfloat16(input[i]));}
struct Control {int token;int bad;};
#include "selection.cuh"
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#include "sampling.cuh"
struct Buffer {
  void* p=nullptr;std::size_t bytes=0;
  explicit Buffer(std::size_t n):bytes(n){Check(cudaMalloc(&p,n));}
  ~Buffer(){if(p)cudaFree(p);}
  Buffer(const Buffer&)=delete;
  template<class T>T* As(){return static_cast<T*>(p);}
};
struct Stream{cudaStream_t s=nullptr;Stream(){Check(cudaStreamCreate(&s));}~Stream(){if(s)cudaStreamDestroy(s);}};
struct Pinned{Control* p=nullptr;Pinned(){Check(cudaMallocHost(&p,sizeof(Control)));}~Pinned(){if(p)cudaFreeHost(p);}};
#include "decode_control.cuh"
#include "attention_decode_fast.cuh"
#include "attention_gqa8.cuh"
#include "attention_merge_quantize.cuh"
#include "attention_gqa8_fragment_validation.cuh"
#include "attention_merge_quantize_exact.cuh"
#include "attention_merge_quantize_wide_exact.cuh"
#include "kv_fp8.cuh"
#include "kv_q8.cuh"
#include "attention_gqa8_q8.cuh"
#include "attention_prefill_q8.cuh"
#include "attention_gqa8_fp8.cuh"
#include "attention_prefill_fp8.cuh"
float BHost(float x){std::uint32_t bits;std::memcpy(&bits,&x,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;std::memcpy(&x,&bits,4);return x;}
void InitializeCodecTables(){
    std::array<float,256> fp8{};std::array<float,16> fp4{};const float magnitudes[]={0,.5,1,1.5,2,3,4,6};
    for(int i=0;i<256;++i){const int exp=(i&127)>>3,mant=i&7;fp8[i]=std::ldexp(exp?1.f+mant/8.f:mant/8.f,exp?exp-7:-6);if(i&128)fp8[i]=-fp8[i];}
    for(int i=0;i<16;++i)fp4[i]=(i&8)?-magnitudes[i&7]:magnitudes[i&7];
    Check(cudaMemcpyToSymbol(e4m3,fp8.data(),sizeof(fp8)));Check(cudaMemcpyToSymbol(e2m1,fp4.data(),sizeof(fp4)));
}
std::vector<float> RopeTable(int capacity,bool local){
  std::array<float,64> frequencies{};
  const double pi=std::acos(-1.0),base=500000.;
  const float low=static_cast<float>(std::max(0.,std::floor(128*std::log(8192/(32*2*pi))/(2*std::log(base)))));
  const float high=static_cast<float>(std::min(127.,std::ceil(128*std::log(8192/(2*pi))/(2*std::log(base)))));
  for(int i=0;i<64;++i){const float freq=std::pow(500000.f,static_cast<float>(2*i)/128);
    const float ramp=std::clamp((i-low)/(high-low),0.f,1.f),extra=1-ramp;
    frequencies[i]=local?1/freq:(1/(16*freq))*(1-extra)+(1/freq)*extra;
  }
  std::vector<float> table(static_cast<std::size_t>(capacity)*128);
  const float scale=local?1.f:static_cast<float>(1+0.1*std::log(16.));
  for(int p=0;p<capacity;++p)for(int i=0;i<64;++i){const float angle=static_cast<float>(p)*frequencies[i];
    table[p*128+i]=BHost(std::cos(angle)*scale);table[p*128+64+i]=BHost(std::sin(angle)*scale);}
  return table;
}
#include "prefill_kv_commit.cuh"
}
// Operator-probe entry points are excluded from the serving source.
static Result<InferenceResult> RunCudaInferenceCore(const InferenceOptions& o,CudaModelWorker* worker=nullptr){
  const auto begin=std::chrono::steady_clock::now();
  try{
    const bool q8Cache=o.kv_cache_format==KVCacheFormat::Q8TokenHead128F32ScaleV1;
    if(o.server_kv_checkpoint && (!worker || !o.controlled_chunk_graphs || q8Cache || o.fp8_kv_cache || !o.padded_bf16_prefill_attention || !o.packed_bf16_prefill_attention || o.capacity<1024))return Status(StatusCode::kUnsupported,"checkpoint requires explicit selected controlled BF16/packed current chunk");
    if(o.kv_cache_format!=KVCacheFormat::BF16 && !q8Cache)return Status(StatusCode::kUnsupported,"unknown typed KV format");
    if(q8Cache && (!worker || !o.controlled_chunk_graphs || o.capacity!=8192 || o.fp8_kv_cache || !o.padded_bf16_prefill_attention || !o.wide_attention_merge || o.capture_trace || o.snapshot_layer>=0))return Status(StatusCode::kUnsupported,"Q8 requires explicit controlled8192 selected runtime; no FP8 alias, trace/snapshot or unqualified long mode");
    const int contextMaximum=worker && o.controlled_chunk_graphs?o.capacity:kW4A8FContextCapacity;
    if(worker){
      const auto manifestPath=o.artifact/"manifest.json";
      if(std::filesystem::file_size(manifestPath)>16*1024*1024)return Status(StatusCode::kDataLoss,"persistent manifest size unsupported");
      std::ifstream source(manifestPath,std::ios::binary);std::string manifest{std::istreambuf_iterator<char>(source),{}};
      if(compiler::Sha256Hex(manifest.data(),manifest.size())!="0f30d4ead6ae3d0683ec1eb9ad7af1adfebe343d804e3313e8c01d6c31f76f0b")return Status(StatusCode::kDataLoss,"persistent runtime manifest identity mismatch");
    }
    auto loaded=internal::LoadCompiledModel(o.artifact);if(!loaded.ok())return loaded.status();
    auto model=std::move(loaded).value();
    if(worker && o.capacity>kW4A8FContextCapacity && (!o.controlled_chunk_graphs || o.capacity>131072 || o.tc_prefill_attention || o.capture_trace || o.snapshot_layer>=0 || o.fp8_kv_cache || !o.wide_attention_merge || !o.padded_bf16_prefill_attention))return Status(StatusCode::kUnsupported,"long-context profile requires selected K4/BF16 online attention; materialized scores/traces/old cache/profile paths unsupported");
    if(o.controlled_chunk_graphs && !worker)return Status(StatusCode::kUnsupported,"controlled chunk graphs require the persistent selected runtime");
    if(o.diagnostic_exl3_layer>=0 && (!o.exl3_experts || o.exl3_rate!=4 || o.diagnostic_exl3_layer>=28 || o.generate!=0 || o.input_ids.size()>2048 || o.input_ids.size()>static_cast<unsigned>(o.prefill_chunk) || o.prune_last_prefill_ffn || o.exl3_prefill_fp16))return Status(StatusCode::kUnsupported,"X1 fixture requires frozen K4,one chunk<=2048,all rows,G0,valid layer");
    if(o.diagnostic_exl3_layer < -1)return Status(StatusCode::kInvalidArgument,"invalid EXL3 fixture layer");
    if(o.exl3_decode_dual && !o.exl3_decode_lane)return Status(StatusCode::kInvalidArgument,"paired gate/up requires lane decoder parent");
    if(o.wide_rms_dataflow && (!o.exl3_experts||o.exl3_rate!=4||!o.native_decode||!o.exact_fp8_encoding||!o.exl3_bf16_postnorm||o.ordinary_rms_norm))return Status(StatusCode::kUnsupported,"wide RMS requires native K4/exact E4/BF16-postnorm/compensated profile");
    if(o.wide_attention_merge && (!o.exact_fp8_encoding||!o.attention_fragment_validation))return Status(StatusCode::kUnsupported,"wide merge requires exact BF16 fragment profile");
    if(o.paired_attention_o && (!o.exl3_experts||o.exl3_rate!=4||!o.native_decode||!o.ordinary_fp8_simt||o.attention_fp8_mma))return Status(StatusCode::kUnsupported,"paired O requires native K4 ordinary-SIMT attention control");
    if(o.exact_router_rank && (!o.exl3_experts||o.exl3_rate!=4||!o.native_decode||!o.fast_w4a8))return Status(StatusCode::kUnsupported,"exact router rank requires native K4 W4A8F decode control");
    if(o.attention_fragment_validation && !o.exact_fp8_encoding)return Status(StatusCode::kUnsupported,"fragment validation requires exact K4/N3a BF16 encoding profile");
    if(o.exact_fp8_encoding && (!o.exl3_experts||o.exl3_rate!=4||!o.native_decode||!o.fused_attention_o||o.ordinary_rms_norm||o.reciprocal_fused_quantization||!o.diagnostic_attention.empty()))return Status(StatusCode::kUnsupported,"exact E4 codes requires K4/native/compensated input norm/N4a BF16 O control,no older reciprocal/X0 override");
    if(o.exl3_decode_lane && !o.exl3_decode_fused)return Status(StatusCode::kInvalidArgument,"lane decoder requires fused-M1 parent");
    if(o.exl3_prefill_warp && !o.exl3_prefill_dot)return Status(StatusCode::kInvalidArgument,"warp input requires dot-prefill parent");
    if((o.exl3_silu_table||o.exl3_bf16_postnorm) && (!o.exl3_experts||o.exl3_rate!=4||!o.native_decode||o.reciprocal_fused_quantization||o.capture_quantization_rows||!o.fused_residual_norm))return Status(StatusCode::kUnsupported,"exact table/postnorm research requires K4/native/N4b direct-div and no FP8 capture");
    if(o.attention_fp8_mma && (!o.exl3_experts || o.exl3_rate!=4 || !o.native_decode || !o.ordinary_fp8_simt || !o.diagnostic_attention.empty()))return Status(StatusCode::kUnsupported,"attention FP8-MMA override requires explicit K4/native/controlSIMT parent/no X0 attention override");
    if(o.exl3_prefill_dot && !o.exl3_prefill_register)return Status(StatusCode::kInvalidArgument,"dot codec prefill requires register-B");
    if((o.exl3_decode_dot||o.exl3_decode_fused) && !o.exl3_decode_mma)return Status(StatusCode::kInvalidArgument,"dot/fused decode requires explicit M1-FP16-MMA parent");
    if(o.exl3_decode_mma && (o.exl3_decode_selected || o.exl3_decode_n32 || !o.exl3_experts || o.exl3_rate!=4 || !o.native_decode || !o.diagnostic_attention.empty() || o.diagnostic_h128_fp32 || o.diagnostic_routes))return Status(StatusCode::kUnsupported,"M1 FP16-MMA requires K4,native decode,no X2/N32/X0 overrides;explicit BF16-to-FP16 input rounding");
    if(o.exl3_prefill_register && (!o.exl3_prefill_fp16 || !o.exl3_prefill_masked))return Status(StatusCode::kInvalidArgument,"register-B requires masked FP16 prefill");
    if(o.exl3_decode_n32 && (o.exl3_decode_selected || !o.exl3_experts || o.exl3_rate!=4 || !o.native_decode || !o.diagnostic_attention.empty() || o.diagnostic_h128_fp32 || o.diagnostic_routes))return Status(StatusCode::kUnsupported,"R-D1 N32 requires K4,native decode,no X2/X0 overrides");
    if(o.exl3_decode_selected && (!o.exl3_experts || o.exl3_rate!=4 || !o.native_decode || !o.diagnostic_attention.empty() || o.diagnostic_h128_fp32 || o.diagnostic_routes))return Status(StatusCode::kUnsupported,"X2 selected N128 requires K4,native decode,no X0 overrides");
    if(o.exl3_prefill_masked && !o.exl3_prefill_fp16)return Status(StatusCode::kInvalidArgument,"masked prefill requires FP16 prefill");
    if(o.exl3_prefill_fp16 && (!o.exl3_experts || o.exl3_rate!=4 || !o.compact_experts || o.prefill_chunk<2 || !o.diagnostic_attention.empty() || o.diagnostic_h128_fp32 || o.diagnostic_routes))return Status(StatusCode::kUnsupported,"X1 FP16 M32 requires K4,compact grouping,chunk>=2,no X0 overrides");
    const bool x0=!o.diagnostic_attention.empty() || o.diagnostic_h128_fp32 || o.diagnostic_routes;
    if(x0 && (!o.exl3_experts || o.generate!=0 || o.input_ids.size()>512 || o.capacity!=512 || o.prefill_chunk!=128 || o.prune_last_prefill_ffn || o.capture_trace || o.snapshot_layer>=0 || o.fp8_kv_cache))return Status(StatusCode::kUnsupported,"X0 requires EXL3,generate0,capacity512,chunk128,all-rows,BF16 KV,no general traces");
    if(!o.diagnostic_attention.empty()){auto status=internal::ApplyDiagnosticAttention(model,o.diagnostic_attention);if(!status.ok())return status;}
    if(worker && (model.weights_sha256!="e614375f2eb26c17321365d4bd8ee7aa4b716f6badee171bb2040b85b754804c" || model.weights.size()!=7092933120ULL))return Status(StatusCode::kDataLoss,"persistent runtime requires pinned K4 checkpoint");
    if(model.exl3_experts!=o.exl3_experts || model.exl3_rate!=o.exl3_rate)return Status(StatusCode::kInvalidArgument,"EXL3 artifact and explicit expert-weight-format must match");
    if(o.exl3_experts && (!o.fast_w4a8 || !o.native_decode || o.scaled_bf16_prefill || o.q8_expert_decode || o.packed_expert_decode || o.capture_trace || o.capture_quantization_rows || o.snapshot_layer>=0))return Status(StatusCode::kUnsupported,"EXL3 K4/K8 requires explicit W4A8F attention/native decode; NVFP4 expert overrides and state snapshots unsupported");
    if(o.tokenizer_sha256!=model.tokenizer_sha256)return Status(StatusCode::kDataLoss,"prompt tokenizer identity mismatch");
    Check(cudaSetDevice(0));
    cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=12 || prop.minor!=0)return Status(StatusCode::kUnsupported,"only SM120 is admitted");
    std::size_t kvbytes=0;for(int l=0;l<28;++l)kvbytes=admission::Sum({kvbytes,admission::Bytes(l%4==3?o.capacity:std::min(1024,o.capacity),q8Cache?1056:o.fp8_kv_cache?512*2+4*sizeof(kv_fp8::Scales):512*2*2)});
    // Admit/load every compiled kernel before the token loop (CUDA lazy loading).
    cudaFuncAttributes attributes{};
    for(const void* kernel:{reinterpret_cast<const void*>(Gemv),reinterpret_cast<const void*>(GemvWarp),reinterpret_cast<const void*>(ExpertWarp),reinterpret_cast<const void*>(ExpertVector),reinterpret_cast<const void*>(ProjectFP8Partial<true>),reinterpret_cast<const void*>(ProjectFP8Tiles<2>),reinterpret_cast<const void*>(ProjectQKVTiles<2>),reinterpret_cast<const void*>(ProjectQKVGrouped<true>),reinterpret_cast<const void*>(ProjectQKVGrouped<false>),reinterpret_cast<const void*>(Embed),
        reinterpret_cast<const void*>(Norm),reinterpret_cast<const void*>(NormFP8),reinterpret_cast<const void*>(QKNormRope),reinterpret_cast<const void*>(Rope),reinterpret_cast<const void*>(Store),
        reinterpret_cast<const void*>(AttentionQKStaged<true>),reinterpret_cast<const void*>(AttentionQKStaged<false>),reinterpret_cast<const void*>(AttentionPVStaged<true>),reinterpret_cast<const void*>(AttentionPVStaged<false>),reinterpret_cast<const void*>(AttentionTCQK<true>),reinterpret_cast<const void*>(AttentionTCQK<false>),reinterpret_cast<const void*>(AttentionTCSoftmax<true>),reinterpret_cast<const void*>(AttentionTCSoftmax<false>),reinterpret_cast<const void*>(AttentionTCPV<true>),reinterpret_cast<const void*>(AttentionTCPV<false>),reinterpret_cast<const void*>(Attend),reinterpret_cast<const void*>(AttendPrefill<true,true>),reinterpret_cast<const void*>(AttendPrefill<true,false>),reinterpret_cast<const void*>(AttendPrefill<false,true>),reinterpret_cast<const void*>(AttendPrefill<false,false>),reinterpret_cast<const void*>(AttendDecode<true>),reinterpret_cast<const void*>(AttendDecode<false>),reinterpret_cast<const void*>(AttendDecodeCTA256<true>),reinterpret_cast<const void*>(AttendDecodeCTA256<false>),reinterpret_cast<const void*>(Add),reinterpret_cast<const void*>(Route),reinterpret_cast<const void*>(RouteParallel),
        reinterpret_cast<const void*>(Expert),reinterpret_cast<const void*>(Activate),reinterpret_cast<const void*>(Combine),
        reinterpret_cast<const void*>(Trace),reinterpret_cast<const void*>(CaptureKV),reinterpret_cast<const void*>(CaptureRouting),reinterpret_cast<const void*>(Select),reinterpret_cast<const void*>(SelectTiles),reinterpret_cast<const void*>(InitializeDecodeState),reinterpret_cast<const void*>(AdvanceDecodeState),reinterpret_cast<const void*>(CancelDecodeState),reinterpret_cast<const void*>(EmbedControlled),reinterpret_cast<const void*>(QKNormRopeControlled),reinterpret_cast<const void*>(StoreControlled),reinterpret_cast<const void*>(AttendDecodeControlled<true>),reinterpret_cast<const void*>(AttendDecodeControlled<false>),reinterpret_cast<const void*>(TraceControlled),reinterpret_cast<const void*>(ProjectBatch),reinterpret_cast<const void*>(ProjectQuantizedAccurate),reinterpret_cast<const void*>(ProjectFP8Vector),reinterpret_cast<const void*>(gem_reuse::ExpertW4A4SIMT<true>),reinterpret_cast<const void*>(gem_reuse::ExpertW4A8SIMT),reinterpret_cast<const void*>(gem_reuse::ExpertW4A8Planes),reinterpret_cast<const void*>(gem_reuse::ExpertPrefillM16<true,true,false>),reinterpret_cast<const void*>(gem_reuse::ExpertPrefillM16<false,true,false>),reinterpret_cast<const void*>(gem_reuse::ExpertPrefillM16<true,false,true>),reinterpret_cast<const void*>(gem_reuse::ExpertPrefillM16<false,false,true>),reinterpret_cast<const void*>(QuantizeFP8Planes),reinterpret_cast<const void*>(ExpertBatch),reinterpret_cast<const void*>(GroupExperts),reinterpret_cast<const void*>(QuantizeFP8),reinterpret_cast<const void*>(QuantizeFP4),reinterpret_cast<const void*>(FP4Divisors),reinterpret_cast<const void*>(ProjectNativeFP8),reinterpret_cast<const void*>(ExpertNativeFP4),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<true>),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<false>),reinterpret_cast<const void*>(ExpandBF16),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<true,true>),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<false,true>),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<true,false,true>),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<false,false,true>),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<true,false,true,true>),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<false,false,true,true>),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<true,true,false,true>),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<false,true,false,true>),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<true,false,false,true>),reinterpret_cast<const void*>(gem_reuse::ExpertSplitGateUp<false,true>),reinterpret_cast<const void*>(gem_reuse::ExpertSplitGateUp<true,false>),reinterpret_cast<const void*>(gem_reuse::ExpertGrouped<false,false,false,true>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.fast_w4a8)for(const auto kernel:{reinterpret_cast<const void*>(fast_attention::SplitOnlineDecodeBF16<true>),reinterpret_cast<const void*>(fast_attention::SplitOnlineDecodeBF16<false>),reinterpret_cast<const void*>(fast_attention::MergeOnlineDecodeBF16<true>),reinterpret_cast<const void*>(fast_attention::MergeOnlineDecodeBF16<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.padded_bf16_prefill_attention)for(const auto kernel:{reinterpret_cast<const void*>(padded_single_prefill_attention::PipelineBF16<true>),reinterpret_cast<const void*>(padded_single_prefill_attention::PipelineBF16<false>),reinterpret_cast<const void*>(padded_single_prefill_attention::EncodeChunk)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.packed_bf16_prefill_attention)for(const auto kernel:{reinterpret_cast<const void*>(packed_prefill_attention::PipelineBF16<true>),reinterpret_cast<const void*>(packed_prefill_attention::PipelineBF16<false>),reinterpret_cast<const void*>(packed_prefill_attention::EncodeChunk)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.gqa8_prefill_attention)for(const auto kernel:{reinterpret_cast<const void*>(g3_prefill_attention::PipelineBF16<true>),reinterpret_cast<const void*>(g3_prefill_attention::PipelineBF16<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.online_prefill_attention)for(const auto kernel:{reinterpret_cast<const void*>(fast_prefill_attention::OnlineBF16<true>),reinterpret_cast<const void*>(fast_prefill_attention::OnlineBF16<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.staged_experts)for(const auto kernel:{reinterpret_cast<const void*>(fast_expert::StagedBF16<true,false>),reinterpret_cast<const void*>(fast_expert::StagedBF16<false,false>),reinterpret_cast<const void*>(fast_expert::StagedBF16<true,true>),reinterpret_cast<const void*>(fast_expert::StagedBF16<false,true>),reinterpret_cast<const void*>(fast_expert::SelectedVector<true>),reinterpret_cast<const void*>(fast_expert::SelectedVector<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.ordinary_prefill_silu)Check(cudaFuncGetAttributes(&attributes,reinterpret_cast<const void*>(fast_expert::CompactM16<true,true,true>)));
    if(o.packed_expert_decode)for(const auto kernel:{reinterpret_cast<const void*>(fast_expert::SelectedPacked4<true>),reinterpret_cast<const void*>(fast_expert::SelectedPacked4<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.q8_expert_decode)for(const auto kernel:{reinterpret_cast<const void*>(expert_q8::Quantize),reinterpret_cast<const void*>(expert_q8::Selected<true>),reinterpret_cast<const void*>(expert_q8::Selected<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.gqa8_decode_attention)for(const auto kernel:{reinterpret_cast<const void*>(gqa8_attention::SplitBF16<true>),reinterpret_cast<const void*>(gqa8_attention::SplitBF16<false>),reinterpret_cast<const void*>(gqa8_attention::MergeBF16<true>),reinterpret_cast<const void*>(gqa8_attention::MergeBF16<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.reciprocal_fused_quantization)for(const auto kernel:{reinterpret_cast<const void*>(ResidualPostNormFP8<true>),reinterpret_cast<const void*>(attention_o_fusion::MergeQuantize<true,true>),reinterpret_cast<const void*>(attention_o_fusion::MergeQuantize<false,true>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.fused_residual_norm)Check(cudaFuncGetAttributes(&attributes,reinterpret_cast<const void*>(ResidualPostNormFP8<false>)));
    if(o.fused_attention_o)for(const auto kernel:{reinterpret_cast<const void*>(attention_o_fusion::MergeQuantize<true>),reinterpret_cast<const void*>(attention_o_fusion::MergeQuantize<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(q8Cache)for(const auto kernel:{reinterpret_cast<const void*>(kv_q8::Encode<false>),reinterpret_cast<const void*>(kv_q8::Encode<true>),reinterpret_cast<const void*>(kv_q8::Commit),reinterpret_cast<const void*>(gqa8_q8_attention::SplitBF16<true>),reinterpret_cast<const void*>(gqa8_q8_attention::SplitBF16<false>),reinterpret_cast<const void*>(q8_prefill_attention::PipelineBF16<true>),reinterpret_cast<const void*>(q8_prefill_attention::PipelineBF16<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.fp8_kv_cache)for(const auto kernel:{reinterpret_cast<const void*>(kv_fp8::Encode<false>),reinterpret_cast<const void*>(kv_fp8::Encode<true>),reinterpret_cast<const void*>(kv_fp8::Commit),reinterpret_cast<const void*>(gqa8_fp8_attention::SplitBF16<true>),reinterpret_cast<const void*>(gqa8_fp8_attention::SplitBF16<false>),reinterpret_cast<const void*>(gqa8_fp8_attention::MergeBF16<true>),reinterpret_cast<const void*>(gqa8_fp8_attention::MergeBF16<false>),reinterpret_cast<const void*>(fp8_prefill_attention::OnlineBF16<true>),reinterpret_cast<const void*>(fp8_prefill_attention::OnlineBF16<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.ordinary_block_experts || o.block_expert_prefill)for(const auto kernel:{reinterpret_cast<const void*>(fast_expert::CompactM16<true,true>),reinterpret_cast<const void*>(fast_expert::CompactM16<false,true>),reinterpret_cast<const void*>(fast_expert::SelectedBlock<true>),reinterpret_cast<const void*>(fast_expert::SelectedBlock<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.compact_experts)for(const auto kernel:{reinterpret_cast<const void*>(fast_expert::Group),reinterpret_cast<const void*>(fast_expert::Describe),reinterpret_cast<const void*>(fast_expert::CompactM16<true>),reinterpret_cast<const void*>(fast_expert::CompactM16<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.ordinary_rms_norm)for(const auto kernel:{reinterpret_cast<const void*>(fast_norm::Rms2304<true>),reinterpret_cast<const void*>(fast_norm::Rms2304<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.ordinary_fp8_projections)for(const auto kernel:{reinterpret_cast<const void*>(fast_projection::Direct),reinterpret_cast<const void*>(fast_projection::QKV),reinterpret_cast<const void*>(fast_projection::Vector),reinterpret_cast<const void*>(fast_projection::QKVVector)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.exl3_decode_dot)Check(cudaFuncGetAttributes(&attributes,exl3_mma_dot::Projection));
    if(o.exl3_decode_fused)Check(cudaFuncGetAttributes(&attributes,exl3_mma_fused::Projection));
    if(o.exl3_decode_dual)Check(cudaFuncGetAttributes(&attributes,exl3_mma_dual::Projection));
    if(o.wide_rms_dataflow)for(const auto kernel:{reinterpret_cast<const void*>(NormFP8WideExact),reinterpret_cast<const void*>(ResidualPostNormWideExact)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.wide_attention_merge)for(const auto kernel:{reinterpret_cast<const void*>(attention_o_wide_exact::MergeQuantize<true>),reinterpret_cast<const void*>(attention_o_wide_exact::MergeQuantize<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.paired_attention_o)Check(cudaFuncGetAttributes(&attributes,fp8_pair_exact::Vector));
    if(o.exact_router_rank)Check(cudaFuncGetAttributes(&attributes,RouteRankExact));
    if(o.attention_fragment_validation)for(const auto kernel:{reinterpret_cast<const void*>(gqa8_fragment_validation::SplitBF16<true>),reinterpret_cast<const void*>(gqa8_fragment_validation::SplitBF16<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.exact_fp8_encoding)for(const auto kernel:{reinterpret_cast<const void*>(NormFP8Exact),reinterpret_cast<const void*>(attention_o_exact::MergeQuantize<true>),reinterpret_cast<const void*>(attention_o_exact::MergeQuantize<false>)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.exl3_decode_lane)Check(cudaFuncGetAttributes(&attributes,exl3_mma_lane::Projection));
    if(o.exl3_prefill_warp)for(const auto kernel:{reinterpret_cast<const void*>(exl3_warp_input::Input),reinterpret_cast<const void*>(exl3_warp_input::Projection)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.pair64_qknorm_prefill)Check(cudaFuncGetAttributes(&attributes,QKNormRopePair64));
    if(o.quad_router_prefill)Check(cudaFuncGetAttributes(&attributes,RouterBF16BatchQuad));
    if(o.exl3_wide_input)Check(cudaFuncGetAttributes(&attributes,exl3_input_wide_vector::Input));
    if(o.exl3_packed_activations)for(const auto kernel:{reinterpret_cast<const void*>(exl3_packed_activations::Projection),reinterpret_cast<const void*>(exl3_packed_activations::InputActivated),reinterpret_cast<const void*>(exl3_packed_activations::Combine)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.exl3_silu_table)Check(cudaFuncGetAttributes(&attributes,exl3_silu_lut::Activate));
#if PADRA_EXL3_PREFILL_ASYNC64
    if(o.exl3_packed_activations)Check(cudaFuncGetAttributes(&attributes,exl3_transport_async64::Projection));
#endif
    if(o.exl3_bf16_postnorm)Check(cudaFuncGetAttributes(&attributes,ResidualPostNormBF16));
    if(o.exl3_prefill_dot)Check(cudaFuncGetAttributes(&attributes,exl3_register_dot::Projection));
    if(o.exl3_decode_mma)Check(cudaFuncGetAttributes(&attributes,exl3_mma_decode::Projection));
    if(o.exl3_prefill_register)Check(cudaFuncGetAttributes(&attributes,exl3_register::Projection));
    if(o.exl3_decode_n32)Check(cudaFuncGetAttributes(&attributes,exl3_n32::Projection));
    if(o.exl3_decode_selected)Check(cudaFuncGetAttributes(&attributes,exl3_selected::Projection));
    if(o.exl3_prefill_fp16)for(const auto kernel:{reinterpret_cast<const void*>(exl3_fp16::Input),reinterpret_cast<const void*>(exl3_fp16::Projection),reinterpret_cast<const void*>(exl3_fp16::Schedule),reinterpret_cast<const void*>(exl3_fp16::TailDot),reinterpret_cast<const void*>(exl3_fp16::TailOutput)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.exl3_prefill_masked)for(const auto kernel:{reinterpret_cast<const void*>(exl3_masked::Input),reinterpret_cast<const void*>(exl3_masked::Projection),reinterpret_cast<const void*>(exl3_masked::Schedule)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.diagnostic_exl3_layer>=0)Check(cudaFuncGetAttributes(&attributes,reinterpret_cast<const void*>(CaptureExl3Fixture)));
    if(o.exl3_experts)for(const auto kernel:{reinterpret_cast<const void*>(exl3_expert::InputHad),reinterpret_cast<const void*>(exl3_expert::Dot),reinterpret_cast<const void*>(exl3_expert::OutputHad)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.scaled_bf16_prefill)for(const auto kernel:{reinterpret_cast<const void*>(scaled_prefill::DescribeFull64Tail16),reinterpret_cast<const void*>(scaled_prefill::GateUp),reinterpret_cast<const void*>(scaled_prefill::Down)})Check(cudaFuncGetAttributes(&attributes,kernel));
    if(o.diagnostic_routes)Check(cudaFuncGetAttributes(&attributes,reinterpret_cast<const void*>(exl3_expert::Range)));
    if(o.native_prefill){const auto status=tc::AdmitFP8();if(!status.ok())return status;}
    std::size_t free=0,total=0;Check(cudaMemGetInfo(&free,&total));
    const auto weightsbytes=model.resident_weights_bytes?model.resident_weights_bytes:model.weights.size();
    const auto routeBytes=o.diagnostic_routes?admission::Bytes(o.input_ids.size(),28*8*sizeof(int)):0;
    const int batch=o.prefill_chunk>1?std::min(o.prefill_chunk,static_cast<int>(o.input_ids.size())):1;
    const std::size_t batchbytes=o.prefill_chunk>1?admission::Bytes(batch,(2304*3+4096*2+512*2+64+8*896*2+8*2304)*4):4;
    const std::size_t quantbytes=o.native_prefill?admission::Sum({admission::Bytes(batch,7168+8*4+8*2304/2+8*2304/16+8*4),15}):4;
    const std::size_t tensorbytes=o.native_prefill?admission::Sum({32*1024*1024,admission::Bytes(batch,4096*2)}):4;
    const std::size_t attentionElements=o.tc_prefill_attention?admission::Bytes(batch,32*((o.capacity+15)/16)*16):1;
    const std::size_t groupbytes=o.native_prefill?admission::Bytes(admission::Sum({admission::Bytes(64,batch),64}),4):4;
    const auto tracebytes=o.capture_trace?admission::Bytes(o.capacity,28*2384*sizeof(float)):sizeof(float);
    auto regions=o.snapshot_layer>=0?OperatorSnapshotLayout(o.snapshot_layer,o.snapshot_position):std::vector<SnapshotRegion>{};
    const auto snapshotbytes=regions.empty()?4:admission::Bytes(admission::Sum({regions.back().offset,regions.back().count}),4);
    const std::size_t samplerTempBytes=worker?gpu_sampling::WorkspaceBytes():0;
    const std::size_t samplerBytes=worker?gpu_sampling::Vocabulary*28+sizeof(gpu_sampling::State):0;
    const std::size_t selectionbytes=kSelectionBlocks*sizeof(SelectionCandidate);
    const std::size_t fastAttentionBytes=o.fast_w4a8?admission::Bytes(o.gqa8_decode_attention?gqa8_attention::WorkspaceElements(o.capacity,contextMaximum):fast_attention::WorkspaceElements(o.capacity),sizeof(float)):0;
    const auto compactBytes=o.compact_experts?admission::Sum({admission::Bytes(fast_expert::DescriptorCapacity(batch),sizeof(fast_expert::Tile)),4}):0;
    const auto scaledMatrixBytes=o.scaled_bf16_prefill?admission::Bytes(28*64*3,sizeof(scaled_prefill::Matrix)):0;
    const auto scaledTileBytes=o.scaled_bf16_prefill?admission::Sum({admission::Bytes(scaled_prefill::WideCapacity(batch),sizeof(scaled_prefill::Tile)),4}):0;
    const std::size_t scratchBytes=(2304*3+4096*2+512*2+64+8*896*2+8*2304+98304)*sizeof(float);
    const std::size_t encodedChunkBytes=(o.fp8_kv_cache || q8Cache)?admission::Bytes(batch,512*2+4*sizeof(kv_fp8::Scales)):0;
    const std::size_t packedChunkBytes=o.packed_bf16_prefill_attention && !q8Cache?admission::Bytes(batch,512*2*sizeof(__nv_bfloat16)):0;
    const std::size_t checkpointBytes=o.server_kv_checkpoint?KVCheckpointLedger::kLocalSnapshotBytes:0;
    const std::size_t quantCaptureBytes=o.capture_quantization_rows?row_quantization::kCaptureBytes:0;
    const auto exl3FixtureBytes=o.diagnostic_exl3_layer>=0?admission::Bytes(o.input_ids.size(),2304*2+8*4+8*2):0;
    const auto exl3ScheduleBytes=o.exl3_prefill_fp16?(o.exl3_prefill_masked?mellum16::exl3_masked_schedule::Bytes(batch):admission::Sum({admission::Bytes(exl3_fp16::Capacity(batch),sizeof(exl3_fp16::Tile)),4})):0;
    const std::size_t exl3SiLUBytes=o.exl3_silu_table?65536*sizeof(unsigned short):0;
    const auto exl3Bytes=o.exl3_experts?admission::Bytes(batch,8*2304*(o.exl3_prefill_fp16?6:8)):0;
    const auto plannedResident=admission::Sum({checkpointBytes,packedChunkBytes,exl3SiLUBytes,exl3ScheduleBytes,exl3FixtureBytes,routeBytes,o.diagnostic_routes?28*3*9*sizeof(unsigned):0,exl3Bytes,weightsbytes,kvbytes,28*64*3*sizeof(DeviceMatrix),admission::Bytes(o.capacity,128*4*2),scratchBytes,
        admission::Bytes(batch,8*sizeof(int)),admission::Bytes(batch,8*sizeof(float)),sizeof(int),sizeof(Control),selectionbytes,
        tracebytes,snapshotbytes,batchbytes,quantbytes,groupbytes,tensorbytes,admission::Bytes(o.input_ids.size(),4),o.device_decode?sizeof(DecodeState):4,
        admission::Bytes(attentionElements,6),fastAttentionBytes,compactBytes,encodedChunkBytes,scaledMatrixBytes,scaledTileBytes,quantCaptureBytes,worker && o.controlled_chunk_graphs?sizeof(PrefillState):0,samplerBytes,samplerTempBytes});
    const auto available=ServerAvailableBytes(free,total,o.server_device_budget_gib);
    const auto required=admission::Sum({plannedResident,700ULL*1024*1024});
    if(worker)worker->RecordAdmission(total,free,available,required);
    if(required>available)return Status(StatusCode::kResourceExhausted,"weights, selected KV cache, exact scratch inventory and700MiB runtime reserve do not fit actual GPU/explicit device ceiling; required="+std::to_string(required)+" available="+std::to_string(available)+" chunk="+std::to_string(o.prefill_chunk)+" capacity="+std::to_string(o.capacity));
    Stream stream;Buffer arena(static_cast<std::size_t>(weightsbytes));Buffer deviceState(o.device_decode?sizeof(DecodeState):4);
    if(model.uploads.empty())Check(cudaMemcpyAsync(arena.p,model.weights.data(),static_cast<std::size_t>(weightsbytes),cudaMemcpyHostToDevice,stream.s));
    else for(const auto& r:model.uploads)Check(cudaMemcpyAsync(arena.As<unsigned char>()+r.device_offset,(r.attention?model.diagnostic_attention.data():model.weights.data())+r.source_offset,r.bytes,cudaMemcpyHostToDevice,stream.s));
    Buffer packedChunk(packedChunkBytes);auto* packedChunkK=packedChunk.As<__nv_bfloat16>();auto* packedChunkV=o.packed_bf16_prefill_attention && !q8Cache?packedChunkK+batch*512:nullptr;
    Buffer checkpointBuffer(checkpointBytes);
    if(o.server_kv_checkpoint)Check(cudaFuncGetAttributes(&attributes,reinterpret_cast<const void*>(kv_checkpoint::CaptureLocal)));
    Buffer exl3Fixture(exl3FixtureBytes);
    Buffer routeCapture(routeBytes);Buffer rangeCapture(o.diagnostic_routes?28*3*9*sizeof(unsigned):0);if(o.diagnostic_routes)Check(cudaMemsetAsync(rangeCapture.p,0,rangeCapture.bytes,stream.s));
    const auto bind=[&](const std::string& name){return BindDeviceMatrix(model,arena.As<unsigned char>(),name);};
    struct LayerWeights{DeviceMatrix norm,post,qn,kn,q,k,v,out,router;};
    struct LayerCacheView{KVCacheView q8;__nv_bfloat16* kc=nullptr;__nv_bfloat16* vc=nullptr;unsigned char* ek=nullptr;unsigned char* ev=nullptr;kv_fp8::Scales* sf=nullptr;int cap;};
    struct Layer:LayerWeights,LayerCacheView{};
    std::array<Layer,28> layers{};std::vector<DeviceMatrix> experts;experts.reserve(28*64*3);Buffer cache(static_cast<std::size_t>(kvbytes));auto* cursor=cache.As<unsigned char>();
    for(int l=0;l<28;++l){const auto p="model.layers."+std::to_string(l)+".";auto& a=layers[l];
      a.norm=bind(p+"input_layernorm.weight");a.post=bind(p+"post_attention_layernorm.weight");a.qn=bind(p+"self_attn.q_norm.weight");a.kn=bind(p+"self_attn.k_norm.weight");
      a.q=bind(p+"self_attn.q_proj.weight");a.k=bind(p+"self_attn.k_proj.weight");a.v=bind(p+"self_attn.v_proj.weight");a.out=bind(p+"self_attn.o_proj.weight");a.router=bind(p+"mlp.gate.weight");
      a.cap=l%4==3?o.capacity:std::min(1024,o.capacity);
      if(q8Cache){auto layout=KVCacheLayout::Make(o.kv_cache_format,a.cap);a.q8={cursor+layout.k_offset,cursor+layout.v_offset,cursor+layout.scales_offset,layout};cursor+=layout.bytes;}
      else if(o.fp8_kv_cache){a.ek=cursor;cursor+=a.cap*512;a.ev=cursor;cursor+=a.cap*512;a.sf=reinterpret_cast<kv_fp8::Scales*>(cursor);cursor+=a.cap*4*sizeof(kv_fp8::Scales);}
      else{a.kc=reinterpret_cast<__nv_bfloat16*>(cursor);cursor+=a.cap*512*2;a.vc=reinterpret_cast<__nv_bfloat16*>(cursor);cursor+=a.cap*512*2;}
      for(int e=0;e<64;++e)for(const char* part:{"gate","up","down"})experts.push_back(bind(p+"mlp.experts."+std::to_string(e)+"."+part+"_proj.weight"));
    }
    if(cursor!=cache.As<unsigned char>()+kvbytes)throw std::runtime_error("selected KV cache inventory mismatch");
    // Q/K/V share the exact norm pointer, row count and activation codec.
    // Admit this before inference, never use a stale or mismatched input.
    if(o.native_prefill || o.native_decode)for(const auto& layer:layers)for(const auto m:{layer.q,layer.k,layer.v})
      if(m.format!=(o.diagnostic_attention.empty()?1:0) || m.cols!=2304)throw std::runtime_error("QKV FP8 reuse requires admitted W8/2304 matrices");
    if(o.accurate_native_prefill || o.fp8_expert_activations)for(const auto& layer:layers)
      if(layer.q.rows!=4096 || layer.k.rows!=512 || layer.v.rows!=512)throw std::runtime_error("compact QKV requires Q4096/K512/V512");
    const auto embed=bind("model.embed_tokens.weight"),norm=bind("model.norm.weight"),head=bind("lm_head.weight");
    Buffer exl3Schedule(exl3ScheduleBytes);auto* exl3Tiles=exl3Schedule.As<exl3_fp16::Tile>();auto* maskedTiles=exl3Schedule.As<exl3_masked::Tile>();auto* exl3TileCount=reinterpret_cast<int*>(exl3Schedule.As<unsigned char>()+(o.exl3_prefill_fp16?exl3ScheduleBytes-sizeof(int):0));
    Buffer exl3SiLU(exl3SiLUBytes);if(o.exl3_silu_table){exl3_silu_lut::Build<<<256,256,0,stream.s>>>(exl3SiLU.As<unsigned short>());Check(cudaStreamSynchronize(stream.s));}
    Buffer exl3Workspace(exl3Bytes);auto* exl3Input=exl3Workspace.As<float>();auto* exl3Output=o.exl3_experts?exl3Input+batch*8*2304:nullptr;
    Buffer descriptors(experts.size()*sizeof(DeviceMatrix));Check(cudaMemcpyAsync(descriptors.p,experts.data(),descriptors.bytes,cudaMemcpyHostToDevice,stream.s));
    // Metadata only: pointers into the one resident native NVFP4 artifact.
    // Divisor scalars are validated and read once from mapped source metadata,
    // never fetched/repacked/allocated on the token path. Gate/up pairs then down.
    Buffer scaledMatrices(scaledMatrixBytes),scaledTiles(scaledTileBytes);
    auto* scaledWide=scaledTiles.As<scaled_prefill::Tile>();
    auto* scaledWideCount=reinterpret_cast<int*>(scaledTiles.As<unsigned char>()+(o.scaled_bf16_prefill?scaled_prefill::WideCapacity(batch)*sizeof(scaled_prefill::Tile):0));
    if(o.scaled_bf16_prefill){
      std::vector<scaled_prefill::Matrix> meta(28*64*3);
      for(int l=0;l<28;++l)for(int e=0;e<64;++e)for(int part=0;part<3;++part){
        const auto& d=experts[(l*64+e)*3+part];
        const auto name="model.layers."+std::to_string(l)+".mlp.experts."+std::to_string(e)+"."+(part==0?"gate":part==1?"up":"down")+"_proj.weight";
        const auto& t=model.tensors.at(name);float divisor;
        std::memcpy(&divisor,model.weights.data()+t.regions.at("divisor").offset,sizeof(divisor));
        if(d.format!=2 || d.rows!=(part==2?2304:896) || d.cols!=(part==2?896:2304) || !std::isfinite(divisor) || divisor<=0 || reinterpret_cast<std::uintptr_t>(d.data)%16 || reinterpret_cast<std::uintptr_t>(d.scales)%4)throw std::runtime_error("scaled-BF16 expert metadata contract unsupported");
        const int index=part==2?28*128+l*64+e:l*128+e*2+part;
        meta[index]={d.data,d.scales,divisor,d.rows,d.cols};
      }
      Check(cudaMemcpyAsync(scaledMatrices.p,meta.data(),scaledMatrices.bytes,cudaMemcpyHostToDevice,stream.s));
      // cudaMemcpyAsync may borrow host data until completion.
      Check(cudaStreamSynchronize(stream.s));
    }
    InitializeCodecTables();
    auto local=RopeTable(o.capacity,true),global=RopeTable(o.capacity,false);Buffer ropeL(local.size()*4),ropeG(global.size()*4);
    Check(cudaMemcpyAsync(ropeL.p,local.data(),ropeL.bytes,cudaMemcpyHostToDevice,stream.s));Check(cudaMemcpyAsync(ropeG.p,global.data(),ropeG.bytes,cudaMemcpyHostToDevice,stream.s));
    // All workspace and host output capacity are fixed before the first token.
    Buffer scratch(scratchBytes);float* ptr=scratch.As<float>();
    auto take=[&](int n){float* p=ptr;ptr+=n;return p;};
    float *x=take(2304),*normalized=take(2304),*residual=take(2304),*q=take(4096),*attn=take(4096),*k=take(512),*v=take(512),*router=take(64),*gate=take(8*896),*up=take(8*896),*down=take(8*2304),*logits=take(98304);
    Buffer quantCapture(quantCaptureBytes);if(o.capture_quantization_rows)Check(cudaMemsetAsync(quantCapture.p,0,quantCapture.bytes,stream.s));
    Buffer encodedChunk(encodedChunkBytes);auto* chunkEK=encodedChunk.As<unsigned char>();
    auto* chunkEV=o.fp8_kv_cache?chunkEK+batch*512:nullptr;
    auto* chunkSF=o.fp8_kv_cache?reinterpret_cast<kv_fp8::Scales*>(chunkEV+batch*512):nullptr;
    KVCacheView q8Chunk;if(q8Cache){auto layout=KVCacheLayout::Make(o.kv_cache_format,batch);q8Chunk={encodedChunk.As<unsigned char>()+layout.k_offset,encodedChunk.As<unsigned char>()+layout.v_offset,encodedChunk.As<unsigned char>()+layout.scales_offset,layout};}
    Buffer trace(tracebytes),snapshot(snapshotbytes),batchScratch(batchbytes),quantScratch(quantbytes),groupScratch(groupbytes),attentionScores(attentionElements*4),attentionProbabilities(attentionElements*2),tensorScratch(tensorbytes),prompt(o.input_ids.size()*4);
    if(!worker)Check(cudaMemcpyAsync(prompt.p,o.input_ids.data(),prompt.bytes,cudaMemcpyHostToDevice,stream.s));
    float *bx=nullptr,*bn=nullptr,*br=nullptr,*bq=nullptr,*ba=nullptr,*bk=nullptr,*bv=nullptr,*brouter=nullptr,*bg=nullptr,*bu=nullptr,*bd=nullptr;
    if(o.prefill_chunk>1){float* cur=batchScratch.As<float>();const auto takeBatch=[&](int n){auto* p=cur;cur+=static_cast<std::size_t>(n)*batch;return p;};
      bx=takeBatch(2304);bn=takeBatch(2304);br=takeBatch(2304);bq=takeBatch(4096);ba=takeBatch(4096);bk=takeBatch(512);bv=takeBatch(512);brouter=takeBatch(64);bg=takeBatch(8*896);bu=takeBatch(8*896);bd=takeBatch(8*2304);}
    auto* fp8Input=quantScratch.As<unsigned char>();auto* fp8Scales=reinterpret_cast<float*>(fp8Input+(o.native_prefill?batch*7168:0));
    auto* fp4Input=fp8Input+(o.native_prefill?batch*7168+((batch*8*4+15)/16)*16:0);auto* fp4Scales=fp4Input+(o.native_prefill?batch*8*2304/2:0);
    auto* fp4Divisors=reinterpret_cast<float*>(fp4Scales+(o.native_prefill?batch*8*2304/16:0));
    // Dead A4 region leased only for native A8 M1 gate/up, no new allocation.
    if(o.native_decode && o.fp8_expert_activations)AdmitW4A8PlaneLease(fp4Input,static_cast<std::size_t>(fp4Scales-fp4Input));
    // Decode-only lease of dead A4 payload; disjoint from live FP8 projection
    // codes/scales. Even batch1 provides9216B, enough for224 Q8 blocks/8064B.
    if(o.q8_expert_decode && (static_cast<std::size_t>(fp4Scales-fp4Input)<(8*896/32)*sizeof(expert_q8::Block) || reinterpret_cast<std::uintptr_t>(fp4Input)%alignof(expert_q8::Block)))throw std::runtime_error("Q8 decode lease extent/alignment unsupported");
    auto* q8Decode=reinterpret_cast<expert_q8::Block*>(fp4Input);
    if(o.native_prefill && reinterpret_cast<std::uintptr_t>(fp4Input)%16!=0)throw std::runtime_error("NVFP4 activation workspace alignment failure");
    auto* tensorOutput=reinterpret_cast<std::uint16_t*>(tensorScratch.As<unsigned char>()+(o.native_prefill?32*1024*1024:0));
    Buffer compactScratch(compactBytes);auto* expertTiles=compactScratch.As<fast_expert::Tile>();auto* expertTileCount=reinterpret_cast<int*>(compactScratch.As<unsigned char>()+(o.compact_experts?fast_expert::DescriptorCapacity(batch)*sizeof(fast_expert::Tile):0));
    auto* groups=groupScratch.As<int>();auto* counts=groups+(o.native_prefill?64*batch:0);
    struct SnapshotBinding{float* pointer;std::size_t bytes;};
    std::map<std::string,SnapshotBinding,std::less<>> bindings;
    for(const auto& region:regions)bindings.emplace(region.name,SnapshotBinding{snapshot.As<float>()+region.offset,region.count*4});
    const auto capture=[&](std::string_view name,const float* source,bool selected){if(selected){const auto& r=bindings.find(name)->second;Check(cudaMemcpyAsync(r.pointer,source,r.bytes,cudaMemcpyDeviceToDevice,stream.s));}};
    Buffer selection(selectionbytes),sampler(samplerBytes),samplerTemp(samplerTempBytes);
    auto* sampleAdjusted=sampler.As<float>();auto* sampleSorted=worker?sampleAdjusted+98304:nullptr;
    auto* sampleIds=worker?reinterpret_cast<int*>(sampleSorted+98304):nullptr;auto* sampleSortedIds=worker?sampleIds+98304:nullptr;
    auto* sampleCumulative=worker?reinterpret_cast<double*>(sampleSortedIds+98304):nullptr;
    auto* sampleState=worker?reinterpret_cast<gpu_sampling::State*>(sampleCumulative+98304):nullptr;
    auto* sampleCounts=worker?reinterpret_cast<int*>(sampleState+1):nullptr;
    Buffer fastAttentionScratch(fastAttentionBytes);
    Buffer ids(batch*8*sizeof(int)),routing(batch*8*sizeof(float)),bad(sizeof(int)),control(sizeof(Control));Pinned host;
    Check(cudaMemsetAsync(bad.p,0,bad.bytes,stream.s));Check(cudaStreamSynchronize(stream.s));
    // The source mapping is released. Runtime has no CPU weight fallback.
    model.weights=internal::MappedFile{};
    Buffer prefillCommand(worker && o.controlled_chunk_graphs?sizeof(PrefillState):0);
    DecodeGraph decodeGraph,samplingGraph,prefillGraph;std::array<DecodeGraph,ChunkShapeSet::kMaxShapes> chunkGraphs;
    std::optional<SamplingOptions> capturedSampling;
    ChunkShapeSet chunkShapes;
    // Unwind drains outstanding kernels before graph/buffer destructors.
    struct Drain {cudaStream_t stream;~Drain(){cudaStreamSynchronize(stream);}} drain{stream.s};
    InferenceResult result;result.prefill_chunk=o.prefill_chunk;result.prefill_execution=o.controlled_chunk_graphs?"chunk_graph_with_eager_tail":o.prefill_graphs?"whole-graph":"stream";
    if(o.native_prefill)result.math_profile="mellum16-prefill-w4a4-rowdiv-block16-w8a8-row-rn-f32mma-decode-w4a16-v1";
    if(o.accurate_native_prefill)result.math_profile="mellum16-prefill-w4a4-rowdiv-block16-compensated-w8a8-compensated-decode-w4a16-v1";
    if(o.fp8_expert_activations)result.math_profile="mellum16-prefill-w4a8-row-fp8mma-block16-compensated-w8a8-compensated-decode-w4a16-v1";
    if(o.native_decode){const auto suffix=result.math_profile.find("-decode-w4a16-v1");
      if(suffix==std::string::npos)throw std::runtime_error("native decode profile identity missing");
      result.math_profile.replace(suffix,std::string::npos,"-decode-same-v1");}
    if(o.tc_prefill_attention)result.math_profile+="+prefill-attention-bf16tc-k16-compensated-tie-repair-rnexp-v1";
    result.kernel_profile=o.warp_projections?"simt-warp4-rn-v1":"simt-block128-rn-v1";
    if(o.prefill_chunk>1){
      if(o.fp8_expert_activations)result.kernel_profile+="+prefill-gem16-w4a8-register-fp8mma-product-residual-fp8-mma-bins-v3";
      else if(o.accurate_native_prefill)result.kernel_profile+="+prefill-gem16-nvfp4-compensated-fp8-mma-bins-v2";
      else result.kernel_profile+=o.native_prefill?"+prefill-gem16-sm120-cutlass-v1":"+prefill-batch-exact-v1";
    }if(o.native_decode)result.kernel_profile+="+decode-selected8-m16-native-v3";
    if(o.native_decode && o.fp8_expert_activations)result.kernel_profile+="+decode-gateup-w4a8-simt-k16-fixed-basis-v1+decode-gateup-w4a8-exact-planes-dp4a-v1";
    if(o.native_prefill && (o.accurate_native_prefill || o.fp8_expert_activations))result.kernel_profile+="+prefill-experts-m16-compensated-v1";
    if(o.native_prefill && (o.accurate_native_prefill || o.fp8_expert_activations))result.kernel_profile+="+prefill-fp8-two-m16-weight-bin-reuse-v1";
    if(o.native_decode && o.accurate_native_prefill && !o.fp8_expert_activations)result.kernel_profile+="+decode-gateup-w4a4-simt-k16-warp-v1+decode-gateup-e2m1-dp4a-basis-v1";
    if(o.native_decode && (o.accurate_native_prefill || o.fp8_expert_activations))result.kernel_profile+="+decode-fp8-simt-vector4-compensated-v1";
    if(o.native_decode && (o.accurate_native_prefill || o.fp8_expert_activations))result.kernel_profile+="+decode-gateup-split-warps8-direct-m1-v1";
    if(o.accurate_native_prefill || o.fp8_expert_activations)result.kernel_profile+="+prefill-fp8-mma-bins-k32-v1";
    result.kernel_profile+="+argmax-hierarchical-v1+attention-decode-specialized-v1";
    if(!o.fast_w4a8)result.kernel_profile+="+decode-attention-cta8-compensated-score-v1";
    result.kernel_profile+="+router-parallel-rn-stable-v1";
    result.kernel_profile+="+qk-norm-rope-fused-compensated-bf16-v1";
    if(o.accurate_native_prefill || o.fp8_expert_activations)result.kernel_profile+="+input-norm-fp8-fused-2304-v1";
    if(o.fp8_expert_activations)result.kernel_profile+="+expert-e2m1-fp8-basis-prmt-v1";
    if(o.native_prefill || o.native_decode)result.kernel_profile+="+qkv-fp8-input-reuse-v1";
    if(o.native_prefill && o.prefill_chunk>1 && (o.accurate_native_prefill || o.fp8_expert_activations))result.kernel_profile+="+prefill-qkv-compact-v1";
    if(o.prefill_chunk>1)result.kernel_profile+="+prefill-source-window-specialized-v1";
    if(o.native_decode && (o.accurate_native_prefill || o.fp8_expert_activations))result.kernel_profile+="+decode-qkv-compact-v1";
    if(o.warp_projections && !o.native_decode)result.kernel_profile+="+expert-nvfp4-w4a16-vector8-v1";
    if(o.tc_prefill_attention)result.kernel_profile+="+prefill-attention-bf16tc-three-pass-m16n8-tie-repair-v1";
    if(o.tc_prefill_attention)result.kernel_profile+="+prefill-attention-shared-qk64n16-pv16n64-cta8-v1";
    if(o.fast_w4a8){result.execution_profile=kW4A8FProfile;result.math_profile=std::string(kW4A8FProfile)+"+f1a-decode-online-fp32-bf16kv-output-v1+prefill-projections-experts-compensated-control"+(o.tc_prefill_attention?std::string("+prefill-attention-bf16tc-k16-compensated-tie-repair-rnexp-v1"):std::string());result.kernel_profile+="+decode-split128-group4-bf16-online-merge32-v1";}
    result.decode_execution=o.full_decode_graph?"full-graph":o.decode_graphs?"graph":o.device_decode?"device":"host";
    if(o.device_decode)result.kernel_profile+="+decode-device-token-position-budget-v1";
    if(o.decode_graphs)result.kernel_profile+=o.full_decode_graph?"+decode-full-step-graph-bf16-head-select-advance-v1+host-8byte-observation-v1":"+decode-28layer-body-graph-v1+host-head-select-transfer-v1";
    if(o.capture_quantization_rows)result.quantization_rows.resize(row_quantization::kCaptureBytes);
    result.generated_ids.reserve(worker?o.capacity:o.generate);result.generated_ids.resize(static_cast<std::size_t>(o.generate));if(o.capture_logits)result.logits.resize(98304);
    result.weights_sha256=model.weights_sha256;result.snapshot_regions=std::move(regions);if(o.snapshot_layer>=0)result.operator_snapshot.resize(snapshotbytes/4);
    result.weights_bytes=weightsbytes;result.kv_bytes=kvbytes;
    if(o.online_prefill_attention){result.math_profile=std::string(kW4A8FProfile)+"+f1a-decode-online-fp32-bf16kv-output-v1+f1b-prefill-online-fp32-qk-bf16-pv-mma-output-v1+projections-experts-compensated-control";result.kernel_profile+="+prefill-online-bf16-m32n32-d128-group4-cta8-v1";}
    if(o.packed_bf16_prefill_attention && !q8Cache){result.kernel_profile+="+prefill-exact-BF16-chunk-16B-async-v1";result.math_profile+=";prefill-KV=unchanged-G3-BF16-QK/PV/roundedP/onlineK32-operation-order;current-chunk-exact-BF16-packed-once/inclusiveEncode,"+(o.prefill_chunk==2048?std::string("4MiB2048"):std::to_string(packedChunkBytes)+"B-chunk"+std::to_string(o.prefill_chunk))+"-transient-buffer/no-shadowprefix";}
    if(o.padded_bf16_prefill_attention){result.kernel_profile+="+prefill-exact-shared136-oneK-v1";result.math_profile+=";prefill-dataflow=unchanged-G3-operations/keys/values,QK-complete-barrier-before-singleK-overwrite,padded-shared-stride136";}
    if(o.gqa8_prefill_attention){const std::string old="prefill-online-bf16-m32n32-d128-group4-cta8-v1";const auto pos=result.kernel_profile.find(old);if(pos==std::string::npos)throw std::runtime_error("missing explicit G3 attention parent profile");result.kernel_profile.replace(pos,old.size(),"g3-prefill-bf16-q8h8-k32-regq-async-cta4-v1");result.math_profile+="+g3-q8-h8-bf16-regq-pipeline-k32-fp32-stats-rounded-p-v1";}
    if(o.scaled_bf16_prefill){result.kernel_profile+="+e3-full64-scalefold-bf16-pipe-m16-e4-tail-v1";result.math_profile+="+expert-e4m3-scalefold-bf16-j64i64k64-pipe-v1;NVFP4-native-SF-fold-before-BF16-MMA;rowScale/original-independent-divisor;expf-two-BF16;old-E4-down-encode-and-ordered-combine;C<=128-old-control";}
    if(o.prune_last_prefill_ffn){result.kernel_profile+="+g4-last-logit-final-layer-ffn-only-after-all-kv-v1";}
    if(o.ordinary_fp8_projections){const auto old=result.math_profile.find("projections-experts-compensated-control");if(old!=std::string::npos)result.math_profile.replace(old,std::string("projections-experts-compensated-control").size(),"experts-compensated-control");result.math_profile+="+f2-projections-ordinary-fp8-mma-fp32-scaled-bf16-v1";result.kernel_profile+="+prefill-cutlass-fp8-128x128x64-scaled-bf16-expand+decode-gem16-direct-fp8-rounded-qkv160-v1";}
    if(o.staged_experts){const auto pos=result.math_profile.find("experts-compensated-control");if(pos!=std::string::npos)result.math_profile.replace(pos,std::string("experts-compensated-control").size(),o.staged_mma_decode?"f3c-experts-bf16-working-tiles-mma-fp32-scale-expf-bf16-v1":"f3c-experts-bf16-working-prefill-mma-decode-simt-fp32-scale-expf-bf16-v1");result.kernel_profile+=o.staged_mma_decode?"+compact-selected-bf16-m16n64k32-working-tiles-v1":"+compact-bf16-m16n64k32-selected-vector4-v1";}
    if(o.block_expert_prefill){const std::string old=o.staged_mma_decode?"f3c-experts-bf16-working-tiles-mma-fp32-scale-expf-bf16-v1":"f3c-experts-bf16-working-prefill-mma-decode-simt-fp32-scale-expf-bf16-v1";const auto pos=result.math_profile.find(old);if(pos!=std::string::npos)result.math_profile.replace(pos,old.size(),o.staged_mma_decode?"f4-prefill-block-fp8-fp32-rnexp-decode-bf16-mma-fp32-expf-bf16-v1":"f4-prefill-block-fp8-fp32-rnexp-decode-vector-fp32-expf-bf16-v1");result.kernel_profile+="+explicit-prefill-override-compact-k16-fp8-block-v1";}
    if(o.ordinary_prefill_silu){const std::string old="f4-prefill-block-fp8-fp32-rnexp-decode-vector-fp32-expf-bf16-v1";const auto pos=result.math_profile.find(old);if(pos==std::string::npos)throw std::runtime_error("missing explicit N1a parent math policy");result.math_profile.replace(pos,old.size(),"n1a-prefill-block-fp8-fp32-expf-two-bf16-decode-vector-fp32-expf-bf16-v1");result.kernel_profile+="+n1a-compact-m16-prefill-expf-silu-v1";}
    if(o.ordinary_block_experts){const auto pos=result.math_profile.find("experts-compensated-control");if(pos!=std::string::npos)result.math_profile.replace(pos,std::string("experts-compensated-control").size(),"f3b-experts-k16-fp8-mma-sf-fp32-fma-bf16-rn-silu-v1");result.kernel_profile+="+compact-selected-m16-ordinary-scaled-blocks-v1";}
    if(o.compact_experts){result.kernel_profile+="+f3a-stable-device-compact-m16-descriptors-v1";}
    if(o.ordinary_rms_norm){result.math_profile+="+f2-input-post-rms-fma-rsqrt-two-bf16-v1";result.kernel_profile+="+input-rms-amax-fp8-fused-ordinary-post-rms-2304-v1";}
    if(o.prefill_chunk>512){result.kernel_profile+="+n1c-chunk2048-admission-local-kv-suffix-commit-v1";}
    if(o.ordinary_fp8_simt){const auto oldmath=result.math_profile.find("f2-projections-ordinary-fp8-mma-fp32-scaled-bf16-v1");if(oldmath!=std::string::npos)result.math_profile.replace(oldmath,std::string("f2-projections-ordinary-fp8-mma-fp32-scaled-bf16-v1").size(),"f2-prefill-fp8-mma-decode-simt-fp32-scale-bf16-v1");const auto old=result.kernel_profile.find("decode-gem16-direct-fp8-rounded-qkv160-v1");if(old!=std::string::npos)result.kernel_profile.replace(old,std::string("decode-gem16-direct-fp8-rounded-qkv160-v1").size(),"decode-fp8-simt-vector4-ordinary-qkv1280-v1");}
    if(o.packed_expert_decode){result.math_profile+="+g2-e4m3-pair-dot-sfk16-two-acc-rn-warp32-bf16-v1";result.kernel_profile+="+g2-selected-packed4-warp-per-row-cta128-native-sfk16-v1";}
    if(o.q8_expert_decode){const std::string old="n1a-prefill-block-fp8-fp32-expf-two-bf16-decode-vector-fp32-expf-bf16-v1";const auto pos=result.math_profile.find(old);if(pos==std::string::npos)throw std::runtime_error("missing N2b parent phase math identity");result.math_profile.replace(pos,old.size(),"n1a-prefill-block-fp8-fp32-expf-two-bf16+n2b-decode-nvfp4-q8-symmetric-block32-halfscale-s32-dot16-fp32-expf-two-bf16-v1");result.kernel_profile+="+n2b-direct-bf16-q8-quantize-decoder-dp4a-row2-warp32-dead-a4-8064byte-lease-v1";}
    if(o.gqa8_decode_attention){const std::string old="f1a-decode-online-fp32-bf16kv-output-v1";const auto pos=result.math_profile.find(old);if(pos==std::string::npos)throw std::runtime_error("missing N3a parent attention math identity");result.math_profile.replace(pos,old.size(),"n3a-gqa8-bf16-qk-pv-mma-fp32-online-softmax-bf16-p-fp32-fixup-bf16-output-v1");const std::string oldkernel="decode-split128-group4-bf16-online-merge32-v1";const auto kp=result.kernel_profile.find(oldkernel);if(kp==std::string::npos)throw std::runtime_error("missing N3a parent attention kernel identity");result.kernel_profile.replace(kp,oldkernel.size(),"n3a-gqa8-async32-visible-split-local16-global32-bf16-mma-merge4-v1");}
    if(o.reciprocal_fused_quantization){result.math_profile+="+n4c-fused-row-fp32-rn-reciprocal-multiply-e4m3-finite-inverse-v1";result.kernel_profile+="+n4c-fused-row-encode-reciprocal-v1";}
    if(o.fused_residual_norm)result.kernel_profile+=o.reciprocal_fused_quantization?"+n4b-attention-residual-postnorm-expert-encode-row128-compensated-reciprocal-v1":"+n4b-attention-residual-postnorm-expert-encode-row128-compensated-direct-div-v1";
    if(o.fused_attention_o)result.kernel_profile+=o.reciprocal_fused_quantization?"+n4a-merge-o-encode-row256-reciprocal-v1":"+n4a-merge-o-encode-row256-direct-div-v1";
    if(o.fp8_kv_cache){result.math_profile+="+n3b-kv-e4m3-satfinite-rne-direct-div-fp32-token-head-k-v-scale-zero-unit-bf16-tile-v1";result.kernel_profile+="+n3b-encoded-old-current-chunk-kv-async-inplace-tile-suffix-commit-v1";}
    if(q8Cache){result.kernel_profile+="+s6-q8-tokenhead128-f32scale-tile-v1";result.math_profile+=";KV-override=q8_token_head128_f32scale_v1/K-after-BF16-QK-RoPE,V-BF16/independent-token-head-KV-FP32-amax127-RNdiv-RNE-clip127/BF16-RN-code-scale/old-and-current-same-codec/no-BF16-shadowprefix/selected-BF16-MMA-softmax-order;encoding-inclusive";}
    result.resident_buffer_bytes=admission::Sum({checkpointBuffer.bytes,packedChunk.bytes,exl3SiLU.bytes,exl3Schedule.bytes,exl3Fixture.bytes,routeCapture.bytes,rangeCapture.bytes,exl3Workspace.bytes,weightsbytes,kvbytes,descriptors.bytes,ropeL.bytes,ropeG.bytes,scratch.bytes,ids.bytes,routing.bytes,bad.bytes,control.bytes,selection.bytes,trace.bytes,snapshot.bytes,batchScratch.bytes,quantScratch.bytes,groupScratch.bytes,tensorScratch.bytes,prompt.bytes,deviceState.bytes,attentionScores.bytes,attentionProbabilities.bytes,fastAttentionScratch.bytes,compactScratch.bytes,encodedChunk.bytes,scaledMatrices.bytes,scaledTiles.bytes,quantCapture.bytes,prefillCommand.bytes,sampler.bytes,samplerTemp.bytes});
    if(result.resident_buffer_bytes!=plannedResident)throw std::runtime_error("admitted allocation inventory differs from actual resident buffers");
    result.device_free_before=free;Check(cudaMemGetInfo(&free,&total));result.device_free_after_initialization=free;
    if(o.capture_trace)result.trace.resize(tracebytes/sizeof(float));
    static_assert(sizeof(DeviceMatrix)==56 && sizeof(Control)==8 && sizeof(SelectionCandidate)==8);
    if(worker){const auto plan=PlanServerMemory(weightsbytes,o.capacity,o.capacity,o.prefill_chunk,true,samplerTempBytes,o.kv_cache_format,o.server_kv_checkpoint);if(admission::Sum({plan.resident,prefillCommand.bytes})!=result.resident_buffer_bytes)throw std::runtime_error("persistent memory inventory mismatch");worker->RecordLoad(result.resident_buffer_bytes);}
    if(worker){ // Prime every sampler/CUB kernel before capture, outside Generate.
      for(const auto kernel:{reinterpret_cast<const void*>(gpu_sampling::CountPrompt),reinterpret_cast<const void*>(gpu_sampling::RecordChoice),reinterpret_cast<const void*>(gpu_sampling::Prepare<true>)})Check(cudaFuncGetAttributes(&attributes,kernel));
      gpu_sampling::State initial{0,0};Check(cudaMemcpyAsync(sampleState,&initial,sizeof(initial),cudaMemcpyHostToDevice,stream.s));
      Check(cudaMemsetAsync(logits,0,98304*4,stream.s));gpu_sampling::Launch(logits,sampleAdjusted,sampleSorted,sampleIds,sampleSortedIds,sampleCumulative,sampleState,control.As<Control>(),bad.As<int>(),samplerTemp.p,samplerTempBytes,stream.s,SamplingOptions{1,1,0,0},sampleCounts);Check(cudaStreamSynchronize(stream.s));
    }
    if(worker && o.diagnostic_decode_token>=0)result.math_profile+=";fixed-teacher-decode-token="+std::to_string(o.diagnostic_decode_token)+";diagnostic-only/no-append/no-sampling";
#if PADRA_EXL3_PREFILL_ASYNC64
    if(o.exl3_packed_activations)result.kernel_profile+="+research-asyncAB-M32N128K64-weightsA-shared-overlay-Tge4-v2;Tlt4-originalSIMT";
#endif
    const auto baseKernelProfile=result.kernel_profile,baseMathProfile=result.math_profile;
    const auto loadSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::size_t preparedLength=0;
    // The entire existing forward frame remains live on its one worker. There
    // are no retained references into a caller/request or expired stack frame.
    do {
    if(worker){
      if(!worker->AwaitPrepare())break;
      if(worker->checkpoint_restore_pending){
        // The CPU worker validated an actual completed identity/prefix. Restore
        // all local windows on the same owned stream before returning Prepared;
        // global rows before the anchor have never been overwritten.
        for(int l=0;l<28;++l)if(l%4!=3){const auto offset=static_cast<std::size_t>(l-l/4)*1024*2048;
          const auto* saved=checkpointBuffer.As<unsigned char>()+offset;
          Check(cudaMemcpyAsync(layers[l].kc,saved,1024*1024,cudaMemcpyDeviceToDevice,stream.s));
          Check(cudaMemcpyAsync(layers[l].vc,saved+1024*1024,1024*1024,cudaMemcpyDeviceToDevice,stream.s));}
        Check(cudaStreamSynchronize(stream.s));worker->RecordCheckpointRestore();
      }
      gpu_sampling::State initial{o.sampling.seed,0};Check(cudaMemcpyAsync(sampleState,&initial,sizeof(initial),cudaMemcpyHostToDevice,stream.s));
      result.kernel_profile=baseKernelProfile;result.math_profile=baseMathProfile;
      result.logits.resize(o.capture_logits?98304:0);result.cancelled=false;result.generated_ids.resize(o.generate);result.decode_steps=0;result.prefill_seconds=result.decode_seconds=0;result.prefill_setup_seconds=result.decode_setup_seconds=0;
      Check(cudaMemsetAsync(bad.p,0,bad.bytes,stream.s));Check(cudaMemsetAsync(deviceState.p,0,deviceState.bytes,stream.s));
      Check(cudaMemcpyAsync(prompt.p,o.input_ids.data(),o.input_ids.size()*sizeof(int),cudaMemcpyHostToDevice,stream.s));Check(cudaStreamSynchronize(stream.s));
      if(!o.controlled_chunk_graphs && preparedLength!=o.input_ids.size()){prefillGraph.Reset();}
    }
    auto initialized=std::chrono::steady_clock::now();result.initialization_seconds=std::chrono::duration<double>(initialized-begin).count();
    auto prefill_finished=initialized,last_token_ready=initialized;
    int generated=0,executed=0,token=o.input_ids[0];
    const auto project=[&](DeviceMatrix m,const float* input,float* output,bool input_quantized=false){
      if(input_quantized && (!o.native_decode || m.format!=1 || !((m.cols==2304 && input==normalized) || (o.fused_attention_o && m.cols==4096 && m.rows==2304 && input==attn))))throw std::runtime_error("FP8 decode input binding contract mismatch");
      if(o.native_decode && m.format==1){
        if(!input_quantized)QuantizeFP8<<<1,128,0,stream.s>>>(input,fp8Input,fp8Scales,m.cols,bad.As<int>());
        if(o.attention_fp8_mma)fast_projection::Direct<<<(m.rows+31)/32,128,0,stream.s>>>(m,fp8Input,fp8Scales,output,bad.As<int>());
        else if(o.paired_attention_o && m.rows==2304 && m.cols==4096)fp8_pair_exact::Vector<<<(m.rows+7)/8,128,0,stream.s>>>(m,fp8Input,fp8Scales,output,bad.As<int>());
        else if(o.ordinary_fp8_simt)fast_projection::Vector<<<(m.rows+3)/4,128,0,stream.s>>>(m,fp8Input,fp8Scales,output,bad.As<int>());
        else if(o.ordinary_fp8_projections)fast_projection::Direct<<<(m.rows+31)/32,128,0,stream.s>>>(m,fp8Input,fp8Scales,output,bad.As<int>());
        else if(o.accurate_native_prefill || o.fp8_expert_activations)ProjectFP8Vector<<<dim3((m.rows+3)/4,1),128,0,stream.s>>>(m,fp8Input,fp8Scales,output,bad.As<int>());
        else {const auto status=tc::ProjectFP8(fp8Input,fp8Scales,m.data,reinterpret_cast<const std::uint16_t*>(m.scales),tensorOutput,1,m.rows,m.cols,tensorScratch.p,32*1024*1024,stream.s);
          if(!status.ok())throw std::runtime_error(status.message());ExpandBF16<<<(m.rows+127)/128,128,0,stream.s>>>(tensorOutput,output,m.rows);}
      }else if(o.warp_projections)GemvWarp<<<(m.rows+3)/4,128,0,stream.s>>>(m,input,output,bad.As<int>());
      else Gemv<<<m.rows,128,0,stream.s>>>(m,input,output,bad.As<int>());};
    if(o.exl3_experts){result.kernel_profile+=o.exl3_rate==8?"+exl3-k8-h128-bf16-input-simt-v1":o.exl3_decode_mma?"+exl3-k4-h128-explicit-f16-mma-v1":"+exl3-k4-h128-bf16-input-simt-v1";result.math_profile+=std::string(";expert-override=EXL3-K")+std::to_string(o.exl3_rate)+"-mul1;FP16-sidecars;FP32-H128-rounded-BF16-input;"+(o.exl3_decode_mma?"FP16-decoded-codebook/explicit-input-FP16-RN/FP32-MMA-dot":"FP16-decoded-codebook-FP32-compensated-dot")+";FP32-H128-BF16-output;rnexp-two-BF16-SiLU;no-expert-E4M3-activation";}
    result.actual_attention_projection=o.diagnostic_attention.empty()?"FP8 weights/E4M3 input selected baseline":"diagnostic BF16 weights/BF16 input compensated FP32";
    result.actual_expert_dispatch=o.exl3_experts?std::string("EXL3-K")+std::to_string(o.exl3_rate)+(o.exl3_decode_mma?" exact-FP16 decoded weights/explicit-FP16-input-FP32-MMA-decode/input-H128-":" exact-FP16 decoded weights/compensated SIMT decode/input-H128-")+(o.diagnostic_h128_fp32?"FP32":"BF16")+"/output-H128-BF16":"NVFP4 selected expert profile";
    if(o.exl3_prefill_fp16&&!o.exl3_prefill_masked){result.kernel_profile+="+exl3-k4-prefill-fp16-m32-n128-k32-simt-tail-x1";result.math_profile+=";X1-prefill=BF16-H128-input-explicit-FP16-cast;exact-FP16-weight-MMA-FP32;declared-BF16-compensated-SIMT-narrow-tail;frozen-SIMT-prefill-tails";result.actual_expert_dispatch+=";prefill=M32-N128-K32-FP16-MMA/full32+exact-SIMT-tail";}
    if(o.exl3_prefill_masked&&!o.exl3_prefill_register){result.kernel_profile+="+exl3-k4-prefill-fp16-m32-masked-rp1";result.math_profile+=";R-P1-prefill=BF16-H128-explicit-FP16-cast/all-valid-pairs;exact-FP16-weight-MMA-FP32/full-and-masked32;narrow-T<4-compensated-SIMT;decode-unchanged";result.actual_expert_dispatch+=";prefill=masked-M32-N128-K32-FP16-MMA/no-SIMT-tail-T>=4";}
    if(o.exl3_prefill_register){result.kernel_profile+="+exl3-k4-prefill-register-m32";result.math_profile+=";P2-prefill=immutable-masked-M32-FP16-operands/MMA-order/H128;exact-adjacent-state-half2-register-B;T<4-compensated-SIMT";result.actual_expert_dispatch+=";prefill=register-B-M32-N128-K32/immutable-RP1-arithmetic";}
    if(o.exl3_decode_mma&&!o.exl3_decode_dot){result.kernel_profile+="+exl3-k4-decode-mma-n64-warps4";result.math_profile+=";D2-decode=BF16-H128-input-explicit-FP16-RN-cast;exact-FP16-K4-weight;M1-duplicated-M16/FP32-MMA;N64/four-warps/four-stage-packed-prefetch;separate-full-H128/Gate-Up/rn-two-BF16-SiLU;no-fallback";result.actual_expert_dispatch+=";decode=M1-FP16-MMA-N64/raw/full-H128-D2";}
    if(o.exl3_prefill_dot){result.kernel_profile+="+prefill-exact-UDP4A-codebook";result.math_profile+=";UDP4A=exact-procedural-hash-byte-sum,no-INT8-activation";result.actual_expert_dispatch+=";prefill=register-B-M32-exact-UDP4A";}
    if(o.exl3_decode_dot){result.kernel_profile+="+decode-exact-UDP4A-codebook";result.math_profile+=";decode-codebook=exact-UDP4A-byte-sum;BF16-H128-explicit-FP16-RN-cast;FP32-MMA-order-immutable;separate-Gate-Up/rn-two-BF16-SiLU/full-H128;no-fallback";result.actual_expert_dispatch+=o.exl3_decode_fused?";decode=dot-codec-M1-FP16-MMA/N128-warps8/fused-H128":";decode=dot-codec-M1-FP16-MMA/N64-warps4/separate-H128";}
    if(o.exl3_decode_fused){result.kernel_profile+="+decode-fused-H128-N128-warps8";result.math_profile+=";fused-input/full-output-H128=immutable-operation-order/BF16-boundary,explicit-FP16-input;MMA-order-unchanged";result.actual_expert_dispatch+=";decode=fused-H128-M1-N128-warps8";}
    if(o.exl3_decode_dual){result.kernel_profile+="+decode-paired-gateup-gridZ2";result.math_profile+=";paired-gate/up=separate-matrices/suh/H128/native-K4-MMA-operation-order,gridZ2,no-sharing-transforms";result.actual_expert_dispatch+=";decode=paired-distinct-gateup-gridZ2+single-down";}
#if PADRA_EXL3_PREFILL_ASYNC64
    if(o.wide_rms_dataflow){result.kernel_profile+="+same128-compensated-input-RMS-256-output-dataflow+postnorm-compensated-tree256-registerrow";result.math_profile+=";input-RMS=unchanged128-product/warp32/four-real-warp-tail-tree/BF16/stats/E4,256-independent-output-ownership";}
#else
    if(o.wide_rms_dataflow){result.kernel_profile+="+same128-compensated-RMS-256-output-dataflow";result.math_profile+=";RMS=unchanged128-product/warp32/four-real-warp-tail-tree/BF16/stats/E4,256-independent-output-ownership";}
#endif
    if(o.wide_attention_merge){result.kernel_profile+="+exact-N4a-merge-512-workers";result.math_profile+=";merge=unchanged-per-head32-split-FMA/BF16/amax/scales/encoder,512ownership";}
    if(o.paired_attention_o){result.kernel_profile+="+exact-O-SIMT-pairedrows";result.math_profile+=";O=unchanged-per-row-ordinary-FMA/reduction/scales/BF16,shared-E4-input-decode";}
    if(o.exact_router_rank){result.kernel_profile+="+exact-probability-rank-router-decode";result.math_profile+=";router=unchanged-RNexp/ascending-compensated-sums/probability-ties/BF16/ascendingIDs,direct-ranks";}
    if(o.attention_fragment_validation){result.kernel_profile+="+same-BF16-fragment-finite-validation";result.math_profile+=";K/V-validation=already-loaded-MMA-fragments,unchanged-N3a-geometry/splits/arithmetic";}
    if(o.exact_fp8_encoding){result.kernel_profile+="+exact-E4-RN-boundary-correction-staged-norm";result.math_profile+=";input-norm/O-final-E4=all1065402240-finite-BF16/native-amax-pairs-byte-exact,cachedRNinverse/same-FP32-RN-division-at-boundaries,unchanged-norm/merge/scales";}
    if(o.exl3_decode_lane){result.kernel_profile+="+decode-transpose-one-shuffle-K4";result.math_profile+=";decode-MMA-ownership=weights-A/replicated-vector-B,one-MMA-per16-outputs,actual-bit-exact-fused-control;one-previous-word-shuffle-native-K4-f16-codebook";result.actual_expert_dispatch+=";decode=transpose-one-shuffle-native-K4-M1-N128";}
    if(o.exl3_prefill_warp){result.kernel_profile+="+prefill-warp-H128";result.math_profile+=";warp-H128=exact-original-order/BF16-F16-cast";result.actual_expert_dispatch+=";prefill=warp-input-H128-exact";}
    if(o.prefill_graphs && !o.controlled_chunk_graphs){result.kernel_profile+="+whole-prefill-graph-v1";result.math_profile+=";prefill=entire-fixed-prompt/all-chunks/same-kernels/ordered-KVcommits,graph-captured-instantiated-uploaded-once-before-timers,no-replay-allocation,no-fallback";}
    if(o.exact_norm_prefill){result.kernel_profile+="+prefill-norm-exact-corrected-batch-v1";result.math_profile+=";prefill-inputnorm=same128-compensatedtree/BF16/scales/E4-bytes,stagedrow/cachedRNinverse/originalRNdivisionatmidpoints";}
    if(o.pair64_qknorm_prefill){result.kernel_profile+="+prefill-qknorm-same128-pair64-large-v1";result.math_profile+=";QKnorm-prefill=same128products/4originalwarp-partials/compensatedtree/BF16-norm-Rope,T>=128-twooppositefeatures-perthread,no-rowsharedbarrier,T<128-originalgeometry,no-precision/contextchange";}
    if(o.quad_router_prefill){result.kernel_profile+="+prefill-router-compensated-quadrows-v1";result.math_profile+=";router-prefill=same-per-row-lane32-stride/compensated-product-sum/tree/BF16,4-independent-rows/input-reuse,no-probability-selection-change";}
    if(o.exl3_wide_input){result.kernel_profile+="+prefill-H128-vector-warps8-v1";result.math_profile+=";Gate-Up-H128=distinct-sidecars/same-butterfly-BF16-F16,8-independentwarps/vector-source-sidecar-store,T<4-originaldeclaredSIMT,no-precisionchange";}
    if(o.exl3_packed_activations){result.kernel_profile+="+prefill-exact-BF16-storage-SiLU-downH128-v1";result.math_profile+=";prefill-expert-storage=same-BF16-values/16bit-physical-gate-up-down,T>=4;unchanged-M32/MMA/H128/FP16-input;fused-exact-table-RN-SiLU/BF16-product/downH128,T<4-unchanged-widened-SIMT;no-newprecision/no-loopallocation";result.actual_expert_dispatch+=";prefill-BF16-packed-storage/exact-SiLU-downH128";}
    if(o.exl3_silu_table){result.kernel_profile+="+exact-BF16-SiLU-table";result.math_profile+=";SiLU=exact-finite-BF16-RN-exp-add-div/BF16-product,128KiB-table-built-before-timers,no-loop-allocation";}
#if PADRA_EXL3_PREFILL_ASYNC64
    if(o.exl3_bf16_postnorm){result.kernel_profile+="+EXL3-BF16-only-postnorm";result.math_profile+=";postnorm=compensated-FP32/BF16-residual-and-two-BF16-output-stages,no-unused-FP8-encoding";}
#else
    if(o.exl3_bf16_postnorm){result.kernel_profile+="+EXL3-BF16-only-postnorm";result.math_profile+=";postnorm=exact-compensated-N4b-BF16,no-unused-FP8-encoding";}
#endif
    const auto activateExl3=[&](float* g,const float* u,int tokens){if(o.exl3_silu_table)exl3_silu_lut::Activate<<<tokens*56,128,0,stream.s>>>(g,u,exl3SiLU.As<unsigned short>(),tokens*8*896,bad.As<int>());else Activate<<<tokens*56,128,0,stream.s>>>(g,u,tokens);};
    if(o.attention_fp8_mma){result.kernel_profile+="+attention-decode-FP8-MMA-explicit-override";result.math_profile+=";attention-override=unchanged-FP8-weights/E4M3-input/scales,FP32-MMA-accumulation/BF16-output,new-reduction-order";result.actual_attention_projection="explicit decode FP8-MMA override; unchanged weights/activation codec; numerical variant";}
    if(o.exl3_decode_n32){result.kernel_profile+="+exl3-k4-decode-n32-warps4-kpart2-rd1";result.math_profile+=";R-D1-decode=BF16-H128-input-widened-FP32;exact-FP16-frozen-codec;ordinary-FP32-FMA/even-odd-K16/four-lane-reduction/two-part-RN-add;separate-full-H128-BF16;separate-Gate-Up/rn-two-BF16-SiLU";result.actual_expert_dispatch+=";decode=N32/four-warps/two-K-partitions/raw-scratch/full-H128-R-D1";}
    if(o.exl3_decode_selected){result.kernel_profile+="+exl3-k4-decode-selected-n128-warps8-x2";result.math_profile+=";X2-decode=BF16-H128-input-widened-FP32;exact-FP16-half2-codec;ordinary-FP32-FMA-four-lane-reduction;FP32-inverse-H128-BF16;separate-gate-up;rn-two-BF16-SiLU";result.actual_expert_dispatch+=";decode=selected-N128/eight-warps/ordinary-FP32-FMA-X2;prefill-contract-unchanged";}
    result.diagnostic_attention_sha256=model.diagnostic_attention_sha256;
    const auto trellisProject=[&](int layer,int part,const float* input,float* output,int tokens){
      exl3_expert::Launch(descriptors.As<DeviceMatrix>(),layer,part,ids.As<int>(),input,output,exl3Input,o.exl3_prefill_fp16?exl3Input+tokens*8*(part==2?896:2304):exl3Output,tokens,bad.As<int>(),stream.s,o.diagnostic_h128_fp32,o.diagnostic_routes?rangeCapture.As<unsigned>():nullptr);
    };
    const auto expert=[&](int layer,int part,const float* input,float* output,bool input_quantized=false){const int rows=part==2?2304:896,cols=part==2?896:2304;
      if(input_quantized && (!o.fused_residual_norm || !o.native_decode || !o.staged_experts || o.staged_mma_decode || part!=0 || input!=normalized))throw std::runtime_error("fused postnorm expert input binding mismatch");
      if(o.exl3_experts && o.exl3_decode_dual){exl3_mma_dual::Launch(descriptors.As<DeviceMatrix>(),layer,part,ids.As<int>(),input,output,exl3Input,8,bad.As<int>(),stream.s,nullptr,part==0?up:nullptr);if(part==0)activateExl3(output,up,1);return;}
      if(o.exl3_experts){const auto selectedProject=[&](int p,float* target){if(o.exl3_decode_lane)exl3_mma_lane::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),input,target,exl3Input,8,bad.As<int>(),stream.s);else if(o.exl3_decode_fused)exl3_mma_fused::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),input,target,exl3Input,8,bad.As<int>(),stream.s);else if(o.exl3_decode_dot)exl3_mma_dot::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),input,target,exl3Input,8,bad.As<int>(),stream.s);else if(o.exl3_decode_mma)exl3_mma_decode::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),input,target,exl3Input,8,bad.As<int>(),stream.s);else if(o.exl3_decode_n32)exl3_n32::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),input,target,exl3Input,8,bad.As<int>(),stream.s);else if(o.exl3_decode_selected)exl3_selected::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),input,target,exl3Input,8,bad.As<int>(),stream.s);else trellisProject(layer,p,input,target,1);};selectedProject(part,output);if(part==0){selectedProject(1,up);activateExl3(output,up,1);}return;}
      if(o.native_decode){
        if(o.fp8_expert_activations){
          if(o.q8_expert_decode){if(part!=0 && part!=2)throw std::runtime_error("Q8 expert decoder requires fused gate/up or down");const int activationRows=part==0?1:8;expert_q8::Quantize<<<dim3((cols+127)/128,activationRows),128,0,stream.s>>>(input,q8Decode,activationRows,cols,bad.As<int>());if(part==0)expert_q8::Selected<true><<<dim3(rows/2,8),32,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),q8Decode,output,bad.As<int>());else expert_q8::Selected<false><<<dim3(rows/2,8),32,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),q8Decode,output,bad.As<int>());return;}
          if(!input_quantized){
          if(o.ordinary_block_experts || o.staged_experts)QuantizeFP8<<<part==0?1:8,128,0,stream.s>>>(input,fp8Input,fp8Scales,cols,bad.As<int>());
          else if(part==0)QuantizeFP8Planes<<<1,128,0,stream.s>>>(input,fp8Input,fp8Scales,fp4Input,bad.As<int>());
          else QuantizeFP8<<<8,128,0,stream.s>>>(input,fp8Input,fp8Scales,cols,bad.As<int>());
          }
          if(o.packed_expert_decode){if(part!=0 && part!=2)throw std::runtime_error("G2 packed expert decoder requires fused gate/up or down");if(part==0)fast_expert::SelectedPacked4<true><<<dim3(rows/4,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),fp8Input,fp8Scales,output,bad.As<int>());else fast_expert::SelectedPacked4<false><<<dim3(rows/4,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),fp8Input,fp8Scales,output,bad.As<int>());}
          else if(o.staged_experts){if(o.staged_mma_decode){if(part==0)fast_expert::StagedBF16<true,true><<<dim3(rows/64,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),nullptr,nullptr,nullptr,fp8Input,fp8Scales,output,1,bad.As<int>());else fast_expert::StagedBF16<false,true><<<dim3(rows/64,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),nullptr,nullptr,nullptr,fp8Input,fp8Scales,output,1,bad.As<int>());}else{if(part==0)fast_expert::SelectedVector<true><<<dim3(rows/4,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),fp8Input,fp8Scales,output,bad.As<int>());else fast_expert::SelectedVector<false><<<dim3(rows/4,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),fp8Input,fp8Scales,output,bad.As<int>());}}
          else if(o.ordinary_block_experts){if(part==0)fast_expert::SelectedBlock<true><<<dim3(rows/64,1,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),nullptr,fp8Input,nullptr,output,1,bad.As<int>(),fp8Scales);else fast_expert::SelectedBlock<false><<<dim3(rows/64,1,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),nullptr,fp8Input,nullptr,output,1,bad.As<int>(),fp8Scales);}
          else if(part==0){gem_reuse::ExpertW4A8Planes<<<dim3(224,8),256,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),fp4Input,fp8Scales,output,up,bad.As<int>());Activate<<<56,128,0,stream.s>>>(output,up);}
          else gem_reuse::ExpertGrouped<false,false,true,true><<<dim3((rows+63)/64,1,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),nullptr,fp8Input,nullptr,output,1,bad.As<int>(),fp8Scales);
        }else{
          FP4Divisors<<<part==2?8:1,128,0,stream.s>>>(input,fp4Divisors,cols,bad.As<int>());
          QuantizeFP4<<<((part==2?8:1)*cols/16+127)/128,128,0,stream.s>>>(input,fp4Input,fp4Scales,(part==2?8:1)*cols,bad.As<int>(),fp4Divisors,cols);
          if(o.accurate_native_prefill){
            if(part==0){gem_reuse::ExpertW4A4SIMT<true><<<dim3(224,8),256,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),fp4Input,fp4Scales,fp4Divisors,output,up,bad.As<int>());Activate<<<56,128,0,stream.s>>>(output,up);}
            else gem_reuse::ExpertGrouped<false,true,false,true><<<dim3((rows+63)/64,1,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),nullptr,fp4Input,fp4Scales,output,1,bad.As<int>(),fp4Divisors);
          }else if(part==0)gem_reuse::ExpertGrouped<true,false,false,true><<<dim3((rows+63)/64,1,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),nullptr,fp4Input,fp4Scales,output,1,bad.As<int>(),fp4Divisors);
          else gem_reuse::ExpertGrouped<false,false,false,true><<<dim3((rows+63)/64,1,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,ids.As<int>(),nullptr,fp4Input,fp4Scales,output,1,bad.As<int>(),fp4Divisors);
        }
      }else if(o.warp_projections)ExpertVector<<<dim3((rows+3)/4,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,part,ids.As<int>(),input,output,bad.As<int>());
      else Expert<<<dim3(rows,8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,part,ids.As<int>(),input,output,bad.As<int>());};
    if(o.controlled_chunk_graphs){result.kernel_profile+="+s2-controlled-chunk-graphs-eager-tail-v1";result.math_profile+=";prefill-execution=chunk_graph_with_eager_tail;device-base/prompt-offset/full-layer-commit/request-binding-epochs;startup-shapes"+std::to_string(o.prefill_chunk)+"/"+std::to_string(4000%o.prefill_chunk)+",other-shapes-identical-body-eager;no-per-request-capture";}
    if(o.server_kv_checkpoint){result.kernel_profile+="+s8-local-bf16-checkpoint-v1";result.math_profile+=";checkpoint=one42MiB-local-ring-bitcopy/unchanged-global-prefix;only-containing-chunk-identical-eager-body/no-chunk-split/no-arithmetic-change;restore-before-Prepared/validated-canonical-prefix-runtime-generation-capacity-BF16";}
    if(worker && o.sampling.selection_graph()){result.kernel_profile+="+sampling-cub-stable-sort-fp64-cdf-splitmix64-v1";result.math_profile+=";selection=graph-bound-penalties/temperature/top-k/top-p-cub-fp64-cdf-splitmix64-v2;unmodified-logits";}
    if(worker)worker->RecordIdentity(result.math_profile,result.kernel_profile);
    const int steps=static_cast<int>(o.input_ids.size())+std::max(0,o.generate-1);
    const auto finishDeviceToken=[&](int position,bool initialize,bool sampled=false,SamplingOptions fixed={}){
      Norm<<<1,128,0,stream.s>>>(x,normalized,norm,2304,1);project(head,normalized,logits);
      if(sampled && fixed.stochastic())gpu_sampling::Launch(logits,sampleAdjusted,sampleSorted,sampleIds,sampleSortedIds,sampleCumulative,sampleState,control.As<Control>(),bad.As<int>(),samplerTemp.p,samplerTempBytes,stream.s,fixed,sampleCounts);
      else {
      const float* selectionLogits=logits;
      if(sampled && fixed.penalties()){gpu_sampling::PrepareLaunch(logits,sampleAdjusted,sampleIds,sampleCounts,kVocabulary,bad.As<int>(),fixed,stream.s);selectionLogits=sampleAdjusted;}
      SelectTiles<<<kSelectionBlocks,kSelectionThreads,0,stream.s>>>(selectionLogits,selection.As<SelectionCandidate>(),kVocabulary,bad.As<int>());
      Select<<<1,kSelectionThreads,0,stream.s>>>(selection.As<SelectionCandidate>(),control.As<Control>(),kSelectionBlocks,bad.As<int>());
      }
      if(sampled && fixed.penalties())gpu_sampling::RecordChoice<<<1,1,0,stream.s>>>(control.As<Control>(),sampleCounts,kVocabulary,bad.As<int>());
      if(o.device_decode){
        if(initialize)InitializeDecodeState<<<1,1,0,stream.s>>>(deviceState.As<DecodeState>(),control.As<Control>(),position+1,o.capacity,std::max(0,o.generate-1),bad.As<int>(),contextMaximum);
        else AdvanceDecodeState<<<1,1,0,stream.s>>>(deviceState.As<DecodeState>(),control.As<Control>(),bad.As<int>(),contextMaximum);
      }
    };
    const auto observeToken=[&](int position){
      Check(cudaGetLastError());Check(cudaMemcpyAsync(host.p,control.p,sizeof(Control),cudaMemcpyDeviceToHost,stream.s));Check(cudaStreamSynchronize(stream.s));
      if(host.p->bad)throw std::runtime_error("nonfinite or unrepresentable native forward; no fallback attempted");
      token=host.p->token;last_token_ready=std::chrono::steady_clock::now();
      if(position+1==static_cast<int>(o.input_ids.size()))prefill_finished=last_token_ready;
      if(o.generate>0)result.generated_ids[static_cast<std::size_t>(generated++)]=token;
      if(worker && !worker->Emit(token,position+1==static_cast<int>(o.input_ids.size())))result.cancelled=true;
      return o.generate>0 && token==28 && o.diagnostic_decode_token<0;
    };
    const auto finishToken=[&](int position){finishDeviceToken(position,position+1==static_cast<int>(o.input_ids.size()),worker && o.sampling.selection_graph(),o.sampling);return observeToken(position);};
    const auto decodeBody=[&](int position){
      if(o.diagnostic_decode_token>=0)fast_attention::ForceDiagnosticDecodeToken<<<1,1,0,stream.s>>>(deviceState.As<DecodeState>(),o.diagnostic_decode_token,bad.As<int>());
      if(o.device_decode)EmbedControlled<<<1,128,0,stream.s>>>(embed,deviceState.As<DecodeState>(),x,bad.As<int>(),contextMaximum);
      else Embed<<<1,128,0,stream.s>>>(embed,token,x);
      for(int l=0;l<28;++l){const auto& a=layers[l];const bool selected=l==o.snapshot_layer && position==o.snapshot_position;
        capture("input",x,selected);
        if(o.ordinary_rms_norm)fast_norm::Rms2304<true><<<1,128,0,stream.s>>>(x,normalized,a.norm,fp8Input,fp8Scales,bad.As<int>());
        else if(o.wide_rms_dataflow)NormFP8WideExact<<<1,256,0,stream.s>>>(x,normalized,a.norm,fp8Input,fp8Scales,bad.As<int>(),selected?bindings.find("input_norm_stats")->second.pointer:nullptr);
        else if(o.exact_fp8_encoding)NormFP8Exact<<<1,128,0,stream.s>>>(x,normalized,a.norm,fp8Input,fp8Scales,bad.As<int>(),selected?bindings.find("input_norm_stats")->second.pointer:nullptr);
        else if(o.native_decode && (o.accurate_native_prefill || o.fp8_expert_activations))NormFP8<<<1,128,0,stream.s>>>(x,normalized,a.norm,fp8Input,fp8Scales,bad.As<int>(),selected?bindings.find("input_norm_stats")->second.pointer:nullptr);
        else Norm<<<1,128,0,stream.s>>>(x,normalized,a.norm,2304,1,selected?bindings.find("input_norm_stats")->second.pointer:nullptr);capture("input_norm",normalized,selected);
        if(!o.diagnostic_attention.empty()){project(a.q,normalized,q);project(a.k,normalized,k);project(a.v,normalized,v);}
        else if(o.native_decode && (o.accurate_native_prefill || o.fp8_expert_activations)){
          if(o.attention_fp8_mma)fast_projection::QKV<<<160,128,0,stream.s>>>(a.q,a.k,a.v,fp8Input,fp8Scales,q,k,v,bad.As<int>());
          else if(o.ordinary_fp8_simt)fast_projection::QKVVector<<<1280,128,0,stream.s>>>(a.q,a.k,a.v,fp8Input,fp8Scales,q,k,v,bad.As<int>());
          else if(o.ordinary_fp8_projections)fast_projection::QKV<<<160,128,0,stream.s>>>(a.q,a.k,a.v,fp8Input,fp8Scales,q,k,v,bad.As<int>());
          else ProjectQKVGrouped<false><<<1280,128,0,stream.s>>>(a.q,a.k,a.v,fp8Input,fp8Scales,q,k,v,1,bad.As<int>());
        }else {project(a.q,normalized,q);project(a.k,normalized,k,o.native_decode);project(a.v,normalized,v,o.native_decode);}
        capture("q_proj",q,selected);capture("k_proj",k,selected);capture("v_proj",v,selected);
        if(o.device_decode)QKNormRopeControlled<<<36,128,0,stream.s>>>(q,k,a.qn,a.kn,(l%4==3?ropeG:ropeL).As<float>(),deviceState.As<DecodeState>(),bad.As<int>(),nullptr,nullptr,contextMaximum);
        else QKNormRope<<<36,128,0,stream.s>>>(q,k,a.qn,a.kn,(l%4==3?ropeG:ropeL).As<float>(),position,bad.As<int>(),selected?bindings.find("q_norm")->second.pointer:nullptr,selected?bindings.find("k_norm")->second.pointer:nullptr);
        capture("q_rope",q,selected);capture("k_rope",k,selected);
        if(q8Cache)kv_q8::Encode<true><<<1,128,0,stream.s>>>(k,v,static_cast<signed char*>(a.q8.k),static_cast<signed char*>(a.q8.v),static_cast<kv_q8::Scales*>(a.q8.scales),1,bad.As<int>(),deviceState.As<DecodeState>(),a.cap,contextMaximum);
        else if(o.fp8_kv_cache)kv_fp8::Encode<true><<<1,128,0,stream.s>>>(k,v,a.ek,a.ev,a.sf,1,a.cap,deviceState.As<DecodeState>(),bad.As<int>());
        else if(o.device_decode)StoreControlled<<<4,128,0,stream.s>>>(k,v,a.kc,a.vc,deviceState.As<DecodeState>(),a.cap,bad.As<int>(),contextMaximum);
        else Store<<<4,128,0,stream.s>>>(k,v,a.kc,a.vc,position,a.cap);
        if(selected){const auto& r=bindings.find("key_cache")->second;const int visible=l%4==3?position+1:std::min(position+1,1024);CaptureKV<<<(visible*512+127)/128,128,0,stream.s>>>(r.pointer,a.kc,a.vc,position,a.cap,l%4!=3);}
        if(q8Cache){gqa8_q8_attention::Launch(q,a.q8,attn,fp8Input,fp8Scales,deviceState.As<DecodeState>(),l%4!=3,fastAttentionScratch.As<float>(),bad.As<int>(),stream.s,contextMaximum);}
        else if(o.wide_attention_merge){attention_o_wide_exact::Launch(q,a.kc,a.vc,attn,fp8Input,fp8Scales,deviceState.As<DecodeState>(),a.cap,l%4!=3,fastAttentionScratch.As<float>(),bad.As<int>(),stream.s,true,contextMaximum);}
        else if(o.exact_fp8_encoding){attention_o_exact::Launch(q,a.kc,a.vc,attn,fp8Input,fp8Scales,deviceState.As<DecodeState>(),a.cap,l%4!=3,fastAttentionScratch.As<float>(),bad.As<int>(),stream.s,o.attention_fragment_validation);}
        else if(o.reciprocal_fused_quantization){attention_o_fusion::Launch<true>(q,a.kc,a.vc,attn,fp8Input,fp8Scales,deviceState.As<DecodeState>(),a.cap,l%4!=3,fastAttentionScratch.As<float>(),bad.As<int>(),stream.s);}
        else if(o.fused_attention_o){attention_o_fusion::Launch(q,a.kc,a.vc,attn,fp8Input,fp8Scales,deviceState.As<DecodeState>(),a.cap,l%4!=3,fastAttentionScratch.As<float>(),bad.As<int>(),stream.s);}
        else if(o.fp8_kv_cache){gqa8_fp8_attention::Launch(q,a.ek,a.ev,a.sf,attn,deviceState.As<DecodeState>(),a.cap,l%4!=3,fastAttentionScratch.As<float>(),bad.As<int>(),stream.s);}
        else if(o.gqa8_decode_attention){gqa8_attention::Launch(q,a.kc,a.vc,attn,deviceState.As<DecodeState>(),a.cap,l%4!=3,fastAttentionScratch.As<float>(),bad.As<int>(),stream.s);}
        else if(o.fast_w4a8){fast_attention::Launch(q,a.kc,a.vc,attn,deviceState.As<DecodeState>(),a.cap,l%4!=3,fastAttentionScratch.As<float>(),bad.As<int>(),stream.s);}
        else if(o.device_decode){
          if(l%4!=3)AttendDecodeControlled<true><<<32,256,a.cap*sizeof(float),stream.s>>>(q,a.kc,a.vc,attn,deviceState.As<DecodeState>(),a.cap,bad.As<int>());
          else AttendDecodeControlled<false><<<32,256,a.cap*sizeof(float),stream.s>>>(q,a.kc,a.vc,attn,deviceState.As<DecodeState>(),a.cap,bad.As<int>());
        }else DecodeAttention(q,a.kc,a.vc,attn,position,a.cap,l%4!=3,bad.As<int>(),stream.s,selected?bindings.find("attention_scores")->second.pointer:nullptr,selected?bindings.find("attention_probabilities")->second.pointer:nullptr);
        if(o.capture_quantization_rows && l==0){auto* dst=quantCapture.As<unsigned char>();Check(cudaMemcpyAsync(dst,attn,4096*4,cudaMemcpyDeviceToDevice,stream.s));Check(cudaMemcpyAsync(dst+25600,fp8Input,4096,cudaMemcpyDeviceToDevice,stream.s));Check(cudaMemcpyAsync(dst+32000,fp8Scales,4,cudaMemcpyDeviceToDevice,stream.s));}
        capture("attention",attn,selected);
        project(a.out,attn,residual,o.fused_attention_o && o.diagnostic_attention.empty());capture("o_proj",residual,selected);
        if(o.wide_rms_dataflow)ResidualPostNormWideExact<<<1,256,0,stream.s>>>(x,residual,normalized,a.post,bad.As<int>(),selected?bindings.find("post_norm_stats")->second.pointer:nullptr);
        else if(o.exl3_bf16_postnorm)ResidualPostNormBF16<<<1,128,0,stream.s>>>(x,residual,normalized,a.post,bad.As<int>(),selected?bindings.find("post_norm_stats")->second.pointer:nullptr);
        else if(o.reciprocal_fused_quantization)ResidualPostNormFP8<true><<<1,128,0,stream.s>>>(x,residual,normalized,a.post,fp8Input,fp8Scales,bad.As<int>(),selected?bindings.find("post_norm_stats")->second.pointer:nullptr);
        else if(o.fused_residual_norm)ResidualPostNormFP8<<<1,128,0,stream.s>>>(x,residual,normalized,a.post,fp8Input,fp8Scales,bad.As<int>(),selected?bindings.find("post_norm_stats")->second.pointer:nullptr);
        else{Add<<<18,128,0,stream.s>>>(x,residual,2304,bad.As<int>());
          if(o.ordinary_rms_norm)fast_norm::Rms2304<false><<<1,128,0,stream.s>>>(x,normalized,a.post,nullptr,nullptr,bad.As<int>());
          else Norm<<<1,128,0,stream.s>>>(x,normalized,a.post,2304,1,selected?bindings.find("post_norm_stats")->second.pointer:nullptr);}
        if(o.capture_quantization_rows && l==0){auto* dst=quantCapture.As<unsigned char>();Check(cudaMemcpyAsync(dst+16384,normalized,2304*4,cudaMemcpyDeviceToDevice,stream.s));Check(cudaMemcpyAsync(dst+29696,fp8Input,2304,cudaMemcpyDeviceToDevice,stream.s));Check(cudaMemcpyAsync(dst+32004,fp8Scales,4,cudaMemcpyDeviceToDevice,stream.s));}
        capture("after_attention",x,selected);project(a.router,normalized,router);
        capture("post_norm",normalized,selected);capture("router_logits",router,selected);
        if(o.exact_router_rank)RouteRankExact<<<1,64,0,stream.s>>>(router,ids.As<int>(),routing.As<float>(),bad.As<int>());
        else RouteParallel<<<1,64,0,stream.s>>>(router,ids.As<int>(),routing.As<float>(),bad.As<int>());
        if(selected)CaptureRouting<<<1,8,0,stream.s>>>(bindings.find("router_ids")->second.pointer,ids.As<int>(),routing.As<float>());
        if(o.native_decode){expert(l,0,normalized,gate,o.fused_residual_norm);}
        else {expert(l,0,normalized,gate);expert(l,1,normalized,up);
          capture("expert_gate",gate,selected);capture("expert_up",up,selected);
          Activate<<<56,128,0,stream.s>>>(gate,up);}

        capture("expert_activation",gate,selected);
        expert(l,2,gate,down);
        capture("expert_down",down,selected);
        Combine<<<18,128,0,stream.s>>>(x,down,routing.As<float>(),bad.As<int>());
        capture("output",x,selected);
        if(o.capture_trace){
          if(o.device_decode)TraceControlled<<<19,128,0,stream.s>>>(trace.As<float>(),x,router,ids.As<int>(),routing.As<float>(),deviceState.As<DecodeState>(),l,bad.As<int>());
          else Trace<<<19,128,0,stream.s>>>(trace.As<float>()+(position*28+l)*2384,x,router,ids.As<int>(),routing.As<float>());
        }
      }
    };
    if(o.decode_graphs && !decodeGraph.executable){
      const auto setup=std::chrono::steady_clock::now();decodeGraph.Capture(stream.s,[&]{decodeBody(0);if(o.full_decode_graph)finishDeviceToken(0,false);});
      if(worker)worker->RecordCapture();
      result.decode_setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup).count();
      initialized=std::chrono::steady_clock::now();prefill_finished=initialized;last_token_ready=initialized;
      result.initialization_seconds=std::chrono::duration<double>(initialized-begin).count();
      Check(cudaMemGetInfo(&free,&total));result.device_free_after_initialization=free;
    }
    if(worker && o.sampling.selection_graph() && (!capturedSampling || !capturedSampling->SameGraph(o.sampling))){
      // Prepare boundary only: one bounded selection executable, replaced when
      // fixed options change. RNG seed/counter alone do not rebuild this graph.
      Check(cudaStreamSynchronize(stream.s));samplingGraph.Reset();
      const auto fixed=o.sampling;
      samplingGraph.Capture(stream.s,[&]{decodeBody(0);finishDeviceToken(0,false,true,fixed);});
      capturedSampling=fixed;worker->RecordCapture();
    }
    if(worker && o.sampling.penalties()){
      Check(cudaMemsetAsync(sampleCounts,0,98304*sizeof(int),stream.s));
      gpu_sampling::CountPrompt<<<(static_cast<int>(o.input_ids.size())+255)/256,256,0,stream.s>>>(prompt.As<int>(),static_cast<int>(o.input_ids.size()),sampleCounts,98304,bad.As<int>());
      Check(cudaGetLastError());Check(cudaStreamSynchronize(stream.s));
    }
    bool stopped=false;
    if(o.prefill_chunk>1){
      const auto prefillChunk=[&](int base,int n,PrefillState* prefill,int checkpoint_anchor=0){
        if(prefill)EmbedPrefillControlled<<<n,128,0,stream.s>>>(embed,prompt.As<int>(),bx,prefill,n,bad.As<int>(),contextMaximum);
        else Embed<<<n,128,0,stream.s>>>(embed,base,bx,prompt.As<int>());
        const auto batchProject=[&](DeviceMatrix m,const float* input,float* output,bool input_quantized=false,int tokenCount=-1){
          const int projectTokens=tokenCount<0?n:tokenCount;
          if(input_quantized && (!o.native_prefill || m.format!=1 || m.cols!=2304 || input!=bn))throw std::runtime_error("QKV FP8 prefill input contract mismatch");
          if(o.native_prefill && m.format==1){
            if(!input_quantized)QuantizeFP8<<<projectTokens,128,0,stream.s>>>(input,fp8Input,fp8Scales,m.cols,bad.As<int>());
            if(!o.ordinary_fp8_projections && (o.accurate_native_prefill || o.fp8_expert_activations))ProjectFP8Tiles<2><<<dim3((m.rows+31)/32,(projectTokens+31)/32),128,0,stream.s>>>(m,fp8Input,fp8Scales,output,projectTokens,bad.As<int>());
            else {const auto status=tc::ProjectFP8(fp8Input,fp8Scales,m.data,reinterpret_cast<const std::uint16_t*>(m.scales),tensorOutput,projectTokens,m.rows,m.cols,tensorScratch.p,32*1024*1024,stream.s);
            if(!status.ok())throw std::runtime_error(status.message());
            ExpandBF16<<<(projectTokens*m.rows+127)/128,128,0,stream.s>>>(tensorOutput,output,projectTokens*m.rows);}
          }else if(o.quad_router_prefill && m.format==0 && m.rows==64 && m.cols==2304)RouterBF16BatchQuad<<<dim3(4,projectTokens),128,0,stream.s>>>(m,input,output,projectTokens,bad.As<int>());
          else ProjectBatch<<<dim3((m.rows+3)/4,projectTokens),128,0,stream.s>>>(m,input,output,bad.As<int>());
        };
        const auto batchExpert=[&](int layer,int part,const float* input,float* output,bool selected=false,int tokenCount=-1){
          const int expertTokens=tokenCount<0?n:tokenCount;
          const int rows=part==2?2304:896,cols=part==2?896:2304;
          if(o.exl3_experts){
            const auto project=[&](int p,float* target){if(o.exl3_prefill_warp)exl3_warp_input::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),groups,counts,maskedTiles,exl3TileCount,input,target,exl3Workspace.As<__half>(),nullptr,exl3Input,expertTokens,bad.As<int>(),stream.s);else if(o.exl3_prefill_dot)exl3_register_dot::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),groups,counts,maskedTiles,exl3TileCount,input,target,exl3Workspace.As<__half>(),nullptr,exl3Input,expertTokens,bad.As<int>(),stream.s);else if(o.exl3_prefill_register)exl3_register::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),groups,counts,maskedTiles,exl3TileCount,input,target,exl3Workspace.As<__half>(),nullptr,exl3Input,expertTokens,bad.As<int>(),stream.s);else if(o.exl3_prefill_masked)exl3_masked::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),groups,counts,maskedTiles,exl3TileCount,input,target,exl3Workspace.As<__half>(),nullptr,exl3Input,expertTokens,bad.As<int>(),stream.s);else if(o.exl3_prefill_fp16)exl3_fp16::Launch(descriptors.As<DeviceMatrix>(),layer,p,ids.As<int>(),groups,counts,exl3Tiles,exl3TileCount,input,target,exl3Workspace.As<__half>(),nullptr,exl3Input,expertTokens,bad.As<int>(),stream.s);else trellisProject(layer,p,input,target,expertTokens);};
            project(part,output);if(part==0){project(1,bu);activateExl3(output,bu,expertTokens);}return;}
          if(o.native_prefill){const int elements=expertTokens*(part==2?8:1)*cols;
            if(o.fp8_expert_activations){
              QuantizeFP8<<<expertTokens*(part==2?8:1),128,0,stream.s>>>(input,fp8Input,fp8Scales,cols,bad.As<int>());
              if(o.compact_experts){
                if(o.scaled_bf16_prefill && expertTokens>128){
                  const int cap=scaled_prefill::WideCapacity(expertTokens);
                  if(part==0){
                    scaled_prefill::GateUp<<<dim3(14,cap),256,0,stream.s>>>(scaledMatrices.As<scaled_prefill::Matrix>()+layer*128,groups,scaledWide,cap,fp8Input,fp8Scales,output,expertTokens,896,2304,bad.As<int>(),nullptr,nullptr,scaledWideCount);
                    fast_expert::CompactM16<true,true,true><<<dim3(14,fast_expert::DescriptorCapacity(expertTokens)),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,expertTiles,expertTileCount,fp8Input,nullptr,output,expertTokens,bad.As<int>(),fp8Scales);
                  }else if(part==2){
                    scaled_prefill::Down<<<dim3(36,cap),256,0,stream.s>>>(scaledMatrices.As<scaled_prefill::Matrix>()+28*128+layer*64,groups,scaledWide,cap,scaledWideCount,fp8Input,fp8Scales,output,expertTokens,bad.As<int>());
                    fast_expert::CompactM16<false,true><<<dim3(36,fast_expert::DescriptorCapacity(expertTokens)),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,expertTiles,expertTileCount,fp8Input,nullptr,output,expertTokens,bad.As<int>(),fp8Scales);
                  }else throw std::runtime_error("scaled-BF16 prefill gate/up must be fused; no independent up launch");
                }
                else if(o.staged_experts && !o.block_expert_prefill){if(part==0)fast_expert::StagedBF16<true,false><<<dim3(rows/64,fast_expert::DescriptorCapacity(expertTokens)),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,expertTiles,expertTileCount,fp8Input,fp8Scales,output,expertTokens,bad.As<int>());else fast_expert::StagedBF16<false,false><<<dim3(rows/64,fast_expert::DescriptorCapacity(expertTokens)),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,expertTiles,expertTileCount,fp8Input,fp8Scales,output,expertTokens,bad.As<int>());}
                else if(o.ordinary_block_experts || o.block_expert_prefill){if(part==0 && o.ordinary_prefill_silu)fast_expert::CompactM16<true,true,true><<<dim3(rows/64,fast_expert::DescriptorCapacity(expertTokens)),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,expertTiles,expertTileCount,fp8Input,nullptr,output,expertTokens,bad.As<int>(),fp8Scales);else if(part==0)fast_expert::CompactM16<true,true><<<dim3(rows/64,fast_expert::DescriptorCapacity(expertTokens)),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,expertTiles,expertTileCount,fp8Input,nullptr,output,expertTokens,bad.As<int>(),fp8Scales);else fast_expert::CompactM16<false,true><<<dim3(rows/64,fast_expert::DescriptorCapacity(expertTokens)),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,expertTiles,expertTileCount,fp8Input,nullptr,output,expertTokens,bad.As<int>(),fp8Scales);}
                else if(part==0)fast_expert::CompactM16<true><<<dim3((rows+63)/64,fast_expert::DescriptorCapacity(expertTokens)),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,expertTiles,expertTileCount,fp8Input,nullptr,output,expertTokens,bad.As<int>(),fp8Scales);
                else fast_expert::CompactM16<false><<<dim3((rows+63)/64,fast_expert::DescriptorCapacity(expertTokens)),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,expertTiles,expertTileCount,fp8Input,nullptr,output,expertTokens,bad.As<int>(),fp8Scales);
              }else if(part==0)gem_reuse::ExpertPrefillM16<true,false,true><<<dim3((rows+63)/64,(expertTokens+15)/16,64),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,fp8Input,nullptr,output,expertTokens,bad.As<int>(),fp8Scales,selected?bindings.find("expert_gate")->second.pointer:nullptr,selected?bindings.find("expert_up")->second.pointer:nullptr);
              else gem_reuse::ExpertPrefillM16<false,false,true><<<dim3((rows+63)/64,(expertTokens+15)/16,64),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,fp8Input,nullptr,output,expertTokens,bad.As<int>(),fp8Scales);
            }else{
            FP4Divisors<<<expertTokens*(part==2?8:1),128,0,stream.s>>>(input,fp4Divisors,cols,bad.As<int>());
            QuantizeFP4<<<(elements/16+127)/128,128,0,stream.s>>>(input,fp4Input,fp4Scales,elements,bad.As<int>(),fp4Divisors,cols);
            if(o.accurate_native_prefill){
              if(part==0)gem_reuse::ExpertPrefillM16<true,true,false><<<dim3((rows+63)/64,(expertTokens+15)/16,64),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,fp4Input,fp4Scales,output,expertTokens,bad.As<int>(),fp4Divisors,selected?bindings.find("expert_gate")->second.pointer:nullptr,selected?bindings.find("expert_up")->second.pointer:nullptr);
              else gem_reuse::ExpertPrefillM16<false,true,false><<<dim3((rows+63)/64,(expertTokens+15)/16,64),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,fp4Input,fp4Scales,output,expertTokens,bad.As<int>(),fp4Divisors);
            }else if(part==0)gem_reuse::ExpertGrouped<true><<<dim3((rows+63)/64,(expertTokens+31)/32,64),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,fp4Input,fp4Scales,output,expertTokens,bad.As<int>(),fp4Divisors,selected?bindings.find("expert_gate")->second.pointer:nullptr,selected?bindings.find("expert_up")->second.pointer:nullptr);
            else gem_reuse::ExpertGrouped<false><<<dim3((rows+63)/64,(expertTokens+31)/32,64),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,groups,counts,fp4Input,fp4Scales,output,expertTokens,bad.As<int>(),fp4Divisors);
            }
          }else ExpertBatch<<<dim3((rows+3)/4,expertTokens*8),128,0,stream.s>>>(descriptors.As<DeviceMatrix>(),layer,part,ids.As<int>(),input,output,bad.As<int>());
        };
        for(int l=0;l<28;++l){const auto& a=layers[l];const bool selected=l==o.snapshot_layer;
          capture("input",bx,selected);
          if(o.ordinary_rms_norm)fast_norm::Rms2304<true><<<n,128,0,stream.s>>>(bx,bn,a.norm,fp8Input,fp8Scales,bad.As<int>());
          else if(o.exact_norm_prefill)NormFP8Exact<<<n,128,0,stream.s>>>(bx,bn,a.norm,fp8Input,fp8Scales,bad.As<int>(),selected?bindings.find("input_norm_stats")->second.pointer:nullptr);
          else if(o.native_prefill && (o.accurate_native_prefill || o.fp8_expert_activations))NormFP8<<<n,128,0,stream.s>>>(bx,bn,a.norm,fp8Input,fp8Scales,bad.As<int>(),selected?bindings.find("input_norm_stats")->second.pointer:nullptr);
          else Norm<<<dim3(1,n),128,0,stream.s>>>(bx,bn,a.norm,2304,1,selected?bindings.find("input_norm_stats")->second.pointer:nullptr);capture("input_norm",bn,selected);
          if(!o.diagnostic_attention.empty()){batchProject(a.q,bn,bq);batchProject(a.k,bn,bk);batchProject(a.v,bn,bv);}
          else if(o.native_prefill && (o.accurate_native_prefill || o.fp8_expert_activations)){
            if(o.ordinary_fp8_projections){batchProject(a.q,bn,bq,true);batchProject(a.k,bn,bk,true);batchProject(a.v,bn,bv,true);}
            else ProjectQKVTiles<2><<<dim3(160,(n+31)/32),128,0,stream.s>>>(a.q,a.k,a.v,fp8Input,fp8Scales,bq,bk,bv,n,bad.As<int>());
          }else {batchProject(a.q,bn,bq);batchProject(a.k,bn,bk,o.native_prefill);batchProject(a.v,bn,bv,o.native_prefill);}capture("q_proj",bq,selected);capture("k_proj",bk,selected);capture("v_proj",bv,selected);
          if(o.pair64_qknorm_prefill && n>=128)QKNormRopePair64<<<dim3(36,n),64,0,stream.s>>>(bq,bk,a.qn,a.kn,(l%4==3?ropeG:ropeL).As<float>(),base,bad.As<int>(),selected?bindings.find("q_norm")->second.pointer:nullptr,selected?bindings.find("k_norm")->second.pointer:nullptr,prefill,contextMaximum);
          else QKNormRope<<<dim3(36,n),128,0,stream.s>>>(bq,bk,a.qn,a.kn,(l%4==3?ropeG:ropeL).As<float>(),base,bad.As<int>(),selected?bindings.find("q_norm")->second.pointer:nullptr,selected?bindings.find("k_norm")->second.pointer:nullptr,prefill,contextMaximum);capture("q_rope",bq,selected);capture("k_rope",bk,selected);
          // Read old cache and current chunk separately. Commit only after all
          // chunk queries have finished, preventing premature local-ring overwrite.
          if(q8Cache){kv_q8::Encode<false><<<n,128,0,stream.s>>>(bk,bv,static_cast<signed char*>(q8Chunk.k),static_cast<signed char*>(q8Chunk.v),static_cast<kv_q8::Scales*>(q8Chunk.scales),n,bad.As<int>());q8_prefill_attention::LaunchEncoded(bq,a.q8,q8Chunk,ba,base,n,l%4!=3,bad.As<int>(),stream.s,prefill,contextMaximum);}
          else if(o.fp8_kv_cache){kv_fp8::Encode<false><<<n,128,0,stream.s>>>(bk,bv,chunkEK,chunkEV,chunkSF,n,batch,nullptr,bad.As<int>());
            fp8_prefill_attention::Launch(bq,a.ek,a.ev,a.sf,chunkEK,chunkEV,chunkSF,ba,base,n,a.cap,l%4!=3,bad.As<int>(),stream.s);}
          else if(o.padded_bf16_prefill_attention){padded_single_prefill_attention::Launch(bq,a.kc,a.vc,bk,bv,ba,base,n,a.cap,l%4!=3,bad.As<int>(),stream.s,packedChunkK,packedChunkV,prefill,contextMaximum);}
          else if(o.packed_bf16_prefill_attention){packed_prefill_attention::Launch(bq,a.kc,a.vc,bk,bv,ba,base,n,a.cap,l%4!=3,bad.As<int>(),stream.s,packedChunkK,packedChunkV);}
          else if(o.gqa8_prefill_attention){g3_prefill_attention::Launch(bq,a.kc,a.vc,bk,bv,ba,base,n,a.cap,l%4!=3,bad.As<int>(),stream.s);}
          else if(o.online_prefill_attention){fast_prefill_attention::Launch(bq,a.kc,a.vc,bk,bv,ba,base,n,a.cap,l%4!=3,bad.As<int>(),stream.s);}
          else if(o.tc_prefill_attention){LaunchAttentionTC(bq,a.kc,a.vc,bk,bv,ba,attentionScores.As<float>(),attentionProbabilities.As<__nv_bfloat16>(),base,n,a.cap,l%4!=3,bad.As<int>(),stream.s);}
          else {
          if(l%4!=3){if(base==0){AttendPrefill<true,true><<<dim3(32,n),128,std::min(base+n,1024)*sizeof(float),stream.s>>>(bq,a.kc,a.vc,ba,base,a.cap,bad.As<int>(),selected?bindings.find("attention_scores")->second.pointer:nullptr,selected?bindings.find("attention_probabilities")->second.pointer:nullptr,bk,bv,base);}else{AttendPrefill<true,false><<<dim3(32,n),128,std::min(base+n,1024)*sizeof(float),stream.s>>>(bq,a.kc,a.vc,ba,base,a.cap,bad.As<int>(),selected?bindings.find("attention_scores")->second.pointer:nullptr,selected?bindings.find("attention_probabilities")->second.pointer:nullptr,bk,bv,base);}}else{if(base==0){AttendPrefill<false,true><<<dim3(32,n),128,(base+n)*sizeof(float),stream.s>>>(bq,a.kc,a.vc,ba,base,a.cap,bad.As<int>(),selected?bindings.find("attention_scores")->second.pointer:nullptr,selected?bindings.find("attention_probabilities")->second.pointer:nullptr,bk,bv,base);}else{AttendPrefill<false,false><<<dim3(32,n),128,(base+n)*sizeof(float),stream.s>>>(bq,a.kc,a.vc,ba,base,a.cap,bad.As<int>(),selected?bindings.find("attention_scores")->second.pointer:nullptr,selected?bindings.find("attention_probabilities")->second.pointer:nullptr,bk,bv,base);}}}
          capture("attention",ba,selected);
          if(checkpoint_anchor && l%4!=3){
            const auto offset=static_cast<std::size_t>(l-l/4)*1024*2048;
            auto* saved=reinterpret_cast<std::uint16_t*>(checkpointBuffer.As<unsigned char>()+offset);
            kv_checkpoint::CaptureLocal<<<kv_checkpoint::Blocks,kv_checkpoint::Threads,0,stream.s>>>(reinterpret_cast<const std::uint16_t*>(a.kc),reinterpret_cast<const std::uint16_t*>(a.vc),reinterpret_cast<const std::uint16_t*>(packedChunkK),reinterpret_cast<const std::uint16_t*>(packedChunkV),saved,saved+1024*512,prefill,n,checkpoint_anchor,bad.As<int>(),contextMaximum);
          }
          if(q8Cache){const int first=l%4!=3?std::max(0,n-a.cap):0;kv_q8::Commit<<<n-first,128,0,stream.s>>>(static_cast<const signed char*>(q8Chunk.k),static_cast<const signed char*>(q8Chunk.v),static_cast<const kv_q8::Scales*>(q8Chunk.scales),static_cast<signed char*>(a.q8.k),static_cast<signed char*>(a.q8.v),static_cast<kv_q8::Scales*>(a.q8.scales),prefill,n,a.cap,first,bad.As<int>(),contextMaximum);}
          else if(o.fp8_kv_cache)kv_fp8::CommitChunk(chunkEK,chunkEV,chunkSF,a.ek,a.ev,a.sf,base,n,a.cap,l%4!=3,bad.As<int>(),stream.s);
          else if(prefill){const int first=l%4!=3?std::max(0,n-a.cap):0;StorePrefillControlled<<<dim3(4,n-first),128,0,stream.s>>>(bk,bv,a.kc,a.vc,prefill,n,a.cap,first,bad.As<int>(),contextMaximum);}
          else CommitPrefillKV(bk,bv,a.kc,a.vc,base,n,a.cap,l%4!=3,bad.As<int>(),stream.s);
          if(selected)CaptureKV<<<4,128,0,stream.s>>>(bindings.find("key_cache")->second.pointer,a.kc,a.vc,base,a.cap,l%4!=3);
          batchProject(a.out,ba,br);capture("o_proj",br,selected);Add<<<(n*2304+127)/128,128,0,stream.s>>>(bx,br,n*2304,bad.As<int>());capture("after_attention",bx,selected);
          // Every layer27 K/V row is already committed above. No later layer
          // consumes these FFN rows; only final request row reaches the head.
          const bool lastRowOnly=o.prune_last_prefill_ffn && l==27;
          if(lastRowOnly && base+n!=static_cast<int>(o.input_ids.size()))continue;
          const int ffnTokens=lastRowOnly?1:n;
          float* ffnX=bx+(lastRowOnly?(n-1)*2304:0);
          if(o.ordinary_rms_norm)fast_norm::Rms2304<false><<<ffnTokens,128,0,stream.s>>>(ffnX,bn,a.post,nullptr,nullptr,bad.As<int>());
          else Norm<<<dim3(1,ffnTokens),128,0,stream.s>>>(ffnX,bn,a.post,2304,1,selected?bindings.find("post_norm_stats")->second.pointer:nullptr);batchProject(a.router,bn,brouter,false,ffnTokens);capture("post_norm",bn,selected);capture("router_logits",brouter,selected);
          RouteParallel<<<ffnTokens,64,0,stream.s>>>(brouter,ids.As<int>(),routing.As<float>(),bad.As<int>());
          if(l==o.diagnostic_exl3_layer)CaptureExl3Fixture<<<(ffnTokens*2304+127)/128,128,0,stream.s>>>(exl3Fixture.As<unsigned char>(),bn,ids.As<int>(),routing.As<float>(),ffnTokens);
          if(o.diagnostic_routes)Check(cudaMemcpyAsync(routeCapture.As<int>()+(l*o.input_ids.size()+base)*8,ids.p,ffnTokens*8*sizeof(int),cudaMemcpyDeviceToDevice,stream.s));
          if(selected)CaptureRouting<<<1,8,0,stream.s>>>(bindings.find("router_ids")->second.pointer,ids.As<int>(),routing.As<float>());
          if(o.compact_experts){fast_expert::Group<<<64,128,0,stream.s>>>(ids.As<int>(),groups,counts,ffnTokens,bad.As<int>());
            if(o.scaled_bf16_prefill && ffnTokens>128)scaled_prefill::DescribeFull64Tail16<<<1,64,0,stream.s>>>(counts,scaledWide,reinterpret_cast<scaled_prefill::Tail*>(expertTiles),scaledWideCount,expertTileCount,ffnTokens,scaled_prefill::WideCapacity(ffnTokens),fast_expert::DescriptorCapacity(ffnTokens),bad.As<int>());
            else fast_expert::Describe<<<1,64,0,stream.s>>>(counts,expertTiles,expertTileCount,ffnTokens,fast_expert::DescriptorCapacity(ffnTokens),bad.As<int>());
          }
          else if(o.native_prefill)GroupExperts<<<64,1,0,stream.s>>>(ids.As<int>(),groups,counts,ffnTokens);
          if(o.exl3_prefill_masked)exl3_masked::Schedule<<<1,1,0,stream.s>>>(counts,maskedTiles,exl3TileCount,ffnTokens,bad.As<int>());else if(o.exl3_prefill_fp16)exl3_fp16::Schedule<<<1,1,0,stream.s>>>(counts,exl3Tiles,exl3TileCount,ffnTokens,bad.As<int>());
          if(o.exl3_packed_activations && ffnTokens>=4){
            for(int part=0;part<3;++part){auto* target=reinterpret_cast<unsigned short*>(part==0?bg:part==1?bu:bd);
#if PADRA_EXL3_PREFILL_ASYNC64
              exl3_transport_async64::Launch(
#else
              exl3_packed_activations::Launch(
#endif
                descriptors.As<DeviceMatrix>(),l,part,ids.As<int>(),groups,counts,maskedTiles,exl3TileCount,bn,target,reinterpret_cast<const unsigned short*>(bg),reinterpret_cast<const unsigned short*>(bu),exl3SiLU.As<unsigned short>(),exl3Workspace.As<__half>(),ffnTokens,bad.As<int>(),stream.s,nullptr,o.exl3_wide_input);}
            exl3_packed_activations::Combine<<<dim3(18,ffnTokens),128,0,stream.s>>>(ffnX,reinterpret_cast<const unsigned short*>(bd),routing.As<float>(),ffnTokens,bad.As<int>(),true);
          }else{
          batchExpert(l,0,bn,bg,selected,ffnTokens);
          if(!o.native_prefill){batchExpert(l,1,bn,bu,false,ffnTokens);capture("expert_gate",bg,selected);capture("expert_up",bu,selected);Activate<<<ffnTokens*56,128,0,stream.s>>>(bg,bu,ffnTokens);}
          capture("expert_activation",bg,selected);batchExpert(l,2,bg,bd,false,ffnTokens);capture("expert_down",bd,selected);
          Combine<<<dim3(18,ffnTokens),128,0,stream.s>>>(ffnX,bd,routing.As<float>(),bad.As<int>());
          }
          capture("output",bx,selected);
          if(o.capture_trace)Trace<<<dim3(19,n),128,0,stream.s>>>(trace.As<float>()+(base*28+l)*2384,bx,brouter,ids.As<int>(),routing.As<float>());
        }
        if(!prefill && base+n==static_cast<int>(o.input_ids.size()))Check(cudaMemcpyAsync(x,bx+(n-1)*2304,2304*4,cudaMemcpyDeviceToDevice,stream.s));
        executed=base+n;
        if(prefill)CompletePrefillChunk<<<1,1,0,stream.s>>>(prefill,n,bad.As<int>(),contextMaximum);
      };
      const auto prefillBody=[&]{for(int base=0;base<static_cast<int>(o.input_ids.size());base+=batch)prefillChunk(base,std::min(batch,static_cast<int>(o.input_ids.size())-base),nullptr);};

      if(o.controlled_chunk_graphs){
        const auto startupShapes=ServerStartupShapes(batch);
        const auto setup=std::chrono::steady_clock::now();
        if(chunkShapes.size()==0){
          for(std::size_t index=0;index<startupShapes.size();++index){
            std::size_t beforeFree=0,afterFree=0;Check(cudaMemGetInfo(&beforeFree,&total));
            const int active=startupShapes[index];chunkGraphs[index].Capture(stream.s,[&]{prefillChunk(0,active,prefillCommand.As<PrefillState>());});
            Check(cudaMemGetInfo(&afterFree,&total));
            ChunkGraphKey key{model.weights_sha256,compiler::Sha256Hex(result.math_profile.data(),result.math_profile.size()),__DATE__ " " __TIME__ "/SM120", q8Cache?"q8_token_head128_f32scale_v1":"bf16",o.capacity,batch,active,1};
            const auto status=chunkShapes.Admit(std::move(key),chunkGraphs[index].nodes,beforeFree>afterFree?beforeFree-afterFree:0);if(!status.ok())throw std::runtime_error(status.message());worker->RecordCapture();
          }
        }
        result.prefill_setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup).count();
        Check(cudaMemGetInfo(&free,&total));const auto postGraphAvailable=ServerAvailableBytes(free,total,o.server_device_budget_gib);worker->RecordGraphAdmission(postGraphAvailable);
        if(postGraphAvailable<700ULL*1024*1024)return Status(StatusCode::kResourceExhausted,"post-graph device admission violates700MiB reserve; available="+std::to_string(postGraphAvailable));
        if(!worker->AwaitGenerate())break;
        if(worker->dropped)continue;
        initialized=prefill_finished=last_token_ready=std::chrono::steady_clock::now();result.initialization_seconds=loadSeconds;
        if(worker->cancel.load()){result.cancelled=true;result.generated_ids.clear();result.prefill_tokens=0;result.inference_seconds=0;worker->Publish(result);continue;}
        for(int base=worker->append_base;base<static_cast<int>(o.input_ids.size());base+=batch){
          const int active=std::min(batch,static_cast<int>(o.input_ids.size())-base);
          PrefillState state{base,base,active,o.capacity,base,1,worker->request_epoch,1};
          Check(cudaMemcpyAsync(prefillCommand.p,&state,sizeof(state),cudaMemcpyHostToDevice,stream.s));
          auto found=std::find(startupShapes.begin(),startupShapes.end(),active);
          const bool capturing=worker->checkpoint_anchor>base && worker->checkpoint_anchor<=base+active;
          if(capturing)worker->BeginCheckpointCapture();
          if(!capturing && found!=startupShapes.end())chunkGraphs[std::distance(startupShapes.begin(),found)].Launch(stream.s);
          else prefillChunk(0,active,prefillCommand.As<PrefillState>(),capturing?worker->checkpoint_anchor:0); // unchanged containing chunk or same-body tail
          Check(cudaMemcpyAsync(&state,prefillCommand.p,sizeof(state),cudaMemcpyDeviceToHost,stream.s));
          Check(cudaMemcpyAsync(host.p,bad.p,sizeof(int),cudaMemcpyDeviceToHost,stream.s));Check(cudaStreamSynchronize(stream.s));
          if(host.p->token || state.committed!=base+active)throw std::runtime_error("invalid/partial controlled prefill commit; session poisoned");
          executed=base+active;worker->CommitChunk(executed,capturing || found==startupShapes.end());
          if(capturing)worker->CompleteCheckpointCapture();
          if(worker->cancel.load()){result.cancelled=true;break;}
          if(executed==static_cast<int>(o.input_ids.size()))Check(cudaMemcpyAsync(x,bx+(active-1)*2304,2304*4,cudaMemcpyDeviceToDevice,stream.s));
        }
        if(result.cancelled){result.generated_ids.clear();result.prefill_tokens=executed;result.decode_steps=0;result.prefill_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-initialized).count();result.inference_seconds=result.prefill_seconds;worker->Publish(result);continue;}
      }else if(o.prefill_graphs){
        const auto setup=std::chrono::steady_clock::now();
        if(!prefillGraph.executable){prefillGraph.Capture(stream.s,prefillBody);if(worker)worker->RecordCapture();}
        preparedLength=o.input_ids.size();
        result.prefill_setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup).count();
        initialized=std::chrono::steady_clock::now();prefill_finished=initialized;last_token_ready=initialized;
        result.initialization_seconds=std::chrono::duration<double>(initialized-begin).count();
        Check(cudaMemGetInfo(&free,&total));result.device_free_after_initialization=free;
        if(worker){
          if(!worker->AwaitGenerate())break;
          if(worker->dropped)continue;
          result.initialization_seconds=loadSeconds;
          initialized=prefill_finished=last_token_ready=std::chrono::steady_clock::now();
          if(worker->cancel.load()){
            result.cancelled=true;result.generated_ids.clear();result.prefill_tokens=0;result.inference_seconds=0;worker->Publish(result);continue;
          }
        }
        prefillGraph.Launch(stream.s);
        executed=static_cast<int>(o.input_ids.size());
      }else prefillBody();
      stopped=finishToken(static_cast<int>(o.input_ids.size())-1);
    }
    for(int position=o.prefill_chunk>1?static_cast<int>(o.input_ids.size()):0;position<steps && !stopped && !result.cancelled;++position){
      if(position<static_cast<int>(o.input_ids.size()))token=o.input_ids[static_cast<std::size_t>(position)];
      if(o.device_decode && ((o.cancellation_requested && o.cancellation_requested->load(std::memory_order_relaxed)) || (o.diagnostic_cancel_after>=0 && generated>=o.diagnostic_cancel_after))){
        CancelDecodeState<<<1,1,0,stream.s>>>(deviceState.As<DecodeState>(),control.As<Control>(),bad.As<int>(),contextMaximum);
        Check(cudaMemcpyAsync(host.p,control.p,sizeof(Control),cudaMemcpyDeviceToHost,stream.s));Check(cudaStreamSynchronize(stream.s));
        if(host.p->bad)throw std::runtime_error("invalid decode state at cancellation boundary");result.cancelled=true;break;
      }
      executed=position+1;
      if(o.decode_graphs){if(worker && o.sampling.selection_graph())samplingGraph.Launch(stream.s);else decodeGraph.Launch(stream.s);}else decodeBody(position);
      if(position+1>=static_cast<int>(o.input_ids.size()))stopped=o.full_decode_graph?observeToken(position):finishToken(position);
    }
    Check(cudaGetLastError());Check(cudaStreamSynchronize(stream.s));
    result.prefill_tokens=static_cast<int>(o.input_ids.size());result.decode_steps=executed-result.prefill_tokens;
    result.prefill_seconds=std::chrono::duration<double>(prefill_finished-initialized).count();
    if(result.decode_steps>0)result.decode_seconds=std::chrono::duration<double>(last_token_ready-prefill_finished).count();
    Check(cudaMemGetInfo(&free,&total));result.device_free_after_execution=free;
    result.inference_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-initialized).count();
    if(o.capture_quantization_rows){if(result.decode_steps!=1)return Status(StatusCode::kUnsupported,"quantization-row capture did not execute its single ordinary decode step");Check(cudaMemcpy(result.quantization_rows.data(),quantCapture.p,quantCapture.bytes,cudaMemcpyDeviceToHost));}
    if(o.capture_logits)Check(cudaMemcpy(result.logits.data(),logits,98304*sizeof(float),cudaMemcpyDeviceToHost));
    if(o.capture_trace){const auto n=static_cast<std::size_t>(executed)*28*2384;Check(cudaMemcpy(result.trace.data(),trace.p,n*sizeof(float),cudaMemcpyDeviceToHost));result.trace.resize(n);}
    if(o.diagnostic_exl3_layer>=0){result.diagnostic_exl3_fixture.resize(exl3Fixture.bytes);Check(cudaMemcpy(result.diagnostic_exl3_fixture.data(),exl3Fixture.p,exl3Fixture.bytes,cudaMemcpyDeviceToHost));}
    if(o.snapshot_layer>=0)Check(cudaMemcpy(result.operator_snapshot.data(),snapshot.p,snapshot.bytes,cudaMemcpyDeviceToHost));
    if(o.diagnostic_routes){result.diagnostic_h128_ranges.resize(28*3*9);Check(cudaMemcpy(result.diagnostic_h128_ranges.data(),rangeCapture.p,rangeCapture.bytes,cudaMemcpyDeviceToHost));result.diagnostic_route_ids.resize(routeBytes/sizeof(int));Check(cudaMemcpy(result.diagnostic_route_ids.data(),routeCapture.p,routeBytes,cudaMemcpyDeviceToHost));}
    result.generated_ids.resize(static_cast<std::size_t>(generated));
    if(!worker)return result;
    worker->Publish(result);
    }while(worker);
    return result;
  }catch(const std::overflow_error& e){return Status(StatusCode::kResourceExhausted,e.what());}catch(const std::exception& e){return Status(StatusCode::kInternal,e.what());}
}
Result<InferenceResult> RunCudaInference(const InferenceOptions& o){return RunCudaInferenceCore(o);}
Result<std::unique_ptr<RuntimeBackend>> LoadCudaModelRuntime(InferenceOptions o){
  auto owner=std::make_unique<CudaModelWorker>(std::move(o));auto* worker=owner.get();
  worker->thread=std::thread([worker]{
    try{auto result=RunCudaInferenceCore(worker->options,worker);if(!result.ok())worker->Fail(result.status());}
    catch(const std::exception& error){worker->Fail({StatusCode::kInternal,error.what()});}
  });
  std::unique_lock lock(worker->mutex);worker->cv.wait(lock,[&]{return worker->loaded || worker->failed;});
  if(worker->failed){auto failure=worker->fault;lock.unlock();return failure;}
  lock.unlock();return std::unique_ptr<RuntimeBackend>(std::move(owner));
}

}
