// Selection mechanics adapted from Gem16 sampling.cu, Apache-2.0.
// Donor pins/adaptations: NOTICE. CUB headers precede this file.
#pragma once
#include <climits>
namespace gpu_sampling {
constexpr int Threads=256,Vocabulary=98304;
// Only evolving RNG state is device-resident. Fixed options are kernel values
// stored by CUDA in graph nodes, never fetched from a device settings pointer.
struct State {std::uint64_t seed,step;};
static_assert(sizeof(State)==16);
inline std::size_t WorkspaceBytes(int n=Vocabulary){
 std::size_t sort=0,scan=0;
 Check(cub::DeviceRadixSort::SortPairsDescending(nullptr,sort,static_cast<float*>(nullptr),static_cast<float*>(nullptr),static_cast<int*>(nullptr),static_cast<int*>(nullptr),n));
 Check(cub::DeviceScan::InclusiveSum(nullptr,scan,static_cast<double*>(nullptr),static_cast<double*>(nullptr),n));return std::max(sort,scan);
}
__device__ std::uint64_t Mix(std::uint64_t v){v+=0x9E3779B97F4A7C15ULL;v=(v^(v>>30))*0xBF58476D1CE4E5B9ULL;v=(v^(v>>27))*0x94D049BB133111EBULL;return v^(v>>31);}
__global__ void CountPrompt(const int* prompt,int tokens,int* counts,int n,int* bad){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tokens)return;int id=prompt[i];if(id<0 || id>=n){atomicExch(bad,1);return;}atomicAdd(counts+id,1);
}
__global__ void RecordChoice(const Control* chosen,int* counts,int n,int* bad){
 if(blockIdx.x || threadIdx.x)return;if(*bad || chosen->bad)return;int id=chosen->token;if(id<0 || id>=n || counts[id]==INT_MAX){atomicExch(bad,1);return;}++counts[id];
}
template<bool Penalties> __global__ void Prepare(const float* logits,float* adjusted,int* ids,const int* counts,int n,int* bad,float frequency,float presence,float repetition){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float v=logits[i];
 if constexpr(Penalties){int count=counts[i];if(count<0){atomicExch(bad,1);count=0;}if(count){v=v<0?__fmul_rn(v,repetition):__fdiv_rn(v,repetition);v=__fsub_rn(v,__fmul_rn(frequency,static_cast<float>(count)));v=__fsub_rn(v,presence);}}
 if(!isfinite(v)){atomicExch(bad,1);v=-FLT_MAX;}adjusted[i]=v==0?0:v;ids[i]=i;
}
__global__ void Probabilities(const float* sorted,double* probabilities,int n,int* bad,float temperature,int top_k){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const int limit=top_k?min(n,top_k):n;
 if(!isfinite(temperature) || temperature<=0 || temperature>2 || top_k<0 || top_k>n){atomicExch(bad,1);probabilities[i]=0;return;}
 probabilities[i]=i<limit?static_cast<double>(expf((sorted[i]-sorted[0])/temperature)):0.;
}
__global__ void Choose(const double* cumulative,const int* sortedIds,State* state,Control* chosen,int n,int* bad,int top_k,float top_p){
 if(threadIdx.x || blockIdx.x)return;int eligible=top_k?min(n,top_k):n;
 if(*bad || eligible<1 || !isfinite(top_p) || top_p<=0 || top_p>1 || !isfinite(cumulative[eligible-1]) || cumulative[eligible-1]<=0 || state->step==UINT64_MAX){atomicExch(bad,1);*chosen={0,1};return;}
 if(top_p<1){double threshold=static_cast<double>(top_p)*cumulative[eligible-1];int low=0,high=eligible;while(low<high){int mid=low+(high-low)/2;if(cumulative[mid]<threshold)low=mid+1;else high=mid;}eligible=min(low+1,eligible);}
 const auto bits=Mix(state->seed^Mix(state->step));const double u=fmin((static_cast<double>(bits>>11)+0.5)*(1.0/9007199254740992.0),0x1.fffffffffffffp-1);
 const double target=u*cumulative[eligible-1];int low=0,high=eligible;while(low<high){int mid=low+(high-low)/2;if(cumulative[mid]<=target)low=mid+1;else high=mid;}
 chosen->token=sortedIds[min(low,eligible-1)];chosen->bad=*bad;++state->step;
}
inline void PrepareLaunch(const float* logits,float* adjusted,int* ids,const int* counts,int n,int* bad,SamplingOptions options,cudaStream_t stream){
 if(options.penalties())Prepare<true><<<(n+Threads-1)/Threads,Threads,0,stream>>>(logits,adjusted,ids,counts,n,bad,options.frequency_penalty,options.presence_penalty,options.repetition_penalty);
 else Prepare<false><<<(n+Threads-1)/Threads,Threads,0,stream>>>(logits,adjusted,ids,counts,n,bad,0,0,1);
}
inline void Launch(const float* logits,float* adjusted,float* sorted,int* ids,int* sortedIds,double* cumulative,State* state,Control* chosen,int* bad,void* temp,std::size_t bytes,cudaStream_t stream,SamplingOptions options,const int* counts,int n=Vocabulary){
 PrepareLaunch(logits,adjusted,ids,counts,n,bad,options,stream);
 Check(cub::DeviceRadixSort::SortPairsDescending(temp,bytes,adjusted,sorted,ids,sortedIds,n,0,32,stream));
 Probabilities<<<(n+Threads-1)/Threads,Threads,0,stream>>>(sorted,cumulative,n,bad,options.temperature,options.top_k);
 Check(cub::DeviceScan::InclusiveSum(temp,bytes,cumulative,cumulative,n,stream));
 Choose<<<1,1,0,stream>>>(cumulative,sortedIds,state,chosen,n,bad,options.top_k,options.top_p);Check(cudaGetLastError());
}
}
