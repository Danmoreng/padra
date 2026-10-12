// N3b Mellum cache contract: FP32 amax/448, direct RN division, E4M3
// satfinite/RNE, separate token/head K,V scales. No checkpoint scale reuse.
// Packed conversion adapted from Gem16 c01789a2 attention/decode_sm120.cu
// DecodeFp8x4 (Apache-2.0); all cache encoding/chronology is Mellum-specific.
namespace kv_fp8 {
constexpr int kDim=128,kHeads=4,kFeatures=512,kThreads=128;
using Scales=float2; // x=K scale, y=V scale; one pair per token/KV head.
static_assert(sizeof(Scales)==8 && alignof(Scales)==8);
__device__ __forceinline__ float WarpMax(float v){for(int d=16;d;d/=2)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,d));return v;}
__device__ __forceinline__ float Decode(unsigned char code,float scale,int* bad){
 if((code&127)==127 || !(scale>0.f) || !isfinite(scale)){atomicExch(bad,1);return NAN;}
 __nv_fp8_e4m3 q;q.__x=code;const float x=__fmul_rn(static_cast<float>(q),scale);
 if(!isfinite(x))atomicExch(bad,1);return __bfloat162float(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float4 Decode4(unsigned bits,float scale,int* bad){
 bool invalid=!(scale>0.f)||!isfinite(scale);
 #pragma unroll
 for(int i=0;i<4;++i)invalid|=((bits>>(8*i))&127)==127;
 if(invalid){atomicExch(bad,1);return make_float4(NAN,NAN,NAN,NAN);}
 __nv_fp8x4_e4m3 q;q.__x=bits;float4 x=static_cast<float4>(q);
 x.x=__fmul_rn(x.x,scale);x.y=__fmul_rn(x.y,scale);x.z=__fmul_rn(x.z,scale);x.w=__fmul_rn(x.w,scale);
 if(!isfinite(x.x)||!isfinite(x.y)||!isfinite(x.z)||!isfinite(x.w))atomicExch(bad,1);
 return x;
}
template<bool Controlled> __launch_bounds__(kThreads)
__global__ void Encode(const float* k,const float* v,unsigned char* ck,unsigned char* cv,
 Scales* scales,int rows,int cap,const DecodeState* state,int* bad){
 if(!k||!v||!ck||!cv||!scales||!bad||rows<1||rows>2048||cap<rows||cap>8192||
 blockDim.x!=kThreads||blockDim.y!=1||blockDim.z!=1||gridDim.x!=unsigned(rows)||gridDim.y!=1||gridDim.z!=1||
 (Controlled && (rows!=1 || !DecodeReady(state) ||
 (cap!=state->capacity && cap!=min(1024,state->capacity))))){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 const int row=blockIdx.x,head=threadIdx.x/32,lane=threadIdx.x%32,feature=head*kDim+lane*4;
 const float4 x=reinterpret_cast<const float4*>(k+row*kFeatures)[feature/4],y=reinterpret_cast<const float4*>(v+row*kFeatures)[feature/4];
 const float a[4]={x.x,x.y,x.z,x.w},b[4]={y.x,y.y,y.z,y.w};bool invalid=false;float km=0,vm=0;
 #pragma unroll
 for(int i=0;i<4;++i){invalid|=!isfinite(a[i])||!isfinite(b[i])||(__float_as_uint(a[i])&65535)||(__float_as_uint(b[i])&65535);km=fmaxf(km,fabsf(a[i]));vm=fmaxf(vm,fabsf(b[i]));}
 if(__any_sync(0xffffffff,invalid)){if(lane==0)atomicExch(bad,1);return;}
 km=WarpMax(km);vm=WarpMax(vm);const float ks=km>0?__fdiv_rn(km,448.f):1.f,vs=vm>0?__fdiv_rn(vm,448.f):1.f;
 if(!(ks>0)||!(vs>0)||!isfinite(ks)||!isfinite(vs)){if(lane==0)atomicExch(bad,1);return;}
 const int target=Controlled?state->position%cap:row;
 if(lane==0)scales[target*kHeads+head]=make_float2(ks,vs);
 unsigned kb=0,vb=0;
 #pragma unroll
 for(int i=0;i<4;++i){kb|=unsigned(__nv_cvt_float_to_fp8(__fdiv_rn(a[i],ks),__NV_SATFINITE,__NV_E4M3))<<(8*i);vb|=unsigned(__nv_cvt_float_to_fp8(__fdiv_rn(b[i],vs),__NV_SATFINITE,__NV_E4M3))<<(8*i);}
 *reinterpret_cast<unsigned*>(ck+target*kFeatures+feature)=kb;
 *reinterpret_cast<unsigned*>(cv+target*kFeatures+feature)=vb;
}
__global__ void Commit(const unsigned char* k,const unsigned char* v,const Scales* scales,
 unsigned char* kc,unsigned char* vc,Scales* cached,int base,int tokens,int cap,int first,int* bad){
 if(!k||!v||!scales||!kc||!vc||!cached||!bad||base<0||tokens<1||tokens>2048||base>8192-tokens||
 cap<1||cap>8192||first<0||first>=tokens||tokens-first>cap||blockDim.x!=128||blockDim.y!=1||blockDim.z!=1||gridDim.x!=unsigned(tokens-first)||gridDim.y!=1||gridDim.z!=1){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 const int row=first+blockIdx.x,target=(base+row)%cap,d=threadIdx.x*4;
 *reinterpret_cast<unsigned*>(kc+target*kFeatures+d)=*reinterpret_cast<const unsigned*>(k+row*kFeatures+d);
 *reinterpret_cast<unsigned*>(vc+target*kFeatures+d)=*reinterpret_cast<const unsigned*>(v+row*kFeatures+d);
 if(threadIdx.x<kHeads)cached[target*kHeads+threadIdx.x]=scales[row*kHeads+threadIdx.x];
}
void CommitChunk(const unsigned char* k,const unsigned char* v,const Scales* scales,
 unsigned char* kc,unsigned char* vc,Scales* cached,int base,int tokens,int cap,bool local,int* bad,cudaStream_t stream){
 if(!k||!v||!scales||!kc||!vc||!cached||!bad||base<0||tokens<1||tokens>2048||base>8192-tokens||cap<1||cap>8192||
 (local?(cap>1024||cap<std::min(1024,base+tokens)):cap<base+tokens))throw std::runtime_error("FP8 KV suffix commit shape/binding unsupported");
 const int first=local?std::max(0,tokens-cap):0;
 Commit<<<tokens-first,128,0,stream>>>(k,v,scales,kc,vc,cached,base,tokens,cap,first,bad);
}
} // namespace kv_fp8
