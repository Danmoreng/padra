// Exact transient E4M3 fixed-basis planes for Mellum M1 gate/up only.
// Existing Gem16 row-warp/packed-load scheduling; no INT8 model/weight profile.
namespace gem_reuse {
__device__ __forceinline__ void W4A8StorePlanes(unsigned code,unsigned char* p,int index,int cols,int* bad){
  if((code&127)==127){if(bad)atomicExch(bad,1);code=0;}
  const int value=W4A8FixedE4M3(code),sign=value<0?-1:1;
  const unsigned mag=static_cast<unsigned>(value<0?-value:value);
  p[index]=static_cast<unsigned char>(sign*static_cast<int>(mag&127));
  p[cols+index]=static_cast<unsigned char>(sign*static_cast<int>((mag>>7)&127));
  p[2*cols+index]=static_cast<unsigned char>(sign*static_cast<int>(mag>>14));
}
__device__ __forceinline__ float W4A8PlanesDot16(const unsigned char* weight,const unsigned char* planes,int cols){
  int d0=0,d1=0,d2=0;
  #pragma unroll
  for(unsigned group=0;group<4;++group){
    const unsigned w=W4A4SignedBasis4(static_cast<unsigned>(weight[group*2])|(static_cast<unsigned>(weight[group*2+1])<<8));
    d0=__dp4a(static_cast<int>(w),static_cast<int>(LoadU32(planes+group*4)),d0);
    d1=__dp4a(static_cast<int>(w),static_cast<int>(LoadU32(planes+cols+group*4)),d1);
    d2=__dp4a(static_cast<int>(w),static_cast<int>(LoadU32(planes+2*cols+group*4)),d2);
  }
  // Signed multiplies, no undefined negative shifts. Exact fixed basis /1024;
  // conservative plane bound <47.2M and actual dot bound44040192, int32 safe.
  return __fmul_rn(__int2float_rn(d0+d1*128+d2*16384),1.f/1024.f);
}
__global__ void ExpertW4A8Planes(const DeviceMatrix* bindings,int layer,const int* ids,
    const unsigned char* activation,const float* activationScales,float* gate,float* up,int* bad){
  constexpr unsigned rows=896,cols=2304;
  if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.x!=224 ||
      gridDim.y!=8 || gridDim.z!=1 || layer<0 || layer>=28 || !bindings || !ids ||
      !activation || !activationScales || !gate || !up || (reinterpret_cast<std::uintptr_t>(activation)&3)){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  const unsigned expert=static_cast<unsigned>(ids[blockIdx.y]);
  if(expert>=64){if(threadIdx.x==0)atomicExch(bad,1);return;}
  const unsigned logicalRow=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
  const bool isUp=logicalRow>=rows;const unsigned row=logicalRow-(isUp?rows:0);
  const auto m=bindings[(layer*64+expert)*3+(isUp?1:0)];
  if(m.rows!=rows || m.cols!=cols || m.format!=2 || !m.data || !m.scales || !m.divisor ||
      (reinterpret_cast<std::uintptr_t>(m.data)&3)){
    if(lane==0)atomicExch(bad,1);return;
  }
  const float divisor=*m.divisor,activationScale=activationScales[0];
  if(!isfinite(divisor) || divisor<=0 || !isfinite(activationScale) || activationScale<=0){
    if(lane==0)atomicExch(bad,1);return;
  }
  float sum=0,error=0;const auto* w=m.data+row*(cols/2);const auto* sf=m.scales+row*(cols/16);
  for(unsigned block=lane;block<cols/16;block+=32){
    const float dot=W4A8PlanesDot16(w+block*8,activation+block*16,cols);
    AccumulateScaledFP8RN(dot,DecodeFP8(sf[block]),sum,error);
  }
  sum=WarpAccurateReduce(sum,error);
  if(lane==0){const float value=B(__fdiv_rn(__fmul_rn(sum,activationScale),divisor));
    (isUp?up:gate)[blockIdx.y*rows+row]=value;Finite(value,bad);}
}
}

// Same amax partition/scale/RN E4M3 codes as QuantizeFP8; planes written once.
__global__ void QuantizeFP8Planes(const float* x,unsigned char* packed,float* scales,
    unsigned char* planes,int* bad){
  if(blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 || gridDim.x!=1 || gridDim.y!=1 || gridDim.z!=1 || !x || !packed || !scales || !planes ||
      (reinterpret_cast<std::uintptr_t>(planes)&3)){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  constexpr int cols=2304;__shared__ float scratch[4];__shared__ float scale;float amax=0;
  for(int c=threadIdx.x;c<cols;c+=128){const float value=x[c];Finite(value,bad);amax=fmaxf(amax,fabsf(value));}
  amax=MaxReduce(amax,scratch);if(threadIdx.x==0){scale=amax==0?1.f:__fdiv_rn(amax,448.f);scales[0]=scale;}__syncthreads();
  for(int c=threadIdx.x;c<cols;c+=128){
    const unsigned code=__nv_cvt_float_to_fp8(__fdiv_rn(x[c],scale),__NV_SATFINITE,__NV_E4M3);
    packed[c]=static_cast<unsigned char>(code);gem_reuse::W4A8StorePlanes(code,planes,c,cols,bad);
  }
}
void AdmitW4A8PlaneLease(const unsigned char* pointer,std::size_t bytes){
  if(!pointer || (reinterpret_cast<std::uintptr_t>(pointer)&15) || bytes<3*2304)
    throw std::runtime_error("W4A8 exact activation plane workspace admission failure");
}
