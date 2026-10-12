// Mechanical packed loads / one-row-per-warp scheduling from read-only Gem16 GEMV.
// Mellum W4A4 math: exact K16 E2M1 dot, E4M3 scale product, compensated FP32,
// RN global-divisor product/division and final BF16. No A16 dequantization.
namespace gem_reuse {
__device__ __forceinline__ float W4A4Dot16Float(const unsigned char* w,const unsigned char* a){
  float dot=0;
  #pragma unroll
  for(unsigned j=0;j<4;++j){
    const auto weight=DecodeFP8Four(BasisFP8(w+j*2));
    const auto activation=DecodeFP8Four(BasisFP8(a+j*2));
    dot=__fadd_rn(dot,__fmul_rn(weight.x,activation.x));
    dot=__fadd_rn(dot,__fmul_rn(weight.y,activation.y));
    dot=__fadd_rn(dot,__fmul_rn(weight.z,activation.z));
    dot=__fadd_rn(dot,__fmul_rn(weight.w,activation.w));
  }
  return dot;
}
#include "w4a4_dp4a_basis.cuh"
template<bool GateUp>
__global__ void ExpertW4A4SIMT(const DeviceMatrix* bindings,int layer,const int* ids,
    const unsigned char* activation,const unsigned char* scales,const float* activationDivisors,
    float* output,float* up,int* bad){
  constexpr unsigned rows=GateUp?896:2304,cols=GateUp?2304:896;
  constexpr unsigned totalRows=GateUp?rows*2:rows;
  if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.x!=(totalRows+7)/8 ||
      gridDim.y!=8 || gridDim.z!=1 || layer<0 || layer>=28 || !bindings || !ids ||
      !activation || !scales || !activationDivisors || !output || (GateUp && !up) ||
      (reinterpret_cast<std::uintptr_t>(activation)&3)){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  const unsigned expert=static_cast<unsigned>(ids[blockIdx.y]);
  if(expert>=64){if(threadIdx.x==0)atomicExch(bad,1);return;}
  const unsigned logicalRow=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
  if(logicalRow>=totalRows)return;
  const bool isUp=GateUp && logicalRow>=rows;
  const unsigned row=logicalRow-(isUp?rows:0),part=GateUp?(isUp?1:0):2;
  const auto m=bindings[(layer*64+expert)*3+part];
  if(m.rows!=rows || m.cols!=cols || m.format!=2 || !m.data || !m.scales || !m.divisor ||
      (reinterpret_cast<std::uintptr_t>(m.data)&3)){
    if(lane==0)atomicExch(bad,1);return;
  }
  const unsigned inputRow=GateUp?0:blockIdx.y;
  const float divisor=__fmul_rn(*m.divisor,activationDivisors[inputRow]);
  if(!isfinite(*m.divisor) || *m.divisor<=0 || !isfinite(activationDivisors[inputRow]) ||
      activationDivisors[inputRow]<=0 || !isfinite(divisor) || divisor<=0){
    if(lane==0)atomicExch(bad,1);return;
  }
  float sum=0,error=0;
  const auto* a=activation+inputRow*(cols/2);
  const auto* sa=scales+inputRow*(cols/16);
  const auto* w=m.data+row*(cols/2);
  const auto* sw=m.scales+row*(cols/16);
  for(unsigned block=lane;block<cols/16;block+=32){
    const float dot=W4A4Dot16(w+block*8,a+block*8);
    const float scale=__fmul_rn(DecodeFP8(sw[block]),DecodeFP8(sa[block]));
    Accumulate(__fmul_rn(dot,scale),sum,error);
  }
  sum=WarpAccurateReduce(sum,error);
  if(lane==0){const float value=B(__fdiv_rn(sum,divisor));
    (isUp?up:output)[blockIdx.y*rows+row]=value;Finite(value,bad);}
}
}
