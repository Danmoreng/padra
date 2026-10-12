// Packed U32/eight-element load scheduling follows gem16's NVFP4 GEMV mechanics.
// Mellum BF16 weight dequantization, A16 inputs and compensated math are its own.
// See NOTICE and the pinned read-only mechanical source reference.
__global__ void ExpertVector(const DeviceMatrix* experts,int layer,int part,
    const int* ids,const float* x,float* y,int* bad){
  const unsigned slot=blockIdx.y;const int expert=ids[slot];
  if(expert<0 || expert>=64){if(threadIdx.x==0)atomicExch(bad,1);return;}
  const DeviceMatrix m=experts[(layer*64+expert)*3+part];
  if(m.format!=2 || m.cols<=0 || m.cols%16 || m.rows<=0){if(threadIdx.x==0)atomicExch(bad,1);return;}
  const int lane=threadIdx.x&31,r=blockIdx.x*4+threadIdx.x/32;if(r>=m.rows)return;
  const float* input=part==2?x+slot*896:x;float sum=0,error=0;
  const float divisor=*m.divisor;
  for(int c=lane*8;c<m.cols;c+=256){
    const unsigned word=*reinterpret_cast<const unsigned*>(m.data+static_cast<std::size_t>(r)*m.cols/2+c/2);
    const float scale=DecodeFP8(m.scales[static_cast<std::size_t>(r)*m.cols/16+c/16]);
    const float4 low=*reinterpret_cast<const float4*>(input+c),high=*reinterpret_cast<const float4*>(input+c+4);
    const float activation[8]={low.x,low.y,low.z,low.w,high.x,high.y,high.z,high.w};
    #pragma unroll
    for(int j=0;j<8;++j){
      const float weight=B(__fdiv_rn(__fmul_rn(DecodeFP4((word>>(j*4))&15),scale),divisor));
      Accumulate(__fmul_rn(weight,activation[j]),sum,error);
    }
  }
  sum=WarpAccurateReduce(sum,error);
  if(lane==0){const float value=B(sum);y[slot*m.rows+r]=value;Finite(value,bad);}
}
