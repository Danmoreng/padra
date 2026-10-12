// NVFP4 x Q8 activation experiment, adapted from llama.cpp (MIT).
// vecdotq.cuh::vec_dot_nvfp4_q8_1 and mmvq.cu::mul_mat_vec_q_moe,
// pin10a60cf303566e10d6a7a2774c17d2085503d87b. See NOTICE.
// Native adjacent low/high nibbles and separate K16 SF/divisor are retained.
// This header uses the same local DeviceMatrix/B/DecodeFP8/Finite ABI as the
// existing expert kernels. It is initially compiled only by the isolated probe.
namespace expert_q8 {
struct alignas(4) Block {__half2 ds; signed char q[32];};
static_assert(sizeof(Block)==36 && alignof(Block)==4);
// Exact signed 2*E2M1 in four contiguous int8 lanes. Unlike the donor's qs,
// two native bytes encode four adjacent values, not two separated subblocks.
// The paired positive/negative tables avoid byte carries at signed zero.
__device__ __forceinline__ int Basis2(const unsigned char* p){
 const unsigned nib=unsigned(p[0])|(unsigned(p[1])<<8);
 const unsigned pos=__byte_perm(0x03020100u,0x0c080604u,nib&0x7777u);
 const unsigned neg=__byte_perm(0xfdfeff00u,0xf4f8fafcu,nib&0x7777u);
 const unsigned mask=__byte_perm(0u,0xffffffffu,(nib&0x8888u)>>1);
 return static_cast<int>((pos&~mask)|(neg&mask));
}
// Donor q8_1: q=round-away(x/(amax/127)) using unrounded FP32 d,
// stored d rounds to FP16. ds.high is explicitly unused/zero (not a sum).
// Zero block uses d=0,q=0. FP16 underflow is preserved; overflow rejects via
// sticky bad, with no alternate scale precision. Source is already BF16.
__global__ void Quantize(const float* input,Block* encoded,int rows,int cols,int* bad){
 if(!input||!encoded||!bad||rows<1||rows>16384||cols<32||cols>2304||cols%32 ||
    blockDim.x!=128||blockDim.y!=1||blockDim.z!=1||gridDim.x!=unsigned((cols+127)/128)||gridDim.y!=unsigned(rows)||gridDim.z!=1){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 const int row=blockIdx.y,c=blockIdx.x*128+threadIdx.x;if(c>=cols)return;
 float x=input[row*cols+c];const bool invalid=!isfinite(x)||(__float_as_uint(x)&0xffffu)!=0;
 if(invalid){atomicExch(bad,1);x=0;}
 float amax=fabsf(x);
 for(int d=16;d;d/=2)amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d));
 const float scale=__fdiv_rn(amax,127.f);const __half stored=__float2half_rn(scale);
 if(!isfinite(__half2float(stored)))atomicExch(bad,1);
 const int value=amax==0?0:__float2int_rz(roundf(__fdiv_rn(x,scale)));
 Block& out=encoded[row*(cols/32)+c/32];out.q[c%32]=static_cast<signed char>(max(-127,min(127,value)));
 if((c%32)==0)out.ds=__halves2half2(stored,__float2half_rn(0.f));
}
// Per-K16 dot is exact integer arithmetic (|dot|<=16*12*127).
// Half of the E4M3 weight scale compensates doubled basis exactly once;
// one Q8 scale covers two independent K16 weight scales. Divisor is applied
// once after the row sum. No reinterpretation as donor block_nvfp4.
__device__ __forceinline__ float Dot16(DeviceMatrix m,int row,int c,const Block* activation){
 const unsigned char* w=m.data+row*(m.cols/2)+c/2;
 const Block& a=activation[c/32];const int* q=reinterpret_cast<const int*>(a.q+c%32);
 int dot=0;
 #pragma unroll
 for(int i=0;i<4;++i)dot=__dp4a(Basis2(w+i*2),q[i],dot);
 const float sf=DecodeFP8(m.scales[row*(m.cols/16)+c/16]);
 const float d=__fmul_rn(__fmul_rn(sf,.5f),__low2float(a.ds));
 return __fmul_rn(d,static_cast<float>(dot));
}
// Same donor geometry: two rows per warp, one selected expert per grid.y.
// Two lanes per K64 block; each owns two K16 groups. Gate and up reuse Q8.
template<bool Fused>
__global__ void Selected(const DeviceMatrix* bindings,int layer,const int* ids,
 const Block* activation,float* output,int* bad){
 constexpr int rows=Fused?896:2304,cols=Fused?2304:896;
 if(!bindings||!ids||!activation||!output||!bad||layer<0||layer>=28||blockDim.x!=32||blockDim.y!=1||blockDim.z!=1||gridDim.x!=rows/2||gridDim.y!=8||gridDim.z!=1){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 const int slot=blockIdx.y,expert=ids[slot],lane=threadIdx.x;
 if(expert<0||expert>=64){if(lane==0)atomicExch(bad,1);return;}
 const DeviceMatrix m=bindings[(layer*64+expert)*3+(Fused?0:2)],u=bindings[(layer*64+expert)*3+1];
 if(m.format!=2||m.rows!=rows||m.cols!=cols||!m.data||!m.scales||!m.divisor||!isfinite(*m.divisor)||*m.divisor<=0 ||
    (Fused&&(u.format!=2||u.rows!=rows||u.cols!=cols||!u.data||!u.scales||!u.divisor||!isfinite(*u.divisor)||*u.divisor<=0))){if(lane==0)atomicExch(bad,1);return;}
 const Block* x=activation+(Fused?0:slot*(cols/32));float sums[2]={},ups[2]={};
 for(int kb=lane/2;kb<cols/64;kb+=16){const int c=kb*64+(lane%2)*32;
  #pragma unroll
  for(int r=0;r<2;++r){const int row=blockIdx.x*2+r;
   sums[r]=__fadd_rn(sums[r],__fadd_rn(Dot16(m,row,c,x),Dot16(m,row,c+16,x)));
   if constexpr(Fused)ups[r]=__fadd_rn(ups[r],__fadd_rn(Dot16(u,row,c,x),Dot16(u,row,c+16,x)));}}
 #pragma unroll
 for(int r=0;r<2;++r){for(int d=16;d;d/=2){sums[r]=__fadd_rn(sums[r],__shfl_down_sync(0xffffffff,sums[r],d));if constexpr(Fused)ups[r]=__fadd_rn(ups[r],__shfl_down_sync(0xffffffff,ups[r],d));}
  if(lane==0){float value=B(__fdiv_rn(sums[r],*m.divisor));
   if constexpr(Fused){const float up=B(__fdiv_rn(ups[r],*u.divisor));value=B(B(value/(1.f+expf(-value)))*up);}
   output[slot*rows+blockIdx.x*2+r]=value;Finite(value,bad);}}
}
} // namespace expert_q8
