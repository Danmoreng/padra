// Selected expert row/packed-load mechanism from pinned Gem16 NVFP4 GEMV,
// adapted to A8 codes and lossless undivided BF16 weight basis; FP32 FMA compute.
namespace fast_expert {
template<bool Fused>
__global__ void SelectedVector(const DeviceMatrix* bindings,int layer,const int* ids,
 const unsigned char* activation,const float* rowScales,float* output,int* bad){
 constexpr int rows=Fused?896:2304,cols=Fused?2304:896;
 if(!bindings||!ids||!activation||!rowScales||!output||!bad||layer<0||layer>=28||blockDim.x!=128||blockDim.y!=1||blockDim.z!=1||gridDim.x!=rows/4||gridDim.y!=8||gridDim.z!=1||(reinterpret_cast<std::uintptr_t>(activation)&3)){if(threadIdx.x==0&&bad)atomicExch(bad,1);return;}
 const int slot=blockIdx.y,expert=ids[slot];if(expert<0||expert>=64){if(threadIdx.x==0)atomicExch(bad,1);return;}
 const DeviceMatrix m=bindings[(layer*64+expert)*3+(Fused?0:2)],u=bindings[(layer*64+expert)*3+1];
 if(m.format!=2||m.rows!=rows||m.cols!=cols||!m.data||!m.scales||!m.divisor||(Fused&&(u.format!=2||u.rows!=rows||u.cols!=cols||!u.data||!u.scales||!u.divisor))){if(threadIdx.x==0)atomicExch(bad,1);return;}
 const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;const unsigned char* x=activation+(Fused?0:slot*cols);float sum=0,up=0;
 for(int c=lane*4;c<cols;c+=128){const auto a=DecodeFP8Four(Load32(x+c));const auto w=DecodeFP8Four(gem_reuse::BasisFP8(m.data+row*(cols/2)+c/2));const float sf=DecodeFP8(m.scales[row*(cols/16)+c/16]);
  sum=fmaf(a.x,__fmul_rn(w.x,sf),sum);sum=fmaf(a.y,__fmul_rn(w.y,sf),sum);sum=fmaf(a.z,__fmul_rn(w.z,sf),sum);sum=fmaf(a.w,__fmul_rn(w.w,sf),sum);
  if constexpr(Fused){const auto v=DecodeFP8Four(gem_reuse::BasisFP8(u.data+row*(cols/2)+c/2));const float uf=DecodeFP8(u.scales[row*(cols/16)+c/16]);up=fmaf(a.x,__fmul_rn(v.x,uf),up);up=fmaf(a.y,__fmul_rn(v.y,uf),up);up=fmaf(a.z,__fmul_rn(v.z,uf),up);up=fmaf(a.w,__fmul_rn(v.w,uf),up);}}
 for(int d=16;d;d/=2){sum+=__shfl_down_sync(0xffffffff,sum,d);if constexpr(Fused)up+=__shfl_down_sync(0xffffffff,up,d);}
 if(lane==0){const float sf=rowScales[Fused?0:slot];float value=B((sum*sf)/(*m.divisor));if constexpr(Fused){const float v=B((up*sf)/(*u.divisor));value=B(B(value/(1.f+expf(-value)))*v);}output[slot*rows+row]=value;Finite(value,bad);}
}
} // namespace fast_expert
