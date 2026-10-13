#pragma once
namespace fast_projection {
// FP32 FMA mechanism from Gem16 fp8/reference.cu:138-167. Parallelized with
// existing Mellum vector4 packed codec/warp ownership; no compensation.
__device__ __forceinline__ void VectorRow(DeviceMatrix m,const unsigned char* x,
    const float* scales,float* y,int row,int* bad,const float* table=nullptr){
  if(row>=m.rows)return;const int lane=threadIdx.x%32;float sum=0;
  for(int c=lane*4;c<m.cols;c+=128){const unsigned input=Load32(x+c),weight=Load32(m.data+row*m.cols+c);const auto a=table?DecodeFP8Lookup(input,table):DecodeFP8Four(input),b=table?DecodeFP8Lookup(weight,table):DecodeFP8Four(weight);
    sum=fmaf(a.x,b.x,sum);sum=fmaf(a.y,b.y,sum);sum=fmaf(a.z,b.z,sum);sum=fmaf(a.w,b.w,sum);}
  for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
  if(lane==0){const float v=B((sum*scales[0])*BF(m.scales,row));y[row]=v;Finite(v,bad);}
}
__global__ void Vector(DeviceMatrix m,const unsigned char* x,const float* scales,float* y,int* bad){
  if(!m.data || !m.scales || !x || !scales || !y || !bad || m.format!=1 || m.rows<1 || m.rows>4096 ||
      (m.cols!=2304 && m.cols!=4096) || blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 ||
      gridDim.y!=1 || gridDim.z!=1 || gridDim.x!=static_cast<unsigned>((m.rows+3)/4) ||
      (reinterpret_cast<std::uintptr_t>(m.data)&3) || (reinterpret_cast<std::uintptr_t>(x)&3)){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
  __shared__ float table[256];InitFP8Lookup(table);VectorRow(m,x,scales,y,blockIdx.x*4+threadIdx.x/32,bad,table);
#else
  VectorRow(m,x,scales,y,blockIdx.x*4+threadIdx.x/32,bad);
#endif
}
__global__ void QKVVector(DeviceMatrix q,DeviceMatrix k,DeviceMatrix v,
    const unsigned char* x,const float* scales,float* oq,float* ok,float* ov,int* bad){
  if(!q.data || !k.data || !v.data || !q.scales || !k.scales || !v.scales || !x || !scales || !oq || !ok || !ov || !bad ||
      q.format!=1 || k.format!=1 || v.format!=1 || q.rows!=4096 || k.rows!=512 || v.rows!=512 ||
      q.cols!=2304 || k.cols!=2304 || v.cols!=2304 || blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 ||
      gridDim.x!=1280 || gridDim.y!=1 || gridDim.z!=1){if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
  const int block=blockIdx.x;const auto m=block<1024?q:(block<1152?k:v);
  float* out=block<1024?oq:(block<1152?ok:ov);const int local=block<1024?block:(block<1152?block-1024:block-1152);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
  __shared__ float table[256];InitFP8Lookup(table);VectorRow(m,x,scales,out,local*4+threadIdx.x/32,bad,table);
#else
  VectorRow(m,x,scales,out,local*4+threadIdx.x/32,bad);
#endif
}
} // namespace fast_projection
