// Ordinary register FP8 MMA from Gem16 c01789a2, fp8/sm120.cu:172-273,
// Apache-2.0. Mellum rounded-output variant; Q4096/KV512/H2304/O2304x4096.
namespace fast_projection {
__device__ __forceinline__ void RowTile(DeviceMatrix m,const unsigned char* x,
    const float* scales,float* y,int tile,int* bad){
  const int lane=threadIdx.x%32,group=lane/4,quarter=lane%4,row=tile*8+group;
  float acc[4]={};
  for(int k=0;k<m.cols;k+=32){
    const unsigned a0=Load32(x+k+quarter*4),a1=Load32(x+k+16+quarter*4);
    const unsigned b0=row<m.rows?Load32(m.data+row*m.cols+k+quarter*4):0;
    const unsigned b1=row<m.rows?Load32(m.data+row*m.cols+k+16+quarter*4):0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%4,%5,%5}, {%6,%7}, {%0,%1,%2,%3};\n"
      : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
      : "r"(a0),"r"(a1),"r"(b0),"r"(b1));
  }
  if(lane<4){const int r=tile*8+lane*2;
    if(r<m.rows){const float value=B((acc[0]*scales[0])*BF(m.scales,r));y[r]=value;Finite(value,bad);}
    if(r+1<m.rows){const float value=B((acc[1]*scales[0])*BF(m.scales,r+1));y[r+1]=value;Finite(value,bad);}
  }
}
__global__ void Direct(DeviceMatrix m,const unsigned char* x,const float* scales,float* y,int* bad){
  if(!m.data || !m.scales || !x || !scales || !y || !bad || m.format!=1 ||
      m.rows<1 || m.rows>4096 || (m.cols!=2304 && m.cols!=4096) ||
      blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 || gridDim.y!=1 || gridDim.z!=1 ||
      gridDim.x!=static_cast<unsigned>((m.rows+31)/32) || (reinterpret_cast<std::uintptr_t>(m.data)&3) || (reinterpret_cast<std::uintptr_t>(x)&3)){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
  const int tile=blockIdx.x*4+threadIdx.x/32;if(tile*8<m.rows)RowTile(m,x,scales,y,tile,bad);
}
__global__ void QKV(DeviceMatrix q,DeviceMatrix k,DeviceMatrix v,
    const unsigned char* x,const float* scales,float* oq,float* ok,float* ov,int* bad){
  if(!q.data || !k.data || !v.data || !q.scales || !k.scales || !v.scales || !x || !scales || !oq || !ok || !ov || !bad ||
      q.format!=1 || k.format!=1 || v.format!=1 || q.rows!=4096 || k.rows!=512 || v.rows!=512 ||
      q.cols!=2304 || k.cols!=2304 || v.cols!=2304 || blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 ||
      gridDim.x!=160 || gridDim.y!=1 || gridDim.z!=1){if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
  // Flatten128 Q +16 K +16 V CTAs; no Q-sized empty K/V planes.
  const int block=blockIdx.x;const DeviceMatrix m=block<128?q:(block<144?k:v);
  float* out=block<128?oq:(block<144?ok:ov);const int local=block<128?block:(block<144?block-128:block-144);
  RowTile(m,x,scales,out,local*4+threadIdx.x/32,bad);
}
} // namespace fast_projection
#include "fp8_projection_simt.cuh"
