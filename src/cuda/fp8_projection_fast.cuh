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
// FP32 FMA mechanism from Gem16 fp8/reference.cu:138-167. Parallelized with
// existing Mellum vector4 packed codec/warp ownership; no compensation.
__device__ __forceinline__ void VectorRow(DeviceMatrix m,const unsigned char* x,
    const float* scales,float* y,int row,int* bad){
  if(row>=m.rows)return;const int lane=threadIdx.x%32;float sum=0;
  for(int c=lane*4;c<m.cols;c+=128){const auto a=DecodeFP8Four(Load32(x+c)),b=DecodeFP8Four(Load32(m.data+row*m.cols+c));
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
  VectorRow(m,x,scales,y,blockIdx.x*4+threadIdx.x/32,bad);
}
__global__ void QKVVector(DeviceMatrix q,DeviceMatrix k,DeviceMatrix v,
    const unsigned char* x,const float* scales,float* oq,float* ok,float* ov,int* bad){
  if(!q.data || !k.data || !v.data || !q.scales || !k.scales || !v.scales || !x || !scales || !oq || !ok || !ov || !bad ||
      q.format!=1 || k.format!=1 || v.format!=1 || q.rows!=4096 || k.rows!=512 || v.rows!=512 ||
      q.cols!=2304 || k.cols!=2304 || v.cols!=2304 || blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 ||
      gridDim.x!=1280 || gridDim.y!=1 || gridDim.z!=1){if(threadIdx.x==0 && bad)atomicExch(bad,1);return;}
  const int block=blockIdx.x;const auto m=block<1024?q:(block<1152?k:v);
  float* out=block<1024?oq:(block<1152?ok:ov);const int local=block<1024?block:(block<1152?block-1024:block-1152);
  VectorRow(m,x,scales,out,local*4+threadIdx.x/32,bad);
}
} // namespace fast_projection
