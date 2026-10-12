// Compact Q/K/V CTA-range selection adapted from read-only gem16 FP8 sm120.cu.
// Revision/hash/license and Mellum arithmetic adaptations: NOTICE.
// Existing Mellum compensated row / exponent-binned tile bodies are reused.
template<bool Prefill>
__global__ void ProjectQKVGrouped(DeviceMatrix query,DeviceMatrix key,DeviceMatrix value,
                                 const unsigned char* x,const float* scales,
                                 float* q,float* k,float* v,int tokens,int* bad){
  constexpr unsigned qtiles=Prefill?128:1024,kvtiles=Prefill?16:128;
  if(query.format!=1 || key.format!=1 || value.format!=1 ||
     query.cols!=2304 || key.cols!=2304 || value.cols!=2304 ||
     query.rows!=4096 || key.rows!=512 || value.rows!=512 ||
     !query.data || !query.scales || !key.data || !key.scales || !value.data || !value.scales ||
     !x || !scales || !q || !k || !v || !bad || tokens<=0 || tokens>(Prefill?512:1) ||
     blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 || blockIdx.x>=qtiles+2*kvtiles ||
     blockIdx.y>=static_cast<unsigned>(Prefill?(tokens+15)/16:1) || blockIdx.z!=0){
    if(threadIdx.x==0 && bad)atomicExch(bad,1);return;
  }
  const unsigned block=blockIdx.x;
  const bool is_query=block<qtiles,is_key=!is_query && block<qtiles+kvtiles;
  const DeviceMatrix matrix=is_query?query:(is_key?key:value);
  float* output=is_query?q:(is_key?k:v);
  const unsigned local_block=is_query?block:(is_key?block-qtiles:block-qtiles-kvtiles);
  if constexpr(Prefill)
    ProjectFP8PartialTile<true>(matrix,x,scales,output,tokens,bad,
                               blockIdx.y*16,(local_block*4+threadIdx.x/32)*8);
  else
    ProjectFP8VectorRow(matrix,x,scales,output,bad,0,local_block*4+threadIdx.x/32);
}
