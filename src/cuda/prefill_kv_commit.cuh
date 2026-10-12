// Whole-chunk attention reads old cache/current K,V separately. Commit only
// after attention completes. A local chunk wider than its ring stores its last
// cap rows exactly once, so different CTAs never compete for one cache slot.
void CommitPrefillKV(const float* k,const float* v,__nv_bfloat16* kc,__nv_bfloat16* vc,
    int base,int tokens,int cap,bool local,int* bad,cudaStream_t stream){
 if(!k || !v || !kc || !vc || !bad || base<0 || tokens<1 || tokens>kW4A8FMaxPrefillChunk ||
     base>kW4A8FContextCapacity-tokens || cap<1 || cap>kW4A8FContextCapacity ||
     (local?(cap>1024 || cap<std::min(1024,base+tokens)):cap<base+tokens))
  throw std::runtime_error("prefill KV commit binding/shape unsupported");
 const int first=local?std::max(0,tokens-cap):0;
 Store<<<dim3(4,tokens-first),128,0,stream>>>(k+first*512,v+first*512,kc,vc,base+first,cap);
}
