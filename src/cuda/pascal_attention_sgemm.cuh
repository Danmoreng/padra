#pragma once
namespace pascal_attention_sgemm {
// One KV head supplies eight query heads. Reuse the projection's FP32 scratch;
// keep the old ring and current chunk separate until Commit has completed.
__global__ void Gather(const __nv_bfloat16* keys,const __nv_bfloat16* values,const __nv_bfloat16* current,float* decoded,int count,int kvhead,bool local,const SimtBatch* batch){
  const int i=blockIdx.x*256+threadIdx.x;if(i>=count*128)return;
  const int first=local?max(0,batch->base-1023):0,absolute=first+i/128,d=i%128;
  float k=0,v=0;
  if(absolute<batch->base+batch->rows){
    if(absolute>=batch->base){k=__bfloat162float(current[(absolute-batch->base)*512+kvhead*128+d]);v=__bfloat162float(current[(kSimtBatch+absolute-batch->base)*512+kvhead*128+d]);}
    else {const int at=(local?(absolute&1023):absolute)*512+kvhead*128+d;k=__bfloat162float(keys[at]);v=__bfloat162float(values[at]);}
  }
  decoded[i]=k;decoded[count*128+i]=v;
}
__global__ void Queries(const float* queries,float* packed,int kvhead,int first,int rows,const SimtBatch* batch){
  const int i=blockIdx.x*256+threadIdx.x;if(i>=8*rows*128)return;
  const int head=i/(rows*128),row=first+(i/128)%rows,d=i%128;
  packed[i]=row<batch->rows?queries[row*4096+(kvhead*8+head)*128+d]:0.f;
}
__global__ void Softmax(float* scores,int count,bool local,int first_query,int rows,const SimtBatch* batch){
  const int row=first_query+blockIdx.x,head=blockIdx.y;float* values=scores+(head*rows+blockIdx.x)*count;
  if(row>=batch->rows){for(int k=threadIdx.x;k<count;k+=256)values[k]=0.f;return;}
  const int first=local?max(0,batch->base-1023):0,pos=batch->base+row,start=local?max(0,pos-1023):0;
  __shared__ float scratch[16],total;float maximum=-INFINITY;
  for(int k=threadIdx.x;k<count;k+=256){const int absolute=first+k;const float score=absolute>=start && absolute<=pos?B(B(values[k])*.08838834764831845f):-INFINITY;values[k]=score;maximum=fmaxf(maximum,score);}
  maximum=pascal_attention::Reduce(maximum,scratch,true);float sum=0.f,error=0.f;
  for(int k=threadIdx.x;k<count;k+=256){const float p=expf(values[k]-maximum);values[k]=p;Accumulate(p,sum,error);}
  sum=AttentionAccurateReduce8(sum,error,scratch);if(!threadIdx.x)total=sum;__syncthreads();
  for(int k=threadIdx.x;k<count;k+=256)values[k]=B(values[k]/total);
}
__global__ void Scatter(const float* packed,float* output,int kvhead,int first,int rows,int* bad,const SimtBatch* batch){
  const int i=blockIdx.x*256+threadIdx.x;if(i>=8*rows*128)return;
  const int head=i/(rows*128),row=first+(i/128)%rows,d=i%128;if(row>=batch->rows)return;
  const float value=B(packed[i]);output[row*4096+(kvhead*8+head)*128+d]=value;Finite(value,bad);
}
inline void Prefill(pascal_sgemm::Handle& handle,const float* q,const __nv_bfloat16* k,const __nv_bfloat16* v,const __nv_bfloat16* current,float* out,const SimtBatch* batch,int capacity,bool local,float* weights,float* queries,float* scores,int* bad,cudaStream_t stream,int maximum_keys=0){
  const int count=local?1280:(maximum_keys?min(capacity,maximum_keys):capacity),rows=capacity>8192?64:kSimtBatch;const float alpha=1.f,beta=0.f;
  float* result=queries+8*rows*128;
  for(int kvhead=0;kvhead<4;++kvhead){
    Gather<<<count*128/256,256,0,stream>>>(k,v,current,weights,count,kvhead,local,batch);
    for(int first=0;first<kSimtBatch;first+=rows){
      Queries<<<8*rows*128/256,256,0,stream>>>(q,queries,kvhead,first,rows,batch);
      pascal_sgemm::CheckBlas(cublasSgemmStridedBatched(handle.value,CUBLAS_OP_T,CUBLAS_OP_N,count,rows,128,&alpha,weights,128,0,queries,128,rows*128,&beta,scores,count,rows*count,8));
      Softmax<<<dim3(rows,8),256,0,stream>>>(scores,count,local,first,rows,batch);
      pascal_sgemm::CheckBlas(cublasSgemmStridedBatched(handle.value,CUBLAS_OP_N,CUBLAS_OP_N,128,rows,count,&alpha,weights+count*128,128,0,scores,count,rows*count,&beta,result,128,rows*128,8));
      Scatter<<<8*rows*128/256,256,0,stream>>>(result,out,kvhead,first,rows,bad,batch);
    }
  }
}
}
