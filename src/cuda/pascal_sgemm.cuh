#pragma once
// FP8 is a storage format on Pascal. Decode one matrix into bounded, reusable
// scratch and let cuBLAS compute a real batch in FP32. Never used for Decode.
namespace pascal_sgemm {
inline void CheckBlas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Pascal cuBLAS: "+std::to_string(int(status)));}
struct Handle {
  cublasHandle_t value=nullptr;
  Handle(cudaStream_t stream,void* workspace,std::size_t bytes){
    CheckBlas(cublasCreate(&value));
    try {CheckBlas(cublasSetStream(value,stream));CheckBlas(cublasSetMathMode(value,CUBLAS_DEFAULT_MATH));CheckBlas(cublasSetWorkspace(value,workspace,bytes));}
    catch(...){cublasDestroy(value);throw;}
  }
  ~Handle(){if(value)cublasDestroy(value);}
  Handle(const Handle&)=delete;Handle& operator=(const Handle&)=delete;
};
__global__ void Weights(DeviceMatrix matrix,float* decoded){
  const int i=blockIdx.x*256+threadIdx.x;if(i<matrix.rows*matrix.cols)decoded[i]=DecodeFP8Pascal(matrix.data[i]);
}
__global__ void Inputs(const unsigned char* input,float* decoded,int cols,const SimtBatch* batch){
  const int i=blockIdx.x*256+threadIdx.x;
  if(i<kSimtBatch*cols)decoded[i]=i<batch->rows*cols?DecodeFP8Pascal(input[i]):0.f;
}
__global__ void Scale(DeviceMatrix matrix,float* output,const float* scales,int* bad,const SimtBatch* batch){
  const int i=blockIdx.x*256+threadIdx.x;if(i>=batch->rows*matrix.rows)return;
  const float value=B(__fmul_rn(__fmul_rn(output[i],scales[i/matrix.rows]),BF(matrix.scales,i%matrix.rows)));
  output[i]=value;Finite(value,bad);
}
inline void Projection(Handle& handle,DeviceMatrix matrix,const unsigned char* input,const float* scales,float* output,float* weights,float* values,int* bad,const SimtBatch* batch,cudaStream_t stream){
  Weights<<<(matrix.rows*matrix.cols+255)/256,256,0,stream>>>(matrix,weights);
  Inputs<<<kSimtBatch*matrix.cols/256,256,0,stream>>>(input,values,matrix.cols,batch);
  const float alpha=1,beta=0;
  // Physical M stays fixed during graph replay; inactive input rows are zero.
  CheckBlas(cublasSgemm(handle.value,CUBLAS_OP_T,CUBLAS_OP_N,matrix.rows,kSimtBatch,matrix.cols,&alpha,weights,matrix.cols,values,matrix.cols,&beta,output,matrix.rows));
  Scale<<<kSimtBatch*matrix.rows/256,256,0,stream>>>(matrix,output,scales,bad,batch);
}
}
