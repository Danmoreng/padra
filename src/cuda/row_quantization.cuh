// N4c explicit, bounded arithmetic policy. Existing scales remain FP32 amax/448.
// Reciprocal overflow is an error, never a switch back to direct division.
namespace row_quantization {
constexpr std::size_t kCaptureBytes=(4096+2304)*sizeof(float)+4096+2304+2*sizeof(float);
template<bool Reciprocal> __device__ __forceinline__ float Inverse(float scale){
 if constexpr(Reciprocal)return __fdiv_rn(1.f,scale);
 else return 1.f;
}
template<bool Reciprocal> __device__ __forceinline__ unsigned char Encode(float value,float scale,float inverse){
 if constexpr(Reciprocal)return __nv_cvt_float_to_fp8(__fmul_rn(value,inverse),__NV_SATFINITE,__NV_E4M3);
 else return __nv_cvt_float_to_fp8(__fdiv_rn(value,scale),__NV_SATFINITE,__NV_E4M3);
}
} // namespace row_quantization
