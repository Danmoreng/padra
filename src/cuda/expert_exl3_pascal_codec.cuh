#pragma once
namespace exl3_pascal {
// GP104's FP16 FMA is slow. The exact half constants and final RN half cast
// preserve the original codec for every 16-bit state; DP4A only sums hash bytes.
__device__ __forceinline__ float Decode(unsigned state){
  const unsigned hash=state*0x83DCD12DU;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  const unsigned sum=__dp4a(hash,0x01010101U,0U);
#else
  const unsigned sum=(hash&255)+((hash>>8)&255)+((hash>>16)&255)+(hash>>24);
#endif
  const float value=__fmaf_rn(float(1024+sum),__half2float(__ushort_as_half(0x1eee)),__half2float(__ushort_as_half(0xc931)));
  return __half2float(__float2half_rn(value));
}
__device__ __forceinline__ void InitLookup(float* lookup){for(unsigned sum=threadIdx.x;sum<1024;sum+=blockDim.x)lookup[sum]=__half2float(__float2half_rn(__fmaf_rn(float(1024+sum),__half2float(__ushort_as_half(0x1eee)),__half2float(__ushort_as_half(0xc931)))));__syncthreads();}
__device__ __forceinline__ float DecodeLookup(unsigned state,const float* lookup){const unsigned hash=state*0x83DCD12DU;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return lookup[__dp4a(hash,0x01010101U,0U)];
#else
  return lookup[(hash&255)+((hash>>8)&255)+((hash>>16)&255)+(hash>>24)];
#endif
}
// With col=lane/4 and ki=(lane%4)*2, all eight weights owned by this
// lane are in its own compressed word and the preceding word. Reuse that
// window instead of performing two arbitrary shuffles for every weight.
// The two states in each pair overlap by twelve bits.
__device__ __forceinline__ float2 PairLookup(unsigned window,const float* lookup,int rate=4){
  return make_float2(DecodeLookup((window>>rate)&65535U,lookup),DecodeLookup(window&65535U,lookup));
}
__device__ __forceinline__ float2 Pair(unsigned window,int rate=4){
  return make_float2(Decode((window>>rate)&65535U),Decode(window&65535U));
}
__device__ __forceinline__ unsigned Previous(unsigned word){
  return __shfl_sync(0xffffffffU,word,((threadIdx.x&31)+31)&31);
}
}

namespace exl3_pascal {
// A lane owns eight consecutive codes. K3 occupies three logical bytes;
// XOR restores the native stream's byte order without a word overread.
struct Windows {
  unsigned word, previous;
  int rate;
  __device__ __forceinline__ Windows(DeviceMatrix matrix, std::uint64_t tile, unsigned lane, int fixed_rate=0) {
    rate = fixed_rate ? fixed_rate : matrix.format == 6 ? 3 : 4;
    const auto* p = matrix.data + tile * (32 * rate);
    const unsigned i = lane * 3;
    if(rate==3){
      // Native K3 tiles contain 24 aligned words. Join the logical byte
      // stream with two readonly word loads rather than three byte addresses.
      // Lane31 needs no bytes from another word: clamp its unused second load
      // so even the last tile never reads beyond the native payload.
      const auto* words=reinterpret_cast<const unsigned*>(p);
      const unsigned at=i/4;
      const unsigned a=__ldg(words+at),b=__ldg(words+min(at+1,23U));
      word=__funnelshift_l(b,a,(i&3)*8)>>8;
    }else word=reinterpret_cast<const unsigned*>(p)[lane];
    previous = Previous(word);
  }
  __device__ __forceinline__ unsigned Low(int half) const {
    return rate == 3 ? (half == 0 ? (previous << 6) | (word >> 18) :
        (previous << 12) | (word >> 12)) :
        __funnelshift_r(word, previous, half == 0 ? 24 : 16);
  }
  __device__ __forceinline__ unsigned High(int half) const {
    return half ? word : rate == 3 ? (previous << 18) | (word >> 6) :
        __funnelshift_r(word, word, 8);
  }
};
}
