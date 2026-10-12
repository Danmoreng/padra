// Adapted from Danmoreng/gem16, Apache-2.0. See NOTICE.
namespace gem_reuse {
constexpr unsigned kWarpSize=32,kWarpsPerBlock=4;
constexpr std::uint64_t kRowsPerWarp=8,kElementsPerKBlock=64,kTokensPerMma=16,
    kGroupedRowTilesPerWarp=2,kGroupedTokenTiles=2,kGroupedAssignmentsPerTile=32;
__device__ unsigned LoadU32(const std::uint8_t* p){return Load32(p);}
template<bool compensated>
__device__ void MmaReused(float& d0,float& d1,float& d2,float& d3,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,unsigned sa,unsigned sb,float* error){
  unsigned a[4]={a0,a1,a2,a3},b[2]={b0,b1};float sum[4]={d0,d1,d2,d3};
  if constexpr(compensated){
    // E2M1 block-16 integer dots times two represented E4M3 scales are exact
    // FP32 partials. Mask other scale bytes, then compensate the partial sum.
    #pragma unroll
    for(int block=0;block<4;++block){float part[4]={};MmaFP4(a,b,sa & (0xffu<<(block*8)),sb,part);
      #pragma unroll
      for(int j=0;j<4;++j)Accumulate(part[j],sum[j],error[j]);}
  }else MmaFP4(a,b,sa,sb,sum);
  d0=sum[0];d1=sum[1];d2=sum[2];d3=sum[3];
}
template <unsigned kBytes>
__device__ __forceinline__ void CopyAsyncZeroFill(
    void* shared_destination, const void* global_source, int source_bytes) {
  static_assert(kBytes == 4U || kBytes == 16U);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
               :
               : "r"(static_cast<unsigned>(__cvta_generic_to_shared(shared_destination))), "l"(global_source),
                 "n"(kBytes), "r"(source_bytes));
#else
  (void)shared_destination;
  (void)global_source;
  (void)source_bytes;
#endif
}

__device__ __forceinline__ void CommitAsyncCopies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void WaitForAsyncCopies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("cp.async.wait_group 0;\n");
#endif
}

template<bool kA8=false>
__device__ __forceinline__ void StageGroupedActivationAsync(
    uint4 (*staged_activation)[kA8?4:2],
    std::uint32_t* staged_activation_scales,
    const std::uint32_t* staged_activation_rows,
    const std::uint32_t* staged_activation_valid,
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* activation_scales_e4m3fn,
    std::uint64_t packed_row_bytes, std::uint64_t scale_row_bytes,
    std::uint64_t k_block) {
  if (threadIdx.x < (kA8?64U:32U)) {
    const unsigned activation_index = threadIdx.x / (kA8?4U:2U);
    const unsigned half = threadIdx.x % (kA8?4U:2U);
    const bool valid = staged_activation_valid[activation_index] != 0U;
    const std::uint8_t* source = valid
        ? packed_activation_e2m1 +
              static_cast<std::uint64_t>(
                  staged_activation_rows[activation_index]) *
                  packed_row_bytes +
              k_block * (kA8?64U:32U) + static_cast<std::uint64_t>(half) * 16U
        : packed_activation_e2m1;
    // Grouped launch validation guarantees a 16-byte base; K64 makes the
    // packed row stride and every block/half offset 16-byte aligned.
    CopyAsyncZeroFill<16U>(&staged_activation[activation_index][half],
                           source, valid ? 16 : 0);
  }
  if (!kA8 && threadIdx.x < 16U) {
    const bool valid = staged_activation_valid[threadIdx.x] != 0U;
    const std::uint8_t* source = valid
        ? activation_scales_e4m3fn +
              static_cast<std::uint64_t>(
                  staged_activation_rows[threadIdx.x]) *
                  scale_row_bytes +
              k_block * 4U
        : activation_scales_e4m3fn;
    CopyAsyncZeroFill<4U>(&staged_activation_scales[threadIdx.x], source,
                          valid ? 4 : 0);
  }
}

// W4A8 retains the serialized NVFP4 weights. Convert four E2M1 values to
// exactly represented E4M3 basis values in registers; apply each original
// block-16 weight scale after FP8 MMA, then compensate across blocks.
__device__ unsigned BasisFP8Scalar(const std::uint8_t* p){
  const unsigned packed=static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1])<<8);unsigned word=0;
  #pragma unroll
  for(int j=0;j<4;++j){const unsigned code=(packed>>(j*4))&15,mag=code&7;
    const unsigned byte=(mag==0?0:mag==1?48:48+4*mag) | ((code&8)<<4);word|=byte<<(j*8);}
  return word;
}
#include "basis_fp8.cuh"
// Preserve the product rounding remainder before the compensated cross-block sum.
// Otherwise a correctly rounded K16 dot can still cross a BF16 midpoint after
// row rescaling, despite compensation of the rounded products themselves.
__device__ __forceinline__ void AccumulateScaledFP8RN(float part,float scale,float& sum,float& error){
  const float product=__fmul_rn(part,scale);
  const float remainder=__fmaf_rn(part,scale,-product);
  Accumulate(product,sum,error);
  Accumulate(remainder,sum,error);
}

template <bool kFusedGateUp,bool kCompensated=false,bool kA8=false,bool kSelectedM1=false>
__global__ void ExpertGrouped(
    const DeviceMatrix* bindings,int layer,const int* groups,const int* counts,
    const std::uint8_t* packed_activation_e2m1,const std::uint8_t* activation_scales_e4m3fn,
    float* output,int tokens,int* bad,const float* activationDivisors=nullptr,float* capturedGate=nullptr,float* capturedUp=nullptr) {
  // Selected M1 needs only its first M16 tile. Grouped prefill keeps M32.
  // K partials, product remainders, rescaling and BF16/SiLU stay unchanged.
  constexpr unsigned kTokenTiles=kSelectedM1?1:kGroupedTokenTiles;
  constexpr unsigned kAssignments=kTokenTiles*kTokensPerMma;
  if constexpr(kSelectedM1){
    if(tokens!=1 || gridDim.y!=1 || gridDim.z!=8){
      if(threadIdx.x==0)atomicExch(bad,1);return;
    }
  }
  const unsigned expert=kSelectedM1?static_cast<unsigned>(groups[blockIdx.z]):blockIdx.z;
  if constexpr(kSelectedM1){if(expert>=64){if(threadIdx.x==0)atomicExch(bad,1);return;}}
  const unsigned grouped_base=blockIdx.y*32,grouped_end=kSelectedM1?1U:counts[expert];
  if(grouped_base>=grouped_end)return;
  const auto m=bindings[(layer*64+expert)*3+(kFusedGateUp?0:2)];
  const auto u=bindings[(layer*64+expert)*3+1];
  const auto rows=m.rows,contracting_elements=m.cols;
  const auto* packed_weight_e2m1=m.data;const auto* weight_scales_e4m3fn=m.scales;
  const float output_divisor=*m.divisor;
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned group = lane >> 2U;
  const unsigned thread_in_group = lane & 3U;
  const std::uint64_t logical_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::uint64_t row_tiles = rows / kRowsPerWarp;
  const std::uint64_t first_row_tile =
      logical_warp * kGroupedRowTilesPerWarp;
  const bool active_warp = first_row_tile < row_tiles;

  const std::uint64_t k_blocks =
      contracting_elements / kElementsPerKBlock;
  const std::uint64_t weight_row_bytes = contracting_elements / 2U;
  const std::uint64_t packed_row_bytes = kA8?contracting_elements:weight_row_bytes;
  const std::uint64_t scale_row_bytes = contracting_elements / 16U;
  const std::uint64_t first_weight_row=first_row_tile*kRowsPerWarp;

  float accumulator[kTokenTiles][kGroupedRowTilesPerWarp][4] = {};
  float up_accumulator[kTokenTiles][kGroupedRowTilesPerWarp][4] = {};
  float error[kTokenTiles][kGroupedRowTilesPerWarp][4]={},up_error[kTokenTiles][kGroupedRowTilesPerWarp][4]={};

  __shared__ alignas(16)
      uint4 staged_activation[2][kTokenTiles][16][kA8?4:2];
  __shared__ std::uint32_t
      staged_activation_scales[2][kTokenTiles][16];
  __shared__ std::uint32_t
      staged_activation_rows[kAssignments];
  __shared__ std::uint32_t
      staged_activation_valid[kAssignments];
  if (threadIdx.x < kAssignments) {
    const std::uint32_t grouped = grouped_base + threadIdx.x;
    const bool valid=grouped<grouped_end;
    const auto slot=valid?(kSelectedM1?blockIdx.z:groups[expert*tokens+grouped]):0;
    const std::uint32_t activation_row=kFusedGateUp?slot/8:slot;
    staged_activation_rows[threadIdx.x] = activation_row;
    staged_activation_valid[threadIdx.x] = valid ? 1U : 0U;
  }
  __syncthreads();

#pragma unroll
  for (std::uint64_t token_tile = 0U;
       token_tile < kTokenTiles; ++token_tile) {
    StageGroupedActivationAsync<kA8>(
        staged_activation[0][token_tile],
        staged_activation_scales[0][token_tile],
        staged_activation_rows + token_tile * kTokensPerMma,
        staged_activation_valid + token_tile * kTokensPerMma,
        packed_activation_e2m1, activation_scales_e4m3fn,
        packed_row_bytes, scale_row_bytes, 0U);
  }
  CommitAsyncCopies();
  WaitForAsyncCopies();
  __syncthreads();

  for (std::uint64_t k_block = 0; k_block < k_blocks; ++k_block) {
    const unsigned stage = static_cast<unsigned>(k_block & 1U);
    if (k_block + 1U < k_blocks) {
      const unsigned next_stage = stage ^ 1U;
#pragma unroll
      for (std::uint64_t token_tile = 0U;
           token_tile < kTokenTiles; ++token_tile) {
        StageGroupedActivationAsync<kA8>(
            staged_activation[next_stage][token_tile],
            staged_activation_scales[next_stage][token_tile],
            staged_activation_rows + token_tile * kTokensPerMma,
            staged_activation_valid + token_tile * kTokensPerMma,
            packed_activation_e2m1, activation_scales_e4m3fn,
            packed_row_bytes, scale_row_bytes, k_block + 1U);
      }
      CommitAsyncCopies();
    }

#pragma unroll
    for (std::uint64_t row_tile = 0U;
         row_tile < kGroupedRowTilesPerWarp; ++row_tile) {
      if (!active_warp || first_row_tile + row_tile >= row_tiles) continue;
      const std::uint64_t tile_first_weight_row =
          first_weight_row + row_tile * kRowsPerWarp;
      if constexpr(kA8){
        #pragma unroll
        for(unsigned quarter=0;quarter<4;++quarter){
          const auto offset=(tile_first_weight_row+group)*weight_row_bytes+k_block*32+quarter*8+thread_in_group*2;
          unsigned b[2]={};b[quarter&1]=BasisFP8(m.data+offset);
          unsigned ub[2]={};if constexpr(kFusedGateUp)ub[quarter&1]=BasisFP8(u.data+offset);
          #pragma unroll
          for(unsigned token_tile=0;token_tile<kTokenTiles;++token_tile){
            const auto* low=reinterpret_cast<const unsigned*>(staged_activation[stage][token_tile][group]);
            const auto* high=reinterpret_cast<const unsigned*>(staged_activation[stage][token_tile][group+8]);
            const unsigned index=thread_in_group+(quarter/2)*8;
            unsigned a[4]={low[index],high[index],low[index+4],high[index+4]};
            a[(quarter&1)?0:2]=0;a[(quarter&1)?1:3]=0;
            float part[4]={},upart[4]={};MmaFP8(a,b,part);if constexpr(kFusedGateUp)MmaFP8(a,ub,upart);
            #pragma unroll
            for(unsigned j=0;j<4;++j){const auto col=tile_first_weight_row+thread_in_group*2+(j&1);
              const auto scale=col*scale_row_bytes+k_block*4+quarter;
              AccumulateScaledFP8RN(part[j],DecodeFP8(m.scales[scale]),accumulator[token_tile][row_tile][j],error[token_tile][row_tile][j]);
              if constexpr(kFusedGateUp)AccumulateScaledFP8RN(upart[j],DecodeFP8(u.scales[scale]),up_accumulator[token_tile][row_tile][j],up_error[token_tile][row_tile][j]);
            }
          }
        }
      }else{
      const auto weight_offset=(tile_first_weight_row+group)*weight_row_bytes+k_block*32U+thread_in_group*4U;
      const auto scale_offset=(tile_first_weight_row+group)*scale_row_bytes+k_block*4U;
      const std::uint32_t b0 = LoadU32(packed_weight_e2m1 + weight_offset);
      const std::uint32_t b1 =
          LoadU32(packed_weight_e2m1 + weight_offset + 16U);
      const std::uint32_t scale_b =
          LoadU32(weight_scales_e4m3fn + scale_offset);
      if constexpr (kFusedGateUp) {
        const auto up_b0=LoadU32(u.data+weight_offset),up_b1=LoadU32(u.data+weight_offset+16U);
        const auto up_scale_b=LoadU32(u.scales+scale_offset);
#pragma unroll
        for (std::uint64_t token_tile = 0U;
             token_tile < kTokenTiles; ++token_tile) {
          const auto* low_words = reinterpret_cast<const std::uint32_t*>(
              staged_activation[stage][token_tile][group]);
          const auto* high_words = reinterpret_cast<const std::uint32_t*>(
              staged_activation[stage][token_tile][group + 8U]);
          const std::uint32_t a0 = low_words[thread_in_group];
          const std::uint32_t a1 = high_words[thread_in_group];
          const std::uint32_t a2 = low_words[thread_in_group + 4U];
          const std::uint32_t a3 = high_words[thread_in_group + 4U];
          const std::uint32_t scale_a = thread_in_group < 2U
              ? staged_activation_scales[stage][token_tile]
                                                [group + thread_in_group * 8U]
              : 0U;
          MmaReused<kCompensated>(accumulator[token_tile][row_tile][0],
                    accumulator[token_tile][row_tile][1],
                    accumulator[token_tile][row_tile][2],
                    accumulator[token_tile][row_tile][3], a0, a1, a2, a3,
                    b0, b1, scale_a, scale_b,error[token_tile][row_tile]);
          MmaReused<kCompensated>(up_accumulator[token_tile][row_tile][0],
                    up_accumulator[token_tile][row_tile][1],
                    up_accumulator[token_tile][row_tile][2],
                    up_accumulator[token_tile][row_tile][3], a0, a1, a2, a3,
                    up_b0, up_b1, scale_a, up_scale_b,up_error[token_tile][row_tile]);
        }
      } else {
#pragma unroll
        for (std::uint64_t token_tile = 0U;
             token_tile < kTokenTiles; ++token_tile) {
          const auto* low_words = reinterpret_cast<const std::uint32_t*>(
              staged_activation[stage][token_tile][group]);
          const auto* high_words = reinterpret_cast<const std::uint32_t*>(
              staged_activation[stage][token_tile][group + 8U]);
          const std::uint32_t a0 = low_words[thread_in_group];
          const std::uint32_t a1 = high_words[thread_in_group];
          const std::uint32_t a2 = low_words[thread_in_group + 4U];
          const std::uint32_t a3 = high_words[thread_in_group + 4U];
          const std::uint32_t scale_a = thread_in_group < 2U
              ? staged_activation_scales[stage][token_tile]
                                                [group + thread_in_group * 8U]
              : 0U;
          MmaReused<kCompensated>(accumulator[token_tile][row_tile][0],
                    accumulator[token_tile][row_tile][1],
                    accumulator[token_tile][row_tile][2],
                    accumulator[token_tile][row_tile][3], a0, a1, a2, a3,
                    b0, b1, scale_a, scale_b,error[token_tile][row_tile]);
        }
      }
      }
    }
    if (k_block + 1U < k_blocks) {
      WaitForAsyncCopies();
      __syncthreads();
    }
  }

#pragma unroll
  for (std::uint64_t token_tile = 0U;
       token_tile < kTokenTiles; ++token_tile) {
    const std::uint32_t grouped_low =
        grouped_base + token_tile * kTokensPerMma + group;
    const std::uint32_t grouped_high = grouped_low + 8U;
#pragma unroll
    for (std::uint64_t row_tile = 0U;
         row_tile < kGroupedRowTilesPerWarp; ++row_tile) {
      const std::uint64_t output_column =
          (first_row_tile + row_tile) * kRowsPerWarp + thread_in_group * 2U;
#pragma unroll
      for (unsigned pair = 0; pair < 4U; ++pair) {
        const bool high = (pair & 2U) != 0U;
        const std::uint32_t grouped = high ? grouped_high : grouped_low;
        const std::uint64_t column = output_column + (pair & 1U);
        if (grouped >= grouped_end || column >= rows) continue;
        const auto output_row=kSelectedM1?blockIdx.z:groups[expert*tokens+grouped];
        const float activationDivisor=activationDivisors?activationDivisors[kFusedGateUp?output_row/8:output_row]:1.f;
        const auto rescale=[&](float sum,float err,float divisor){const float total=__fadd_rn(sum,err);
          if constexpr(kA8)return B(__fdiv_rn(__fmul_rn(total,activationDivisor),divisor));
          else return B(__fdiv_rn(total,__fmul_rn(divisor,activationDivisor)));};
        if constexpr (kFusedGateUp) {
          const float gate = rescale(accumulator[token_tile][row_tile][pair],error[token_tile][row_tile][pair],output_divisor);
          const float up = rescale(up_accumulator[token_tile][row_tile][pair],up_error[token_tile][row_tile][pair],*u.divisor);
          if(capturedGate)capturedGate[static_cast<std::uint64_t>(output_row)*rows+column]=gate;
          if(capturedUp)capturedUp[static_cast<std::uint64_t>(output_row)*rows+column]=up;
          // Mellum SiLU, with the explicit rounded-exponential BF16 stages.
          const float value=B(B(__fdiv_rn(gate,__fadd_rn(1.f,ExpRN(-gate))))*up);
          output[static_cast<std::uint64_t>(output_row)*rows+column]=value;Finite(value,bad);
        } else {
          const float value=rescale(accumulator[token_tile][row_tile][pair],error[token_tile][row_tile][pair],output_divisor);
          output[static_cast<std::uint64_t>(output_row)*rows+column]=value;Finite(value,bad);
        }
      }
    }
  }
}
}
