// BF16 fragment packing and FP32 Tensor Core accumulation shared by attention.
#pragma once
__device__ __forceinline__ unsigned AttentionPackBF16(float a, float b) {
  return static_cast<unsigned>(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
         (static_cast<unsigned>(__bfloat16_as_ushort(__float2bfloat16_rn(b))) << 16);
}
__device__ __forceinline__ void AttentionMmaBF16(const unsigned *a, const unsigned *b, float *c) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
