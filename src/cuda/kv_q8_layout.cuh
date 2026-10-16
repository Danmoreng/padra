#pragma once
namespace q8_native {
// One token:16 FP16 scales followed by512 codes; each head starts on a
// 32-byte boundary. Logical32-element block coefficients are unchanged.
struct alignas(16) CacheRow {
  __half d[16];
  signed char qs[512];
};
struct QueryBlock {
  float d;
  int qs[8];
};
static_assert(sizeof(CacheRow) == 544 && alignof(CacheRow) == 16);
static_assert(sizeof(QueryBlock) == 36 && alignof(QueryBlock) == 4);
} // namespace q8_native
