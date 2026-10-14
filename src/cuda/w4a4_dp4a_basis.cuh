// Exact doubled E2M1 basis in temporary signed bytes; not an INT8 model profile.
// Keep represented NVFP4 weights/activations, E4M3 scales and compensated F32.
__device__ __forceinline__ unsigned W4A4SignedBasis4(unsigned packed) {
  const unsigned magnitude = __byte_perm(0x03020100u, 0x0c080604u, packed & 0x7777u);
  const unsigned sign = __byte_perm(0u, 0xffffffffu, (packed & 0x8888u) >> 1);
  const unsigned negative = __vsub4(0u, magnitude);
  return (magnitude & ~sign) | (negative & sign);
}
__device__ __forceinline__ float W4A4Dot16(const unsigned char *w, const unsigned char *a) {
  const unsigned weight0 = LoadU32(w), weight1 = LoadU32(w + 4);
  const unsigned activation0 = LoadU32(a), activation1 = LoadU32(a + 4);
  int dot = 0;
  dot = __dp4a(static_cast<int>(W4A4SignedBasis4(weight0)),
               static_cast<int>(W4A4SignedBasis4(activation0)), dot);
  dot = __dp4a(static_cast<int>(W4A4SignedBasis4(weight0 >> 16)),
               static_cast<int>(W4A4SignedBasis4(activation0 >> 16)), dot);
  dot = __dp4a(static_cast<int>(W4A4SignedBasis4(weight1)),
               static_cast<int>(W4A4SignedBasis4(activation1)), dot);
  dot = __dp4a(static_cast<int>(W4A4SignedBasis4(weight1 >> 16)),
               static_cast<int>(W4A4SignedBasis4(activation1 >> 16)), dot);
  // |dot| <=16*12*12=2304, conversion and quarter scaling are exact in F32.
  return __fmul_rn(static_cast<float>(dot), .25f);
}
