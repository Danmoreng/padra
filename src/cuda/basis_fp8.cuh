// Exact E2M1 -> E4M3 basis for existing Gem16-derived row-linear W4A8 kernels.
// New byte-permutation adaptation; checkpoint, scale math and negative zero preserved.
__device__ __forceinline__ unsigned BasisFP8(const std::uint8_t* p){
  const unsigned packed=static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1])<<8);
  const unsigned magnitude=__byte_perm(0x3c383000u,0x4c484440u,packed & 0x7777u);
  const unsigned sign=__byte_perm(0u,0x80808080u,(packed & 0x8888u)>>1);
  return magnitude | sign;
}
