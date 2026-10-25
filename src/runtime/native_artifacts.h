#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace mellum16 {
struct NativeArtifactIdentity {
  std::string_view manifest_sha256, weights_sha256;
  std::uint64_t weights_bytes;
  int expert_rate;
  bool q8_head = false;
};
inline constexpr std::array<NativeArtifactIdentity, 6> kNativeArtifacts{{
    {"0f30d4ead6ae3d0683ec1eb9ad7af1adfebe343d804e3313e8c01d6c31f76f0b",
     "e614375f2eb26c17321365d4bd8ee7aa4b716f6badee171bb2040b85b754804c", 7092933120ULL, 4},
    {"2a19089d1d50eb7f9dfec63ebcd3e066591db34bcada038cf5072a7602056f57",
     "dfea2118feb666d2ee150ea91699cbb200b39edab95562111a4d5d6e8e402549", 9867465216ULL, 6},
    {"6aa0adfb5ba3b137ac547a15cba5d8c80a0020737c26c4be7dcc695fbf887271",
     "f931bae3cc6a3e275da0cda8a3fca5f2c3a01121a9ba878642cea936ce0ac7bd", 5705667072ULL, 3},
    {"7b8f11faebcb9137f751e93ccb393c6919a34d6193e80794745777efb495f197",
     "5e9eb70fb97fc0915c3691314d7dce154e501d45b404d39309b3383d472530b1", 9655128576ULL, 6, true},
    {"966172e06e2ee42576bd4cf08c0b94f97e00ef4e35fff182640bb9cd5fbb3a28",
     "69ff11ae4c0a4e15349092e9e9a3cc28369f5840112f520afb2518da94dacb82", 6880596480ULL, 4, true},
    {"9032f99eab3b9adf536f722a3721185ee32daa07f96254e9449ef7c69e5c507d",
     "bfa5157b1bcf8919399023f9f943129044835844cc8354cb58ec9dada65299a9", 5493330432ULL, 3, true},
}};
inline constexpr const NativeArtifactIdentity *FindNativeArtifact(std::string_view manifest) {
  for (const auto &identity : kNativeArtifacts)
    if (identity.manifest_sha256 == manifest)
      return &identity;
  return nullptr;
}
} // namespace mellum16
