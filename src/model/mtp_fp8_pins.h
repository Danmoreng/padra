#pragma once
#include <cstdint>
// Separate immutable expert-only W8A16 profile, produced by quantize_mtp_experts.py.
namespace mellum16::internal {
inline constexpr char kMtpFp8Profile[] = "mellum21-mtp-fp8-experts-bf16-f32-v1";
inline constexpr char kMtpFp8Conversion[] =
    "e4m3fn-rne-f32div-ceilbf16-maxabs448-zero1-positivezero-bf16decode-v1";
inline constexpr char kMtpFp8WeightsSha256[] =
    "2b6b785cea1296bdf5a45abb89d7306208288d761951c16a6d7ace0ab1252d30";
inline constexpr std::uint64_t kMtpFp8WeightsBytes = 461223936;
struct MtpFp8TensorPin {
  const char *name, *source, *codes, *scales;
};
inline constexpr MtpFp8TensorPin kMtpFp8TensorPins[] = {
    {"blk.28.ffn_down_exps.weight",
     "81b2b76a5d4e0e202a9f1d7a46401048b9cac324bfc37e74bb1e9b3335989e1f",
     "b99f96c46153f82f5a32f9c25ba25aafa1c4ea35a435533f5ee910739f0fdefe",
     "d79df82b4a3de0cfb1055746e085be594787f21732091499a5c8ee72393cfda0"},
    {"blk.28.ffn_gate_exps.weight",
     "9492519e2ab6da1da1e8870cfe5945d5390c09af1810f165e354adbbd2dab0d8",
     "c8afeb4ff5705de07b173a808524227405d413614ae99918a898ef6ee24ad745",
     "2dd6db5282b22b13f8d4bfc7bd683456efdc6febbec379b52f317bfe2b0e1e34"},
    {"blk.28.ffn_up_exps.weight",
     "1e484c4b0113af1e1c341c12d4568b5e648a9d1f0217e98be3e4b509500752a6",
     "f35660def61c30098b3cfa65750c4476e4b73b797d0ce5a82e325d47b1037ace",
     "cf10330aedd5dc8cf8da88ea6a77c6d7ed511a84e572df9dc966b37f1386f052"},
};
} // namespace mellum16::internal
