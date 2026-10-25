#pragma once
#include <cstdint>
namespace mellum16::internal {
inline constexpr char kMtpExl3Profile[] = "mellum21-mtp-exl3-k4-experts-bf16-f32-v1";
inline constexpr char kMtpExl3Conversion[] = "exl3-k4-h128-mul1-f16-sidecars-routed-hessian025-v1";
inline constexpr char kMtpExl3WeightsSha256[] =
    "985312eda174b32499468fb60291584e63da3b930674ccb0c9a59a700d8e2464";
inline constexpr char kMtpExl3CalibrationSha256[] =
    "de82774663535c621ac18bbf9c3f9fc5657f74045ed915593700629af6152e81";
inline constexpr std::uint64_t kMtpExl3WeightsBytes = 263747584;
struct MtpExl3TensorPin {
  const char *name, *source, *packed, *suh, *svh;
};
inline constexpr MtpExl3TensorPin kMtpExl3TensorPins[] = {
    {"blk.28.ffn_down_exps.weight",
     "81b2b76a5d4e0e202a9f1d7a46401048b9cac324bfc37e74bb1e9b3335989e1f",
     "c51d41baba43ed69049e8593ce273a0cb0c94db7ce35e78bddfd841ee991e845",
     "7a49c3876395a13b476d74d3970cc7579a058ef7a6dea48133c523a2b6d04db5",
     "f537953f49947742a45549f49f695f5f88bed29d06aeec6c8c317bf9717b59e5"},
    {"blk.28.ffn_gate_exps.weight",
     "9492519e2ab6da1da1e8870cfe5945d5390c09af1810f165e354adbbd2dab0d8",
     "5283f2344e23a37be717b9daaf4574d26cde115a66773c69a2e59f4635ba7cf3",
     "5974661aa7c7941e75f0948d691c329d281b53b15e3c13af47e2d2516194433c",
     "fc72b142a9c1ecbf5df05f10ee3aeb68ea14e72a8f79b6ccd96b54316d2395c5"},
    {"blk.28.ffn_up_exps.weight",
     "1e484c4b0113af1e1c341c12d4568b5e648a9d1f0217e98be3e4b509500752a6",
     "0bafcf69f2758578024302fbef97b7cd217578e43654fa6be840f8751e653cb2",
     "c7058829dc1e1c3cb7eaf5d31f5f45db04eb3b7bebd46d71f4b29a20de62718e",
     "979ebd4d31a6029240a8240ed5727037a452a6a3ace08bf239a907ba5bee6177"},
};
} // namespace mellum16::internal
