#pragma once
#include <algorithm>
#include <map>
#include <string>
#include "size_checked.h"
#include "prefill_profile.h"
#include "kv_cache_layout.h"
#include "exl3_masked_schedule.h"
#include "kv_checkpoint.h"
#include "mtp_memory.h"
#include "head_memory.h"
namespace mellum16 {
struct ServerMemoryPlan {
  std::map<std::string, std::size_t> buffers;
  std::size_t resident = 0, reserve = 700ULL * 1024 * 1024;
  std::size_t required() const {
    return admission::Sum({resident, reserve});
  }
  bool Fits(std::size_t free) const {
    return required() <= free;
  }
};
// Selected all-row K3/K4/K6 profiles; exact native cudaMalloc inventory, not a
// generic estimator. These ABI widths are independently asserted by CUDA.
inline ServerMemoryPlan PlanServerMemory(std::size_t weights, int capacity, int prompt, int chunk,
                                         bool persistent = false,
                                         std::size_t sampling_workspace = 0,
                                         KVCacheFormat kv_format = KVCacheFormat::BF16,
                                         bool checkpoint = false, int expert_rate = 4,
                                         bool mtp = false, bool mtp_fp8_experts = false,
                                         bool q8_head = false, bool mtp_exl3_experts = false) {
  if (capacity < 1 || capacity > 131072 || prompt < 1 || prompt > capacity || chunk < 2 ||
      chunk > 2048 || (expert_rate != 3 && expert_rate != 4 && expert_rate != 6))
    throw std::invalid_argument("invalid selected-profile memory shape");
  if ((mtp_fp8_experts || mtp_exl3_experts) && !mtp)
    throw std::invalid_argument("quantized draft experts require MTP");
  const std::size_t batch = persistent ? chunk : std::min(chunk, prompt);
  ServerMemoryPlan plan;
  const auto add = [&](const char *name, std::size_t bytes) {
    plan.buffers.emplace(name, bytes);
    plan.resident = admission::Sum({plan.resident, bytes});
  };
  add("weights", weights);
  if (q8_head)
    add("head_input", kHeadQ8InputBytes);
  if (mtp) {
    if (!persistent)
      throw std::invalid_argument("MTP inventory requires persistent execution");
    add("mtp_draft", MtpDraftResidentBytes(mtp_fp8_experts, mtp_exl3_experts));
    add("mtp_verifier", kMtpVerifyResidentBytes);
    add("mtp_hidden", MtpHiddenBytes(chunk));
    if (checkpoint)
      add("mtp_checkpoint", kMtpCheckpointBytes);
  }
  add("kv", admission::Bytes(7ULL * capacity + 21ULL * std::min(capacity, 1024),
                             KVCacheLayout::Make(kv_format, 1).bytes));
  add("descriptors", 28 * 64 * 3 * 56);
  add("rope", admission::Bytes(capacity, 128 * 4 * 2));
  add("scratch", (2304 * 3 + 4096 * 2 + 512 * 2 + 64 + 8 * 896 * 2 + 8 * 2304 + 98304) * 4);
  add("batch",
      admission::Bytes(batch, (2304 * 3 + 4096 * 2 + 512 * 2 + 64 + 8 * 896 * 2 + 8 * 2304) * 4));
  add("quant",
      admission::Sum(
          {admission::Bytes(batch, 7168 + 8 * 4 + 8 * 2304 / 2 + 8 * 2304 / 16 + 8 * 4), 15}));
  add("tensor", admission::Sum({32 * 1024 * 1024, admission::Bytes(batch, 4096 * 2)}));
  add("groups", admission::Bytes(admission::Sum({admission::Bytes(batch, 64), 64}), 4));
  add("ids_and_routing", admission::Bytes(batch, 8 * 4 * 2));
  add("bad_and_control", 12);
  add("selection", 96 * 8);
  add("trace_and_snapshot_placeholders", 8);
  add("prompt", admission::Bytes(persistent ? capacity : prompt, 4));
  add("device_state", 32);
  add("attention_placeholders", 6);
  add("attention_scratch",
      admission::Bytes(std::min((capacity + 31) / 32, 32), 32 * (128 + 1) * 4));
  const auto compact = std::min((batch * 8 + 15) / 16 + 63, 64 * ((batch + 15) / 16));
  add("compact", admission::Sum({admission::Bytes(compact, 8), 4}));
  add("packed_chunk", (kv_format == KVCacheFormat::BF16 ||
                       kv_format == KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2)
                          ? admission::Bytes(batch, 2048)
                          : 0);
  if (kv_format == KVCacheFormat::Q8TokenHead128F32ScaleV1) {
    add("encoded_q8_chunk", admission::Bytes(batch, 1056));
  }
  if (kv_format == KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2) {
    add("encoded_q8_0_chunk", admission::Bytes(batch, 1088));
    add("q8_0_prefix_workspace", admission::Bytes(capacity, 2048));
    add("q8_0_query", 128 * 36);
  }
  add("expert_schedule", exl3_masked_schedule::Bytes(static_cast<int>(batch)));
  add("silu", 65536 * 2);
  add("expert_workspace", admission::Bytes(batch, 8 * 2304 * 6));
  if (checkpoint) {
    if (!persistent || capacity < 1024)
      throw std::invalid_argument("checkpoint inventory requires persistent capacity>=1024");
    add("local_checkpoint", KVCheckpointLedger::LocalSnapshotBytes(kv_format));
  }
  if (sampling_workspace) {
    add("sampling_arrays_state", 98304 * 28 + 16);
    add("sampling_algorithm_workspace", sampling_workspace);
  }
  return plan;
}
} // namespace mellum16
