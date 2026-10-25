#include "runtime/context_capacity.h"
#include "runtime/chunk_graph_contract.h"
#include "runtime/server_memory.h"
#include <iostream>
#include <stdexcept>

namespace {
void Require(bool condition) {
  if (!condition)
    throw std::runtime_error("KV memory contract failed");
}
} // namespace

int main() {
  try {
    using namespace mellum16;
    constexpr std::size_t weights = 5705667072ULL, ceiling = 8ULL << 30;
    for (int chunk : {256, 512, 2048})
      for (int boundary : {chunk, 2 * chunk, 3 * chunk}) {
        Require(CheckpointCaptureAnchor(boundary, chunk) == boundary);
        Require(CheckpointCaptureAnchor(boundary - 4, chunk) == boundary - 4);
        for (int tail : {1, 2, 3})
          Require(CheckpointCaptureAnchor(boundary - tail, chunk) == boundary - 4);
        Require(CheckpointCaptureAnchor(boundary + 1, chunk) == boundary + 1);
      }
    Require(CheckpointCaptureAnchor(0, 256) == 0 && CheckpointCaptureAnchor(1, 256) == 1);
    for (int capacity : {8192, 16384, 32768, 65536, 131072}) {
      Require(AdmitNativeContext(capacity).ok());
      const auto layout = KVCacheLayout::Make(KVCacheFormat::Q8TokenHead128F32ScaleV1, capacity);
      Require(layout.v_offset == std::size_t(capacity) * 512 &&
              layout.scales_offset == std::size_t(capacity) * 1024 &&
              layout.bytes == std::size_t(capacity) * 1056 && layout.scales_offset % 8 == 0);
      for (int chunk : {256, 512, 2048}) {
        const auto q8 = PlanServerMemory(weights, capacity, capacity, chunk, true, 0,
                                         KVCacheFormat::Q8TokenHead128F32ScaleV1, false, 3);
        const auto bf16 = PlanServerMemory(weights, capacity, capacity, chunk, true, 0,
                                           KVCacheFormat::BF16, false, 3);
        const auto rows = std::size_t(7) * capacity + 21 * 1024;
        Require(q8.buffers.at("kv") == rows * 1056 && bf16.buffers.at("kv") == rows * 2048);
        Require(q8.buffers.at("packed_chunk") == 0 &&
                q8.buffers.at("encoded_q8_chunk") == std::size_t(chunk) * 1056);
        Require(bf16.resident - q8.resident == (rows + chunk) * (2048 - 1056));
        Require(q8.Fits(q8.required()) && !q8.Fits(q8.required() - 1));
      }
    }
    const auto q8 = PlanServerMemory(weights, 131072, 131072, 256, true, 0,
                                     KVCacheFormat::Q8TokenHead128F32ScaleV1, false, 3);
    const auto bf16 =
        PlanServerMemory(weights, 131072, 131072, 256, true, 0, KVCacheFormat::BF16, false, 3);
    // The allocator plan includes its700 MiB reserve. Driver/graph admission is
    // checked against actual free memory by the runtime, not emulated here.
    Require(q8.Fits(ceiling) && !bf16.Fits(ceiling));
    bool rejected = false;
    Require(!AdmitNativeContext(131073).ok());
    const auto native =
        PlanServerMemory(weights, 131072, 131072, 256, true, 0,
                         KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2, false, 3);
    Require(native.buffers.at("kv") == std::size_t(7 * 131072 + 21 * 1024) * 1088);
    Require(native.buffers.at("q8_0_prefix_workspace") == 131072ULL * 2048);
    Require(native.buffers.at("q8_0_query") == 128 * 36 && native.Fits(ceiling));
    const auto layout =
        KVCacheLayout::Make(KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2, 131072);
    Require(layout.v_offset == 131072ULL * 544 && layout.bytes == 131072ULL * 1088);
    ChunkShapeSet shapes;
    ChunkGraphKey key{"checkpoint", "execution", "build", layout.name(), 131072, 256, 256, 1};
    Require(shapes.Admit(key, 100, 1024).ok() && shapes.Find(key));
    key.kv_format = "q8_token_head128_f32scale_v1";
    Require(!shapes.Find(key));
    for (int chunk : {256, 512, 2048}) {
      const auto ordinary = PlanServerMemory(7092933120ULL, 8192, 8192, chunk, true);
      const auto mtp = PlanServerMemory(7092933120ULL, 8192, 8192, chunk, true, 0,
                                        KVCacheFormat::BF16, false, 4, true);
      Require(mtp.resident - ordinary.resident ==
              kMtpDraftResidentBytes + kMtpVerifyResidentBytes + MtpHiddenBytes(chunk));
      const auto fp8 = PlanServerMemory(7092933120ULL, 8192, 8192, chunk, true, 0,
                                        KVCacheFormat::BF16, false, 4, true, true);
      Require(mtp.resident - fp8.resident == kMtpFp8ExpertWeightSavingBytes &&
              fp8.buffers.at("kv") == mtp.buffers.at("kv") &&
              fp8.buffers.at("weights") == mtp.buffers.at("weights") &&
              fp8.reserve == mtp.reserve && fp8.Fits(fp8.required()) &&
              !fp8.Fits(fp8.required() - 1));
      Require(mtp.buffers.at("weights") == ordinary.buffers.at("weights") &&
              mtp.buffers.at("kv") == ordinary.buffers.at("kv") && mtp.Fits(mtp.required()) &&
              !mtp.Fits(mtp.required() - 1));
    }
    for (int chunk : {256, 512, 2048})
      for (bool draft : {false, true}) {
        const auto bf = PlanServerMemory(7092933120ULL, 8192, 8192, chunk, true, 0,
                                         KVCacheFormat::BF16, false, 4, draft);
        const auto q8 =
            PlanServerMemory(7092933120ULL - kHeadQ8WeightSavingBytes, 8192, 8192, chunk, true, 0,
                             KVCacheFormat::BF16, false, 4, draft, false, true);
        Require(kHeadQ8WeightBytes == 240648192 && kHeadQ8InputBytes == 12240 &&
                bf.resident - q8.resident == 212324400 &&
                q8.buffers.at("head_input") == kHeadQ8InputBytes &&
                bf.buffers.at("kv") == q8.buffers.at("kv") && bf.reserve == q8.reserve &&
                q8.Fits(q8.required()) && !q8.Fits(q8.required() - 1));
        if (draft)
          Require(bf.buffers.at("mtp_draft") == q8.buffers.at("mtp_draft"));
      }
    for (int rate : {3, 4, 6})
      for (auto format : {KVCacheFormat::BF16, KVCacheFormat::Q8TokenHead128F32ScaleV1,
                          KVCacheFormat::Q8Token512Block32F16ScaleRotatedV2}) {
        const auto ordinary =
            PlanServerMemory(weights, 131072, 131072, 256, true, 0, format, false, rate);
        const auto mtp = PlanServerMemory(weights, 131072, 131072, 256, true, 0, format, false,
                                          rate, true, false, false, true);
        Require(mtp.resident - ordinary.resident ==
                MtpDraftResidentBytes(false, true) + kMtpVerifyResidentBytes + MtpHiddenBytes(256));
        Require(mtp.buffers.at("kv") == ordinary.buffers.at("kv"));
        const auto snapshotBytes = format == KVCacheFormat::BF16 ? 44040192ULL
                                   : format == KVCacheFormat::Q8TokenHead128F32ScaleV1
                                       ? 22708224ULL
                                       : 23396352ULL;
        const auto checkpoint = PlanServerMemory(weights, 131072, 131072, 256, true, 0, format,
                                                 true, rate, true, false, false, true);
        Require(checkpoint.buffers.at("local_checkpoint") == snapshotBytes &&
                checkpoint.resident - mtp.resident == snapshotBytes + kMtpCheckpointBytes &&
                checkpoint.reserve == mtp.reserve);
        KVCheckpointLedger ledger(131072, 42, format);
        std::vector<int> tokens(2048, 123);
        Require(ledger.Capture(tokens, 1025).ok());
        const auto handle = *ledger.handle();
        Require(handle.format == format && ledger.RestoreBase(tokens, handle).value() == 1025);
        auto wrong = handle;
        wrong.format = format == KVCacheFormat::BF16 ? KVCacheFormat::Q8TokenHead128F32ScaleV1
                                                     : KVCacheFormat::BF16;
        Require(!ledger.RestoreBase(tokens, wrong).ok());
        wrong = handle;
        ++wrong.runtime_binding;
        Require(!ledger.RestoreBase(tokens, wrong).ok());
        tokens[1024] = 124;
        Require(!ledger.RestoreBase(tokens, handle).ok());
        tokens[1024] = 123;
        tokens[1025] = 124;
        Require(ledger.RestoreBase(tokens, handle).ok());
        Require(ledger.Capture(tokens, 1537).ok());
        Require(!ledger.RestoreBase(tokens, handle).ok());
        const auto latest = *ledger.handle();
        ledger.Invalidate();
        Require(!ledger.RestoreBase(tokens, latest).ok());
      }
    rejected = false;
    try {
      (void)PlanServerMemory(weights, 8192, 8192, 256, false, 0, KVCacheFormat::BF16, false, 4,
                             true);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    Require(rejected);
    for (bool head : {false, true})
      for (int chunk : {256, 512, 2048}) {
        const auto bf = PlanServerMemory(7092933120ULL, 8192, 8192, chunk, true, 0,
                                         KVCacheFormat::BF16, false, 4, true, false, head);
        const auto exl3 = PlanServerMemory(7092933120ULL, 8192, 8192, chunk, true, 0,
                                           KVCacheFormat::BF16, false, 4, true, false, head, true);
        Require(bf.resident - exl3.resident == 593313792 &&
                bf.buffers.at("weights") == exl3.buffers.at("weights") &&
                bf.buffers.at("kv") == exl3.buffers.at("kv") && exl3.Fits(exl3.required()) &&
                !exl3.Fits(exl3.required() - 1));
      }
    rejected = false;
    try {
      (void)MtpDraftResidentBytes(true, true);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    Require(rejected);
    std::cout << "KV memory layouts, Q8 inventory and admission boundaries passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
