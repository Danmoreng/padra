#pragma once
#include "server/job.h"
#include "runtime/model_runtime.h"
namespace mellum16::server {
// Worker-only token processing. No KV preparation, store commit or HTTP writes.
void GenerateOutput(ModelRuntime &, const text::Tokenizer &, Job &,
                    std::chrono::steady_clock::time_point started,
                    std::chrono::steady_clock::time_point deadline, bool &runtime_failed);
void PublishTerminal(Job &);
} // namespace mellum16::server
