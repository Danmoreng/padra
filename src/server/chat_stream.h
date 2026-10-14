#pragma once
#include "server/generation_event.h"
#include "server/sse_chunk.h"
namespace mellum16::server {
struct Job;
struct GenerationResult;
std::string ChatUsage(const GenerationResult &);
std::string CompletionJson(const Job &);
// Per HTTP consumer encoder. The coordinator publishes protocol-neutral events.
class ChatStream {
public:
  explicit ChatStream(const Job &job);
  std::span<const char> Encode(const GenerationEvent &);

private:
  void Need(bool);
  void Envelope();
  const Job &job_;
  SseChunkBuilder builder_{8190};
};
} // namespace mellum16::server
