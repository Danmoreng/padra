#include "server/chat_stream.h"
#include "server/coordinator.h"
#include "server/job_error.h"
#include <stdexcept>
namespace mellum16::server {
ChatStream::ChatStream(const Job &job) : job_(job) {}
void ChatStream::Need(bool ok) {
  if (!ok)
    throw std::runtime_error("bounded Chat SSE frame encoding exceeded");
}
void ChatStream::Envelope() {
  builder_.Reset();
  Need(builder_.Append("{\"id\":"));
  Need(builder_.AppendJsonString(job_.request.id));
  Need(builder_.Append(",\"object\":\"chat.completion.chunk\",\"created\":"));
  Need(builder_.AppendUnsigned(job_.created_at));
  Need(builder_.Append(",\"model\":"));
  Need(builder_.AppendJsonString(kModelId));
  Need(builder_.Append(",\"choices\":"));
}
std::span<const char> ChatStream::Encode(const GenerationEvent &event) {
  if (!event.Valid())
    throw std::logic_error("invalid generation event");
  if (event.kind == EventKind::Done) {
    builder_.Reset();
    Need(builder_.Append("[DONE]"));
  } else if (event.kind == EventKind::Error) {
    builder_.Reset();
    const auto error = ClassifyJobError(job_.failure_cause);
    Need(builder_.Append(ErrorJson(event.Text(), error.code, error.type)));
  } else {
    Envelope();
    switch (event.kind) {
    case EventKind::Begin:
      Need(builder_.Append("[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
                           "\"finish_reason\":null,\"logprobs\":null}]"));
      break;
    case EventKind::Text:
    case EventKind::Reasoning:
      Need(builder_.Append("[{\"index\":0,\"delta\":{"));
      Need(builder_.Append(event.kind == EventKind::Text ? "\"content\":"
                                                         : "\"reasoning_content\":"));
      Need(builder_.AppendJsonString(event.Text()));
      Need(builder_.Append("},\"finish_reason\":null,\"logprobs\":null}]"));
      break;
    case EventKind::ToolStart:
      Need(builder_.Append("[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":"));
      Need(builder_.AppendUnsigned(event.index));
      Need(builder_.Append(",\"id\":"));
      Need(builder_.AppendJsonString(event.Id()));
      Need(builder_.Append(",\"type\":\"function\",\"function\":{\"name\":"));
      Need(builder_.AppendJsonString(event.Name()));
      Need(builder_.Append(",\"arguments\":\"\"}}]},\"finish_reason\":null,\"logprobs\":null}]"));
      break;
    case EventKind::ToolArguments:
      Need(builder_.Append("[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":"));
      Need(builder_.AppendUnsigned(event.index));
      Need(builder_.Append(",\"function\":{\"arguments\":"));
      Need(builder_.AppendJsonString(event.Text()));
      Need(builder_.Append("}}]},\"finish_reason\":null,\"logprobs\":null}]"));
      break;
    case EventKind::Finish:
      Need(builder_.Append("[{\"index\":0,\"delta\":{},\"finish_reason\":"));
      Need(builder_.AppendJsonString(job_.finish_reason));
      Need(builder_.Append(",\"logprobs\":null}]"));
      break;
    case EventKind::Usage:
      Need(builder_.Append("[],\"usage\":"));
      Need(builder_.Append(ChatUsage(job_)));
      break;
    default:
      throw std::logic_error("unknown logical event kind");
    }
    if (event.kind != EventKind::Usage && job_.request.include_usage)
      Need(builder_.Append(",\"usage\":null"));
    Need(builder_.Append('}'));
  }
  auto frame = builder_.FinishRecord();
  Need(!frame.empty());
  return frame;
}
} // namespace mellum16::server
