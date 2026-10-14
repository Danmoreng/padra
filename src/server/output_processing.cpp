#include "server/output_processing.h"
#include "server/secure_id.h"
#include "text/stream.h"
#include <stdexcept>
namespace mellum16::server {
namespace {
using Clock = std::chrono::steady_clock;
class Events {
public:
  explicit Events(Job &job) : job_(job) {}
  void Begin() {
    Send(GenerationEvent::Make(EventKind::Begin));
  }
  void Delta(std::string_view content, std::string_view reasoning) {
    if (!reasoning.empty())
      job_.ObserveOutput(OutputKind::Reasoning);
    if (!content.empty())
      job_.ObserveOutput(OutputKind::Message);
    Parts(EventKind::Reasoning, reasoning);
    Parts(EventKind::Text, content);
  }
  void Terminal() {
    job_.PushEnding(GenerationEvent::Make(EventKind::Finish));
    if (job_.request.include_usage)
      job_.PushEnding(GenerationEvent::Make(EventKind::Usage));
    job_.PushEnding(GenerationEvent::Make(EventKind::Done));
  }
  void ToolBegin(std::size_t index, std::string_view id, std::string_view name) {
    job_.ObserveOutput(OutputKind::Tool, index);
    Send(GenerationEvent::Make(EventKind::ToolStart, {}, index, id, name));
  }
  void ToolArguments(std::size_t index, std::string_view args) {
    Parts(EventKind::ToolArguments, args, index);
  }
  void Tool(std::size_t index, const ToolCall &call) {
    ToolBegin(index, call.id, call.name);
    ToolArguments(index, call.arguments);
  }

private:
  void Send(const GenerationEvent &event) {
    if (job_.request.stream && !job_.Push(event))
      job_.Cancel(CancelCause::Backpressure);
  }
  void Parts(EventKind kind, std::string_view text, std::size_t index = 0) {
    if (!job_.request.stream)
      return;
    while (!text.empty() && job_.cancellation.load() == CancelCause::None) {
      auto size = std::min<std::size_t>(1024, text.size());
      while (size < text.size() && (static_cast<unsigned char>(text[size]) & 0xc0) == 0x80)
        --size;
      Send(GenerationEvent::Make(kind, text.substr(0, size), index));
      text.remove_prefix(size);
    }
  }
  Job &job_;
};
} // namespace
void GenerateOutput(ModelRuntime &runtime, const text::Tokenizer &tokenizer, Job &job,
                    std::chrono::steady_clock::time_point started,
                    std::chrono::steady_clock::time_point deadline, bool &runtime_failed) {
  const auto lease = job.session_lease;
  text::StreamDecoder decoder(tokenizer);
  const auto maximum = (tokenizer.MaxRawTokenBytes() + 3) * 3;
  text::StopMatcher stops(job.request.stops, maximum);
  text::ThinkingParser thinking(maximum + 256);
  ToolStreamParser tools(job.request.tools.parse, maximum + 256,
                         job.request.tools.parallel ? 8 : 1);
  std::array<std::string, 8> call_ids;
  if (job.request.tools.parse)
    for (auto &id : call_ids) {
      auto generated = MakeSecureId("call_");
      if (!generated.ok())
        throw std::runtime_error(generated.status().message());
      id = std::move(generated).value();
    }
  Events events(job);
  if (job.request.stream)
    events.Begin();
  if (job.request.tools.parse)
    tools.StreamDeltas(job.request.tools, [&](std::size_t index, std::string_view name,
                                              std::string_view arguments) {
      if (job.visible_seconds == 0)
        job.visible_seconds = std::chrono::duration<double>(Clock::now() - started).count();
      if (!name.empty())
        events.ToolBegin(index, call_ids[index], name);
      if (!arguments.empty())
        events.ToolArguments(index, arguments);
      return job.cancellation.load() == CancelCause::None;
    });
  const auto accept_output = [&](text::ThinkingOutput output) {
    auto visible = tools.Push(output.content);
    if (!visible.empty() && job.visible_seconds == 0)
      job.visible_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    job.content.append(visible);
    job.reasoning.append(output.reasoning);
    events.Delta(visible, output.reasoning);
    tools.FlushDeltas();
    if (tools.delivery_failed())
      job.Cancel(CancelCause::Backpressure);
    if (tools.error())
      job.Cancel(CancelCause::ToolInvalid);
  };
  const auto accept = [&](std::string_view text) {
    accept_output(thinking.Push(stops.Push(text)));
  };
  const auto allowed = [&] {
    if (Clock::now() > deadline)
      job.Cancel(CancelCause::Deadline);
    if (lease && lease->record->deleted.load())
      job.Cancel(CancelCause::SessionDiscard);
    return !stops.stopped() && job.cancellation.load() == CancelCause::None;
  };
  auto result = runtime.Generate(
      [&](int token) {
        if (job.completion_tokens++ == 0)
          job.first_token_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        const bool was_reasoning = thinking.in_reasoning();
        accept(decoder.Push(token));
        if (was_reasoning || thinking.in_reasoning())
          ++job.reasoning_tokens;
        return allowed();
      },
      [&](int) { return allowed(); });
  if (!result.ok()) {
    runtime_failed = true;
    throw std::runtime_error(result.status().message());
  }
  job.prefill_seconds = result.value().prefill_seconds;
  job.decode_seconds = result.value().decode_seconds;
  if (tools.error())
    throw std::runtime_error(tools.error());
  const auto cause = job.cancellation.load();
  if (cause != CancelCause::None)
    throw std::runtime_error("generation cancelled/deadline/disconnect/backpressure");
  accept(decoder.Finish());
  accept_output(thinking.Push(stops.Finish()));
  accept_output(thinking.Finish());
  auto visible = tools.Finish();
  job.content.append(visible);
  if (!visible.empty() && job.visible_seconds == 0)
    job.visible_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  events.Delta(visible, {});
  if (tools.error())
    throw std::runtime_error(tools.error());
  job.finish_reason = (stops.stopped() || (!result.value().generated_ids.empty() &&
                                           result.value().generated_ids.back() == 28))
                          ? "stop"
                          : "length";
  if (tools.incomplete() && job.finish_reason != "length") {
    job.Cancel(CancelCause::ToolInvalid);
    throw std::runtime_error("model ended/stopped with incomplete tool call");
  }
  for (std::size_t i = 0; i < tools.count(); ++i) {
    const bool complete = !(tools.incomplete() && i + 1 == tools.count());
    auto parsed =
        ParseGeneratedTool(tools.raw(i), job.request.tools, std::move(call_ids[i]), complete);
    if (!parsed.ok()) {
      job.Cancel(CancelCause::ToolInvalid);
      throw std::runtime_error(parsed.status().message());
    }
    job.tool_calls.push_back(std::move(parsed).value());
  }
  job.partial_tool_call = tools.incomplete();
  if (!job.tool_calls.empty() && !job.partial_tool_call)
    job.finish_reason = "tool_calls";
  if (!job.tool_calls.empty() && job.visible_seconds == 0)
    job.visible_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  for (std::size_t i = 0; i < job.tool_calls.size(); ++i)
    if (!tools.identity_sent(i))
      events.Tool(i, job.tool_calls[i]);
  if (!job.output_count && job.finish_reason != "length")
    job.ObserveOutput(OutputKind::Message);
  if (job.cancellation.load() != CancelCause::None)
    throw std::runtime_error("generation cancelled during output finalization");
}
void PublishTerminal(Job &job) {
  if (!job.request.stream)
    return;
  Events(job).Terminal();
  if (job.cancellation.load() != CancelCause::None)
    throw std::runtime_error("terminal stream backpressure");
}
} // namespace mellum16::server
