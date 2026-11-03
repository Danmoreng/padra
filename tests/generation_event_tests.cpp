#include "server_test_support.h"
#include <cstdlib>
#include <new>
#include <type_traits>
namespace {
std::atomic<std::size_t> allocations{0};
constexpr bool count_allocations = PADRA_COUNT_ALLOCATIONS;
} // namespace
#if PADRA_COUNT_ALLOCATIONS
void *operator new(std::size_t size) {
  allocations.fetch_add(1, std::memory_order_relaxed);
  if (auto *p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size) {
  return ::operator new(size);
}
void operator delete(void *p) noexcept {
  std::free(p);
}
void operator delete[](void *p) noexcept {
  std::free(p);
}
void operator delete(void *p, std::size_t) noexcept {
  std::free(p);
}
void operator delete[](void *p, std::size_t) noexcept {
  std::free(p);
}
#endif

int main() {
  try {
    auto large_request = Request("lazy_output");
    large_request.outputs = 131072;
    Job lazy(std::move(large_request), 128);
    Require(lazy.content.capacity() + lazy.reasoning.capacity() < 1024,
            "unused output budget allocated eagerly");
    lazy.AppendReasoning("short reasoning");
    Require(lazy.content.empty() && lazy.reasoning == "short reasoning" &&
                lazy.content.capacity() + lazy.reasoning.capacity() < 4096,
            "short output allocated the maximum token budget");
    auto bounded_request = Request("output_ceiling");
    Job bounded(std::move(bounded_request), 0);
    const std::string full(1033, 'x');
    for (char c : full)
      bounded.AppendContent(std::string_view(&c, 1));
    Require(bounded.content == full && bounded.content.capacity() <= full.size(),
            "incremental growth exceeded admitted capacity or changed bytes");
    bool rejected = false;
    try {
      bounded.AppendContent("x");
    } catch (const std::length_error &) {
      rejected = true;
    }
    Require(rejected && bounded.content == full, "output ceiling did not reject atomically");
    auto mixed_request = Request("mixed_output_ceiling");
    Job mixed(std::move(mixed_request), 0);
    for (int i = 0; i < 1033; ++i) {
      if (i % 2)
        mixed.AppendContent("c");
      else
        mixed.AppendReasoning("r");
    }
    Require(mixed.content.size() + mixed.reasoning.size() == 1033 &&
                mixed.content.capacity() + mixed.reasoning.capacity() <= 2066,
            "mixed output growth changed bytes or exceeded admission");
    rejected = false;
    try {
      mixed.AppendReasoning("r");
    } catch (const std::length_error &) {
      rejected = true;
    }
    Require(rejected, "mixed channels exceeded the combined decoded byte limit");
    static_assert(std::is_trivially_copyable_v<GenerationEvent>);
    auto request = Request("event_ownership");
    request.stream = true;
    request.include_usage = true;
    Job job(std::move(request), 4);
    ChatStream encoder(job);
    std::string source(1024, '\0');
    const auto before = allocations.load();
    auto original = GenerationEvent::Make(EventKind::Text, source);
    for (int i = 0; i < 64; ++i)
      Require(job.Push(original), "ordinary event ring rejected admitted slot");
    Require(!count_allocations || allocations.load() == before,
            "logical event construction/publication allocated");
    source.assign(1024, 'x');
    Require(!job.Push(original) && job.cancellation.load() == CancelCause::Backpressure,
            "full ordinary ring did not signal backpressure");
    job.finish_reason = "length";
    Require(job.PushEnding(GenerationEvent::Make(EventKind::Finish)) &&
                job.PushEnding(GenerationEvent::Make(EventKind::Usage)) &&
                job.PushEnding(GenerationEvent::Make(EventKind::Done)),
            "ordinary saturation consumed reserved completion lane");
    {
      std::lock_guard lock(job.mutex);
      job.complete = true;
    }
    GenerationEvent event;
    for (int i = 0; i < 64; ++i) {
      Require(job.Pop(event, 0ms) && event.kind == EventKind::Text && event.size == 1024 &&
                  event.data[0] == '\0',
              "published event borrowed mutable source or changed order");
      const auto count = allocations.load();
      auto wire = encoder.Encode(event);
      Require(!count_allocations || allocations.load() == count,
              "ordinary Chat encoding allocated after reservation");
      Require(wire.size() <= 8192 && wire.size() > 6144,
              "worst-case JSON escaping exceeded frame contract");
    }
    for (auto kind : {EventKind::Finish, EventKind::Usage, EventKind::Done})
      Require(job.Pop(event, 0ms) && event.kind == kind && !encoder.Encode(event).empty(),
              "completion lane lost order after saturated ring");
    Require(!job.Pop(event, 0ms), "event ring did not drain");
    auto invalid = original;
    invalid.size = invalid.data.size() + 1;
    Require(!job.Push(invalid), "invalid typed slot accepted");
    bool oversized = false;
    try {
      (void)GenerationEvent::Make(EventKind::Text, std::string(1025, 'x'));
    } catch (const std::length_error &) {
      oversized = true;
    }
    Require(oversized, "oversized logical fragment was not rejected");
    auto loaded = text::Tokenizer::Load({}, {});
    Require(loaded.ok(), "test tokenizer failed");
    auto state = std::make_shared<MockState>();
    state->prepare_failure = true;
    Coordinator coordinator(Runtime(state), *loaded.value());
    auto failed = Submit(coordinator, Request("frozen_failure"));
    Wait(failed);
    auto response = ParseResponseRequest(ResponseBody);
    Require(response.ok() && !failed->failure.ok(), "failure snapshot setup failed");
    auto frozen = ResponseEnvelope(response.value(), failed.get());
    failed->Cancel(CancelCause::Disconnect);
    Require(ResponseEnvelope(response.value(), failed.get()) == frozen,
            "transport cancellation rewrote published generation error");
    std::cout
        << "event_bytes=" << sizeof(GenerationEvent)
        << " ring_bytes=" << sizeof(job.events) + sizeof(job.endings)
        << " job_bytes=" << sizeof(Job)
        << (count_allocations
                ? "; 64 publications and ordinary Chat encodes: zero counted new allocations\n"
                : "; functional checks passed; allocation probe disabled under TSan\n");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
