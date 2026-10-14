#pragma once
#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string_view>
namespace mellum16::server {
enum class EventKind {
  Begin,
  Text,
  Reasoning,
  ToolStart,
  ToolArguments,
  Finish,
  Usage,
  Done,
  Error
};
// A slot owns all bytes crossing the worker/HTTP thread boundary. Text and raw
// arguments are split at UTF-8 boundaries before publication. No heap allocation
// or reference to a reusable decoder/parser buffer is stored in the ring.
struct GenerationEvent {
  EventKind kind = EventKind::Begin;
  std::size_t index = 0, size = 0, id_size = 0, name_size = 0;
  std::array<char, 1024> data{};
  std::array<char, 128> id{};
  std::array<char, 64> name{};
  static GenerationEvent Make(EventKind kind, std::string_view text = {}, std::size_t index = 0,
                              std::string_view id = {}, std::string_view name = {}) {
    GenerationEvent event;
    if (text.size() > event.data.size() || id.size() > event.id.size() ||
        name.size() > event.name.size() || index >= 8)
      throw std::length_error("logical generation event exceeds fixed slot");
    event.kind = kind;
    event.index = index;
    event.size = text.size();
    event.id_size = id.size();
    event.name_size = name.size();
    if (!text.empty())
      std::memcpy(event.data.data(), text.data(), text.size());
    if (!id.empty())
      std::memcpy(event.id.data(), id.data(), id.size());
    if (!name.empty())
      std::memcpy(event.name.data(), name.data(), name.size());
    return event;
  }
  bool Valid() const {
    return kind >= EventKind::Begin && kind <= EventKind::Error && size <= data.size() &&
           id_size <= id.size() && name_size <= name.size() && index < 8;
  }
  std::string_view Text() const {
    return {data.data(), size};
  }
  std::string_view Id() const {
    return {id.data(), id_size};
  }
  std::string_view Name() const {
    return {name.data(), name_size};
  }
  bool Terminal() const {
    return kind == EventKind::Finish || kind == EventKind::Usage || kind == EventKind::Done ||
           kind == EventKind::Error;
  }
  bool Visible() const {
    return kind == EventKind::Text || kind == EventKind::ToolStart ||
           kind == EventKind::ToolArguments;
  }
};
static_assert(sizeof(GenerationEvent) <= 1280);
} // namespace mellum16::server
