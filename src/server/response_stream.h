#pragma once
#include "server/responses.h"
#include "server/generation_event.h"
#include <functional>
#include <vector>
namespace mellum16::server {
// Network-thread adapter only. GPU callback remains the existing bounded ring.
class ResponseStream {
public:
  using Send = std::function<bool(std::string_view)>;
  using Written = std::function<void(std::size_t, bool)>;
  ResponseStream(const ResponseRequest &, Send, Written = {});
  bool Begin();
  bool OnEvent(const GenerationEvent &);
  bool Finish(const GenerationResult &);
  bool Fail(std::string_view message);

private:
  struct Item {
    std::string id, type;
    std::size_t index;
  };
  bool Event(std::string_view type, std::string_view fields, bool visible = false);
  Result<std::size_t> Ensure(std::string type, std::string id, std::string item,
                             std::string part = {});
  bool Delta(std::string_view type, std::string_view id, std::string_view text);
  const ResponseRequest &request_;
  Send send_;
  Written written_;
  std::size_t sequence_ = 0;
  std::vector<Item> items_;
};
} // namespace mellum16::server
