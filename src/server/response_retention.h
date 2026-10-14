#pragma once
#include "server/response_store.h"
#include "server/responses.h"
#include "server/generation_result.h"
namespace mellum16::server {
// Request/provider-owned reservation. Dropping an uncommitted instance rolls
// back the ticket and root through their existing RAII ownership.
class ResponseRetention {
public:
  ResponseRetention(std::shared_ptr<ResponseStore> store,
                    std::shared_ptr<ResponseReservation> reservation,
                    std::shared_ptr<const ResponseRequest> spec, std::shared_ptr<ResponseRoot> root)
      : store_(std::move(store)), reservation_(std::move(reservation)), spec_(std::move(spec)),
        root_(std::move(root)) {}
  Status Commit(const GenerationResult &) const;

private:
  std::shared_ptr<ResponseStore> store_;
  std::shared_ptr<ResponseReservation> reservation_;
  std::shared_ptr<const ResponseRequest> spec_;
  std::shared_ptr<ResponseRoot> root_;
};
} // namespace mellum16::server
