#include "server/response_retention.h"
namespace mellum16::server {
Status ResponseRetention::Commit(const GenerationResult &result) const {
  if (!reservation_)
    return {};
  return store_->Commit(reservation_, ResponseEnvelope(*spec_, &result),
                        result.partial_tool_call ? spec_->history
                                                 : ResponseCompletedHistory(*spec_, result),
                        root_, !result.partial_tool_call);
}
} // namespace mellum16::server
