#include "server/http_request.h"
#include "compiler/sha256.h"
namespace mellum16::server::http_detail {
Result<std::string> SessionOwner(const httplib::Request &request) {
  if (request.get_header_value_count("Authorization") > 1)
    return Status(StatusCode::kInvalidArgument, "duplicate Authorization namespace header");
  const auto auth = request.get_header_value("Authorization");
  if (auth.size() > 512 || (!auth.empty() && (!auth.starts_with("Bearer ") || auth.size() == 7)))
    return Status(StatusCode::kInvalidArgument, "invalid loopback credential namespace");
  const auto value = auth.empty() ? std::string("anonymous-loopback") : auth;
  return compiler::Sha256Hex(value.data(), value.size());
}
} // namespace mellum16::server::http_detail
