// Adapted from Gem16 (Apache-2.0), pin c0a807afcbcd2528f304bada513afd11d1e9c979.
// See NOTICE and third_party/gem16-server-reference/LICENSE.
#pragma once

#include <string>
#include <string_view>

#include "mellum16/status.h"

namespace mellum16::server {

// Returns prefix followed by 128 bits of OS-generated randomness encoded as
// lowercase hexadecimal. IDs are opaque protocol handles, not counters.
[[nodiscard]] Result<std::string> MakeSecureId(std::string_view prefix);

}  // namespace mellum16::server
