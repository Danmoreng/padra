#include "parser_properties.h"
#include <cstddef>
#include <cstdint>
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
  if (size <= 131072)
    mellum16::server::test::CheckParserProperties({reinterpret_cast<const char *>(data), size});
  return 0;
}
