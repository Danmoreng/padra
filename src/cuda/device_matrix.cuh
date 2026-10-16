#pragma once
#include "model/compiled.h"
#include <limits>
#include <stdexcept>
namespace mellum16 {
struct DeviceMatrix {
  const unsigned char *data = nullptr;
  const unsigned char *scales = nullptr;
  const float *divisor = nullptr;
  int rows = 0, cols = 0, format = 0;
  const std::uint16_t *suh = nullptr;
  const std::uint16_t *svh = nullptr;
};
inline DeviceMatrix BindDeviceMatrix(const internal::CompiledModel &model, unsigned char *arena,
                                     const std::string &name) {
  const auto &m = model.tensors.at(name);
  DeviceMatrix d;
  d.rows = static_cast<int>(m.shape.at(0));
  d.cols = m.shape.size() > 1 ? static_cast<int>(m.shape.at(1)) : 1;
  switch (m.format) {
  case internal::WeightFormat::Bf16:
    d.format = 0;
    break;
  case internal::WeightFormat::Fp8:
    d.format = 1;
    break;
  case internal::WeightFormat::Nvfp4:
    d.format = 2;
    break;
  case internal::WeightFormat::Exl3K4:
    d.format = 3;
    break;
  case internal::WeightFormat::Exl3K8:
    d.format = 4;
    break;
  case internal::WeightFormat::Exl3K6:
    d.format = 5;
    break;
  default:
    throw std::runtime_error("invalid production matrix format");
  }
  const auto size =
      model.resident_weights_bytes ? model.resident_weights_bytes : model.weights.size();
  const auto pointer = [&](const char *part) {
    const auto r = m.regions.at(part);
    if (r.offset > size || r.bytes > size - r.offset)
      throw std::runtime_error("invalid production matrix region");
    return arena + r.offset;
  };
  d.data = pointer(d.format == 0 ? "data" : "packed");
  if (d.format == 1 || d.format == 2)
    d.scales = pointer("scales");
  if (d.format == 3 || d.format == 4 || d.format == 5) {
    d.suh = reinterpret_cast<const std::uint16_t *>(pointer("suh"));
    d.svh = reinterpret_cast<const std::uint16_t *>(pointer("svh"));
  }
  if (d.format == 2)
    d.divisor = reinterpret_cast<const float *>(pointer("divisor"));
  return d;
}
} // namespace mellum16
