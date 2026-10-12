#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstddef>
#include "mellum16/status.h"
namespace mellum16::tc {
Status AdmitFP8();
Status ProjectFP8(const std::uint8_t* activation,const float* activation_scales,
    const std::uint8_t* weight,const std::uint16_t* weight_scales,std::uint16_t* output,
    std::uint64_t tokens,std::uint64_t rows,std::uint64_t cols,void* workspace,
    std::size_t workspace_bytes,cudaStream_t stream);
}
