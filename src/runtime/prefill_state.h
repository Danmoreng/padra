#pragma once
#include <cstdint>
namespace mellum16 {
// Stable device command slot. Active geometry is part of the graph key; only
// absolute base/content/prefix advance between replays of a prepared shape.
struct PrefillState {
  int base=0,prompt_offset=0,tokens=0,capacity=0,committed=0;
  std::uint32_t binding_epoch=1,request_epoch=1,valid=0;
};
static_assert(sizeof(PrefillState)==32);
#if defined(__CUDACC__)
#define MELLUM_PREFILL_HD __host__ __device__
#else
#define MELLUM_PREFILL_HD
#endif
MELLUM_PREFILL_HD inline bool ValidPrefillState(const PrefillState& state,int active_tokens,int capacity,std::uint32_t binding_epoch,int maximum=8192){
  return state.valid==1 && state.binding_epoch==binding_epoch && state.request_epoch!=0 &&
    maximum>0 && maximum<=131072 && state.capacity==capacity && capacity>0 && capacity<=maximum && state.tokens==active_tokens &&
    active_tokens>0 && active_tokens<=2048 && state.base>=0 && state.base<=capacity-active_tokens &&
    state.committed==state.base && state.prompt_offset>=0 && state.prompt_offset<=capacity-active_tokens;
}
#undef MELLUM_PREFILL_HD
}
