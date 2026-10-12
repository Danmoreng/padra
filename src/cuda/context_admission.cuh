#pragma once
// Explicit validated profile bound; defaults remain the old8192 contract.
// State layout and arithmetic do not change. Standalone legacy probes can use
// this helper without redefining their one-argument DecodeReady function.
template<class State> __device__ __forceinline__ bool DecodeReadyAtLimit(const State* s,int maximum){
 return maximum>0 && maximum<=131072 && s && !s->stopped && !s->eos && s->remaining>0 && s->token>=0 && s->token<98304 && s->capacity>0 && s->capacity<=maximum && s->position>=0 && s->position<s->capacity && s->global_valid==s->position && s->local_valid==min(s->position,1024);
}
