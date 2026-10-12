#pragma once
#include "mellum16/status.h"
#include <cmath>
#include <cstdint>
namespace mellum16 {
// Source logits remain immutable. Counts cover canonical prompt plus chosen tokens.
struct SamplingOptions {
 float temperature=0,top_p=1;int top_k=0;std::uint64_t seed=0;
 float frequency_penalty=0,presence_penalty=0,repetition_penalty=1;
 bool stochastic()const{return temperature>0;}
 bool penalties()const{return frequency_penalty!=0 || presence_penalty!=0 || repetition_penalty!=1;}
 bool selection_graph()const{return stochastic() || penalties();}
 bool SameGraph(const SamplingOptions& other)const{return temperature==other.temperature && top_p==other.top_p && top_k==other.top_k && frequency_penalty==other.frequency_penalty && presence_penalty==other.presence_penalty && repetition_penalty==other.repetition_penalty;}
};
inline Status ValidateSampling(const SamplingOptions& s){
 if(!std::isfinite(s.temperature) || s.temperature<0 || s.temperature>2 || (s.temperature>0 && s.temperature<0.0001f))return {StatusCode::kInvalidArgument,"temperature must be0 or finite[0.0001,2]"};
 if(!std::isfinite(s.top_p) || s.top_p<=0 || s.top_p>1)return {StatusCode::kInvalidArgument,"top_p must be finite(0,1]"};
 if(s.top_k<0 || s.top_k>98304)return {StatusCode::kInvalidArgument,"top_k must be integer0..98304 (0 disables)"};
 if(!std::isfinite(s.frequency_penalty) || std::abs(s.frequency_penalty)>2 || !std::isfinite(s.presence_penalty) || std::abs(s.presence_penalty)>2)return {StatusCode::kInvalidArgument,"frequency/presence penalty must be finite[-2,2]"};
 if(!std::isfinite(s.repetition_penalty) || s.repetition_penalty<0.01f || s.repetition_penalty>100)return {StatusCode::kInvalidArgument,"repetition_penalty must be finite[0.01,100]"};
 return {};
}
// Adapted from pinned Gem16 sampler. Counter-based; one draw per chosen token,
// including first prefill token/EOS. Thread/network scheduling cannot advance it.
inline std::uint64_t SamplingMix(std::uint64_t v){v+=0x9E3779B97F4A7C15ULL;v=(v^(v>>30))*0xBF58476D1CE4E5B9ULL;v=(v^(v>>27))*0x94D049BB133111EBULL;return v^(v>>31);}
inline double SamplingUniform(std::uint64_t seed,std::uint64_t step){return std::fmin((static_cast<double>(SamplingMix(seed^SamplingMix(step))>>11)+0.5)*(1.0/9007199254740992.0),0x1.fffffffffffffp-1);}
}
