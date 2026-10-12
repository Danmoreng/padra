#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include "size_checked.h"
namespace mellum16 {
// Explicit owner-selected memory classes. Zero means actual device admission;
// a ceiling constrains the actual device, it does not emulate another GPU/SM.
inline bool ValidServerChunk(int chunk){return chunk==2048 || chunk==512 || chunk==256;}
inline bool ValidServerBudget(int gib){return gib==0 || gib==8 || gib==12;}
inline std::array<int,2> ServerStartupShapes(int chunk){
 if(!ValidServerChunk(chunk))throw std::invalid_argument("unsupported server chunk; select2048,512 or256");
 return {chunk,4000%chunk};
}
inline std::size_t ServerAvailableBytes(std::size_t free,std::size_t total,int budget_gib){
 if(free>total || !ValidServerBudget(budget_gib))throw std::invalid_argument("invalid device memory admission");
 if(!budget_gib)return free;
 const auto ceiling=admission::Bytes(static_cast<std::size_t>(budget_gib),1024ULL*1024*1024);
 const auto used=total-free;
 return std::min(free,used>=ceiling?std::size_t{0}:ceiling-used);
}
}
