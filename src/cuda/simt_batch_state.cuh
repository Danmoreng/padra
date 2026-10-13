#pragma once
struct SimtBatch {int base,rows,capacity;};
inline constexpr int kSimtBatch=256;
__global__ void SetSimtBatch(SimtBatch* state,int base,int rows,int capacity,int* bad){if(!threadIdx.x){*state={base,rows,capacity};if(base<0 || rows<1 || rows>kSimtBatch || base>capacity-rows)atomicExch(bad,1);}}
