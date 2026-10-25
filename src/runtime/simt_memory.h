#pragma once
#include "runtime/server_memory.h"
namespace mellum16 {
// Single-token graph workspace, independent of prefill commit chunk size.
// All entries correspond to owned allocations in inference_simt.cu.
inline ServerMemoryPlan PlanSimtMemory(std::size_t weights,int capacity,std::size_t sampling_workspace,bool checkpoint,bool batch=true,bool fast_math=false,bool q8_head=false,bool q8_kv=false){
  if(capacity<1024 || capacity>131072)throw std::invalid_argument("invalid SIMT context capacity");
  ServerMemoryPlan plan;
  const auto add=[&](const char* name,std::size_t bytes){plan.buffers.emplace(name,bytes);plan.resident=admission::Sum({plan.resident,bytes});};
  add("weights",weights);
  if(q8_head)add("head_input",kHeadQ8InputBytes);
  add("kv",admission::Bytes(7ULL*capacity+21ULL*1024,q8_kv?1088:2048));
  if(q8_kv){
    add("q8_state",32);add("q8_query",32*4*36);add("q8_partial",32*4096*4);add("q8_lse",32*32*4);
    if(batch){add("q8_prefix",admission::Bytes(capacity,2048));add("q8_chunk",256*1088);}
  }
  add("rope",admission::Bytes(capacity,128*4*2));
  add("descriptors",28*64*3*56);
  add("scratch",(2304*3+4096*2+512*2+64+8*896*2+8*2304+98304)*4);
  add("routing",8*4);add("ids",8*4);add("selection",96*8);
  add("command",8);add("bad",4);add("control",8);
  add("encoded",4096);add("scale",4);add("transform",8*2304*4);
  add("silu",65536*2);add("attention_scores",admission::Bytes(capacity,32*4));
  add("prompt",admission::Bytes(capacity,4));
  add("sampling",98304*28+16);add("sampling_workspace",sampling_workspace);
  if(checkpoint)add("local_checkpoint",KVCheckpointLedger::kLocalSnapshotBytes);
  if(fast_math)add("attention_partials",4*4096*4);
  if(fast_math && batch){
    add("sgemm_weights",4096ULL*2304*4);
    add("sgemm_inputs",256ULL*4096*4);
    add("sgemm_workspace",16ULL*1024*1024);
  }
  if(batch){
    add("batch_state",12);
    add("batch_activations",256ULL*(2304*3+4096*2+512*2+64)*4);
    add("batch_experts",256ULL*8*(896*2+2304)*4);
    add("batch_transform",256ULL*8*2304*4);
    add("batch_encoded",256ULL*4096);add("batch_scales",256*4);
    add("batch_kv",256ULL*512*2*2);
    // Pascal SGEMM Attention batches 256 queries for one GQA group at a time.
    add("batch_scores",admission::Bytes(fast_math && capacity<=16384?std::max(capacity,1280):capacity,fast_math && capacity<=16384?8*(capacity>8192?64:256)*4:8*32*4));
    add("batch_ids",256*8*4);add("batch_routing",256*8*4);
    add("batch_groups",64*256*4);add("batch_counts",64*4);
    add("batch_tiles",319*8);add("batch_tile_count",4);
  }
  return plan;
}
}
