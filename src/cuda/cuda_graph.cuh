#pragma once
// Capture/instantiate/upload once; replay allocates nothing. No silent fallback.
struct DecodeGraph {
  cudaGraphExec_t executable=nullptr;std::size_t nodes=0;
  ~DecodeGraph(){if(executable)cudaGraphExecDestroy(executable);}
  void Reset(){if(executable){Check(cudaGraphExecDestroy(executable));executable=nullptr;}}
  DecodeGraph()=default;DecodeGraph(const DecodeGraph&)=delete;
  template<class Launcher> void Capture(cudaStream_t stream,Launcher&& launch){
    Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));cudaGraph_t graph=nullptr;
    try{launch();}catch(...){cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);throw;}
    const auto ended=cudaStreamEndCapture(stream,&graph);
    if(ended!=cudaSuccess){if(graph)cudaGraphDestroy(graph);Check(ended);}
    Check(cudaGraphGetNodes(graph,nullptr,&nodes));
    const auto created=cudaGraphInstantiate(&executable,graph,0);cudaGraphDestroy(graph);Check(created);
    Check(cudaGraphUpload(executable,stream));Check(cudaStreamSynchronize(stream));
  }
  void Launch(cudaStream_t stream){if(!executable)throw std::runtime_error("decode graph not prepared");Check(cudaGraphLaunch(executable,stream));}
};
