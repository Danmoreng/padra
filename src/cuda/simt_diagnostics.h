#pragma once
// Explicit offline qualification only. Never enabled by the HTTP server.
struct SimtDiagnostics {
  struct Timing {cudaEvent_t begin=nullptr,end=nullptr;std::string name;int layer=0;};
  std::vector<Timing> timings;std::filesystem::path directory;
  bool active=false,capture_tensors=true;int position=0,rows=1;std::string phase="decode";std::size_t cursor=0;
  explicit SimtDiagnostics(const InferenceOptions& options){
    capture_tensors=options.simt_capture_tensors;
    if(options.simt_diagnostics.empty())return;
    timings.resize(28*12+4);
    for(auto& t:timings){Check(cudaEventCreate(&t.begin));Check(cudaEventCreate(&t.end));}
  }
  ~SimtDiagnostics(){for(auto& t:timings){if(t.begin)cudaEventDestroy(t.begin);if(t.end)cudaEventDestroy(t.end);}}
  template<class F> void Measure(const char* name,int layer,cudaStream_t stream,F&& launch){
    if(!active){launch();return;}
    if(cursor>=timings.size())throw std::runtime_error("SIMT diagnostic timing inventory exhausted");
    auto& t=timings[cursor++];t.name=name;t.layer=layer;Check(cudaEventRecord(t.begin,stream));launch();Check(cudaEventRecord(t.end,stream));
  }
  void Dump(const char* name,const void* source,std::size_t bytes,cudaStream_t stream){
    if(!active || !capture_tensors)return;
    std::vector<unsigned char> data(bytes);Check(cudaMemcpyAsync(data.data(),source,bytes,cudaMemcpyDeviceToHost,stream));Check(cudaStreamSynchronize(stream));
    std::ofstream out(directory/("position-"+std::to_string(position)+"-"+name+".bin"),std::ios::binary);out.write(reinterpret_cast<const char*>(data.data()),data.size());if(!out)throw std::runtime_error("cannot write SIMT diagnostic tensor");
  }
  void Flush(cudaStream_t stream){
    if(!active)return;Check(cudaStreamSynchronize(stream));
    std::ofstream out(directory/"operators.jsonl",std::ios::app);
    for(std::size_t i=0;i<cursor;++i){const auto& t=timings[i];float ms=0;Check(cudaEventElapsedTime(&ms,t.begin,t.end));out<<"{\"position\":"<<position<<",\"phase\":"<<("\""+phase+"\"")<<",\"rows\":"<<rows<<",\"layer\":"<<t.layer<<",\"operator\":\""<<t.name<<"\",\"milliseconds\":"<<ms<<"}\n";}
    if(!out)throw std::runtime_error("cannot write SIMT diagnostic timings");cursor=0;active=false;
  }
};
