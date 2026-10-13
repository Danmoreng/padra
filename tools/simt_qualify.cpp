#include "runtime/model_runtime.h"
#include "util/json.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace mellum16 {Result<std::unique_ptr<RuntimeBackend>> LoadCudaModelRuntime(InferenceOptions);}
int main(int argc,char** argv){try{
  if(argc!=5)throw std::runtime_error("usage: padra-simt-qualify ARTIFACT TOKENIZER_SHA256 CONFIG_JSON OUTPUT_DIRECTORY");
  std::ifstream input(argv[3]);std::string source{std::istreambuf_iterator<char>(input),{}};
  auto config=mellum16::json::Parse(source);if(!config.ok())throw std::runtime_error(config.status().message());
  const auto& root=config.value();
  auto flag=[&](const char* name){auto* p=root.find(name);return p && p->as_bool();};
  auto number=[&](const char* name,int fallback){auto* p=root.find(name);return p?static_cast<int>(p->as_integer()):fallback;};
  auto ids=[](const mellum16::json::Value* value){std::vector<int> result;if(value)for(const auto& v:value->as_array())result.push_back(static_cast<int>(v.as_integer()));return result;};
  auto o=mellum16::SelectedServerOptions(argv[1],argv[2]);o.capacity=number("capacity",8192);o.prefill_chunk=number("chunk",256);o.server_device_budget_gib=8;o.controlled_chunk_graphs=true;
  o.input_ids={0};o.generate=0;o.capture_logits=true;o.simt_eager=flag("eager");o.simt_reference_codec=flag("reference_codec");o.simt_reference_prefill=flag("reference_prefill");
  const std::filesystem::path output(argv[4]);std::filesystem::create_directories(output);
  if(flag("diagnostics"))o.simt_diagnostics=output/"tensors";
  o.simt_diagnostic_layer=number("layer",0);o.simt_diagnostic_positions=ids(root.find("positions"));o.simt_teacher_tokens=ids(root.find("teacher_tokens"));
  auto loaded=mellum16::LoadCudaModelRuntime(o);if(!loaded.ok())throw std::runtime_error(loaded.status().message());
  mellum16::ModelRuntime runtime(std::move(loaded).value());std::ofstream report(output/"results.jsonl");int index=0;
  const auto* requests=root.find("requests");if(!requests)throw std::runtime_error("configuration requires requests array");
  for(const auto& request:requests->as_array()){
    auto prompt=ids(request.find("ids"));const int generate=static_cast<int>(request.find("generate")->as_integer());
    if(!o.simt_teacher_tokens.empty() && o.simt_teacher_tokens.size()<static_cast<std::size_t>(std::max(0,generate-1)))throw std::runtime_error("too few teacher tokens");
    auto status=runtime.PrepareRequestShape(prompt,generate,true);if(!status.ok())throw std::runtime_error(status.message());
    const auto started=std::chrono::steady_clock::now();
    auto result=runtime.Generate({},[](int count){std::cerr<<"committed "<<count<<'\n';return true;});if(!result.ok())throw std::runtime_error(result.status().message());
    const auto& r=result.value();const auto stats=runtime.Stats();
    const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    report<<"{\"request\":"<<index<<",\"prefill_tokens\":"<<prompt.size()<<",\"generated_ids\":[";
    for(std::size_t i=0;i<r.generated_ids.size();++i){if(i)report<<',';report<<r.generated_ids[i];}
    report<<"],\"decode_steps\":"<<r.decode_steps<<",\"prefill_seconds\":"<<r.prefill_seconds<<",\"decode_seconds\":"<<r.decode_seconds<<",\"wall_seconds\":"<<wall<<",\"math_profile\":"<<mellum16::json::Quote(r.math_profile)<<",\"kernel_profile\":"<<mellum16::json::Quote(r.kernel_profile)<<",\"resident_bytes\":"<<stats.resident_bytes<<",\"post_graph_free_bytes\":"<<stats.post_graph_available_bytes<<"}\n";report.flush();
    std::ofstream logits(output/("logits-"+std::to_string(index++)+".f32"),std::ios::binary);logits.write(reinterpret_cast<const char*>(r.logits.data()),r.logits.size()*4);
    std::cerr<<"prefill="<<r.prefill_seconds<<" decode="<<r.decode_seconds<<" generated="<<r.generated_ids.size()<<'\n';
    if(!report || !logits)throw std::runtime_error("cannot write qualification results");
    status=runtime.Reset();if(!status.ok())throw std::runtime_error(status.message());
  }
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
