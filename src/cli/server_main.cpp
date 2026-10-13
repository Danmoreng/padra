#include "server/http_server.h"
#include "runtime/model_runtime.h"
#include "runtime/gpu_build_config.h"
#include "text/tokenizer.h"
#include "util/json.h"
#include <charconv>
#include <array>
#include <csignal>
#include <fstream>
#include <iostream>
#include <thread>
namespace {volatile std::sig_atomic_t interrupted=0;void Signal(int){interrupted=1;}}
int main(int argc,char** argv){try{
 if(argc<5 || argc>9)throw std::runtime_error("usage: padra-server ARTIFACT TOKENIZER_JSON TOKENIZER_SHA256 PORT [GENERATION_TIMEOUT_MS] [CONTEXT_CAPACITY] [PREFILL_CHUNK] [DEVICE_BUDGET_GIB]");
 int port=0;auto [end,error]=std::from_chars(argv[4],argv[4]+std::char_traits<char>::length(argv[4]),port);if(error!=std::errc{} || *end || port<1024 || port>65535)throw std::runtime_error("port must be1024..65535");
 mellum16::server::ServerLimits limits;
 if(argc>5){int milliseconds=0;auto [stop,code]=std::from_chars(argv[5],argv[5]+std::char_traits<char>::length(argv[5]),milliseconds);if(code!=std::errc{} || *stop || milliseconds<1 || milliseconds>600000)throw std::runtime_error("generation timeout must be1..600000ms");limits.generation_timeout=std::chrono::milliseconds(milliseconds);}
 int capacity=mellum16::kDefaultServerContext;if(argc<=5)limits.generation_timeout=std::chrono::milliseconds(600000);
 if(argc>6){auto [stop,code]=std::from_chars(argv[6],argv[6]+std::char_traits<char>::length(argv[6]),capacity);if(code!=std::errc{} || *stop)throw std::runtime_error("invalid context capacity integer");}
 int chunk=mellum16::kDefaultServerChunk,budget=0;
 for(int argument=7;argument<argc;++argument){int& value=argument==7?chunk:budget;auto [stop,code]=std::from_chars(argv[argument],argv[argument]+std::char_traits<char>::length(argv[argument]),value);if(code!=std::errc{} || *stop)throw std::runtime_error("invalid explicit chunk/device budget");}
 if(!mellum16::ValidServerChunk(chunk) || !mellum16::ValidServerBudget(budget))throw std::runtime_error("server requires chunk2048/512/256 and budget0/8/12GiB");
 auto context=mellum16::AdmitNativeContext(capacity);if(!context.ok())throw std::runtime_error(context.status().message());limits.queued_tokens=static_cast<std::size_t>(capacity)*8;
 auto text=mellum16::text::Tokenizer::Load(argv[2],argv[3]);if(!text.ok())throw std::runtime_error(text.status().message());auto tokenizer=std::move(text).value();
 std::unique_ptr<mellum16::ModelRuntime> runtime;
 auto loaded=mellum16::ModelRuntime::Load(argv[1],tokenizer->sha256(),true,context.value(),mellum16::KVCacheFormat::BF16,-1,chunk,budget,true);if(!loaded.ok())throw std::runtime_error(loaded.status().message());runtime=std::move(loaded).value();
 // Prepare startup graphs without claiming model inference; discard request.
 const std::array<int,1> primer{0};auto prepared=runtime->PrepareRequestShape(primer,0);if(!prepared.ok())throw std::runtime_error(prepared.message());auto reset=runtime->Reset();if(!reset.ok())throw std::runtime_error(reset.message());
 mellum16::server::Coordinator coordinator(std::move(runtime),*tokenizer,limits);mellum16::server::HttpServer http(coordinator,*tokenizer,port,capacity);
 std::signal(SIGINT,Signal);std::signal(SIGTERM,Signal);std::atomic<bool> finished{false};
 std::jthread shutdown([&](std::stop_token stop){while(!stop.stop_requested() && !finished.load() && !interrupted)std::this_thread::sleep_for(std::chrono::milliseconds(20));if(interrupted)http.Stop();});
 std::cout<<"{\"listener\":"<<mellum16::json::Quote("127.0.0.1:"+std::to_string(port))<<",\"model\":\"mellum21-exl3-k4-research\",\"quality_status\":\"unqualified\"}"<<std::endl;
 const bool listened=http.Listen();finished.store(true);shutdown.join();coordinator.Shutdown();if(!listened)throw std::runtime_error("loopback listener failed");return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
