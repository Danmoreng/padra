#include "server/http_server.h"
#include "server/http_policy.h"
#include "server/secure_id.h"
#include "util/json.h"
#include "compiler/sha256.h"
#include "server/responses.h"
#include "server/response_stream.h"
#include "runtime/size_checked.h"
#include <chrono>
#include <sstream>
#ifndef _WIN32
#include <sys/socket.h>
#endif
namespace mellum16::server {
namespace {
Result<std::string> SessionOwner(const httplib::Request& request){
 if(request.get_header_value_count("Authorization")>1)return Status(StatusCode::kInvalidArgument,"duplicate Authorization namespace header");
 const auto auth=request.get_header_value("Authorization");if(auth.size()>512 || (!auth.empty() && (!auth.starts_with("Bearer ") || auth.size()==7)))return Status(StatusCode::kInvalidArgument,"invalid loopback credential namespace");
 const auto value=auth.empty()?std::string("anonymous-loopback"):auth;return compiler::Sha256Hex(value.data(),value.size());
}
}
void HttpServer::Error(httplib::Response& response,int status,std::string_view message,std::string_view code){response.status=status;response.set_content(ErrorJson(message,code,status>=500?"server_error":"invalid_request_error"),"application/json");}
HttpServer::HttpServer(Coordinator& coordinator,const text::Tokenizer& tokenizer,int port,int capacity):coordinator_(coordinator),tokenizer_(tokenizer),port_(port),capacity_(capacity){
 http_.new_task_queue=[] {return new httplib::ThreadPool(16,16,32);};
 if(coordinator_.RuntimeMetrics().capacity!=capacity_)throw std::invalid_argument("HTTP/runtime context capacity mismatch");
 http_.set_payload_max_length(RequestBodyLimit(capacity_));http_.set_read_timeout(5);http_.set_write_timeout(2);http_.set_keep_alive_timeout(2);http_.set_keep_alive_max_count(16);http_.set_tcp_nodelay(true);
#ifndef _WIN32
 http_.set_socket_options([](socket_t socket){int bytes=16384;(void)setsockopt(socket,SOL_SOCKET,SO_SNDBUF,&bytes,sizeof(bytes));});
#endif
 http_.set_pre_routing_handler([this](const httplib::Request& request,httplib::Response& response){auto error=ValidateLocalHttpRequest(request,port_);if(!error.empty()){Error(response,403,error,"http_policy");return httplib::Server::HandlerResponse::Handled;}return httplib::Server::HandlerResponse::Unhandled;});
 http_.set_error_handler([this](const httplib::Request&,httplib::Response& response){if(response.get_header_value("Content-Type").find("application/json")!=0)Error(response,response.status,"endpoint/request unsupported",response.status==404?"not_found":"http_error");});
 http_.set_exception_handler([this](const httplib::Request&,httplib::Response& response,std::exception_ptr error){try{if(error)std::rethrow_exception(error);}catch(const std::exception& e){Error(response,500,e.what(),"server_error");}});
 http_.Get("/healthz",[](const httplib::Request&,httplib::Response& response){response.set_content("{\"status\":\"alive\"}","application/json");});
 http_.Get("/readyz",[this](const httplib::Request&,httplib::Response& response){auto state=coordinator_.Metrics();response.status=state.ready?200:503;response.set_content(std::string("{\"ready\":")+(state.ready?"true":"false")+",\"quality_status\":\"unqualified\"}","application/json");});
 http_.Get("/v1/models",[this](const httplib::Request&,httplib::Response& response){
  const auto r=coordinator_.RuntimeMetrics();std::ostringstream out;
  out<<"{\"object\":\"list\",\"data\":[{\"id\":"<<json::Quote(kModelId)<<",\"object\":\"model\",\"created\":0,\"owned_by\":\"mellum16\",\"context_capacity\":"<<capacity_
     <<",\"quality_status\":\"unqualified\",\"sampling_profiles\":[\"greedy\",\"gpu-graph-parameters-penalties-splitmix64-v2\"],\"kv_format\":"<<json::Quote(KVCacheLayout::Make(r.kv_format,1).name())
     <<",\"execution_profile\":"<<json::Quote(ServerExecutionProfile(capacity_,true,r.kv_format,r.prefill_chunk,r.device_budget_gib,r.checkpoint_enabled))<<",\"prefill_chunk\":"<<r.prefill_chunk<<",\"device_budget_gib\":"<<r.device_budget_gib
     <<",\"active_gpu_sessions\":1,\"logical_session_limit\":32,\"session_ttl_seconds\":1800,\"native_kv_reuse\":"<<(r.checkpoint_enabled && r.model_loads?"true":"false")<<",\"checkpoint_snapshot_bytes\":"<<r.checkpoint_bytes
     <<",\"tokenizer_sha256\":"<<json::Quote(tokenizer_.sha256())<<",\"checkpoint_sha256\":\"e614375f2eb26c17321365d4bd8ee7aa4b716f6badee171bb2040b85b754804c\",\"manifest_sha256\":\"0f30d4ead6ae3d0683ec1eb9ad7af1adfebe343d804e3313e8c01d6c31f76f0b\",\"math_profile\":"<<json::Quote(r.math_profile)<<",\"test_backend\":"<<(r.model_loads==0?"true":"false")<<"}]}";
  response.set_content(out.str(),"application/json");});
 http_.Get("/metrics",[this](const httplib::Request&,httplib::Response& response){auto m=coordinator_.Metrics();auto r=coordinator_.RuntimeMetrics();std::ostringstream out;
  out<<"mellum_requests_submitted_total "<<m.submitted<<"\nmellum_requests_completed_total "<<m.completed<<"\nmellum_requests_failed_total "<<m.failed<<"\nmellum_requests_cancelled_total "<<m.cancelled<<"\nmellum_queue_rejected_total "<<m.rejected<<"\nmellum_disconnects_total "<<m.disconnects<<"\nmellum_backpressure_total "<<m.backpressure<<"\nmellum_queue_requests "<<m.queued<<"\nmellum_queue_bytes "<<m.queued_bytes<<"\nmellum_queue_tokens "<<m.queued_tokens<<"\nmellum_active_requests "<<m.active<<"\nmellum_ready "<<m.ready<<"\nmellum_model_loads_total "<<r.model_loads<<"\nmellum_weight_uploads_total "<<r.weight_uploads<<"\nmellum_graph_captures_total "<<r.graph_captures<<"\nmellum_resident_device_bytes "<<r.resident_bytes<<"\nmellum_required_device_bytes "<<r.required_bytes<<"\nmellum_post_graph_available_device_bytes "<<r.post_graph_available_bytes<<"\nmellum_admission_available_device_bytes "<<r.admission_available_bytes<<"\nmellum_device_budget_gib "<<r.device_budget_gib<<"\nmellum_prefill_chunk "<<r.prefill_chunk<<"\nmellum_tokenization_seconds_total "<<m.tokenize_seconds<<"\nmellum_queue_wait_seconds_total "<<m.queue_seconds<<"\nmellum_prefill_seconds_total "<<m.prefill_seconds<<"\nmellum_decode_seconds_total "<<m.decode_seconds<<"\nmellum_first_token_seconds_total "<<m.first_token_seconds<<"\nmellum_first_visible_seconds_total "<<m.visible_seconds<<'\n';response.set_content(out.str(),"text/plain; version=0.0.4");});
 http_.Post("/v1/chat/completions",[this](const httplib::Request& request,httplib::Response& response){Chat(request,response);});
 http_.Post("/v1/sessions",[this](const httplib::Request& request,httplib::Response& response){Session(request,response,"create");});
 http_.Get(R"(/v1/sessions/(sess-[0-9a-f]{32}))",[this](const httplib::Request& request,httplib::Response& response){Session(request,response,"get");});
 http_.Delete(R"(/v1/sessions/(sess-[0-9a-f]{32}))",[this](const httplib::Request& request,httplib::Response& response){Session(request,response,"delete");});
 http_.Post("/v1/responses",[this](const httplib::Request& request,httplib::Response& response){Responses(request,response);});
 http_.Get(R"(/v1/responses/(resp_[0-9a-f]{32}))",[this](const httplib::Request& request,httplib::Response& response){StoredResponses(request,response,false);});
 http_.Delete(R"(/v1/responses/(resp_[0-9a-f]{32}))",[this](const httplib::Request& request,httplib::Response& response){StoredResponses(request,response,true);});
}
bool HttpServer::Listen(){return http_.listen("127.0.0.1",port_);}
void HttpServer::Stop(){coordinator_.BeginDrain();http_.stop();}
void HttpServer::StoredResponses(const httplib::Request& request,httplib::Response& response,bool remove){
 auto owner=SessionOwner(request);if(!owner.ok()){Error(response,400,owner.status().message());return;}const auto id=request.matches[1].str();response.set_header("Cache-Control","no-store");
 if(remove){auto deleted=responses_->Delete(id,owner.value());if(!deleted.ok()){Error(response,404,deleted.message(),"response_not_found");return;}response.set_content("{\"id\":"+json::Quote(id)+",\"object\":\"response.deleted\",\"deleted\":true}","application/json");return;}
 auto record=responses_->Get(id,owner.value());if(!record.ok()){Error(response,404,record.status().message(),"response_not_found");return;}response.set_content(record.value()->envelope,"application/json");
}
void HttpServer::Responses(const httplib::Request& request,httplib::Response& response){
 auto owner=SessionOwner(request);if(!owner.ok()){Error(response,400,owner.status().message());return;}
 std::shared_ptr<const StoredResponse> parent;
 // First validate shape before resolving an owned previous ID. No GPU mutation.
 auto raw=json::Parse(request.body);if(!raw.ok() || !raw.value().is_object()){Error(response,400,"Responses body must be JSON object");return;}
 if(auto* previous=raw.value().find("previous_response_id");previous && !previous->is_null()){
  if(!previous->is_string()){Error(response,400,"previous_response_id must be string or null");return;}auto stored=responses_->Get(previous->as_string(),owner.value());if(!stored.ok()){Error(response,404,stored.status().message(),"response_not_found");return;}parent=std::move(stored).value();
 }
 auto parsed=ParseResponseRequest(request.body,parent?parent->history:"[]");if(!parsed.ok()){Error(response,parsed.status().code()==StatusCode::kNotFound?404:parsed.status().code()==StatusCode::kResourceExhausted?413:400,parsed.status().message());return;}
 auto spec=std::make_shared<ResponseRequest>(std::move(parsed).value());std::shared_ptr<ResponseRoot> root=parent?parent->root:nullptr;
 if(root && !coordinator_.Sessions().Get(root->session_id,root->owner).ok())root.reset();
 if(!root && spec->store){auto session=coordinator_.Sessions().Create(owner.value());if(!session.ok()){Error(response,429,session.status().message(),"session_unavailable");return;}root=std::make_shared<ResponseRoot>();root->session_id=session.value().id;root->owner=owner.value();root->discard=[coordinator=&coordinator_,id=root->session_id,who=root->owner]{(void)coordinator->DiscardSession(id,who);};}
 if(root){const auto end=spec->chat_body.find_last_not_of(" \t\r\n");spec->chat_body.insert(end,",\"session_id\":"+json::Quote(root->session_id));}
 auto chat=ParseChatRequest(spec->chat_body,tokenizer_,capacity_,spec->id);if(!chat.ok()){Error(response,chat.status().code()==StatusCode::kResourceExhausted?413:400,chat.status().message());return;}
 chat.value().session_owner=owner.value();chat.value().adapter_bytes=admission::Sum({sizeof(ResponseRequest),spec->chat_body.size(),spec->history.size(),spec->tools.size(),spec->metadata.size(),spec->instructions.size(),2048});
 spec->outputs=chat.value().outputs;spec->temperature=chat.value().sampling.temperature;spec->top_p=chat.value().sampling.top_p;
 std::shared_ptr<ResponseReservation> reservation;
 if(spec->store){const auto budget=admission::Sum({admission::Bytes(spec->chat_body.size(),2),admission::Bytes(chat.value().outputs,admission::Bytes(admission::Sum({tokenizer_.MaxRawTokenBytes(),3}),18)),131072});auto reserved=responses_->Reserve(spec->id,owner.value(),budget);if(!reserved.ok()){Error(response,429,reserved.status().message(),"response_store_full");return;}reservation=std::move(reserved).value();}
 auto admitted=coordinator_.Submit(std::move(chat).value());if(!admitted.ok()){Error(response,admitted.status().code()==StatusCode::kNotFound?404:503,admitted.status().message(),"queue_full");return;}auto job=std::move(admitted).value();response.set_header("X-Request-Id",spec->id);response.set_header("Cache-Control","no-store");
 const auto retain=[store=responses_,reservation,spec,root](const Job& job)->Status {if(!reservation)return {};return store->Commit(reservation,ResponseEnvelope(*spec,&job),ResponseCompletedHistory(*spec,job),root);};
 if(!spec->stream){
  std::unique_lock lock(job->mutex);while(!job->complete){job->changed.wait_for(lock,std::chrono::milliseconds(20));if(request.is_connection_closed()){job->Cancel(CancelCause::Disconnect);break;}}
  if(!job->complete){Error(response,499,"client disconnected","cancelled");return;}if(!job->failure.ok()){Error(response,job->cancellation.load()==CancelCause::Deadline?504:500,job->failure.message(),"generation_error");return;}
  auto stored=retain(*job);if(!stored.ok()){Error(response,500,stored.message(),"response_store_error");return;}response.set_header("X-Mellum-Cached-Tokens",std::to_string(job->cached_tokens));response.set_header("X-Mellum-Prefill-Ms",std::to_string(job->prefill_seconds*1000));response.set_header("X-Mellum-Decode-Ms",std::to_string(job->decode_seconds*1000));response.set_content(ResponseEnvelope(*spec,job.get()),"application/json");return;
 }
 response.set_header("X-Accel-Buffering","no");
 response.set_chunked_content_provider("text/event-stream",[job,spec,root,reservation,retain,closed=request.is_connection_closed](std::size_t,httplib::DataSink& sink){
  try{
   ResponseStream events(*spec,[&sink](std::string_view frame){return sink.write(frame.data(),frame.size());});if(!events.Begin()){job->Cancel(CancelCause::Disconnect);return false;}EventFrame frame;
   for(;;){if(closed()){job->Cancel(CancelCause::Disconnect);return false;}if(job->Pop(frame,std::chrono::milliseconds(20))){if(!events.ChatFrame(std::string_view(frame.data.data(),frame.size))){job->Cancel(CancelCause::Disconnect);return false;}continue;}
    std::unique_lock lock(job->mutex);if(job->complete && !job->count){if(job->failure.ok()){auto stored=retain(*job);if(!stored.ok()){if(!events.Fail(stored.message()))return false;sink.done();return true;}}if(!events.Finish(*job)){job->Cancel(CancelCause::Disconnect);return false;}sink.done();return true;}
   }
  }catch(...){job->Cancel(CancelCause::Disconnect);return false;}
 },[job](bool success){if(!success)job->Cancel(CancelCause::Disconnect);});
}
void HttpServer::Session(const httplib::Request& request,httplib::Response& response,std::string_view operation){
 auto owner=SessionOwner(request);if(!owner.ok()){Error(response,400,owner.status().message());return;}
 auto& store=coordinator_.Sessions();
 const auto id=operation=="create"?std::string():request.matches[1].str();
 if(operation=="delete"){auto status=coordinator_.DiscardSession(id,owner.value());if(!status.ok()){Error(response,404,status.message(),"session_not_found");return;}response.set_content("{\"id\":"+json::Quote(id)+",\"object\":\"mellum.session.deleted\",\"deleted\":true}","application/json");return;}
 Result<SessionSnapshot> snapshot=Status{StatusCode::kInternal,"session operation not completed"};
 if(operation=="create"){
  auto body=json::Parse(request.body);if(!body.ok() || !body.value().is_object()){Error(response,400,"session creation requires JSON object");return;}
  auto* model=body.value().find("model");if(!model || !model->is_string() || model->as_string()!=kModelId || body.value().as_object().size()!=1){Error(response,400,"session creation supports exactly the selected model field");return;}
  snapshot=store.Create(owner.value());
 }else snapshot=store.Get(id,owner.value());
 if(!snapshot.ok()){Error(response,snapshot.status().code()==StatusCode::kNotFound?404:429,snapshot.status().message(),"session_unavailable");return;}
 const auto& s=snapshot.value();response.set_header("Cache-Control","no-store");
 response.set_content("{\"id\":"+json::Quote(s.id)+",\"object\":\"mellum.session\",\"model\":"+json::Quote(kModelId)+",\"ttl_seconds\":"+std::to_string(store.ttl_seconds())+",\"active\":"+(s.active?"true":"false")+",\"stored_history_bytes\":"+std::to_string(s.bytes)+",\"kv_cache_available\":"+(coordinator_.SessionCacheAvailable(s.id,owner.value())?"true":"false")+",\"messages\":"+s.history+"}","application/json");
}
void HttpServer::Chat(const httplib::Request& request,httplib::Response& response){
 auto id=MakeSecureId("chatcmpl-");if(!id.ok()){Error(response,500,id.status().message(),"id_failure");return;}
 response.set_header("X-Request-Id",id.value());
 auto parsed=ParseChatRequest(request.body,tokenizer_,capacity_,std::move(id).value());if(!parsed.ok()){Error(response,parsed.status().code()==StatusCode::kNotFound?404:parsed.status().code()==StatusCode::kResourceExhausted?413:400,parsed.status().message());return;}
 if(!parsed.value().session_id.empty()){auto owner=SessionOwner(request);if(!owner.ok()){Error(response,400,owner.status().message());return;}parsed.value().session_owner=std::move(owner).value();}
 const bool stream=parsed.value().stream;auto admitted=coordinator_.Submit(std::move(parsed).value());if(!admitted.ok()){response.set_header("Retry-After","1");Error(response,admitted.status().code()==StatusCode::kNotFound?404:503,admitted.status().message(),admitted.status().code()==StatusCode::kNotFound?"session_not_found":"queue_full");return;}
 auto job=std::move(admitted).value();response.set_header("X-Request-Id",job->request.id);response.set_header("Cache-Control","no-cache");
 if(!stream){
  std::unique_lock lock(job->mutex);
  while(!job->complete){job->changed.wait_for(lock,std::chrono::milliseconds(20));if(request.is_connection_closed()){job->Cancel(CancelCause::Disconnect);break;}}
  if(!job->complete){Error(response,499,"client disconnected","cancelled");return;}
  if(!job->failure.ok()){const auto cause=job->cancellation.load();Error(response,cause==CancelCause::Deadline?504:cause==CancelCause::QueueDeadline || cause==CancelCause::Shutdown?503:500,job->failure.message(),"generation_error");return;}
  response.set_header("X-Mellum-Queue-Ms",std::to_string(job->queue_seconds*1000));response.set_header("X-Mellum-Tokenization-Ms",std::to_string(job->request.tokenize_seconds*1000));response.set_header("X-Mellum-Prefill-Ms",std::to_string(job->prefill_seconds*1000));response.set_header("X-Mellum-Decode-Ms",std::to_string(job->decode_seconds*1000));response.set_header("X-Mellum-First-Token-Ms",std::to_string(job->first_token_seconds*1000));response.set_header("X-Mellum-First-Visible-Ms",std::to_string(job->visible_seconds*1000));
  response.set_header("X-Mellum-Cached-Tokens",std::to_string(job->cached_tokens));
  response.set_content(CompletionJson(*job),"application/json");return;
 }
 response.set_header("X-Accel-Buffering","no");
 response.set_chunked_content_provider("text/event-stream",[job,closed=request.is_connection_closed](std::size_t,httplib::DataSink& sink){
  try{
   EventFrame frame;
   for(;;){
    if(closed()){job->Cancel(CancelCause::Disconnect);return false;}
    if(job->Pop(frame,std::chrono::milliseconds(20))){if(!sink.write(frame.data.data(),frame.size)){job->Cancel(CancelCause::Disconnect);return false;}continue;}
    {std::lock_guard lock(job->mutex);if(job->complete && job->count==0){sink.done();return true;}}
   }
  }catch(...){job->Cancel(CancelCause::Disconnect);return false;}
 },[job](bool success){if(!success)job->Cancel(CancelCause::Disconnect);});
}
}
