#pragma once
#include "util/json.h"
#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace mellum16::server {
inline constexpr std::size_t kMaximumToolCalls=8,kMaximumToolJson=65536;
struct ToolConfig {std::vector<std::string> names;bool parse=false,parallel=true;};
struct ToolCall {std::string id,name,arguments;bool complete=true;};
Result<ToolConfig> ValidateTools(const json::Value&);
Status ValidateToolHistory(const json::Value::Array&);
bool ValidFunctionName(std::string_view);
// Offsets into original JSON, including preserved nested property order.
struct JsonFieldSpan{std::size_t key_begin=0,value_begin=0,value_end=0,following=0;};
Result<JsonFieldSpan> FindJsonField(std::string_view,std::string_view,bool allow_partial=false);
Result<std::string> OmitJsonField(std::string_view,std::string_view);
class ToolStreamParser {
 public:
  ToolStreamParser(bool enabled,std::size_t maximum_fragment,std::size_t maximum_calls=8);
  std::string_view Push(std::string_view);
  std::string_view Finish();
  const char* error()const{return error_;}
  std::size_t count()const{return count_+(inside_?1:0);}
  bool incomplete()const{return inside_;}
  std::string_view raw(std::size_t i)const{return raw_[i];}
 private:
  void Plain(char);void Emit(char);void Fail(const char* message){if(!error_)error_=message;}
  bool enabled_,inside_=false,in_string_=false,escaped_=false,started_=false,ended_=false;
  int depth_=0;std::size_t count_=0,maximum_calls_;
  std::array<std::string,8> raw_;std::string visible_;
  std::array<char,16> held_{};std::size_t held_size_=0;
  const char* error_=nullptr;
};
Result<ToolCall> ParseGeneratedTool(std::string_view raw,const ToolConfig&,std::string id,bool complete);
std::string ToolCallsJson(std::span<const ToolCall>);
}
