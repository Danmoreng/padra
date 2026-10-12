#pragma once
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "mellum16/status.h"
namespace mellum16::text {
class Tokenizer {
 public:
  static Result<std::unique_ptr<Tokenizer>> Load(const std::filesystem::path&,const std::string& expected_sha256);
  ~Tokenizer();
  Tokenizer(const Tokenizer&)=delete;Tokenizer& operator=(const Tokenizer&)=delete;
  Result<std::vector<int>> Encode(std::string_view)const;
  Result<std::string> Decode(std::span<const int>,bool skip_special=true)const;
  std::string_view RawToken(int id,bool skip_special=true)const;
  std::size_t MaxRawTokenBytes()const{return max_raw_;}
  const std::string& sha256()const{return sha256_;}
 private:
  Tokenizer()=default;void* handle_=nullptr;std::vector<std::string> raw_;std::vector<bool> special_;std::size_t max_raw_=0;std::string sha256_;
};
Result<std::string> RenderChat(std::string_view request_json);
// Model byte decoder + native UTF8 lossy semantics, with at most3 incomplete
// bytes held between tokens. Output storage is reserved before generation.
class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer&,bool skip_special=true);
  std::string_view Push(int id);
  std::string_view Finish();
 private:
  std::string_view Process(bool final);
  const Tokenizer& tokenizer_;bool skip_special_;std::string pending_,output_;
};
}
