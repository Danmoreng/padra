#include "text/tokenizer.h"
#include "compiler/sha256.h"
#include "util/json.h"
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#if MELLUM16_NATIVE_TEXT
extern "C" {
struct NativeIds{std::uint32_t* data;std::size_t size,capacity;};
struct NativeBytes{unsigned char* data;std::size_t size,capacity;};
int mellum_tokenizer_create(const unsigned char*,std::size_t,void**,char*,std::size_t);
void mellum_tokenizer_destroy(void*);
int mellum_tokenizer_encode(const void*,const unsigned char*,std::size_t,NativeIds*,char*,std::size_t);
int mellum_tokenizer_decode(const void*,const std::uint32_t*,std::size_t,bool,NativeBytes*,char*,std::size_t);
void mellum_tokenizer_free_ids(NativeIds);
void mellum_text_free_bytes(NativeBytes);
int mellum_render_chat(const unsigned char*,std::size_t,NativeBytes*,char*,std::size_t);
}
#endif
namespace mellum16::text {
namespace {
#if MELLUM16_NATIVE_TEXT
Status NativeFailure(const char* message){return {StatusCode::kInvalidArgument,message};}
#endif
std::array<int,512> ByteMap(){std::array<int,512> map;map.fill(-1);int next=256;for(int i=0;i<256;++i){const bool direct=(i>=33&&i<=126)||(i>=161&&i<=172)||(i>=174);map[direct?i:next++]=i;}return map;}
[[maybe_unused]] std::string Raw(std::string_view text){
 static const auto map=ByteMap();std::string result;
 for(std::size_t i=0;i<text.size();){unsigned char lead=text[i];std::uint32_t code=lead;std::size_t n=1;
  if(lead>=0xc2 && lead<=0xdf){code=lead&31;n=2;}else if(lead>=0xe0 && lead<=0xef){code=lead&15;n=3;}else if(lead>=0xf0 && lead<=0xf4){code=lead&7;n=4;}else if(lead>=128)throw std::runtime_error("invalid vocabulary UTF8");
  if(n>text.size()-i)throw std::runtime_error("incomplete vocabulary UTF8");
  for(std::size_t j=1;j<n;++j){unsigned char c=text[i+j];if((c&192)!=128)throw std::runtime_error("invalid vocabulary UTF8 continuation");code=(code<<6)|(c&63);}
  if(code>=map.size() || map[code]<0)return std::string(text); // HF added-token fallback is whole-token raw UTF8.
  result.push_back(static_cast<char>(map[code]));i+=n;
 }return result;
}
}
Tokenizer::~Tokenizer(){
#if MELLUM16_NATIVE_TEXT
 if(handle_)mellum_tokenizer_destroy(handle_);
#endif
}
Result<std::unique_ptr<Tokenizer>> Tokenizer::Load(const std::filesystem::path& path,const std::string& expected){
#if !MELLUM16_NATIVE_TEXT
 (void)path;(void)expected;return Status(StatusCode::kUnsupported,"native text backend not compiled");
#else
 try{
  if(expected.size()!=64 || std::filesystem::file_size(path)>16*1024*1024)return Status(StatusCode::kInvalidArgument,"invalid/bounded tokenizer identity");
  std::ifstream file(path,std::ios::binary);std::string data{std::istreambuf_iterator<char>(file),{}};if(!file)return Status(StatusCode::kIoError,"cannot read tokenizer");
  if(compiler::Sha256Hex(data.data(),data.size())!=expected)return Status(StatusCode::kDataLoss,"tokenizer SHA256 mismatch");
  auto parsed=json::Parse(data);if(!parsed.ok())return parsed.status();const auto& root=parsed.value();
  const auto* model=root.find("model");const auto* vocab=model?model->find("vocab"):nullptr;const auto* fallback=model?model->find("byte_fallback"):nullptr;
  if(!model || !vocab || !vocab->is_object() || vocab->as_object().size()!=98304 || !fallback || !fallback->is_bool() || fallback->as_bool())return Status(StatusCode::kDataLoss,"unexpected Mellum byte-level BPE contract");
  auto tokenizer=std::unique_ptr<Tokenizer>(new Tokenizer);tokenizer->sha256_=expected;tokenizer->raw_.resize(98304);tokenizer->special_.resize(98304);std::vector<bool> seen(98304);
  for(const auto& [token,value]:vocab->as_object()){
   if(!value.is_integer() || value.as_integer()<0 || value.as_integer()>=98304 || seen[value.as_integer()])return Status(StatusCode::kDataLoss,"duplicate/out-of-range vocabulary ID");
   const int id=static_cast<int>(value.as_integer());seen[id]=true;tokenizer->raw_[id]=Raw(token);tokenizer->max_raw_=std::max(tokenizer->max_raw_,tokenizer->raw_[id].size());
  }
  const auto* added=root.find("added_tokens");if(!added || !added->is_array())return Status(StatusCode::kDataLoss,"missing AddedTokens");
  std::vector<bool> added_seen(98304);
  for(const auto& token:added->as_array()){
   const auto* id=token.find("id");const auto* special=token.find("special");const auto* content=token.find("content");
   if(!id || !id->is_integer() || id->as_integer()<0 || id->as_integer()>=98304 || !special || !special->is_bool() || !content || !content->is_string() || added_seen[id->as_integer()])return Status(StatusCode::kDataLoss,"invalid/duplicate AddedToken");
   added_seen[id->as_integer()]=true;tokenizer->special_[id->as_integer()]=special->as_bool();
  }
  std::array<char,512> error{};if(mellum_tokenizer_create(reinterpret_cast<const unsigned char*>(data.data()),data.size(),&tokenizer->handle_,error.data(),error.size()))return NativeFailure(error.data());
  return tokenizer;
 }catch(const std::exception& error){return Status(StatusCode::kDataLoss,error.what());}
#endif
}
Result<std::vector<int>> Tokenizer::Encode(std::string_view text)const{
#if MELLUM16_NATIVE_TEXT
 std::array<char,512> error{};NativeIds ids{};if(mellum_tokenizer_encode(handle_,reinterpret_cast<const unsigned char*>(text.data()),text.size(),&ids,error.data(),error.size()))return NativeFailure(error.data());
 struct Release{NativeIds ids;~Release(){mellum_tokenizer_free_ids(ids);}} release{ids};
 if(ids.size>131072)return Status(StatusCode::kResourceExhausted,"encoded text exceeds model token capacity");
 std::vector<int> result;result.reserve(ids.size);for(std::size_t i=0;i<ids.size;++i){if(ids.data[i]>=98304)return Status(StatusCode::kDataLoss,"native token ID outside vocabulary");result.push_back(static_cast<int>(ids.data[i]));}return result;
#else
 (void)text;return Status(StatusCode::kUnsupported,"native text backend not compiled");
#endif
}
Result<std::string> Tokenizer::Decode(std::span<const int> ids,bool skip)const{
#if MELLUM16_NATIVE_TEXT
 if(ids.size()>131072)return Status(StatusCode::kResourceExhausted,"decode token budget exceeded");
 std::vector<std::uint32_t> native;native.reserve(ids.size());
 for(int id:ids){if(id<0 || id>=98304)return Status(StatusCode::kInvalidArgument,"decode token ID outside vocabulary");native.push_back(static_cast<std::uint32_t>(id));}
 static const std::uint32_t empty=0;std::array<char,512> error{};NativeBytes data{};
 if(mellum_tokenizer_decode(handle_,native.empty()?&empty:native.data(),native.size(),skip,&data,error.data(),error.size()))return NativeFailure(error.data());
 struct Release{NativeBytes data;~Release(){mellum_text_free_bytes(data);}} release{data};return std::string(reinterpret_cast<const char*>(data.data),data.size);
#else
 (void)ids;(void)skip;return Status(StatusCode::kUnsupported,"native text backend not compiled");
#endif
}
std::string_view Tokenizer::RawToken(int id,bool skip)const{if(id<0 || id>=static_cast<int>(raw_.size()))throw std::invalid_argument("raw token ID outside vocabulary");return skip && special_[id]?std::string_view{}:raw_[id];}
Result<std::string> RenderChat(std::string_view request){
#if MELLUM16_NATIVE_TEXT
 if(request.size()>2*1024*1024)return Status(StatusCode::kResourceExhausted,"chat rendering byte budget exceeded");
 auto parsed=json::Parse(request);if(!parsed.ok())return parsed.status(); // strict duplicates/UTF8 before ordered Rust parser
 std::array<char,512> error{};NativeBytes data{};if(mellum_render_chat(reinterpret_cast<const unsigned char*>(request.data()),request.size(),&data,error.data(),error.size()))return NativeFailure(error.data());
 struct Release{NativeBytes data;~Release(){mellum_text_free_bytes(data);}} release{data};return std::string(reinterpret_cast<const char*>(data.data),data.size);
#else
 (void)request;return Status(StatusCode::kUnsupported,"native text backend not compiled");
#endif
}
StreamDecoder::StreamDecoder(const Tokenizer& tokenizer,bool skip):tokenizer_(tokenizer),skip_special_(skip){pending_.reserve(tokenizer.MaxRawTokenBytes()+3);output_.reserve((tokenizer.MaxRawTokenBytes()+3)*3);}
std::string_view StreamDecoder::Push(int id){pending_.append(tokenizer_.RawToken(id,skip_special_));return Process(false);}
std::string_view StreamDecoder::Finish(){return Process(true);}
std::string_view StreamDecoder::Process(bool final){
 output_.clear();std::size_t i=0;
 while(i<pending_.size()){
  const auto lead=static_cast<unsigned char>(pending_[i]);int length=lead<128?1:lead>=0xc2&&lead<=0xdf?2:lead>=0xe0&&lead<=0xef?3:lead>=0xf0&&lead<=0xf4?4:0;
  if(length==1){output_.push_back(pending_[i++]);continue;}
  if(length==0){output_.append("\xef\xbf\xbd");++i;continue;}
  int good=1;bool invalid=false;
  for(int j=1;j<length && i+static_cast<std::size_t>(j)<pending_.size();++j){const auto c=static_cast<unsigned char>(pending_[i+j]);bool valid=c>=128&&c<=191;
   if(j==1){if(lead==0xe0)valid=valid&&c>=0xa0;if(lead==0xed)valid=valid&&c<0xa0;if(lead==0xf0)valid=valid&&c>=0x90;if(lead==0xf4)valid=valid&&c<0x90;}
   if(!valid){invalid=true;break;}++good;
  }
  if(invalid){output_.append("\xef\xbf\xbd");i+=good;continue;}
  if(good<length){if(final){output_.append("\xef\xbf\xbd");i+=good;}break;}
  output_.append(pending_,i,length);i+=length;
 }
 pending_.erase(0,i);return output_;
}
}
