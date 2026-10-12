#include "text/stream.h"
#include "util/json.h"
#include <algorithm>
#include <stdexcept>
namespace mellum16::text {
namespace {
std::size_t HeldSuffix(std::string_view text,std::string_view pattern){for(std::size_t n=std::min(text.size(),pattern.size()-1);n>0;--n)if(text.substr(text.size()-n)==pattern.substr(0,n))return n;return 0;}
}
StopMatcher::StopMatcher(std::vector<std::string> stops,std::size_t max_chunk):stops_(std::move(stops)),max_chunk_(max_chunk){
 if(max_chunk<1 || max_chunk>8192 || stops_.size()>4)throw std::invalid_argument("stop/chunk limits exceeded");
 std::size_t longest=0;
 for(const auto& stop:stops_){if(stop.empty() || stop.size()>256 || !json::Parse(json::Quote(stop)).ok())throw std::invalid_argument("stop must be nonempty valid UTF8 <=256 bytes");longest=std::max(longest,stop.size());}
 pending_.reserve(max_chunk+longest);output_.reserve(max_chunk+longest);
}
std::string_view StopMatcher::Push(std::string_view text){
 output_.clear();if(stopped_)return output_;if(text.size()>max_chunk_)throw std::length_error("decoded chunk exceeds admitted maximum");pending_.append(text);
 std::size_t first=std::string::npos;for(const auto& stop:stops_)first=std::min(first,pending_.find(stop));
 if(first!=std::string::npos){output_.assign(pending_,0,first);pending_.clear();stopped_=true;return output_;}
 std::size_t hold=0;for(const auto& stop:stops_)hold=std::max(hold,HeldSuffix(pending_,stop));output_.assign(pending_,0,pending_.size()-hold);pending_.erase(0,pending_.size()-hold);return output_;
}
std::string_view StopMatcher::Finish(){output_.clear();if(!stopped_)output_.swap(pending_);pending_.clear();return output_;}
ThinkingParser::ThinkingParser(std::size_t maximum):max_chunk_(maximum){if(maximum<1 || maximum>8192)throw std::invalid_argument("thinking chunk bound invalid");pending_.reserve(maximum+16);content_.reserve(maximum+16);reasoning_.reserve(maximum+16);}
void ThinkingParser::Answer(std::string_view text){if(trim_answer_lf_){while(!text.empty() && text.front()=='\n')text.remove_prefix(1);if(!text.empty())trim_answer_lf_=false;}content_.append(text);}
void ThinkingParser::Consume(){
 constexpr std::string_view open="<think>",close="</think>";
 if(phase_==Phase::Detect){
  if(pending_.size()<open.size() && open.starts_with(pending_))return;
  if(pending_.starts_with(open)){pending_.erase(0,open.size());phase_=Phase::Reasoning;}
  else {phase_=Phase::Answer;Answer(pending_);pending_.clear();return;}
 }
 if(phase_==Phase::Reasoning){
  auto end=pending_.find(close);if(end!=std::string::npos){reasoning_.assign(pending_,0,end);pending_.erase(0,end+close.size());phase_=Phase::Answer;trim_answer_lf_=true;Answer(pending_);pending_.clear();}
  else {const auto hold=HeldSuffix(pending_,close);reasoning_.assign(pending_,0,pending_.size()-hold);pending_.erase(0,pending_.size()-hold);}
 }else if(phase_==Phase::Answer){Answer(pending_);pending_.clear();}
}
ThinkingOutput ThinkingParser::Push(std::string_view text){content_.clear();reasoning_.clear();if(text.size()>max_chunk_)throw std::length_error("thinking chunk exceeds admitted maximum");pending_.append(text);Consume();return {content_,reasoning_};}
ThinkingOutput ThinkingParser::Finish(){content_.clear();reasoning_.clear();if(phase_==Phase::Reasoning)reasoning_.assign(pending_);else Answer(pending_);pending_.clear();return {content_,reasoning_};}
}
