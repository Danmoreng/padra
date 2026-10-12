#pragma once
#include "util/json.h"
#include <array>
#include <string>
namespace mellum16::server {
struct Job;
struct ResponseRequest {
 std::string id,previous_id,chat_body,history,instructions="null",metadata="{}",tools="[]",tool_choice="\"none\"";
 std::string reasoning_id,message_id;std::array<std::string,8> function_ids;
 std::int64_t created_at=0;bool store=true,stream=false,parallel=true;
 int outputs=256;float temperature=0,top_p=1;
};
// Previous history excludes the previous request's top-level instructions.
// Original nested tool/schema order is retained in chat_body.
Result<ResponseRequest> ParseResponseRequest(std::string_view body,std::string_view previous_history="[]");
std::string ResponseOutputItems(const ResponseRequest&,const Job&);
std::string ResponseEnvelope(const ResponseRequest&,const Job* job=nullptr,std::string_view status="in_progress");
std::string ResponseCompletedHistory(const ResponseRequest&,const Job&);
}
