#include "server/response_store.h"
#include <algorithm>
#include <vector>
#include <stdexcept>
namespace mellum16::server {
ResponseReservation::~ResponseReservation(){if(store)store->Release(id,owner,committed);}
ResponseStore::ResponseStore(ResponseStoreLimits limits,Now now):limits_(limits),now_(std::move(now)){if(!limits.entries || !limits.bytes || !limits.per_record || limits.per_record>limits.bytes || limits.ttl.count()<1 || !now_)throw std::invalid_argument("invalid response-store limits");}
Status ResponseStore::Missing(){return {StatusCode::kNotFound,"response absent/expired or outside credential namespace"};}
void ResponseStore::Prune(){const auto now=now_();for(auto it=records_.begin();it!=records_.end();){auto& r=it->second;if(!r.pending && (!r.record || r.record.use_count()==1) && (r.deleted || now>=r.expires)){bytes_-=r.bytes;it=records_.erase(it);}else ++it;}}
Result<std::shared_ptr<ResponseReservation>> ResponseStore::Reserve(std::string id,std::string owner,std::size_t predicted){
 if(id.size()!=37 || !id.starts_with("resp_") || id.find_first_not_of("0123456789abcdef",5)!=std::string::npos || owner.empty() || owner.size()>128)return Status(StatusCode::kInvalidArgument,"invalid response identity/owner");
 std::lock_guard lock(mutex_);Prune();if(records_.contains(id))return Status(StatusCode::kInvalidArgument,"duplicate response identity");
 if(!predicted || predicted>limits_.per_record)return Status(StatusCode::kResourceExhausted,"bounded response-store admission exhausted");
 // Each child owns complete canonical history. Retiring an old response does
 // not break its child's lineage or release a still-shared physical root.
 // Plan eviction before changing any records; pending/pinned entries are safe.
 std::vector<decltype(records_)::iterator> eligible;
 for(auto it=records_.begin();it!=records_.end();++it)if(!it->second.pending && it->second.record.use_count()==1)eligible.push_back(it);
 std::sort(eligible.begin(),eligible.end(),[](auto a,auto b){return a->second.expires<b->second.expires;});
 auto count=records_.size(),bytes=bytes_;std::size_t retire=0;
 while((count>=limits_.entries || predicted>limits_.bytes-bytes) && retire<eligible.size()){--count;bytes-=eligible[retire++]->second.bytes;}
 if(count>=limits_.entries || predicted>limits_.bytes-bytes)return Status(StatusCode::kResourceExhausted,"bounded response-store admission exhausted");
 for(std::size_t i=0;i<retire;++i){bytes_-=eligible[i]->second.bytes;records_.erase(eligible[i]);}
 auto ticket=std::make_shared<ResponseReservation>();ticket->id=id;ticket->owner=owner;ticket->budget=predicted;
 Entry entry;entry.owner=owner;entry.bytes=predicted;entry.expires=now_()+limits_.ttl;
 if(!records_.emplace(id,std::move(entry)).second)return Status(StatusCode::kInvalidArgument,"duplicate response identity");bytes_+=predicted;ticket->store=shared_from_this();return ticket;
}
Status ResponseStore::Commit(const std::shared_ptr<ResponseReservation>& ticket,std::string envelope,std::string history,std::shared_ptr<ResponseRoot> root){
 if(!ticket || ticket->store.get()!=this)return {StatusCode::kInvalidArgument,"foreign response reservation"};
 std::lock_guard lock(mutex_);auto it=records_.find(ticket->id);if(it==records_.end() || it->second.owner!=ticket->owner || it->second.deleted)return Missing();auto& e=it->second;
 if(ticket->committed || !e.pending)return {StatusCode::kInvalidArgument,"response already committed"};
 // Both copies plus fixed record identity are admitted; no GPU KV lives here.
 constexpr std::size_t overhead=512;if(history.size()>ticket->budget || envelope.size()>ticket->budget-history.size() || overhead>ticket->budget-history.size()-envelope.size())return {StatusCode::kResourceExhausted,"response exceeds admitted reservation"};
 const auto size=history.size()+envelope.size()+overhead;auto record=std::make_shared<StoredResponse>();record->id=ticket->id;record->owner=ticket->owner;record->envelope=std::move(envelope);record->history=std::move(history);record->root=std::move(root);
 bytes_-=e.bytes-size;e.bytes=size;e.record=std::move(record);e.pending=false;e.expires=now_()+limits_.ttl;ticket->committed=true;return {};
}
Result<std::shared_ptr<const StoredResponse>> ResponseStore::Get(const std::string& id,const std::string& owner){
 std::lock_guard lock(mutex_);Prune();auto it=records_.find(id);if(it==records_.end() || it->second.owner!=owner || it->second.deleted || it->second.pending || now_()>=it->second.expires)return Missing();it->second.expires=now_()+limits_.ttl;return it->second.record;
}
Status ResponseStore::Delete(const std::string& id,const std::string& owner){std::lock_guard lock(mutex_);Prune();auto it=records_.find(id);if(it==records_.end() || it->second.owner!=owner || it->second.deleted || it->second.pending || now_()>=it->second.expires)return Missing();it->second.deleted=true;Prune();return {};}
void ResponseStore::Release(const std::string& id,const std::string& owner,bool committed){std::lock_guard lock(mutex_);auto it=records_.find(id);if(it!=records_.end() && it->second.owner==owner && !committed && it->second.pending){bytes_-=it->second.bytes;records_.erase(it);}Prune();}
std::pair<std::size_t,std::size_t> ResponseStore::Stats(){std::lock_guard lock(mutex_);Prune();return {records_.size(),bytes_};}
}
