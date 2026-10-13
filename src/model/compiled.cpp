#include "model/compiled.h"
#include "compiler/sha256.h"
#include "model/safetensors.h"
#include "util/json.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#if MELLUM16_HAS_OPENSSL
#include <openssl/evp.h>
#endif
namespace mellum16::internal {
namespace {
const json::Value& Field(const json::Value& v,std::string_view key) {
  const auto* p=v.find(key);
  if(!p)throw std::runtime_error("missing field: "+std::string(key));
  return *p;
}
std::string Text(const json::Value& v) {
  if(!v.is_string())throw std::runtime_error("expected string");
  return v.as_string();
}
std::uint64_t Number(const json::Value& v) {
  if(!v.is_integer() || v.as_integer()<0)throw std::runtime_error("expected nonnegative integer");
  return static_cast<std::uint64_t>(v.as_integer());
}
void Require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
std::map<std::string,std::vector<std::uint64_t>,std::less<>> Expected() {
  std::map<std::string,std::vector<std::uint64_t>,std::less<>> r;
  r["model.embed_tokens.weight"]={98304,2304};r["lm_head.weight"]={98304,2304};r["model.norm.weight"]={2304};
  for(unsigned l=0;l<28;++l) {
    const auto p="model.layers."+std::to_string(l)+".";
    for(auto name:{"input_layernorm.weight","post_attention_layernorm.weight"})r[p+name]={2304};
    for(auto name:{"q_norm.weight","k_norm.weight"})r[p+"self_attn."+name]={128};
    r[p+"self_attn.q_proj.weight"]={4096,2304};
    for(auto name:{"k_proj.weight","v_proj.weight"})r[p+"self_attn."+name]={512,2304};
    r[p+"self_attn.o_proj.weight"]={2304,4096};r[p+"mlp.gate.weight"]={64,2304};
    for(unsigned e=0;e<64;++e) {
      const auto ep=p+"mlp.experts."+std::to_string(e)+".";
      for(auto n:{"gate_proj.weight","up_proj.weight"})r[ep+n]={896,2304};
      r[ep+"down_proj.weight"]={2304,896};
    }
  }
  return r;
}
std::string Hash(const void* data,std::size_t bytes) {
#if MELLUM16_HAS_OPENSSL
  unsigned char hash[32];unsigned int size=0;
  Require(EVP_Digest(data,bytes,hash,&size,EVP_sha256(),nullptr)==1 && size==32,"SHA-256 failed");
  std::string out;out.reserve(64);constexpr auto hex="0123456789abcdef";
  for(auto x:hash){out+=hex[x>>4];out+=hex[x&15];}return out;
#else
  return compiler::Sha256Hex(data,bytes);
#endif
}
MappedFile OpenRegular(const std::filesystem::path& path) {
  const auto stat=std::filesystem::symlink_status(path);
  Require(std::filesystem::is_regular_file(stat) && !std::filesystem::is_symlink(stat),"artifact file must be regular and not a symlink");
  auto r=MappedFile::Open(path);
  if(!r.ok())throw std::runtime_error(r.status().message());
  return std::move(r).value();
}
}
Result<CompiledModel> LoadCompiledModel(const std::filesystem::path& directory) {
  try {
    auto file=OpenRegular(directory/"manifest.json");
    Require(file.size()<4*1024*1024,"manifest too large");
    auto parsed=json::Parse({reinterpret_cast<const char*>(file.data()),static_cast<std::size_t>(file.size())},
        {.max_depth=16,.max_values=200000,.max_string_bytes=4*1024*1024});
    if(!parsed.ok())return parsed.status();
    const auto& v=parsed.value();
    Require(Number(Field(v,"schema_version"))==1,"unsupported manifest schema");
    const auto profile=Text(Field(v,"profile"));const bool exl3=profile==kExl3Profile || profile==kExl3K8Profile;const int exl3Rate=profile==kExl3K8Profile?8:exl3?4:0;
    Require(profile==kHybridProfile || exl3,"unsupported precision/math profile");
    Require(Text(Field(v,"model_revision"))==kModelRevision,"wrong Mellum model revision");
    Require(Text(Field(v,"activation_arithmetic"))==(exl3?(exl3Rate==8?"EXL3 K8 mul1; FP16 sidecars; FP32 H128 rounded BF16 input; compensated FP32 dot; FP32 H128 rounded BF16 output":"EXL3 K4 mul1; FP16 sidecars; FP32 H128 rounded BF16 input; compensated FP32 dot; FP32 H128 rounded BF16 output"):"BF16 activations; dequantized weights rounded BF16 in registers; FP32 SIMT accumulation"),"unsupported activation arithmetic");
    Require(Text(Field(v,"kv_dtype"))=="BF16","unsupported cache precision");
    Require(Text(Field(v,"weights_file"))=="weights.bin","unsafe weights path");
    const auto bytes=Number(Field(v,"weights_bytes"));
    Require(bytes>0 && bytes<12ULL*1024*1024*1024,"invalid artifact byte budget");
    CompiledModel result;result.manifest_sha256=compiler::Sha256Hex(file.data(),file.size());result.exl3_experts=exl3;result.exl3_rate=exl3Rate;
    result.weights=OpenRegular(directory/"weights.bin");
    Require(result.weights.size()==bytes,"truncated or oversized weight payload");
    result.weights_sha256=Text(Field(v,"weights_sha256"));
    Require(result.weights_sha256.size()==64 && std::all_of(result.weights_sha256.begin(),result.weights_sha256.end(),[](char c){return (c>='0' && c<='9') || (c>='a' && c<='f');}),"invalid weight hash");
    result.tokenizer_sha256=Text(Field(v,"tokenizer_sha256"));
    Require(result.tokenizer_sha256=="58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a","invalid tokenizer identity");
    const auto expected=Expected();const auto& table=Field(v,"tensors");
    Require(table.is_object() && table.as_object().size()==expected.size(),"incomplete/extra tensor inventory");
    std::vector<Region> ranges;
    for(const auto& [name,shape]:expected) {
      const auto& t=Field(table,name);Matrix matrix;matrix.shape=shape;
      const auto& s=Field(t,"shape");Require(s.is_array() && s.as_array().size()==shape.size(),"wrong tensor rank");
      std::uint64_t elements=1;
      for(std::size_t i=0;i<shape.size();++i){Require(Number(s.as_array()[i])==shape[i],"wrong tensor shape");elements*=shape[i];}
      const bool expert=name.find(".experts.")!=std::string::npos;
      const bool projection=name.find(".self_attn.")!=std::string::npos && name.find("_proj.weight")!=std::string::npos;
      const auto format=expert?(exl3?(exl3Rate==8?"EXL3_K8":"EXL3_K4"):"NVFP4"):projection?"FP8":"BF16";
      Require(Text(Field(t,"format"))==format,"tensor precision does not match selected hybrid");
      matrix.format=expert?(exl3?(exl3Rate==8?WeightFormat::Exl3K8:WeightFormat::Exl3K4):WeightFormat::Nvfp4):projection?WeightFormat::Fp8:WeightFormat::Bf16;
      std::map<std::string,std::uint64_t,std::less<>> sizes;
      if(expert && exl3){Require(Text(Field(t,"codebook"))=="mul1" && Number(Field(t,"hadamard_block"))==128,"unsupported EXL3 codec/transform");sizes={{"packed",elements*exl3Rate/8},{"suh",shape[1]*2},{"svh",shape[0]*2}};}
      else if(expert)sizes={{"packed",elements/2},{"scales",elements/16},{"divisor",4}};
      else if(projection)sizes={{"packed",elements},{"scales",shape[0]*2}};
      else sizes={{"data",elements*2}};
      const auto& regions=Field(t,"regions");Require(regions.is_object() && regions.as_object().size()==sizes.size(),"unknown/missing weight regions");
      for(const auto& [part,length]:sizes) {
        const auto& region=Field(regions,part);
        Region r{Number(Field(region,"offset")),Number(Field(region,"bytes"))};
        Require(r.offset%256==0 && r.bytes==length && r.offset<=bytes && r.bytes<=bytes-r.offset,"invalid region range/alignment/bytes");
        matrix.regions.emplace(part,r);ranges.push_back(r);
      }
      result.tensors.emplace(name,std::move(matrix));
    }
    std::sort(ranges.begin(),ranges.end(),[](auto a,auto b){return a.offset<b.offset;});
    std::uint64_t cursor=0;
    for(const auto& r:ranges){Require(r.offset==(cursor+255)/256*256,"overlapping/noncanonical weight regions");cursor=r.offset+r.bytes;}
    Require(cursor==bytes,"unaccounted payload bytes");
    Require(Hash(result.weights.data(),static_cast<std::size_t>(bytes))==result.weights_sha256,"weight SHA-256 mismatch");
    for(const auto& [name,matrix]:result.tensors) {
      (void)name;const bool expert=matrix.format==WeightFormat::Nvfp4;
      const bool projection=matrix.format==WeightFormat::Fp8;
      for(const auto& [part,r]:matrix.regions) {
        const auto* raw=result.weights.data()+r.offset;
        if((expert && part=="scales") || (projection && part=="packed")) {
          for(std::uint64_t j=0;j<r.bytes;++j) {
            const auto x=std::to_integer<unsigned>(raw[j]);
            Require(expert?x<127:(x&127)!=127,"invalid E4M3 scale/weight");
          }
        } else if(part=="data" || (projection && part=="scales")) {
          for(std::uint64_t j=0;j<r.bytes;j+=2) {
            std::uint16_t x;std::memcpy(&x,raw+j,2);
            Require((x&0x7f80)!=0x7f80,"nonfinite BF16 weight");
            if(part=="scales")Require(x>0 && x<0x7f80,"invalid BF16 row scale");
          }
        } else if(part=="suh" || part=="svh") {
          for(std::uint64_t j=0;j<r.bytes;j+=2){std::uint16_t x;std::memcpy(&x,raw+j,2);Require((x&0x7c00)!=0x7c00,"nonfinite EXL3 FP16 sidecar");Require((x&0x7fff)!=0,"zero EXL3 sidecar");}
        } else if(part=="divisor") {
          float x;std::memcpy(&x,raw,4);Require(std::isfinite(x) && x>0,"invalid NVFP4 divisor");
        }
      }
    }
    return result;
  }catch(const std::exception& e){return Status(StatusCode::kDataLoss,e.what());}
}
// X0-only overlay: source SF is mapped once, all existing expert bytes retained,
// and a single canonical GPU arena is constructed before inference.
Status ApplyDiagnosticAttention(CompiledModel& model,const std::filesystem::path& directory) {
  try {
    Require(model.exl3_experts,"attention diagnostic requires EXL3 model");
    auto plan=OpenRegular(directory/"diagnostic.json");Require(plan.size()<16384,"diagnostic plan too large");
    auto parsed=json::Parse({reinterpret_cast<const char*>(plan.data()),static_cast<std::size_t>(plan.size())});
    if(!parsed.ok())return parsed.status();
    const auto& p=parsed.value();
    Require(Number(Field(p,"schema_version"))==1,"diagnostic schema");
    Require(Text(Field(p,"base_weights_sha256"))==model.weights_sha256,"diagnostic base identity mismatch");
    Require(Text(Field(p,"model_revision"))==kModelRevision,"diagnostic source mismatch");
    Require(Text(Field(p,"weights_file"))=="attention.safetensors","diagnostic unsafe path");
    const auto file=directory/"attention.safetensors";
    auto source=OpenRegular(file);Require(source.size()<=3ULL*1024*1024*1024,"diagnostic source budget");
    const auto hash=Hash(source.data(),source.size());Require(hash==Text(Field(p,"weights_sha256")),"diagnostic source hash mismatch");
    const auto kind=Text(Field(p,"kind"));Require(kind=="original-bf16" || kind=="decoded-fp8-bf16","unknown diagnostic attention");
    auto sf=LoadSafetensorsFile(file);if(!sf.ok())return sf.status();
    std::map<std::string,Region,std::less<>> replacement;
    for(const auto& t:sf.value()) {
      auto found=model.tensors.find(t.name);if(found==model.tensors.end() || found->second.format!=WeightFormat::Fp8)continue;
      Require(t.dtype=="BF16" && t.shape==found->second.shape,"diagnostic attention shape/dtype mismatch");
      Require(t.length==found->second.shape[0]*found->second.shape[1]*2,"diagnostic BF16 bytes");
      Require(t.absolute_offset<=source.size() && t.length<=source.size()-t.absolute_offset,"diagnostic source range");
      for(std::uint64_t i=0;i<t.length;i+=2){std::uint16_t x;std::memcpy(&x,source.data()+t.absolute_offset+i,2);Require((x&0x7f80)!=0x7f80,"nonfinite diagnostic BF16");}
      Require(replacement.emplace(t.name,Region{t.absolute_offset,t.length}).second,"duplicate diagnostic tensor");
    }
    Require(replacement.size()==112,"diagnostic requires all112 Q/K/V/O matrices");
    auto table=model.tensors;std::vector<UploadSlice> slices;std::uint64_t cursor=0;
    for(auto& [name,m]:table){
      const auto found=replacement.find(name);const bool attention=found!=replacement.end();
      if(attention){m.format=WeightFormat::Bf16;m.regions={{"data",found->second}};}
      for(auto& [part,r]:m.regions){(void)part;cursor=(cursor+255)/256*256;
        Require(cursor<14ULL*1024*1024*1024 && r.bytes<=14ULL*1024*1024*1024-cursor,"diagnostic arena overflow");
        slices.push_back({r.offset,cursor,r.bytes,attention});r.offset=cursor;cursor+=r.bytes;
      }
    }
    model.tensors=std::move(table);model.uploads=std::move(slices);model.diagnostic_attention=std::move(source);
    model.resident_weights_bytes=cursor;model.diagnostic_attention_sha256=hash;return Status{};
  }catch(const std::exception& e){return Status(StatusCode::kDataLoss,e.what());}
}

}
