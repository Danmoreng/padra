// Portable Q8_0 attention and speculative cache transactions on the real GPU.
#define main simt_prefill_fixture_main
#include "simt_prefill.cu"
#undef main
#include <functional>
#include "runtime/head_memory.h"
#include "runtime/mtp_memory.h"
#include "model/mtp.h"
namespace mellum16 { namespace {
#include "cuda/head_q8.cuh"
#include "cuda/mtp_simt_ops.cuh"
#define PADRA_MTP_SIMT 1
#include "cuda/mtp.cuh"
#undef PADRA_MTP_SIMT
#include "cuda/kv_q8_simt.cuh"
void Q8Attention(bool local,int position){
  constexpr int capacity=2048;const int cap=local?1024:capacity;
  std::vector<float> source(cap*512),value(source.size()),query(4096);
  for(std::size_t i=0;i<source.size();++i){source[i]=BHost(std::sin(float(i)*.17f));value[i]=BHost(std::cos(float(i)*.13f));}
  for(int i=0;i<4096;++i)query[i]=BHost(std::sin(float(i)*.11f));
  Buffer k(source.size()*4),v(value.size()*4),q(query.size()*4),keys(cap*544),values(cap*544),packed(128*36),state(sizeof(DecodeState)),partial(32*4096*4),lse(32*32*4),output(4096*4),bad(4);
  Check(cudaMemcpy(k.p,source.data(),k.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(v.p,value.data(),v.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(q.p,query.data(),q.bytes,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
  DecodeState host{0,position,capacity,capacity-position,position,std::min(position,1024),0,0};Check(cudaMemcpy(state.p,&host,sizeof(host),cudaMemcpyHostToDevice));
  q8_native::EncodeKV<<<cap,128>>>(k.As<float>(),v.As<float>(),keys.As<q8_native::CacheRow>(),values.As<q8_native::CacheRow>(),cap,bad.As<int>());
  q8_native::EncodeQuery<<<8,128>>>(q.As<float>(),packed.As<q8_native::QueryBlock>(),bad.As<int>());
  const int splits=local?4:32;
  if(local){q8_native::Split<true><<<dim3(8,splits),128>>>(packed.As<q8_native::QueryBlock>(),keys.As<q8_native::CacheRow>(),values.As<q8_native::CacheRow>(),partial.As<float>(),lse.As<float>(),state.As<DecodeState>(),cap,splits,bad.As<int>(),capacity);q8_native::Merge<true,float><<<8,128>>>(partial.As<float>(),lse.As<float>(),output.As<float>(),state.As<DecodeState>(),splits,bad.As<int>(),capacity);}
  else{q8_native::Split<false><<<dim3(8,splits),128>>>(packed.As<q8_native::QueryBlock>(),keys.As<q8_native::CacheRow>(),values.As<q8_native::CacheRow>(),partial.As<float>(),lse.As<float>(),state.As<DecodeState>(),cap,splits,bad.As<int>(),capacity);q8_native::Merge<false,float><<<8,128>>>(partial.As<float>(),lse.As<float>(),output.As<float>(),state.As<DecodeState>(),splits,bad.As<int>(),capacity);}
  Check(cudaDeviceSynchronize());int fault=0;Check(cudaMemcpy(&fault,bad.p,4,cudaMemcpyDeviceToHost));Require(!fault,"Q8 attention fault");
  std::vector<q8_native::CacheRow> hk(cap),hv(cap);std::vector<q8_native::QueryBlock> hq(128);std::vector<float> actual(4096),expected(4096);
  Check(cudaMemcpy(hk.data(),keys.p,keys.bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(hv.data(),values.p,values.bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(hq.data(),packed.p,packed.bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(actual.data(),output.p,output.bytes,cudaMemcpyDeviceToHost));
  const int first=local?std::max(0,position-1023):0,n=position-first+1;
  for(int head=0;head<32;++head){
    std::vector<double> probabilities(n);double maximum=-INFINITY,total=0;
    for(int j=0;j<n;++j){double dot=0;const auto& key=hk[(first+j)%cap];
      for(int b=0;b<4;++b){const auto& qb=hq[head*4+b];const auto* codes=reinterpret_cast<const signed char*>(qb.qs);int sum=0;for(int i=0;i<32;++i)sum+=int(codes[i])*int(key.qs[(head/8)*128+b*32+i]);dot+=double(sum)*qb.d*__half2float(key.d[(head/8)*4+b]);}
      probabilities[j]=dot*.08838834764831845;maximum=std::max(maximum,probabilities[j]);}
    for(auto& p:probabilities){p=std::exp(p-maximum);total+=p;}
    double rotated[128]={};
    for(int j=0;j<n;++j){const auto& val=hv[(first+j)%cap];for(int i=0;i<128;++i)rotated[i]+=probabilities[j]/total*val.qs[(head/8)*128+i]*__half2float(val.d[(head/8)*4+i/32]);}
    for(int base=0;base<128;base+=64){for(int d=1;d<64;d*=2)for(int i=0;i<64;i+=2*d)for(int j=0;j<d;++j){const double a=rotated[base+i+j],b=rotated[base+i+j+d];rotated[base+i+j]=a+b;rotated[base+i+j+d]=a-b;}for(int i=0;i<64;++i)expected[head*128+base+i]=BHost(float(rotated[base+i]*.125));}
  }
  Close(actual,expected,"Q8 attention vs independent FP64 encoded-cache oracle",.003,1e-5,1e-3);
}
void Rollback(int retained){
  constexpr int cap=1024,position=1022,rows=5;
  std::array<mtp::CacheBinding,29> bindings{};std::vector<std::unique_ptr<Buffer>> buffers;std::vector<std::vector<unsigned char>> original;
  for(int i=0;i<29;++i){const int row=i%2?544:1024;buffers.push_back(std::make_unique<Buffer>(cap*row*2));auto& b=*buffers.back();original.emplace_back(b.bytes);for(std::size_t j=0;j<b.bytes;++j)original.back()[j]=(j*13+i*7)%251;Check(cudaMemcpy(b.p,original.back().data(),b.bytes,cudaMemcpyHostToDevice));bindings[i]={b.p,b.As<unsigned char>()+cap*row,cap,nullptr,row,0};}
  Buffer table(sizeof(bindings)),state(sizeof(DecodeState)),saved(29*5*2048),bad(4);DecodeState host{0,position,2048,rows,position,1024,0,0};Check(cudaMemcpy(table.p,bindings.data(),sizeof(bindings),cudaMemcpyHostToDevice));Check(cudaMemcpy(state.p,&host,sizeof(host),cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
  mtp::CopyCacheRows<false><<<dim3(rows,29),128>>>(table.As<mtp::CacheBinding>(),state.As<DecodeState>(),saved.As<__nv_bfloat16>(),rows,0,bad.As<int>());
  for(int i=0;i<29;++i)for(int row=0;row<rows;++row)for(int part=0;part<2;++part)Check(cudaMemset(buffers[i]->As<unsigned char>()+(part*cap+(position+row)%cap)*bindings[i].row_bytes,252,bindings[i].row_bytes));
  mtp::CopyCacheRows<true><<<dim3(rows,29),128>>>(table.As<mtp::CacheBinding>(),state.As<DecodeState>(),saved.As<__nv_bfloat16>(),rows,retained,bad.As<int>());Check(cudaDeviceSynchronize());int fault=0;Check(cudaMemcpy(&fault,bad.p,4,cudaMemcpyDeviceToHost));Require(!fault,"MTP rollback fault");
  for(int i=0;i<29;++i){auto expected=original[i];for(int row=0;row<rows;++row)for(int part=0;part<2;++part){auto* p=expected.data()+(part*cap+(position+row)%cap)*bindings[i].row_bytes;if(row<retained)std::memset(p,252,bindings[i].row_bytes);else if(position+row<cap)std::memset(p,0,bindings[i].row_bytes);}std::vector<unsigned char> actual(expected.size());Check(cudaMemcpy(actual.data(),buffers[i]->p,actual.size(),cudaMemcpyDeviceToHost));Require(actual==expected,"MTP rollback differs from byte oracle");}
}
}}
int main(){try{using namespace mellum16;for(bool local:{false,true})for(int position:{127,1023,1026,2047})Q8Attention(local,position);for(int retained:{0,1,2,5})Rollback(retained);std::cout<<"Q8_0 attention and BF16/Q8_0 speculative rollback passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
