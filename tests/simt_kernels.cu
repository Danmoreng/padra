#include "runtime/inference.h"
#include "cuda/device_matrix.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace mellum16 { namespace {
#include "cuda/simt_kernels.cuh"
#include "cuda/simt_attention_split.cuh"
void Require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
float RoundedBF(float x){auto u=std::bit_cast<unsigned>(x);u=(u+0x7fff+((u>>16)&1))&0xffff0000;return std::bit_cast<float>(u);}
unsigned Bits(float x){return std::bit_cast<unsigned>(x);}
float Half(unsigned bits){return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));}
float Mul1CPU(unsigned state){const unsigned h=state*0x83DCD12DU;const unsigned s=(h&255)+((h>>8)&255)+((h>>16)&255)+(h>>24);
  return __half2float(__float2half_rn(std::fma(Half(0x6400+s),Half(0x1eee),Half(0xc931))));}
__global__ void CodecProbe(float* out){unsigned i=blockIdx.x*256+threadIdx.x;if(i<65536)out[i]=__half2float(exl3_fp16::Mul1(i));}
__global__ void DirectEncoding(const float* x,const float* scale,unsigned char* out,int n){int i=blockIdx.x*128+threadIdx.x;if(i<n)out[i]=__nv_cvt_float_to_fp8(__fdiv_rn(x[i],scale[0]),__NV_SATFINITE,__NV_E4M3);}
void TestCodec(){
  Buffer result(65536*4);CodecProbe<<<256,256>>>(result.As<float>());Check(cudaDeviceSynchronize());std::vector<float> host(65536);Check(cudaMemcpy(host.data(),result.p,result.bytes,cudaMemcpyDeviceToHost));
  for(unsigned i=0;i<65536;++i)Require(Bits(host[i])==Bits(Mul1CPU(i)),"EXL3 half-FMA codec differs from CPU reference");
}
void TestQuantization(){
  Buffer input(4096*4),encoded(4096),reference(4096),scale(4),bad(4);std::vector<float> values(4096);
  for(int pattern=0;pattern<3;++pattern){
    for(int i=0;i<4096;++i)values[i]=pattern==0?0.f:pattern==1?RoundedBF(std::sin(float(i))*.125f):RoundedBF(std::ldexp(float((i%127)-63),i%80-40));
    Check(cudaMemcpy(input.p,values.data(),input.bytes,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
    QuantizeSimt<<<1,128>>>(input.As<float>(),encoded.As<unsigned char>(),scale.As<float>(),4096,bad.As<int>());
    DirectEncoding<<<32,128>>>(input.As<float>(),scale.As<float>(),reference.As<unsigned char>(),4096);Check(cudaDeviceSynchronize());
    std::vector<unsigned char> actual(4096),expected(4096);Check(cudaMemcpy(actual.data(),encoded.p,4096,cudaMemcpyDeviceToHost));Check(cudaMemcpy(expected.data(),reference.p,4096,cudaMemcpyDeviceToHost));
    Require(actual==expected,"SIMT encoding differs from direct RN FP8 encoding");int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"finite quantization flagged bad");
  }
}
float FP8CPU(unsigned char code){const int exponent=(code&127)>>3,mantissa=code&7;float value=std::ldexp(exponent?1.f+mantissa/8.f:mantissa/8.f,exponent?exponent-7:-6);return code&128?-value:value;}
void TestFP8Projection(){
  constexpr int rows=512;
  for(int cols:{2304,4096}){
    std::vector<unsigned char> payload(rows*cols);unsigned random=23;
    for(auto& code:payload){random=random*1664525U+1013904223U;code=static_cast<unsigned char>(random>>24);if((code&127)==127)--code;}
    std::vector<float> input(cols);for(int i=0;i<cols;++i)input[i]=RoundedBF(std::sin(float(i))*.25f);
    std::vector<unsigned short> row_scales(rows,0x3c00); // BF16 1/128
    Buffer weights(payload.size()),wscale(rows*2),source(cols*4),encoded(cols),scale(4),out(rows*4),bad(4);
    Check(cudaMemcpy(weights.p,payload.data(),weights.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(wscale.p,row_scales.data(),wscale.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(source.p,input.data(),source.bytes,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
    QuantizeSimt<<<1,128>>>(source.As<float>(),encoded.As<unsigned char>(),scale.As<float>(),cols,bad.As<int>());
    DeviceMatrix matrix{weights.As<unsigned char>(),wscale.As<unsigned char>(),nullptr,rows,cols,1};fast_projection::Vector<<<rows/4,128>>>(matrix,encoded.As<unsigned char>(),scale.As<float>(),out.As<float>(),bad.As<int>());Check(cudaDeviceSynchronize());
    std::vector<unsigned char> activation(cols);std::vector<float> result(rows);float activation_scale=0;
    Check(cudaMemcpy(activation.data(),encoded.p,encoded.bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(result.data(),out.p,out.bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(&activation_scale,scale.p,4,cudaMemcpyDeviceToHost));
    for(int row=0;row<rows;++row){std::array<float,32> sums{};
      for(int lane=0;lane<32;++lane)for(int c=lane*4;c<cols;c+=128)for(int j=0;j<4;++j)sums[lane]=std::fma(FP8CPU(activation[c+j]),FP8CPU(payload[row*cols+c+j]),sums[lane]);
      for(int d=16;d;d/=2){auto old=sums;for(int lane=0;lane<32-d;++lane)sums[lane]+=old[lane+d];}
      Require(Bits(result[row])==Bits(RoundedBF((sums[0]*activation_scale)*.0078125f)),"FP8 packed decode/FMA/scale projection differs from CPU reference");
    }
    int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"FP8 projection flagged bad");
  }
}
void TestNormalizationAndRopeGraph(){
  std::vector<unsigned short> weights(2304,0x3f80);
  std::vector<float> input(2304,.5f),query(4096,.5f),key(512,.25f),table(256,0);
  for(int d=0;d<64;++d){table[d]=1;table[128+64+d]=1;}
  Buffer w(weights.size()*2),source(input.size()*4),normalized(input.size()*4),encoded(input.size()),scale(4),bad(4);
  Check(cudaMemcpy(w.p,weights.data(),w.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(source.p,input.data(),source.bytes,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
  DeviceMatrix norm{w.As<unsigned char>(),nullptr,nullptr,2304,1,0};
  NormFP8WideExact<<<1,256>>>(source.As<float>(),normalized.As<float>(),norm,encoded.As<unsigned char>(),scale.As<float>(),bad.As<int>());Check(cudaDeviceSynchronize());
  std::vector<float> actual(2304);std::vector<unsigned char> bytes(2304);Check(cudaMemcpy(actual.data(),normalized.p,normalized.bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(bytes.data(),encoded.p,encoded.bytes,cudaMemcpyDeviceToHost));
  for(int i=0;i<2304;++i)Require(actual[i]==1.f && bytes[i]==0x7e,"compensated RMS/FP8 norm differs from constant-vector reference");
  Buffer q(query.size()*4),k(key.size()*4),rope(table.size()*4),command(sizeof(SimtCommand));Check(cudaMemcpy(rope.p,table.data(),rope.bytes,cudaMemcpyHostToDevice));
  DeviceMatrix head_norm{w.As<unsigned char>(),nullptr,nullptr,128,1,0};Stream stream;DecodeGraph graph;
  graph.Capture(stream.s,[&]{QKNormRope<<<36,128,0,stream.s>>>(q.As<float>(),k.As<float>(),head_norm,head_norm,rope.As<float>(),0,bad.As<int>(),nullptr,nullptr,nullptr,2,command.As<int>()+1);});
  for(int pos=0;pos<2;++pos){Check(cudaMemcpyAsync(q.p,query.data(),q.bytes,cudaMemcpyHostToDevice,stream.s));Check(cudaMemcpyAsync(k.p,key.data(),k.bytes,cudaMemcpyHostToDevice,stream.s));SetSimtCommand<<<1,1,0,stream.s>>>(command.As<SimtCommand>(),0,pos);graph.Launch(stream.s);Check(cudaStreamSynchronize(stream.s));std::vector<float> values(4096);Check(cudaMemcpy(values.data(),q.p,q.bytes,cudaMemcpyDeviceToHost));
    for(int i=0;i<4096;++i)Require(values[i]==(pos && i%128<64?-1.f:1.f),"QK norm/RoPE graph used a stale absolute position");}
  int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"normalization/RoPE flagged bad");
}
void H128CPU(std::vector<float>& values){for(int d=1;d<128;d*=2){auto old=values;for(int t=0;t<128;++t)values[t]=(t&d)?old[t^d]-old[t]:old[t]+old[t^d];}for(auto& x:values)x*=.08838834764831845f;}
unsigned Permutation(unsigned input,unsigned output){return ((output&7)*4+((input&7)>>1))*8+(output>=8?4:0)+(input>=8?2:0)+(input&1);}
float WeightCPU(const std::vector<unsigned>& payload,int input,int output){
  const auto tile=std::size_t(input/16)*8+output/16;const int end=int(Permutation(input&15,output&15))*4+4+1024;
  const int last=(end-1)/32,first=(end-16)/32,shift=(last+1)*32-end;
  const unsigned a=payload[tile*32+(last&31)],b=payload[tile*32+(first&31)];
  const unsigned state=shift?((a>>shift)|(b<<(32-shift)))&65535U:a&65535U;
  return Mul1CPU(state);
}
void TestSelectedExpert(){
  std::vector<unsigned> payload(8*8*32);unsigned random=19;for(auto& x:payload){random=random*1664525U+1013904223U;x=random;}
  std::vector<float> input(128);for(int i=0;i<128;++i)input[i]=RoundedBF(std::sin(float(i))*.25f);
  std::vector<unsigned short> side(128,0x3c00);
  Buffer packed(payload.size()*4),suh(256),svh(256),source(512),transform(8*128*4),output(8*128*4),descriptors(64*3*sizeof(DeviceMatrix)),ids(8*4),bad(4);
  Check(cudaMemcpy(packed.p,payload.data(),packed.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(suh.p,side.data(),256,cudaMemcpyHostToDevice));Check(cudaMemcpy(svh.p,side.data(),256,cudaMemcpyHostToDevice));Check(cudaMemcpy(source.p,input.data(),512,cudaMemcpyHostToDevice));
  DeviceMatrix matrix{packed.As<unsigned char>(),nullptr,nullptr,128,128,3,suh.As<unsigned short>(),svh.As<unsigned short>()};std::vector<DeviceMatrix> table(64*3,matrix);Check(cudaMemcpy(descriptors.p,table.data(),descriptors.bytes,cudaMemcpyHostToDevice));Check(cudaMemset(ids.p,0,ids.bytes));Check(cudaMemset(bad.p,0,4));
  exl3_expert::InputHad<<<dim3(1,8),128>>>(descriptors.As<DeviceMatrix>(),0,0,ids.As<int>(),source.As<float>(),transform.As<float>(),8,bad.As<int>());
  exl3_selected::Projection<<<dim3(1,8),256>>>(descriptors.As<DeviceMatrix>(),0,0,ids.As<int>(),transform.As<float>(),output.As<float>(),8,bad.As<int>());Check(cudaDeviceSynchronize());
  std::vector<float> expected_input=input;H128CPU(expected_input);for(auto& x:expected_input)x=RoundedBF(x);
  std::vector<float> transformed(8*128),actual(8*128);Check(cudaMemcpy(transformed.data(),transform.p,transform.bytes,cudaMemcpyDeviceToHost));Check(cudaMemcpy(actual.data(),output.p,output.bytes,cudaMemcpyDeviceToHost));
  for(int p=0;p<8;++p)for(int i=0;i<128;++i)Require(Bits(transformed[p*128+i])==Bits(expected_input[i]),"input H128/BF16 differs");
  std::vector<float> raw(128);
  for(int r=0;r<128;++r){float sums[4]={};for(int lane=0;lane<4;++lane)for(int k=0;k<128;k+=16)for(int half=0;half<2;++half){int at=k+lane*2+half*8;sums[lane]=std::fma(expected_input[at],WeightCPU(payload,at,r),sums[lane]);sums[lane]=std::fma(expected_input[at+1],WeightCPU(payload,at+1,r),sums[lane]);}
    raw[r]=(sums[0]+sums[1])+(sums[2]+sums[3]);}
  H128CPU(raw);for(auto& x:raw)x=RoundedBF(x);
  for(int p=0;p<8;++p)for(int r=0;r<128;++r)Require(Bits(actual[p*128+r])==Bits(raw[r]),"selected K4 projection differs from CPU FMA/H128 reference");
  int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"selected expert flagged bad");
}
void TestAttention(){
  // A context beyond Pascal's shared-memory score limit, plus local ring wrap.
  const int capacity=16385;Buffer query(4096*4),keys(capacity*512*2),values(capacity*512*2),out(4096*4),scores(capacity*32*4),command(sizeof(SimtCommand)),bad(4);
  Check(cudaMemset(query.p,0,query.bytes));Check(cudaMemset(keys.p,0,keys.bytes));
  std::vector<__nv_bfloat16> cache(capacity*512);for(int row=0;row<capacity;++row)for(int d=0;d<512;++d)cache[row*512+d]=__float2bfloat16_rn(float(d%13-6)*.25f);
  Check(cudaMemcpy(values.p,cache.data(),values.bytes,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
  Stream stream;DecodeGraph graph;graph.Capture(stream.s,[&]{AttendSimt<false><<<32,256,0,stream.s>>>(query.As<float>(),keys.As<__nv_bfloat16>(),values.As<__nv_bfloat16>(),out.As<float>(),command.As<SimtCommand>(),capacity,scores.As<float>(),bad.As<int>());});
  for(int pos:{0,123,16384}){
    SetSimtCommand<<<1,1,0,stream.s>>>(command.As<SimtCommand>(),0,pos);graph.Launch(stream.s);Check(cudaStreamSynchronize(stream.s));std::vector<float> result(4096);Check(cudaMemcpy(result.data(),out.p,out.bytes,cudaMemcpyDeviceToHost));
    const float probability=RoundedBF(1.f/(pos+1));for(int h=0;h<32;++h)for(int d=0;d<128;++d){float value=float(((h/8)*128+d)%13-6)*.25f;float expected=RoundedBF(float(double(probability)*value*(pos+1)));Require(Bits(result[h*128+d])==Bits(expected),"global attention/graph command differs from uniform-softmax reference");}
  }
  SetSimtCommand<<<1,1,0,stream.s>>>(command.As<SimtCommand>(),0,17021);AttendSimt<true><<<32,256,0,stream.s>>>(query.As<float>(),keys.As<__nv_bfloat16>(),values.As<__nv_bfloat16>(),out.As<float>(),command.As<SimtCommand>(),1024,scores.As<float>(),bad.As<int>());Check(cudaStreamSynchronize(stream.s));std::vector<float> result(4096);Check(cudaMemcpy(result.data(),out.p,out.bytes,cudaMemcpyDeviceToHost));
  for(int h=0;h<32;++h)for(int d=0;d<128;++d)Require(result[h*128+d]==float(((h/8)*128+d)%13-6)*.25f,"local ring-wrap attention differs");
  int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"attention flagged bad");
}
void TestCausalAttention(){
  // Future rows are poisoned; a causal token must never read them. Nonzero
  // Q/K, per-head routing and differing V test the complete score/softmax/PV.
  constexpr int cap=128,visible=37;std::vector<float> q(4096);
  std::vector<__nv_bfloat16> k(cap*512),v(cap*512);
  for(int i=0;i<4096;++i)q[i]=float(i%7-3)/16;
  for(int row=0;row<cap;++row)for(int d=0;d<512;++d){k[row*512+d]=__float2bfloat16_rn(row<visible?float((row+d)%9-4)/16:NAN);v[row*512+d]=__float2bfloat16_rn(row<visible?float((row*3+d)%11-5)/8:NAN);}
  Buffer query(q.size()*4),keys(k.size()*2),values(v.size()*2),out(4096*4),workspace(cap*32*4),command(sizeof(SimtCommand)),bad(4);
  Check(cudaMemcpy(query.p,q.data(),query.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(keys.p,k.data(),keys.bytes,cudaMemcpyHostToDevice));Check(cudaMemcpy(values.p,v.data(),values.bytes,cudaMemcpyHostToDevice));Check(cudaMemset(bad.p,0,4));
  SetSimtCommand<<<1,1>>>(command.As<SimtCommand>(),0,visible-1);AttendSimt<false><<<32,256>>>(query.As<float>(),keys.As<__nv_bfloat16>(),values.As<__nv_bfloat16>(),out.As<float>(),command.As<SimtCommand>(),cap,workspace.As<float>(),bad.As<int>());Check(cudaDeviceSynchronize());
  std::vector<float> result(4096);Check(cudaMemcpy(result.data(),out.p,out.bytes,cudaMemcpyDeviceToHost));
  for(int h=0;h<32;++h){std::vector<float> p(visible);float mx=-INFINITY;
    for(int row=0;row<visible;++row){double dot=0;for(int d=0;d<128;++d)dot+=double(q[h*128+d])*__bfloat162float(k[row*512+(h/8)*128+d]);p[row]=RoundedBF(RoundedBF(float(dot))*.08838834764831845f);mx=std::max(mx,p[row]);}
    double total=0;for(auto& x:p){x=float(std::exp(double(x-mx)));total+=x;}for(auto& x:p)x=RoundedBF(x/float(total));
    for(int d=0;d<128;++d){double sum=0;for(int row=0;row<visible;++row)sum+=double(p[row])*__bfloat162float(v[row*512+(h/8)*128+d]);Require(Bits(result[h*128+d])==Bits(RoundedBF(float(sum))),"causal nonzero attention differs from CPU reference");}
  }
  simt_attention_split::Launch(query.As<float>(),keys.As<__nv_bfloat16>(),values.As<__nv_bfloat16>(),out.As<float>(),command.As<SimtCommand>(),cap,false,workspace.As<float>(),bad.As<int>(),nullptr);Check(cudaDeviceSynchronize());std::vector<float> split(4096);Check(cudaMemcpy(split.data(),out.p,out.bytes,cudaMemcpyDeviceToHost));Require(!memcmp(split.data(),result.data(),out.bytes),"split attention differs from exact reference");
  int flag=0;Check(cudaMemcpy(&flag,bad.p,4,cudaMemcpyDeviceToHost));Require(!flag,"causal attention read poisoned future rows");
}
void TestSampling(){
  constexpr int n=32;const auto bytes=gpu_sampling::WorkspaceBytes(n);Buffer temp(bytes),input(n*4),adjusted(n*4),sorted(n*4),ids(n*4),sorted_ids(n*4),counts(n*4),cdf(n*8),rng(sizeof(gpu_sampling::State)),control(sizeof(Control)),bad(4);
  std::vector<float> logits(n,-4.f);logits[3]=2;logits[7]=2;Check(cudaMemcpy(input.p,logits.data(),n*4,cudaMemcpyHostToDevice));Check(cudaMemset(counts.p,0,n*4));Check(cudaMemset(bad.p,0,4));Stream stream;
  SamplingOptions options;options.temperature=.7f;options.top_k=8;options.top_p=.9f;options.frequency_penalty=.5f;int chosen=-1;
  for(int attempt=0;attempt<2;++attempt){gpu_sampling::State state{123,0};Check(cudaMemcpy(rng.p,&state,sizeof(state),cudaMemcpyHostToDevice));gpu_sampling::Launch(input.As<float>(),adjusted.As<float>(),sorted.As<float>(),ids.As<int>(),sorted_ids.As<int>(),cdf.As<double>(),rng.As<gpu_sampling::State>(),control.As<Control>(),bad.As<int>(),temp.p,bytes,stream.s,options,counts.As<int>(),n);Check(cudaStreamSynchronize(stream.s));Control result;Check(cudaMemcpy(&result,control.p,sizeof(result),cudaMemcpyDeviceToHost));Require(!result.bad && (result.token==3 || result.token==7),"sampling returned invalid result");if(attempt)Require(chosen==result.token,"sampling seed is not repeatable");chosen=result.token;}
}
} }
int main(){try{mellum16::TestCodec();mellum16::TestQuantization();mellum16::TestFP8Projection();mellum16::TestNormalizationAndRopeGraph();mellum16::TestSelectedExpert();mellum16::TestAttention();mellum16::TestCausalAttention();mellum16::TestSampling();std::cout<<"EXL3/FP8 projections, RMS, RoPE graphs, causal/long/local attention and sampling passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
