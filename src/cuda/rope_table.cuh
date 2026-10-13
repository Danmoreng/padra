// Shared Mellum scalar arithmetic; extracted unchanged from inference.cu.
#pragma once
float BHost(float x){std::uint32_t bits;std::memcpy(&bits,&x,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;std::memcpy(&x,&bits,4);return x;}
std::vector<float> RopeTable(int capacity,bool local){
  std::array<float,64> frequencies{};
  const double pi=std::acos(-1.0),base=500000.;
  const float low=static_cast<float>(std::max(0.,std::floor(128*std::log(8192/(32*2*pi))/(2*std::log(base)))));
  const float high=static_cast<float>(std::min(127.,std::ceil(128*std::log(8192/(2*pi))/(2*std::log(base)))));
  for(int i=0;i<64;++i){const float freq=std::pow(500000.f,static_cast<float>(2*i)/128);
    const float ramp=std::clamp((i-low)/(high-low),0.f,1.f),extra=1-ramp;
    frequencies[i]=local?1/freq:(1/(16*freq))*(1-extra)+(1/freq)*extra;
  }
  std::vector<float> table(static_cast<std::size_t>(capacity)*128);
  const float scale=local?1.f:static_cast<float>(1+0.1*std::log(16.));
  for(int p=0;p<capacity;++p)for(int i=0;i<64;++i){const float angle=static_cast<float>(p)*frequencies[i];
    table[p*128+i]=BHost(std::cos(angle)*scale);table[p*128+64+i]=BHost(std::sin(angle)*scale);}
  return table;
}
