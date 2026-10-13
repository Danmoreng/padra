// CUDA owner helpers shared by the SM120 and SIMT runtimes.
#pragma once
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#include "sampling.cuh"
struct Buffer {
  void* p=nullptr;std::size_t bytes=0;
  explicit Buffer(std::size_t n):bytes(n){Check(cudaMalloc(&p,n));}
  ~Buffer(){if(p)cudaFree(p);}
  Buffer(const Buffer&)=delete;
  template<class T>T* As(){return static_cast<T*>(p);}
};
struct Stream{cudaStream_t s=nullptr;Stream(){Check(cudaStreamCreate(&s));}~Stream(){if(s)cudaStreamDestroy(s);}};
struct Pinned{Control* p=nullptr;Pinned(){Check(cudaMallocHost(&p,sizeof(Control)));}~Pinned(){if(p)cudaFreeHost(p);}};
