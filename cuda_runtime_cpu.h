// cuda_runtime_cpu.h — header-only CPU shim for the CUDA runtime API subset that
// library-based GPU code uses (cudaMalloc/Memcpy/Memset, streams, events, device
// queries). Include INSTEAD of <cuda_runtime.h> to build on a GPU-less box; "device"
// memory is just host memory, streams are no-ops, events time with std::chrono.
//
// Scope: this makes code that drives cuDNN/cuBLAS/Thrust + manual buffers compile and
// run on CPU. It does NOT run raw __global__ kernels launched with <<<>>> — that needs
// nvcc. __host__/__device__ annotations are stripped so device-marked helpers compile.
#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <chrono>

#ifndef __CUDACC__
  #define __host__
  #define __device__
  #define __forceinline__ inline
  #define __restrict__
  struct dim3 { unsigned x=1,y=1,z=1; dim3(unsigned X=1,unsigned Y=1,unsigned Z=1):x(X),y(Y),z(Z){} };
#endif

typedef enum {
  cudaSuccess = 0,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInvalidValue = 11,
} cudaError_t;

typedef enum {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4,
} cudaMemcpyKind;

// ---- error handling ----
inline const char* cudaGetErrorString(cudaError_t e){
  switch(e){ case cudaSuccess: return "no error";
    case cudaErrorMemoryAllocation: return "out of memory";
    default: return "invalid value"; }
}
inline cudaError_t cudaGetLastError(){ return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError(){ return cudaSuccess; }

// ---- memory (device memory == host memory here) ----
inline cudaError_t cudaMalloc(void** p, size_t n){ *p = std::malloc(n ? n : 1); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
inline cudaError_t cudaFree(void* p){ std::free(p); return cudaSuccess; }
inline cudaError_t cudaMallocHost(void** p, size_t n){ *p = std::malloc(n ? n : 1); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
inline cudaError_t cudaFreeHost(void* p){ std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* dst,const void* src,size_t n,cudaMemcpyKind){ std::memcpy(dst,src,n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* p,int v,size_t n){ std::memset(p,v,n); return cudaSuccess; }

// ---- streams (no-op; all work is synchronous on CPU) ----
#ifndef CUDA_STREAM_T_DEFINED
#define CUDA_STREAM_T_DEFINED 1
struct CUstream_st { int _=0; };
typedef CUstream_st* cudaStream_t;
#endif
inline cudaError_t cudaStreamCreate(cudaStream_t* s){ *s=new CUstream_st; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t s){ delete s; return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t){ return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* dst,const void* src,size_t n,cudaMemcpyKind,cudaStream_t=0){ std::memcpy(dst,src,n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* p,int v,size_t n,cudaStream_t=0){ std::memset(p,v,n); return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize(){ return cudaSuccess; }

// ---- events (timed with the host clock) ----
struct CUevent_st { std::chrono::steady_clock::time_point t{}; };
typedef CUevent_st* cudaEvent_t;
inline cudaError_t cudaEventCreate(cudaEvent_t* e){ *e=new CUevent_st; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t e){ delete e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e,cudaStream_t=0){ e->t=std::chrono::steady_clock::now(); return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t){ return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t a,cudaEvent_t b){ *ms=std::chrono::duration<float,std::milli>(b->t-a->t).count(); return cudaSuccess; }

// ---- device management (single fake device) ----
struct cudaDeviceProp { char name[256]="cuda_runtime_cpu (host)"; size_t totalGlobalMem=0; int major=0,minor=0; int multiProcessorCount=1; };
inline cudaError_t cudaGetDeviceCount(int* c){ *c=1; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int){ return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d){ *d=0; return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p,int){ *p=cudaDeviceProp(); return cudaSuccess; }
