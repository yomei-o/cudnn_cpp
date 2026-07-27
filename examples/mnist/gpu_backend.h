// gpu_backend.h — the one switch that makes CPU and GPU builds *the same source*.
//
//   CPU  (g++,  no GPU):   just include this — the repo's header-only shims are used.
//   GPU  (nvcc, real GPU): compile with -DMNIST_GPU — the real CUDA libraries are used.
//
// Nothing else in the MNIST programs changes between CPU and GPU. That is the whole
// point of this repo: you write & debug GPU code on a laptop, then flip one flag.
#pragma once

#if defined(MNIST_GPU)
  // ---- real CUDA toolkit (needs an NVIDIA GPU + nvcc) ----
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #ifdef MNIST_USE_CUDNN
    #include <cudnn.h>
  #endif
  #include <thrust/device_ptr.h>
  #include <thrust/device_vector.h>
  #include <thrust/execution_policy.h>
  #include <thrust/functional.h>
  #include <thrust/transform.h>
  #include <thrust/reduce.h>
  #include <thrust/transform_reduce.h>
  #include <thrust/extrema.h>
  #include <thrust/sequence.h>
  #include <thrust/fill.h>
#else
  // ---- CPU shims from this repo (no GPU, no CUDA install) ----
  #include "cuda_runtime_cpu.h"
  #include "cublas_cpu.h"
  #ifdef MNIST_USE_CUDNN
    #include "cudnn_cpu.h"
  #endif
  #include "thrust_cpu.h"
#endif

#include <cstdio>
#include <cstdlib>

// minimal error checks (kept quiet on success)
#define CUDA_CHECK(x)   do{ cudaError_t e_=(x);   if(e_!=cudaSuccess){ std::fprintf(stderr,"CUDA error %s at %s:%d\n",  cudaGetErrorString(e_),__FILE__,__LINE__); std::exit(1);} }while(0)
#define CUBLAS_CHECK(x) do{ cublasStatus_t s_=(x);if(s_!=CUBLAS_STATUS_SUCCESS){ std::fprintf(stderr,"cuBLAS error %d at %s:%d\n",(int)s_,__FILE__,__LINE__); std::exit(1);} }while(0)

// which backend am I? (for banners in the samples)
#if defined(MNIST_GPU)
  static const char* BACKEND_NAME = "GPU (real CUDA)";
#else
  static const char* BACKEND_NAME = "CPU (cudnn_cpp shims)";
#endif
