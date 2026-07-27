// bench_conv.cpp — time a realistic conv layer. Build twice to compare backends:
//   g++ -std=c++17 -O3 -I. bench_conv.cpp -o b_naive && ./b_naive
//   g++ -std=c++17 -O3 -I. -DCUDNN_CPU_USE_EIGEN bench_conv.cpp -o b_eigen && ./b_eigen
#include "cudnn_cpu.h"
#include <cstdio>
#include <chrono>
#include <random>

int main(){
  int N=1,Cin=64,H=128,W=128,Cout=64,k=3,pad=1;
  std::mt19937 rng(0); std::normal_distribution<float> nd(0,1);
  std::vector<float> X(N*Cin*H*W), Wt(Cout*Cin*k*k), Y(N*Cout*H*W,0);
  for(auto&v:X)v=nd(rng); for(auto&v:Wt)v=nd(rng);
  const float one=1.f,zero=0.f;
  cudnnHandle_t h; cudnnCreate(&h);
  cudnnTensorDescriptor_t xd,yd; cudnnFilterDescriptor_t wd; cudnnConvolutionDescriptor_t cd;
  cudnnCreateTensorDescriptor(&xd);cudnnCreateTensorDescriptor(&yd);cudnnCreateFilterDescriptor(&wd);cudnnCreateConvolutionDescriptor(&cd);
  cudnnSetTensor4dDescriptor(xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,Cin,H,W);
  cudnnSetTensor4dDescriptor(yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,Cout,H,W);
  cudnnSetFilter4dDescriptor(wd,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,Cout,Cin,k,k);
  cudnnSetConvolution2dDescriptor(cd,pad,pad,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);

  auto conv=[&]{ cudnnConvolutionForward(h,&one,xd,X.data(),wd,Wt.data(),cd,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,nullptr,0,&zero,yd,Y.data()); };
  conv(); // warm up
  const int R=10;
  auto t0=std::chrono::high_resolution_clock::now();
  for(int i=0;i<R;++i) conv();
  auto t1=std::chrono::high_resolution_clock::now();
  double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/R;
  double gflop=2.0*N*Cout*Cin*k*k*H*W/1e9;
#ifdef CUDNN_CPU_USE_EIGEN
  const char* be="EIGEN";
#else
  const char* be="naive";
#endif
  printf("[%s] conv %dx%dx%dx%d k%d -> %d ch : %.2f ms/iter  (%.1f GFLOP/s)  checksum=%.3f\n",
         be,N,Cin,H,W,k,Cout,ms,gflop/(ms/1e3),Y[0]+Y[Y.size()/2]);
  cudnnDestroy(h);
  return 0;
}
