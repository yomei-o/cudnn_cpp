// mnist_mlp.cpp — a 2-layer MLP on MNIST using ONLY cuBLAS (matmuls) + Thrust
// (elementwise / reductions) + the CUDA runtime (buffers). NO custom kernels, so the
// SAME source builds on CPU (g++, this repo's shims) and GPU (nvcc, real CUDA):
//
//   CPU:  g++  -std=c++17 -O3 -I<repo> [-DCUDNN_CPU_USE_EIGEN -DCUBLAS_CPU_USE_EIGEN] \
//              examples/mnist/mnist_mlp.cpp -o mnist_mlp
//   GPU:  nvcc -std=c++17 -O3 -x cu -DMNIST_GPU -I<repo> \
//              examples/mnist/mnist_mlp.cpp -o mnist_mlp -lcublas
//
// Net:  784 --(W1)--> 128 --ReLU--> --(W2)--> 10 --softmax--> cross-entropy.
// Column-major throughout (cuBLAS convention): a batch X is [features x B], one example
// per column.  Run with --help for options.
#define MNIST_WITH_STB
#define STB_IMAGE_IMPLEMENTATION
#include "gpu_backend.h"
#include "mnist_data.h"
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cstring>

// ---- Thrust functors (annotations are stripped to nothing on the CPU build) ----
struct ReLU        { __host__ __device__ float operator()(float x) const { return x>0?x:0.f; } };
struct ReLUGradMul { __host__ __device__ float operator()(float grad,float act) const { return act>0?grad:0.f; } };
struct Sub         { __host__ __device__ float operator()(float a,float b) const { return a-b; } };
struct Scale       { float s; __host__ __device__ float operator()(float x) const { return x*s; } };
struct ExpShift    { float m; __host__ __device__ float operator()(float x) const { return expf(x-m); } };
struct ExpDiv      { float m,inv; __host__ __device__ float operator()(float x) const { return expf(x-m)*inv; } };

// softmax over each column (rows contiguous per column in column-major)
static void softmax_cols(float* Z,int rows,int cols){
  for(int j=0;j<cols;++j){ auto p=thrust::device_pointer_cast(Z+(size_t)j*rows);
    float m=thrust::reduce(thrust::device,p,p+rows,-1e30f,thrust::maximum<float>());
    float s=thrust::transform_reduce(thrust::device,p,p+rows,ExpShift{m},0.f,thrust::plus<float>());
    thrust::transform(thrust::device,p,p+rows,p,ExpDiv{m,1.f/s});
  }
}

// device buffer helper
struct Buf { float* d=nullptr; size_t n=0;
  void alloc(size_t k){ n=k; CUDA_CHECK(cudaMalloc((void**)&d,sizeof(float)*k)); }
  void from_host(const float* h){ CUDA_CHECK(cudaMemcpy(d,h,sizeof(float)*n,cudaMemcpyHostToDevice)); }
  void to_host(float* h){ CUDA_CHECK(cudaMemcpy(h,d,sizeof(float)*n,cudaMemcpyDeviceToHost)); }
  void free_(){ if(d)cudaFree(d); d=nullptr; }
};

static const int IN=784, HID=128, OUT=10;

int main(int argc,char**argv){
  std::string dataDir="examples/mnist/data", savePath, loadPath, inferPng;
  int epochs=3, B=64, limit=0; float lr=0.1f;
  for(int i=1;i<argc;++i){ std::string a=argv[i];
    auto next=[&]{ return std::string(i+1<argc?argv[++i]:""); };
    if(a=="--data")dataDir=next(); else if(a=="--epochs")epochs=std::stoi(next());
    else if(a=="--batch")B=std::stoi(next()); else if(a=="--lr")lr=std::stof(next());
    else if(a=="--limit")limit=std::stoi(next()); else if(a=="--save")savePath=next();
    else if(a=="--load")loadPath=next(); else if(a=="--infer")inferPng=next();
    else if(a=="--help"){ std::printf("usage: mnist_mlp [--data DIR] [--epochs N] [--batch B] [--lr F] [--limit N] [--save F] [--load F] [--infer PNG]\n"); return 0; }
  }
  std::printf("MNIST MLP  |  backend: %s\n", BACKEND_NAME);

  cublasHandle_t cub; CUBLAS_CHECK(cublasCreate(&cub));
  const float one=1.f, zero=0.f, negLr=-lr;

  // parameters on host (for init / save / load) and device
  std::vector<float> hW1((size_t)HID*IN), hW2((size_t)OUT*HID), hb1(HID,0.f), hb2(OUT,0.f);
  Buf W1,W2,b1,b2; W1.alloc(hW1.size()); W2.alloc(hW2.size()); b1.alloc(HID); b2.alloc(OUT);

  auto init_params=[&]{
    std::mt19937 rng(0);
    std::normal_distribution<float> g1(0.f,std::sqrt(2.f/IN)), g2(0.f,std::sqrt(2.f/HID));
    for(auto&v:hW1)v=g1(rng); for(auto&v:hW2)v=g2(rng);
    W1.from_host(hW1.data()); W2.from_host(hW2.data()); b1.from_host(hb1.data()); b2.from_host(hb2.data());
  };
  auto save_model=[&](const std::string&p){ W1.to_host(hW1.data());W2.to_host(hW2.data());b1.to_host(hb1.data());b2.to_host(hb2.data());
    std::FILE*f=std::fopen(p.c_str(),"wb"); if(!f){std::fprintf(stderr,"cannot write %s\n",p.c_str());return;}
    std::fwrite(hW1.data(),4,hW1.size(),f);std::fwrite(hb1.data(),4,hb1.size(),f);
    std::fwrite(hW2.data(),4,hW2.size(),f);std::fwrite(hb2.data(),4,hb2.size(),f); std::fclose(f);
    std::printf("saved model -> %s\n",p.c_str()); };
  auto load_model=[&](const std::string&p){ std::FILE*f=std::fopen(p.c_str(),"rb"); if(!f){std::fprintf(stderr,"cannot read %s\n",p.c_str());return false;}
    size_t r=0; r+=std::fread(hW1.data(),4,hW1.size(),f);r+=std::fread(hb1.data(),4,hb1.size(),f);
    r+=std::fread(hW2.data(),4,hW2.size(),f);r+=std::fread(hb2.data(),4,hb2.size(),f); std::fclose(f);
    W1.from_host(hW1.data());W2.from_host(hW2.data());b1.from_host(hb1.data());b2.from_host(hb2.data());
    std::printf("loaded model <- %s\n",p.c_str()); return true; };

  // scratch device buffers (max batch B)
  Buf X,ones,Z1,Z2,P,OH,dZ2,dA1,dW1,dW2,db1,db2;
  X.alloc((size_t)IN*B); ones.alloc(B); Z1.alloc((size_t)HID*B); Z2.alloc((size_t)OUT*B);
  P.alloc((size_t)OUT*B); OH.alloc((size_t)OUT*B); dZ2.alloc((size_t)OUT*B); dA1.alloc((size_t)HID*B);
  dW1.alloc(hW1.size()); dW2.alloc(hW2.size()); db1.alloc(HID); db2.alloc(OUT);
  { std::vector<float> o(B,1.f); ones.from_host(o.data()); }

  // forward: fills Z1(=A1 after ReLU) and P; returns nothing. b = actual batch size.
  auto forward=[&](int b){
    // Z1 = W1 * X ; + b1 ; ReLU
    CUBLAS_CHECK(cublasSgemm(cub,CUBLAS_OP_N,CUBLAS_OP_N,HID,b,IN,&one,W1.d,HID,X.d,IN,&zero,Z1.d,HID));
    CUBLAS_CHECK(cublasSger(cub,HID,b,&one,b1.d,1,ones.d,1,Z1.d,HID));
    { auto p=thrust::device_pointer_cast(Z1.d); thrust::transform(thrust::device,p,p+(size_t)HID*b,p,ReLU{}); }
    // Z2 = W2 * A1 ; + b2 ; softmax -> P
    CUBLAS_CHECK(cublasSgemm(cub,CUBLAS_OP_N,CUBLAS_OP_N,OUT,b,HID,&one,W2.d,OUT,Z1.d,HID,&zero,Z2.d,OUT));
    CUBLAS_CHECK(cublasSger(cub,OUT,b,&one,b2.d,1,ones.d,1,Z2.d,OUT));
    CUDA_CHECK(cudaMemcpy(P.d,Z2.d,sizeof(float)*(size_t)OUT*b,cudaMemcpyDeviceToDevice));
    softmax_cols(P.d,OUT,b);
  };

  // ---------- inference-only path ----------
  if(!loadPath.empty() && !inferPng.empty()){
    if(!load_model(loadPath)) return 1;
    std::vector<float> px; if(!mnist::load_png_digit(inferPng,px)) return 1;
    CUDA_CHECK(cudaMemcpy(X.d,px.data(),sizeof(float)*IN,cudaMemcpyHostToDevice));  // one example
    forward(1);
    std::vector<float> p(OUT); CUDA_CHECK(cudaMemcpy(p.data(),P.d,sizeof(float)*OUT,cudaMemcpyDeviceToHost));
    int best=0; for(int c=1;c<OUT;++c)if(p[c]>p[best])best=c;
    std::printf("\nprediction for %s :  %d   (confidence %.1f%%)\n",inferPng.c_str(),best,100.f*p[best]);
    std::printf("probs:"); for(int c=0;c<OUT;++c)std::printf(" %d:%.2f",c,p[c]); std::printf("\n");
    return 0;
  }

  // ---------- training ----------
  mnist::Dataset tr,te;
  if(!mnist::load(dataDir,tr,te)){ std::fprintf(stderr,"failed to load MNIST from %s\n",dataDir.c_str()); return 1; }
  int ntr = limit>0? std::min(limit,tr.n) : tr.n;
  std::printf("train=%d  test=%d  epochs=%d  batch=%d  lr=%.3f\n",ntr,te.n,epochs,B,lr);
  init_params();

  std::vector<int> order(ntr); for(int i=0;i<ntr;++i)order[i]=i;
  std::mt19937 rng(1);
  std::vector<float> hX((size_t)IN*B), hOH((size_t)OUT*B), hP((size_t)OUT*B);

  cudaEvent_t evStart,evEnd; cudaEventCreate(&evStart); cudaEventCreate(&evEnd);
  cudaEventRecord(evStart);
  for(int e=0;e<epochs;++e){
    std::shuffle(order.begin(),order.end(),rng);
    double loss=0; int correct=0, seen=0;
    for(int s=0;s<ntr;s+=B){
      int b=std::min(B,ntr-s);
      // assemble batch (column-major: example j -> column j, contiguous 784)
      std::memset(hOH.data(),0,sizeof(float)*(size_t)OUT*b);
      for(int j=0;j<b;++j){ int idx=order[s+j];
        std::memcpy(&hX[(size_t)j*IN], &tr.images[(size_t)idx*IN], sizeof(float)*IN);
        hOH[(size_t)j*OUT + tr.labels[idx]] = 1.f;
      }
      CUDA_CHECK(cudaMemcpy(X.d ,hX.data(), sizeof(float)*(size_t)IN*b, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(OH.d,hOH.data(),sizeof(float)*(size_t)OUT*b,cudaMemcpyHostToDevice));

      forward(b);

      // metrics (host)
      CUDA_CHECK(cudaMemcpy(hP.data(),P.d,sizeof(float)*(size_t)OUT*b,cudaMemcpyDeviceToHost));
      for(int j=0;j<b;++j){ int lab=tr.labels[order[s+j]]; loss+=-std::log(std::max(hP[(size_t)j*OUT+lab],1e-8f));
        int best=0; for(int c=1;c<OUT;++c)if(hP[(size_t)j*OUT+c]>hP[(size_t)j*OUT+best])best=c;
        if(best==lab)++correct; } seen+=b;

      // backward:  dZ2 = (P - onehot)/b
      { auto p=thrust::device_pointer_cast(P.d); auto o=thrust::device_pointer_cast(OH.d); auto g=thrust::device_pointer_cast(dZ2.d);
        thrust::transform(thrust::device,p,p+(size_t)OUT*b,o,g,Sub{});
        thrust::transform(thrust::device,g,g+(size_t)OUT*b,g,Scale{1.f/b}); }
      // dW2 = dZ2 * A1^T ; db2 = dZ2 * 1
      CUBLAS_CHECK(cublasSgemm(cub,CUBLAS_OP_N,CUBLAS_OP_T,OUT,HID,b,&one,dZ2.d,OUT,Z1.d,HID,&zero,dW2.d,OUT));
      CUBLAS_CHECK(cublasSgemv(cub,CUBLAS_OP_N,OUT,b,&one,dZ2.d,OUT,ones.d,1,&zero,db2.d,1));
      // dA1 = W2^T * dZ2 ; dZ1 = dA1 .* (A1>0)
      CUBLAS_CHECK(cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,HID,b,OUT,&one,W2.d,OUT,dZ2.d,OUT,&zero,dA1.d,HID));
      { auto ga=thrust::device_pointer_cast(dA1.d); auto a=thrust::device_pointer_cast(Z1.d);
        thrust::transform(thrust::device,ga,ga+(size_t)HID*b,a,ga,ReLUGradMul{}); }
      // dW1 = dZ1 * X^T ; db1 = dZ1 * 1
      CUBLAS_CHECK(cublasSgemm(cub,CUBLAS_OP_N,CUBLAS_OP_T,HID,IN,b,&one,dA1.d,HID,X.d,IN,&zero,dW1.d,HID));
      CUBLAS_CHECK(cublasSgemv(cub,CUBLAS_OP_N,HID,b,&one,dA1.d,HID,ones.d,1,&zero,db1.d,1));
      // SGD:  P -= lr * grad
      CUBLAS_CHECK(cublasSaxpy(cub,(int)hW1.size(),&negLr,dW1.d,1,W1.d,1));
      CUBLAS_CHECK(cublasSaxpy(cub,(int)hW2.size(),&negLr,dW2.d,1,W2.d,1));
      CUBLAS_CHECK(cublasSaxpy(cub,HID,&negLr,db1.d,1,b1.d,1));
      CUBLAS_CHECK(cublasSaxpy(cub,OUT,&negLr,db2.d,1,b2.d,1));
    }
    std::printf("epoch %d/%d  loss %.4f  train-acc %.2f%%\n",e+1,epochs,loss/seen,100.0*correct/seen);
  }
  cudaEventRecord(evEnd); cudaEventSynchronize(evEnd);
  float trainMs=0; cudaEventElapsedTime(&trainMs,evStart,evEnd);
  std::printf(">> trained on %s in %.1f ms  (%.0f images/sec)\n", BACKEND_NAME, trainMs, (double)ntr*epochs/(trainMs/1000.0));

  // test accuracy
  int correct=0;
  for(int s=0;s<te.n;s+=B){ int b=std::min(B,te.n-s);
    for(int j=0;j<b;++j) std::memcpy(&hX[(size_t)j*IN],&te.images[(size_t)(s+j)*IN],sizeof(float)*IN);
    CUDA_CHECK(cudaMemcpy(X.d,hX.data(),sizeof(float)*(size_t)IN*b,cudaMemcpyHostToDevice));
    forward(b);
    CUDA_CHECK(cudaMemcpy(hP.data(),P.d,sizeof(float)*(size_t)OUT*b,cudaMemcpyDeviceToHost));
    for(int j=0;j<b;++j){ int best=0; for(int c=1;c<OUT;++c)if(hP[(size_t)j*OUT+c]>hP[(size_t)j*OUT+best])best=c;
      if(best==te.labels[s+j])++correct; }
  }
  std::printf("TEST ACCURACY: %.2f%%  (%d/%d)\n",100.0*correct/te.n,correct,te.n);

  if(!savePath.empty()) save_model(savePath);
  if(!inferPng.empty()){ std::vector<float> px; if(mnist::load_png_digit(inferPng,px)){ CUDA_CHECK(cudaMemcpy(X.d,px.data(),sizeof(float)*IN,cudaMemcpyHostToDevice)); forward(1);
    std::vector<float> p(OUT); CUDA_CHECK(cudaMemcpy(p.data(),P.d,sizeof(float)*OUT,cudaMemcpyDeviceToHost));
    int best=0; for(int c=1;c<OUT;++c)if(p[c]>p[best])best=c;
    std::printf("infer %s -> %d (%.1f%%)\n",inferPng.c_str(),best,100.f*p[best]); } }

  cublasDestroy(cub);
  return 0;
}
