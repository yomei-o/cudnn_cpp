// mnist_cnn.cpp — a small CNN on MNIST using cuDNN (conv / ReLU / maxpool / softmax) +
// cuBLAS (the final fully-connected layer) + Thrust (loss elementwise) + CUDA runtime
// (buffers). NO custom kernels, so the SAME source builds on CPU and GPU:
//
//   CPU:  g++  -std=c++17 -O3 -I<repo> -DCUDNN_CPU_USE_EIGEN -DCUBLAS_CPU_USE_EIGEN \
//              examples/mnist/mnist_cnn.cpp -o mnist_cnn
//   GPU:  nvcc -std=c++17 -O3 -x cu -DMNIST_GPU -I<repo> \
//              examples/mnist/mnist_cnn.cpp -o mnist_cnn -lcudnn -lcublas
//
// Net:  1x28x28 --conv(8,3x3)+ReLU+pool2--> 8x14x14 --conv(16,3x3)+ReLU+pool2--> 16x7x7
//       --FC--> 10 --softmax--> cross-entropy.
#define MNIST_USE_CUDNN
#define MNIST_WITH_STB
#define STB_IMAGE_IMPLEMENTATION
#include "gpu_backend.h"
#include "mnist_data.h"
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cstring>

#define CUDNN_CHECK(x) do{ cudnnStatus_t s_=(x); if(s_!=CUDNN_STATUS_SUCCESS){ std::fprintf(stderr,"cuDNN error %d at %s:%d\n",(int)s_,__FILE__,__LINE__); std::exit(1);} }while(0)

struct Sub   { __host__ __device__ float operator()(float a,float b) const { return a-b; } };
struct Scale { float s; __host__ __device__ float operator()(float x) const { return x*s; } };

struct Buf { float* d=nullptr; size_t n=0;
  void alloc(size_t k){ n=k; CUDA_CHECK(cudaMalloc((void**)&d,sizeof(float)*k)); }
  void from_host(const float* h,size_t k){ CUDA_CHECK(cudaMemcpy(d,h,sizeof(float)*k,cudaMemcpyHostToDevice)); }
  void to_host(float* h,size_t k){ CUDA_CHECK(cudaMemcpy(h,d,sizeof(float)*k,cudaMemcpyDeviceToHost)); }
};

// sizes
static const int C0=1, S0=28;          // input 1x28x28
static const int C1=8,  S1=28, P1=14;  // conv1 -> 8x28x28 -> pool 8x14x14
static const int C2=16, S2=14, P2=7;   // conv2 -> 16x14x14 -> pool 16x7x7
static const int FEAT=C2*P2*P2;        // 16*7*7 = 784
static const int OUT=10;

int main(int argc,char**argv){
  std::string dataDir="examples/mnist/data", savePath, loadPath, inferPng;
  int epochs=1, B=64, limit=0; float lr=0.05f;
  for(int i=1;i<argc;++i){ std::string a=argv[i]; auto next=[&]{ return std::string(i+1<argc?argv[++i]:""); };
    if(a=="--data")dataDir=next(); else if(a=="--epochs")epochs=std::stoi(next());
    else if(a=="--batch")B=std::stoi(next()); else if(a=="--lr")lr=std::stof(next());
    else if(a=="--limit")limit=std::stoi(next()); else if(a=="--save")savePath=next();
    else if(a=="--load")loadPath=next(); else if(a=="--infer")inferPng=next();
    else if(a=="--help"){ std::printf("usage: mnist_cnn [--data DIR][--epochs N][--batch B][--lr F][--limit N][--save F][--load F][--infer PNG]\n"); return 0; }
  }
  std::printf("MNIST CNN  |  backend: %s\n", BACKEND_NAME);

  cudnnHandle_t cud; CUDNN_CHECK(cudnnCreate(&cud));
  cublasHandle_t cub; CUBLAS_CHECK(cublasCreate(&cub));
  const float one=1.f, zero=0.f, negLr=-lr;

  // ---- parameters (host mirror + device) ----
  std::vector<float> hWc1(C1*C0*9), hbc1(C1,0), hWc2(C2*C1*9), hbc2(C2,0), hWfc(OUT*FEAT), hbfc(OUT,0);
  Buf Wc1,bc1,Wc2,bc2,Wfc,bfc; Wc1.alloc(hWc1.size());bc1.alloc(C1);Wc2.alloc(hWc2.size());bc2.alloc(C2);Wfc.alloc(hWfc.size());bfc.alloc(OUT);
  auto init=[&]{ std::mt19937 r(0);
    auto he=[&](std::vector<float>&w,int fan){ std::normal_distribution<float> g(0,std::sqrt(2.f/fan)); for(auto&v:w)v=g(r); };
    he(hWc1,C0*9); he(hWc2,C1*9); he(hWfc,FEAT);
    Wc1.from_host(hWc1.data(),hWc1.size()); bc1.from_host(hbc1.data(),C1);
    Wc2.from_host(hWc2.data(),hWc2.size()); bc2.from_host(hbc2.data(),C2);
    Wfc.from_host(hWfc.data(),hWfc.size()); bfc.from_host(hbfc.data(),OUT); };
  auto save=[&](const std::string&p){ Wc1.to_host(hWc1.data(),hWc1.size());bc1.to_host(hbc1.data(),C1);Wc2.to_host(hWc2.data(),hWc2.size());bc2.to_host(hbc2.data(),C2);Wfc.to_host(hWfc.data(),hWfc.size());bfc.to_host(hbfc.data(),OUT);
    std::FILE*f=std::fopen(p.c_str(),"wb"); if(!f)return; for(auto*v:{&hWc1,&hbc1,&hWc2,&hbc2,&hWfc,&hbfc})std::fwrite(v->data(),4,v->size(),f); std::fclose(f); std::printf("saved model -> %s\n",p.c_str()); };
  auto load=[&](const std::string&p){ std::FILE*f=std::fopen(p.c_str(),"rb"); if(!f){std::fprintf(stderr,"cannot read %s\n",p.c_str());return false;}
    for(auto*v:{&hWc1,&hbc1,&hWc2,&hbc2,&hWfc,&hbfc}){ size_t rr=std::fread(v->data(),4,v->size(),f); (void)rr; } std::fclose(f);
    Wc1.from_host(hWc1.data(),hWc1.size());bc1.from_host(hbc1.data(),C1);Wc2.from_host(hWc2.data(),hWc2.size());bc2.from_host(hbc2.data(),C2);Wfc.from_host(hWfc.data(),hWfc.size());bfc.from_host(hbfc.data(),OUT);
    std::printf("loaded model <- %s\n",p.c_str()); return true; };

  // ---- activation / gradient buffers (max batch B) ----
  Buf x,c1,p1,c2,p2,logit,P,OH,ones,ws;
  x.alloc((size_t)B*C0*S0*S0); c1.alloc((size_t)B*C1*S1*S1); p1.alloc((size_t)B*C1*P1*P1);
  c2.alloc((size_t)B*C2*S2*S2); p2.alloc((size_t)B*FEAT); logit.alloc((size_t)B*OUT); P.alloc((size_t)B*OUT);
  OH.alloc((size_t)B*OUT); ones.alloc(B); ws.alloc((size_t)8*1024*1024); // 32MB conv workspace
  Buf dlogit,dp2,dc2,dp1,dc1,dWc1,dbc1,dWc2,dbc2,dWfc,dbfc;
  dlogit.alloc((size_t)B*OUT); dp2.alloc((size_t)B*FEAT); dc2.alloc((size_t)B*C2*S2*S2);
  dp1.alloc((size_t)B*C1*P1*P1); dc1.alloc((size_t)B*C1*S1*S1);
  dWc1.alloc(hWc1.size());dbc1.alloc(C1);dWc2.alloc(hWc2.size());dbc2.alloc(C2);dWfc.alloc(hWfc.size());dbfc.alloc(OUT);
  { std::vector<float> o(B,1.f); ones.from_host(o.data(),B); }

  // ---- descriptors ----
  cudnnTensorDescriptor_t xd,c1d,p1d,c2d,p2d,ld,b1d,b2d;
  cudnnFilterDescriptor_t w1d,w2d; cudnnConvolutionDescriptor_t conv; cudnnPoolingDescriptor_t pool; cudnnActivationDescriptor_t relu;
  for(auto*t:{&xd,&c1d,&p1d,&c2d,&p2d,&ld}) cudnnCreateTensorDescriptor(t);
  cudnnCreateTensorDescriptor(&b1d);cudnnCreateTensorDescriptor(&b2d);
  cudnnCreateFilterDescriptor(&w1d);cudnnCreateFilterDescriptor(&w2d);
  cudnnCreateConvolutionDescriptor(&conv);cudnnCreatePoolingDescriptor(&pool);cudnnCreateActivationDescriptor(&relu);
  cudnnSetFilter4dDescriptor(w1d,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,C1,C0,3,3);
  cudnnSetFilter4dDescriptor(w2d,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,C2,C1,3,3);
  cudnnSetConvolution2dDescriptor(conv,1,1,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
  cudnnSetPooling2dDescriptor(pool,CUDNN_POOLING_MAX,CUDNN_NOT_PROPAGATE_NAN,2,2,0,0,2,2);
  cudnnSetActivationDescriptor(relu,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,0);
  cudnnSetTensor4dDescriptor(b1d,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,C1,1,1);
  cudnnSetTensor4dDescriptor(b2d,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,C2,1,1);
  auto set_n=[&](int b){ // tensor descriptors depend on batch size
    cudnnSetTensor4dDescriptor(xd ,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,b,C0,S0,S0);
    cudnnSetTensor4dDescriptor(c1d,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,b,C1,S1,S1);
    cudnnSetTensor4dDescriptor(p1d,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,b,C1,P1,P1);
    cudnnSetTensor4dDescriptor(c2d,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,b,C2,S2,S2);
    cudnnSetTensor4dDescriptor(p2d,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,b,C2,P2,P2);
    cudnnSetTensor4dDescriptor(ld ,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,b,OUT,1,1); };
  const size_t wss = ws.n*sizeof(float);

  // ---- forward (fills p2, logit, P) ----
  auto forward=[&](int b){ set_n(b);
    cudnnConvolutionForward(cud,&one,xd,x.d,w1d,Wc1.d,conv,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,ws.d,wss,&zero,c1d,c1.d);
    cudnnAddTensor(cud,&one,b1d,bc1.d,&one,c1d,c1.d);
    cudnnActivationForward(cud,relu,&one,c1d,c1.d,&zero,c1d,c1.d);
    cudnnPoolingForward(cud,pool,&one,c1d,c1.d,&zero,p1d,p1.d);
    cudnnConvolutionForward(cud,&one,p1d,p1.d,w2d,Wc2.d,conv,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,ws.d,wss,&zero,c2d,c2.d);
    cudnnAddTensor(cud,&one,b2d,bc2.d,&one,c2d,c2.d);
    cudnnActivationForward(cud,relu,&one,c2d,c2.d,&zero,c2d,c2.d);
    cudnnPoolingForward(cud,pool,&one,c2d,c2.d,&zero,p2d,p2.d);
    // FC: logit[OUT x b] = Wfc[OUT x FEAT] * p2[FEAT x b]  (p2 NCHW == col-major FEATxb)
    cublasSgemm(cub,CUBLAS_OP_N,CUBLAS_OP_N,OUT,b,FEAT,&one,Wfc.d,OUT,p2.d,FEAT,&zero,logit.d,OUT);
    cublasSger(cub,OUT,b,&one,bfc.d,1,ones.d,1,logit.d,OUT);
    cudnnSoftmaxForward(cud,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_CHANNEL,&one,ld,logit.d,&zero,ld,P.d);
  };

  // ---- inference-only ----
  if(!loadPath.empty() && !inferPng.empty()){
    if(!load(loadPath)) return 1;
    std::vector<float> px; if(!mnist::load_png_digit(inferPng,px)) return 1;
    CUDA_CHECK(cudaMemcpy(x.d,px.data(),sizeof(float)*C0*S0*S0,cudaMemcpyHostToDevice));
    forward(1);
    std::vector<float> p(OUT); P.to_host(p.data(),OUT);
    int best=0; for(int c=1;c<OUT;++c)if(p[c]>p[best])best=c;
    std::printf("\nprediction for %s :  %d   (confidence %.1f%%)\n",inferPng.c_str(),best,100.f*p[best]);
    std::printf("probs:"); for(int c=0;c<OUT;++c)std::printf(" %d:%.2f",c,p[c]); std::printf("\n");
    return 0;
  }

  // ---- training ----
  mnist::Dataset tr,te; if(!mnist::load(dataDir,tr,te)){ std::fprintf(stderr,"failed to load MNIST from %s\n",dataDir.c_str()); return 1; }
  int ntr = limit>0? std::min(limit,tr.n):tr.n;
  std::printf("train=%d test=%d epochs=%d batch=%d lr=%.3f\n",ntr,te.n,epochs,B,lr);
  init();
  std::vector<int> order(ntr); for(int i=0;i<ntr;++i)order[i]=i; std::mt19937 rng(1);
  std::vector<float> hx((size_t)B*C0*S0*S0), hOH((size_t)B*OUT), hP((size_t)B*OUT);

  cudaEvent_t evStart,evEnd; cudaEventCreate(&evStart); cudaEventCreate(&evEnd);
  cudaEventRecord(evStart);
  for(int e=0;e<epochs;++e){ std::shuffle(order.begin(),order.end(),rng);
    double loss=0; int correct=0,seen=0;
    for(int s=0;s<ntr;s+=B){ int b=std::min(B,ntr-s);
      std::memset(hOH.data(),0,sizeof(float)*(size_t)OUT*b);
      for(int j=0;j<b;++j){ int idx=order[s+j];
        std::memcpy(&hx[(size_t)j*C0*S0*S0],&tr.images[(size_t)idx*784],sizeof(float)*784);
        hOH[(size_t)j*OUT+tr.labels[idx]]=1.f; }
      CUDA_CHECK(cudaMemcpy(x.d ,hx.data(), sizeof(float)*(size_t)b*C0*S0*S0,cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(OH.d,hOH.data(),sizeof(float)*(size_t)b*OUT,cudaMemcpyHostToDevice));

      forward(b);
      P.to_host(hP.data(),(size_t)OUT*b);
      for(int j=0;j<b;++j){ int lab=tr.labels[order[s+j]]; loss+=-std::log(std::max(hP[(size_t)j*OUT+lab],1e-8f));
        int best=0; for(int c=1;c<OUT;++c)if(hP[(size_t)j*OUT+c]>hP[(size_t)j*OUT+best])best=c; if(best==lab)++correct; } seen+=b;

      // dlogit = (P - OH)/b
      { auto p=thrust::device_pointer_cast(P.d),o=thrust::device_pointer_cast(OH.d),g=thrust::device_pointer_cast(dlogit.d);
        thrust::transform(thrust::device,p,p+(size_t)OUT*b,o,g,Sub{});
        thrust::transform(thrust::device,g,g+(size_t)OUT*b,g,Scale{1.f/b}); }
      // FC backward
      cublasSgemm(cub,CUBLAS_OP_N,CUBLAS_OP_T,OUT,FEAT,b,&one,dlogit.d,OUT,p2.d,FEAT,&zero,dWfc.d,OUT);
      cublasSgemv(cub,CUBLAS_OP_N,OUT,b,&one,dlogit.d,OUT,ones.d,1,&zero,dbfc.d,1);
      cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,FEAT,b,OUT,&one,Wfc.d,OUT,dlogit.d,OUT,&zero,dp2.d,FEAT);
      // pool2 bwd -> dc2(as dy of relu2); relu2 bwd (in place, x==y==c2)
      cudnnPoolingBackward(cud,pool,&one,p2d,p2.d,p2d,dp2.d,c2d,c2.d,&zero,c2d,dc2.d);
      cudnnActivationBackward(cud,relu,&one,c2d,c2.d,c2d,dc2.d,c2d,c2.d,&zero,c2d,dc2.d);
      // conv2 bwd: dWc2, dbc2, dp1
      cudnnConvolutionBackwardFilter(cud,&one,p1d,p1.d,c2d,dc2.d,conv,CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,ws.d,wss,&zero,w2d,dWc2.d);
      cudnnConvolutionBackwardBias(cud,&one,c2d,dc2.d,&zero,b2d,dbc2.d);
      cudnnConvolutionBackwardData(cud,&one,w2d,Wc2.d,c2d,dc2.d,conv,CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,ws.d,wss,&zero,p1d,dp1.d);
      // pool1 bwd -> dc1; relu1 bwd
      cudnnPoolingBackward(cud,pool,&one,p1d,p1.d,p1d,dp1.d,c1d,c1.d,&zero,c1d,dc1.d);
      cudnnActivationBackward(cud,relu,&one,c1d,c1.d,c1d,dc1.d,c1d,c1.d,&zero,c1d,dc1.d);
      // conv1 bwd: dWc1, dbc1
      cudnnConvolutionBackwardFilter(cud,&one,xd,x.d,c1d,dc1.d,conv,CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,ws.d,wss,&zero,w1d,dWc1.d);
      cudnnConvolutionBackwardBias(cud,&one,c1d,dc1.d,&zero,b1d,dbc1.d);
      // SGD
      cublasSaxpy(cub,(int)hWc1.size(),&negLr,dWc1.d,1,Wc1.d,1); cublasSaxpy(cub,C1,&negLr,dbc1.d,1,bc1.d,1);
      cublasSaxpy(cub,(int)hWc2.size(),&negLr,dWc2.d,1,Wc2.d,1); cublasSaxpy(cub,C2,&negLr,dbc2.d,1,bc2.d,1);
      cublasSaxpy(cub,(int)hWfc.size(),&negLr,dWfc.d,1,Wfc.d,1); cublasSaxpy(cub,OUT,&negLr,dbfc.d,1,bfc.d,1);
      if(seen % (B*100)==0) std::printf("  epoch %d  %d/%d  loss %.4f  acc %.2f%%\r",e+1,seen,ntr,loss/seen,100.0*correct/seen), std::fflush(stdout);
    }
    std::printf("epoch %d/%d  loss %.4f  train-acc %.2f%%                 \n",e+1,epochs,loss/seen,100.0*correct/seen);
  }
  cudaEventRecord(evEnd); cudaEventSynchronize(evEnd);
  float trainMs=0; cudaEventElapsedTime(&trainMs,evStart,evEnd);
  std::printf(">> trained on %s in %.1f ms  (%.0f images/sec)\n", BACKEND_NAME, trainMs, (double)ntr*epochs/(trainMs/1000.0));

  // test
  int correct=0;
  for(int s=0;s<te.n;s+=B){ int b=std::min(B,te.n-s);
    for(int j=0;j<b;++j) std::memcpy(&hx[(size_t)j*784],&te.images[(size_t)(s+j)*784],sizeof(float)*784);
    CUDA_CHECK(cudaMemcpy(x.d,hx.data(),sizeof(float)*(size_t)b*784,cudaMemcpyHostToDevice));
    forward(b); P.to_host(hP.data(),(size_t)OUT*b);
    for(int j=0;j<b;++j){ int best=0; for(int c=1;c<OUT;++c)if(hP[(size_t)j*OUT+c]>hP[(size_t)j*OUT+best])best=c; if(best==te.labels[s+j])++correct; } }
  std::printf("TEST ACCURACY: %.2f%%  (%d/%d)\n",100.0*correct/te.n,correct,te.n);

  if(!savePath.empty()) save(savePath);
  if(!inferPng.empty()){ std::vector<float> px; if(mnist::load_png_digit(inferPng,px)){ CUDA_CHECK(cudaMemcpy(x.d,px.data(),sizeof(float)*784,cudaMemcpyHostToDevice)); forward(1);
    std::vector<float> p(OUT); P.to_host(p.data(),OUT); int best=0; for(int c=1;c<OUT;++c)if(p[c]>p[best])best=c;
    std::printf("infer %s -> %d (%.1f%%)\n",inferPng.c_str(),best,100.f*p[best]); } }

  cudnnDestroy(cud); cublasDestroy(cub);
  return 0;
}
