// test_backward.cpp — verify every backward op by finite-difference VJP check.
// For a fixed random dy, define L(params) = <dy, forward(params)>. Then the analytic
// gradient returned by the cudnn*Backward call must equal dL/dparams (central diff).
//   g++ -std=c++17 -O2 -I. test_backward.cpp -o tb && ./tb
//   g++ -std=c++17 -O3 -I. -DCUDNN_CPU_USE_EIGEN test_backward.cpp -o tbe && ./tbe
#include "cudnn_cpu.h"
#include <cstdio>
#include <random>
#include <functional>
#include <vector>

static std::mt19937 rng(7);
static std::normal_distribution<float> nd(0,1);
static int fails=0;
// relative max diff between analytic grad and numeric grad
static void check(const char* name, const std::vector<float>& an, const std::vector<float>& num, float tol=2e-2f){
  float md=0,scale=1e-6f;
  for(size_t i=0;i<an.size();++i){ md=std::max(md,std::fabs(an[i]-num[i])); scale=std::max(scale,std::fabs(num[i])); }
  float rel=md/scale;
  printf("  %-34s rel|diff| = %.2e  %s\n",name,rel,rel<tol?"OK":"FAIL"); if(rel>=tol)++fails;
}
// numeric grad of L() w.r.t each entry of param (central difference)
static std::vector<float> numgrad(std::vector<float>& param, std::function<double()> L){
  const float e=1e-3f; std::vector<float> gpr(param.size());
  for(size_t i=0;i<param.size();++i){ float o=param[i];
    param[i]=o+e; double lp=L(); param[i]=o-e; double lm=L(); param[i]=o; gpr[i]=(float)((lp-lm)/(2*e)); }
  return gpr;
}
static double dot(const std::vector<float>& a,const std::vector<float>& b){ double s=0; for(size_t i=0;i<a.size();++i)s+=(double)a[i]*b[i]; return s; }

int main(){
  cudnnHandle_t h; cudnnCreate(&h); const float one=1.f,zero=0.f;

  // ================= convolution backward (data / filter / bias) =================
  {
    int N=2,Cin=4,H=7,W=7,Cout=6,k=3,pad=1,stride=1,G=2; int cig=Cin/G,cog=Cout/G;
    int OH=(H+2*pad-k)/stride+1, OW=(W+2*pad-k)/stride+1;
    std::vector<float> X(N*Cin*H*W),Wt(Cout*cig*k*k),Bias(Cout),dY(N*Cout*OH*OW),Y(N*Cout*OH*OW);
    for(auto&v:X)v=nd(rng); for(auto&v:Wt)v=nd(rng); for(auto&v:Bias)v=nd(rng); for(auto&v:dY)v=nd(rng);
    cudnnTensorDescriptor_t xd,yd,bd; cudnnFilterDescriptor_t wd; cudnnConvolutionDescriptor_t cd;
    cudnnCreateTensorDescriptor(&xd);cudnnCreateTensorDescriptor(&yd);cudnnCreateTensorDescriptor(&bd);
    cudnnCreateFilterDescriptor(&wd);cudnnCreateConvolutionDescriptor(&cd);
    cudnnSetTensor4dDescriptor(xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,Cin,H,W);
    cudnnSetTensor4dDescriptor(yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,Cout,OH,OW);
    cudnnSetTensor4dDescriptor(bd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,Cout,1,1);
    cudnnSetFilter4dDescriptor(wd,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,Cout,cig,k,k);
    cudnnSetConvolution2dDescriptor(cd,pad,pad,stride,stride,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(cd,G);
    // forward that also adds bias, so L exercises bias too
    auto fwd=[&]{ cudnnConvolutionForward(h,&one,xd,X.data(),wd,Wt.data(),cd,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,nullptr,0,&zero,yd,Y.data());
                  cudnnAddTensor(h,&one,bd,Bias.data(),&one,yd,Y.data()); };
    auto L=[&]{ fwd(); return dot(Y,dY); };
    // analytic
    std::vector<float> dX(X.size(),0),dWt(Wt.size(),0),dB(Bias.size(),0);
    cudnnConvolutionBackwardData(h,&one,wd,Wt.data(),yd,dY.data(),cd,CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,nullptr,0,&zero,xd,dX.data());
    cudnnConvolutionBackwardFilter(h,&one,xd,X.data(),yd,dY.data(),cd,CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,nullptr,0,&zero,wd,dWt.data());
    cudnnConvolutionBackwardBias(h,&one,yd,dY.data(),&zero,bd,dB.data());
    check("conv backward data (dX)",  dX,  numgrad(X,L));
    check("conv backward filter (dW)",dWt, numgrad(Wt,L));
    check("conv backward bias (dB)",  dB,  numgrad(Bias,L));
  }

  // ================= activation backward (SiLU-relevant: sigmoid) =================
  {
    int n=48; std::vector<float> X(n),Y(n),dY(n); for(auto&v:X)v=nd(rng); for(auto&v:dY)v=nd(rng);
    cudnnTensorDescriptor_t td; cudnnCreateTensorDescriptor(&td);
    cudnnSetTensor4dDescriptor(td,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,n,1,1);
    for(int mi=0; mi<3; ++mi){
      cudnnActivationMode_t modes[3]={CUDNN_ACTIVATION_SIGMOID,CUDNN_ACTIVATION_RELU,CUDNN_ACTIVATION_TANH};
      const char* nm[3]={"activation bwd sigmoid","activation bwd relu","activation bwd tanh"};
      cudnnActivationDescriptor_t ad; cudnnCreateActivationDescriptor(&ad);
      cudnnSetActivationDescriptor(ad,modes[mi],CUDNN_NOT_PROPAGATE_NAN,0);
      auto L=[&]{ cudnnActivationForward(h,ad,&one,td,X.data(),&zero,td,Y.data()); return dot(Y,dY); };
      L(); std::vector<float> dX(n,0);
      cudnnActivationBackward(h,ad,&one,td,Y.data(),td,dY.data(),td,X.data(),&zero,td,dX.data());
      check(nm[mi], dX, numgrad(X,L));
    }
  }

  // ================= pooling backward (MAX 3x3 s2, AVG) =================
  {
    int N=1,C=2,H=8,W=8,k=3,pad=1,s=2; int OH=(H+2*pad-k)/s+1,OW=(W+2*pad-k)/s+1;
    std::vector<float> X(N*C*H*W),Y(N*C*OH*OW),dY(N*C*OH*OW);
    for(auto&v:X)v=nd(rng); for(auto&v:dY)v=nd(rng);
    cudnnTensorDescriptor_t xd,yd; cudnnPoolingDescriptor_t pd;
    cudnnCreateTensorDescriptor(&xd);cudnnCreateTensorDescriptor(&yd);cudnnCreatePoolingDescriptor(&pd);
    cudnnSetTensor4dDescriptor(xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,H,W);
    cudnnSetTensor4dDescriptor(yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,OH,OW);
    for(int mi=0;mi<2;++mi){
      cudnnPoolingMode_t modes[2]={CUDNN_POOLING_MAX,CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING};
      const char* nm[2]={"pooling bwd MAX","pooling bwd AVG"};
      cudnnSetPooling2dDescriptor(pd,modes[mi],CUDNN_NOT_PROPAGATE_NAN,k,k,pad,pad,s,s);
      auto L=[&]{ cudnnPoolingForward(h,pd,&one,xd,X.data(),&zero,yd,Y.data()); return dot(Y,dY); };
      L(); std::vector<float> dX(X.size(),0);
      cudnnPoolingBackward(h,pd,&one,yd,Y.data(),yd,dY.data(),xd,X.data(),&zero,xd,dX.data());
      check(nm[mi], dX, numgrad(X,L), 3e-2f);
    }
  }

  // ================= softmax backward (channel, DFL) =================
  {
    int N=1,C=16,H=2,W=2; std::vector<float> X(N*C*H*W),Y(X.size()),dY(X.size());
    for(auto&v:X)v=nd(rng); for(auto&v:dY)v=nd(rng);
    cudnnTensorDescriptor_t td; cudnnCreateTensorDescriptor(&td);
    cudnnSetTensor4dDescriptor(td,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,H,W);
    auto L=[&]{ cudnnSoftmaxForward(h,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_CHANNEL,&one,td,X.data(),&zero,td,Y.data()); return dot(Y,dY); };
    L(); std::vector<float> dX(X.size(),0);
    cudnnSoftmaxBackward(h,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_CHANNEL,&one,td,Y.data(),td,dY.data(),&zero,td,dX.data());
    check("softmax backward (channel)", dX, numgrad(X,L));
  }

  // ================= batchnorm forward-training backward (dX, dScale, dBias) =================
  {
    int N=4,C=3,H=4,W=4; double eps=1e-5;
    std::vector<float> X(N*C*H*W),Y(X.size()),dY(X.size()),g(C),bt(C),rm(C,0),rv(C,1),sm(C),si(C);
    for(auto&v:X)v=nd(rng); for(auto&v:dY)v=nd(rng); for(int c=0;c<C;++c){g[c]=nd(rng)+1;bt[c]=nd(rng);}
    cudnnTensorDescriptor_t td,pd; cudnnCreateTensorDescriptor(&td);cudnnCreateTensorDescriptor(&pd);
    cudnnSetTensor4dDescriptor(td,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,H,W);
    cudnnSetTensor4dDescriptor(pd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,C,1,1);
    // L over x (running stats irrelevant to L; recomputed each forward)
    auto Lx=[&]{ std::vector<float> rmx(C,0),rvx(C,1); cudnnBatchNormalizationForwardTraining(h,CUDNN_BATCHNORM_SPATIAL,&one,&zero,td,X.data(),td,Y.data(),pd,g.data(),bt.data(),0.1,rmx.data(),rvx.data(),eps,nullptr,nullptr); return dot(Y,dY); };
    // reference forward to populate saved stats
    cudnnBatchNormalizationForwardTraining(h,CUDNN_BATCHNORM_SPATIAL,&one,&zero,td,X.data(),td,Y.data(),pd,g.data(),bt.data(),0.1,rm.data(),rv.data(),eps,sm.data(),si.data());
    std::vector<float> dX(X.size(),0),dG(C,0),dB(C,0);
    cudnnBatchNormalizationBackward(h,CUDNN_BATCHNORM_SPATIAL,&one,&zero,&one,&zero,td,X.data(),td,dY.data(),td,dX.data(),pd,g.data(),dG.data(),dB.data(),eps,sm.data(),si.data());
    check("batchnorm bwd dX",    dX, numgrad(X, Lx));
    check("batchnorm bwd dScale",dG, numgrad(g, Lx));
    check("batchnorm bwd dBias", dB, numgrad(bt,Lx));
  }

  cudnnDestroy(h);
  printf("\n%s\n", fails? "SOME TESTS FAILED":"ALL BACKWARD TESTS PASSED");
  return fails?1:0;
}
