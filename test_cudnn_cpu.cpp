// test_cudnn_cpu.cpp — verify cudnn_cpu.h against naive references.
//   g++ -std=c++17 -O2 test_cudnn_cpu.cpp -o t && ./t
#include "cudnn_cpu.h"
#include <cstdio>
#include <random>

static float maxdiff(const std::vector<float>& a, const std::vector<float>& b){
  float m=0; for(size_t i=0;i<a.size();++i) m=std::max(m,std::fabs(a[i]-b[i])); return m;
}
static int fails=0;
static void check(const char* name,float d,float tol=1e-5f){
  printf("  %-28s max|diff| = %.3e  %s\n",name,d,d<tol?"OK":"FAIL"); if(d>=tol)++fails;
}

int main(){
  std::mt19937 rng(1); std::normal_distribution<float> nd(0,1);
  cudnnHandle_t h; cudnnCreate(&h);
  const float one=1.f, zero=0.f;

  // ---- 1. convolution forward (grouped) vs naive ----
  {
    int N=2,Cin=4,H=7,W=7,Cout=6,k=3,pad=1,stride=1,G=2;
    int cig=Cin/G, cog=Cout/G;
    std::vector<float> X(N*Cin*H*W), Wt(Cout*cig*k*k);
    for(auto&v:X)v=nd(rng); for(auto&v:Wt)v=nd(rng);
    int OH=(H+2*pad-k)/stride+1, OW=(W+2*pad-k)/stride+1;
    std::vector<float> Y(N*Cout*OH*OW,0), Ref(Y.size(),0);
    cudnnTensorDescriptor_t xd,yd; cudnnFilterDescriptor_t wd; cudnnConvolutionDescriptor_t cd;
    cudnnCreateTensorDescriptor(&xd); cudnnCreateTensorDescriptor(&yd);
    cudnnCreateFilterDescriptor(&wd); cudnnCreateConvolutionDescriptor(&cd);
    cudnnSetTensor4dDescriptor(xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,Cin,H,W);
    cudnnSetFilter4dDescriptor(wd,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,Cout,cig,k,k);
    cudnnSetConvolution2dDescriptor(cd,pad,pad,stride,stride,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(cd,G);
    int on,oc,oh,ow; cudnnGetConvolution2dForwardOutputDim(cd,xd,wd,&on,&oc,&oh,&ow);
    cudnnSetTensor4dDescriptor(yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,on,oc,oh,ow);
    cudnnConvolutionForward(h,&one,xd,X.data(),wd,Wt.data(),cd,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,nullptr,0,&zero,yd,Y.data());
    // naive ref
    for(int n=0;n<N;++n)for(int co=0;co<Cout;++co){int g=co/cog;
      for(int y=0;y<OH;++y)for(int x=0;x<OW;++x){float a=0;
        for(int ci=0;ci<cig;++ci)for(int r=0;r<k;++r)for(int s=0;s<k;++s){
          int ih=y*stride-pad+r,iw=x*stride-pad+s; if(ih<0||ih>=H||iw<0||iw>=W)continue;
          a+=X[((n*Cin+g*cig+ci)*H+ih)*W+iw]*Wt[((co*cig+ci)*k+r)*k+s];}
        Ref[((n*Cout+co)*OH+y)*OW+x]=a;}}
    check("conv fwd (groups=2)",maxdiff(Y,Ref));
  }

  // ---- 2. depthwise conv (groups=Cin) ----
  {
    int N=1,C=5,H=6,W=6,k=3,pad=1;
    std::vector<float> X(N*C*H*W),Wt(C*1*k*k); for(auto&v:X)v=nd(rng); for(auto&v:Wt)v=nd(rng);
    int OH=H,OW=W; std::vector<float> Y(N*C*OH*OW,0),Ref(Y.size(),0);
    cudnnTensorDescriptor_t xd,yd; cudnnFilterDescriptor_t wd; cudnnConvolutionDescriptor_t cd;
    cudnnCreateTensorDescriptor(&xd);cudnnCreateTensorDescriptor(&yd);cudnnCreateFilterDescriptor(&wd);cudnnCreateConvolutionDescriptor(&cd);
    cudnnSetTensor4dDescriptor(xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,H,W);
    cudnnSetFilter4dDescriptor(wd,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,C,1,k,k);
    cudnnSetConvolution2dDescriptor(cd,pad,pad,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(cd,C);
    cudnnSetTensor4dDescriptor(yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,OH,OW);
    cudnnConvolutionForward(h,&one,xd,X.data(),wd,Wt.data(),cd,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,nullptr,0,&zero,yd,Y.data());
    for(int c=0;c<C;++c)for(int y=0;y<OH;++y)for(int x=0;x<OW;++x){float a=0;
      for(int r=0;r<k;++r)for(int s=0;s<k;++s){int ih=y-pad+r,iw=x-pad+s; if(ih<0||ih>=H||iw<0||iw>=W)continue;
        a+=X[(c*H+ih)*W+iw]*Wt[(c*k+r)*k+s];} Ref[(c*OH+y)*OW+x]=a;}
    check("conv fwd (depthwise)",maxdiff(Y,Ref));
  }

  // ---- 3. addTensor (bias broadcast 1,C,1,1) ----
  {
    int N=2,C=3,H=4,W=4; std::vector<float> Y(N*C*H*W),B(C),Ref;
    for(auto&v:Y)v=nd(rng); for(auto&v:B)v=nd(rng); Ref=Y;
    cudnnTensorDescriptor_t yd,bd; cudnnCreateTensorDescriptor(&yd);cudnnCreateTensorDescriptor(&bd);
    cudnnSetTensor4dDescriptor(yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,H,W);
    cudnnSetTensor4dDescriptor(bd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,C,1,1);
    cudnnAddTensor(h,&one,bd,B.data(),&one,yd,Y.data());
    for(int n=0;n<N;++n)for(int c=0;c<C;++c)for(int p=0;p<H*W;++p)Ref[(n*C+c)*H*W+p]+=B[c];
    check("addTensor (bias)",maxdiff(Y,Ref));
  }

  // ---- 4. activation: SiLU = x * sigmoid(x) via Sigmoid + OpTensor(MUL) ----
  {
    int n=64; std::vector<float> X(n),S(n),Y(n),Ref(n); for(auto&v:X)v=nd(rng);
    cudnnTensorDescriptor_t td; cudnnCreateTensorDescriptor(&td);
    cudnnSetTensor4dDescriptor(td,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,n,1,1);
    cudnnActivationDescriptor_t ad; cudnnCreateActivationDescriptor(&ad);
    cudnnSetActivationDescriptor(ad,CUDNN_ACTIVATION_SIGMOID,CUDNN_NOT_PROPAGATE_NAN,0);
    cudnnActivationForward(h,ad,&one,td,X.data(),&zero,td,S.data());
    cudnnOpTensorDescriptor_t od; cudnnCreateOpTensorDescriptor(&od);
    cudnnSetOpTensorDescriptor(od,CUDNN_OP_TENSOR_MUL,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN);
    cudnnOpTensor(h,od,&one,td,X.data(),&one,td,S.data(),&zero,td,Y.data());
    for(int i=0;i<n;++i)Ref[i]=X[i]/(1.f+std::exp(-X[i]));
    check("SiLU (sigmoid+opTensor mul)",maxdiff(Y,Ref));
  }

  // ---- 5. maxpool 5x5 stride1 pad2 (SPPF) ----
  {
    int N=1,C=2,H=8,W=8,k=5,pad=2; std::vector<float> X(N*C*H*W),Y(N*C*H*W,0),Ref(Y.size(),0);
    for(auto&v:X)v=nd(rng);
    cudnnTensorDescriptor_t xd,yd; cudnnPoolingDescriptor_t pd;
    cudnnCreateTensorDescriptor(&xd);cudnnCreateTensorDescriptor(&yd);cudnnCreatePoolingDescriptor(&pd);
    cudnnSetTensor4dDescriptor(xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,H,W);
    cudnnSetTensor4dDescriptor(yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,H,W);
    cudnnSetPooling2dDescriptor(pd,CUDNN_POOLING_MAX,CUDNN_NOT_PROPAGATE_NAN,k,k,pad,pad,1,1);
    cudnnPoolingForward(h,pd,&one,xd,X.data(),&zero,yd,Y.data());
    for(int c=0;c<C;++c)for(int y=0;y<H;++y)for(int x=0;x<W;++x){float m=-1e30f;
      for(int r=0;r<k;++r)for(int s=0;s<k;++s){int ih=y-pad+r,iw=x-pad+s; if(ih<0||ih>=H||iw<0||iw>=W)continue;
        m=std::max(m,X[(c*H+ih)*W+iw]);} Ref[(c*H+y)*W+x]=m;}
    check("maxpool 5x5 (SPPF)",maxdiff(Y,Ref));
  }

  // ---- 6. softmax over channel (DFL: 16 bins) ----
  {
    int N=1,C=16,H=3,W=3; std::vector<float> X(N*C*H*W),Y(X.size()),Ref(X.size());
    for(auto&v:X)v=nd(rng);
    cudnnTensorDescriptor_t td; cudnnCreateTensorDescriptor(&td);
    cudnnSetTensor4dDescriptor(td,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,H,W);
    cudnnSoftmaxForward(h,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_CHANNEL,&one,td,X.data(),&zero,td,Y.data());
    for(int p=0;p<H*W;++p){double s=0; for(int c=0;c<C;++c)s+=std::exp(X[c*H*W+p]);
      for(int c=0;c<C;++c)Ref[c*H*W+p]=std::exp(X[c*H*W+p])/s;}
    check("softmax channel (DFL)",maxdiff(Y,Ref));
  }

  // ---- 7. batchnorm inference (spatial, per-channel) ----
  {
    int N=2,C=3,H=4,W=4; double eps=1e-5;
    std::vector<float> X(N*C*H*W),Y(X.size()),Ref(X.size()),g(C),b(C),mu(C),var(C);
    for(auto&v:X)v=nd(rng); for(int c=0;c<C;++c){g[c]=nd(rng);b[c]=nd(rng);mu[c]=nd(rng);var[c]=std::fabs(nd(rng))+0.1f;}
    cudnnTensorDescriptor_t td,pd; cudnnCreateTensorDescriptor(&td);cudnnCreateTensorDescriptor(&pd);
    cudnnSetTensor4dDescriptor(td,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,H,W);
    cudnnSetTensor4dDescriptor(pd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,C,1,1);
    cudnnBatchNormalizationForwardInference(h,CUDNN_BATCHNORM_SPATIAL,&one,&zero,td,X.data(),td,Y.data(),pd,g.data(),b.data(),mu.data(),var.data(),eps);
    for(int n=0;n<N;++n)for(int c=0;c<C;++c){float inv=1.f/std::sqrt((float)(var[c]+eps));
      for(int p=0;p<H*W;++p){int i=(n*C+c)*H*W+p; Ref[i]=g[c]*(X[i]-mu[c])*inv+b[c];}}
    check("batchnorm inference",maxdiff(Y,Ref));
  }

  cudnnDestroy(h);
  printf("\n%s\n", fails? "SOME TESTS FAILED":"ALL TESTS PASSED");
  return fails?1:0;
}
