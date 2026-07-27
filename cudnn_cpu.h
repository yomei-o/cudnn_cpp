// cudnn_cpu.h — a tiny, header-only, source-available CPU implementation of the cuDNN
// API subset that YOLO-style detectors use. Drop-in for development on GPU-less machines:
// build against this header; on a real GPU swap it for <cudnn.h> and link cuDNN — the
// call sites don't change. Forward subset (v1): conv (+groups/depthwise), addTensor,
// activation, pooling, softmax, batchnorm-inference, opTensor. NCHW / FP32.
//
// Not a bit-exact cuDNN clone: algorithms/workspace are ignored (workspace size = 0),
// results match cuDNN's math to float precision. Only the pieces detectors need.
#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

// Optional Eigen backend: define CUDNN_CPU_USE_EIGEN to route the heavy math
// (conv gemm + elementwise) through Eigen (header-only "eigen_flat"). Without it,
// the header is fully standalone (naive loops). CUDNN_CPU_EIGEN_HEADER lets you
// override the include path (default: the vendored flat single-dir Eigen).
#ifdef CUDNN_CPU_USE_EIGEN
  #ifndef CUDNN_CPU_EIGEN_HEADER
    #define CUDNN_CPU_EIGEN_HEADER "third_party/eigen_flat/Eigen_Core.h"
  #endif
  #include CUDNN_CPU_EIGEN_HEADER
namespace cudnn_cpu_detail {
  using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using CRowMap = Eigen::Map<const RowMat>;
  using RowMap  = Eigen::Map<RowMat>;
  using Arr  = Eigen::Map<Eigen::ArrayXf>;
  using CArr = Eigen::Map<const Eigen::ArrayXf>;
}
#endif

// ---- status / types / enums (names & values chosen to be cuDNN-source-compatible) ----
typedef enum { CUDNN_STATUS_SUCCESS = 0, CUDNN_STATUS_BAD_PARAM = 3 } cudnnStatus_t;
typedef enum { CUDNN_DATA_FLOAT = 0 } cudnnDataType_t;
typedef enum { CUDNN_TENSOR_NCHW = 0 } cudnnTensorFormat_t;
typedef enum { CUDNN_CROSS_CORRELATION = 1, CUDNN_CONVOLUTION = 0 } cudnnConvolutionMode_t;
typedef enum { CUDNN_ACTIVATION_SIGMOID = 0, CUDNN_ACTIVATION_RELU = 1,
               CUDNN_ACTIVATION_TANH = 2, CUDNN_ACTIVATION_CLIPPED_RELU = 3,
               CUDNN_ACTIVATION_IDENTITY = 5 } cudnnActivationMode_t;
typedef enum { CUDNN_POOLING_MAX = 0, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
               CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2 } cudnnPoolingMode_t;
typedef enum { CUDNN_SOFTMAX_FAST = 0, CUDNN_SOFTMAX_ACCURATE = 1 } cudnnSoftmaxAlgorithm_t;
typedef enum { CUDNN_SOFTMAX_MODE_INSTANCE = 0, CUDNN_SOFTMAX_MODE_CHANNEL = 1 } cudnnSoftmaxMode_t;
typedef enum { CUDNN_BATCHNORM_PER_ACTIVATION = 0, CUDNN_BATCHNORM_SPATIAL = 1 } cudnnBatchNormMode_t;
typedef enum { CUDNN_OP_TENSOR_ADD = 0, CUDNN_OP_TENSOR_MUL = 1, CUDNN_OP_TENSOR_MAX = 2 } cudnnOpTensorOp_t;
typedef enum { CUDNN_NOT_PROPAGATE_NAN = 0, CUDNN_PROPAGATE_NAN = 1 } cudnnNanPropagation_t;
typedef enum { CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0 } cudnnConvolutionFwdAlgo_t;

// ---- opaque descriptors (structs behind pointers, like cuDNN) ----
struct cudnnContext { int _ = 0; };
typedef cudnnContext* cudnnHandle_t;
struct cudnnTensorStruct { int n=0,c=0,h=0,w=0; };
typedef cudnnTensorStruct* cudnnTensorDescriptor_t;
struct cudnnFilterStruct { int k=0,c=0,h=0,w=0; };
typedef cudnnFilterStruct* cudnnFilterDescriptor_t;
struct cudnnConvolutionStruct { int ph=0,pw=0,sh=1,sw=1,dh=1,dw=1,groups=1; cudnnConvolutionMode_t mode=CUDNN_CROSS_CORRELATION; };
typedef cudnnConvolutionStruct* cudnnConvolutionDescriptor_t;
struct cudnnActivationStruct { cudnnActivationMode_t mode=CUDNN_ACTIVATION_RELU; double coef=0; };
typedef cudnnActivationStruct* cudnnActivationDescriptor_t;
struct cudnnPoolingStruct { cudnnPoolingMode_t mode=CUDNN_POOLING_MAX; int kh=0,kw=0,ph=0,pw=0,sh=1,sw=1; };
typedef cudnnPoolingStruct* cudnnPoolingDescriptor_t;
struct cudnnOpTensorStruct { cudnnOpTensorOp_t op=CUDNN_OP_TENSOR_ADD; };
typedef cudnnOpTensorStruct* cudnnOpTensorDescriptor_t;

// ---- lifecycle ----
inline cudnnStatus_t cudnnCreate(cudnnHandle_t* h){ *h=new cudnnContext; return CUDNN_STATUS_SUCCESS; }
inline cudnnStatus_t cudnnDestroy(cudnnHandle_t h){ delete h; return CUDNN_STATUS_SUCCESS; }
#define CUDNN_DESC(NAME, STRUCT, T) \
  inline cudnnStatus_t cudnnCreate##NAME##Descriptor(T* d){ *d=new STRUCT; return CUDNN_STATUS_SUCCESS; } \
  inline cudnnStatus_t cudnnDestroy##NAME##Descriptor(T d){ delete d; return CUDNN_STATUS_SUCCESS; }
CUDNN_DESC(Tensor, cudnnTensorStruct, cudnnTensorDescriptor_t)
CUDNN_DESC(Filter, cudnnFilterStruct, cudnnFilterDescriptor_t)
CUDNN_DESC(Convolution, cudnnConvolutionStruct, cudnnConvolutionDescriptor_t)
CUDNN_DESC(Activation, cudnnActivationStruct, cudnnActivationDescriptor_t)
CUDNN_DESC(Pooling, cudnnPoolingStruct, cudnnPoolingDescriptor_t)
CUDNN_DESC(OpTensor, cudnnOpTensorStruct, cudnnOpTensorDescriptor_t)
#undef CUDNN_DESC

inline cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d, cudnnTensorFormat_t, cudnnDataType_t, int n,int c,int h,int w){ d->n=n;d->c=c;d->h=h;d->w=w; return CUDNN_STATUS_SUCCESS; }
inline cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t d, cudnnDataType_t, cudnnTensorFormat_t, int k,int c,int h,int w){ d->k=k;d->c=c;d->h=h;d->w=w; return CUDNN_STATUS_SUCCESS; }
inline cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t d,int ph,int pw,int sh,int sw,int dh,int dw,cudnnConvolutionMode_t m,cudnnDataType_t){ d->ph=ph;d->pw=pw;d->sh=sh;d->sw=sw;d->dh=dh;d->dw=dw;d->mode=m; return CUDNN_STATUS_SUCCESS; }
inline cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t d,int g){ d->groups=g; return CUDNN_STATUS_SUCCESS; }
inline cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t d,cudnnActivationMode_t m,cudnnNanPropagation_t,double coef){ d->mode=m; d->coef=coef; return CUDNN_STATUS_SUCCESS; }
inline cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t d,cudnnPoolingMode_t m,cudnnNanPropagation_t,int kh,int kw,int ph,int pw,int sh,int sw){ d->mode=m;d->kh=kh;d->kw=kw;d->ph=ph;d->pw=pw;d->sh=sh;d->sw=sw; return CUDNN_STATUS_SUCCESS; }
inline cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t d,cudnnOpTensorOp_t op,cudnnDataType_t,cudnnNanPropagation_t){ d->op=op; return CUDNN_STATUS_SUCCESS; }

inline cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t c,const cudnnTensorDescriptor_t x,const cudnnFilterDescriptor_t w,int* n,int* co,int* h,int* ww){
  *n=x->n; *co=w->k;
  *h  = (x->h + 2*c->ph - (c->dh*(w->h-1)+1))/c->sh + 1;
  *ww = (x->w + 2*c->pw - (c->dw*(w->w-1)+1))/c->sw + 1;
  return CUDNN_STATUS_SUCCESS;
}
// workspace not needed on CPU
inline cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t,const cudnnTensorDescriptor_t,const cudnnFilterDescriptor_t,const cudnnConvolutionDescriptor_t,const cudnnTensorDescriptor_t,cudnnConvolutionFwdAlgo_t,size_t* sz){ *sz=0; return CUDNN_STATUS_SUCCESS; }

// ---- convolution forward: y = alpha * (w * x) + beta * y  (grouped, cross-correlation) ----
inline cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t,const void* alpha,
    const cudnnTensorDescriptor_t xd,const void* xp,const cudnnFilterDescriptor_t wd,const void* wp,
    const cudnnConvolutionDescriptor_t cd,cudnnConvolutionFwdAlgo_t,void*,size_t,
    const void* beta,const cudnnTensorDescriptor_t yd,void* yp){
  float a=*(const float*)alpha, b=*(const float*)beta;
  const float* X=(const float*)xp; const float* W=(const float*)wp; float* Y=(float*)yp;
  int N=xd->n, Cin=xd->c, H=xd->h, Wd=xd->w, Cout=yd->c, OH=yd->h, OW=yd->w;
  int kh=wd->h, kw=wd->w, G=cd->groups, cig=Cin/G, cog=Cout/G;
#ifdef CUDNN_CPU_USE_EIGEN
  // im2col + Eigen gemm, per (image, group): Y_g[cog,OHW] = W_g[cog,cig*k*k] * col[cig*k*k,OHW]
  using namespace cudnn_cpu_detail;
  const int OHW=OH*OW, KK=cig*kh*kw;
  RowMat col(KK, OHW);
  for (int n=0;n<N;++n) for (int g=0;g<G;++g){
    for (int ci=0;ci<cig;++ci){ int icin=g*cig+ci;
      for (int r=0;r<kh;++r) for (int s=0;s<kw;++s){ int row=(ci*kh+r)*kw+s;
        for (int oh=0;oh<OH;++oh){ int ih=oh*cd->sh-cd->ph+r*cd->dh;
          for (int ow=0;ow<OW;++ow){ int iw=ow*cd->sw-cd->pw+s*cd->dw;
            col(row, oh*OW+ow) = (ih<0||ih>=H||iw<0||iw>=Wd) ? 0.f : X[((n*Cin+icin)*H+ih)*Wd+iw]; } } } }
    CRowMap Wm(W + (size_t)(g*cog)*KK, cog, KK);
    RowMap  Ym(Y + ((size_t)(n*Cout+g*cog)*OH)*OW, cog, OHW);
    if (b==0.f) Ym.noalias() = a * (Wm * col);
    else        Ym = a * (Wm * col) + b * Ym;
  }
#else
  for (int n=0;n<N;++n) for (int co=0;co<Cout;++co){
    int g=co/cog;
    for (int oh=0;oh<OH;++oh) for (int ow=0;ow<OW;++ow){
      float acc=0;
      for (int ci=0;ci<cig;++ci){ int icin=g*cig+ci;
        for (int r=0;r<kh;++r){ int ih=oh*cd->sh - cd->ph + r*cd->dh; if(ih<0||ih>=H)continue;
          for (int s=0;s<kw;++s){ int iw=ow*cd->sw - cd->pw + s*cd->dw; if(iw<0||iw>=Wd)continue;
            acc += X[((n*Cin+icin)*H+ih)*Wd+iw] * W[((co*cig+ci)*kh+r)*kw+s]; } } }
      float* o=&Y[((n*Cout+co)*OH+oh)*OW+ow]; *o = a*acc + b*(*o);
    }
  }
#endif
  return CUDNN_STATUS_SUCCESS;
}

// ---- addTensor: C = alpha*A + beta*C, A broadcast over dims of size 1 (bias: 1,C,1,1) ----
inline cudnnStatus_t cudnnAddTensor(cudnnHandle_t,const void* alpha,const cudnnTensorDescriptor_t ad,const void* ap,const void* beta,const cudnnTensorDescriptor_t cd,void* cp){
  float a=*(const float*)alpha, b=*(const float*)beta;
  const float* A=(const float*)ap; float* C=(float*)cp;
  int N=cd->n,Cc=cd->c,H=cd->h,W=cd->w;
  for (int n=0;n<N;++n) for (int c=0;c<Cc;++c) for (int h=0;h<H;++h) for (int w=0;w<W;++w){
    int an=(ad->n==1?0:n), ac=(ad->c==1?0:c), ah=(ad->h==1?0:h), aw=(ad->w==1?0:w);
    float av=A[((an*ad->c+ac)*ad->h+ah)*ad->w+aw];
    float* o=&C[((n*Cc+c)*H+h)*W+w]; *o = a*av + b*(*o);
  }
  return CUDNN_STATUS_SUCCESS;
}

// ---- opTensor: C = op(alpha1*A, alpha2*B) + beta*C  (ADD/MUL/MAX; used for residual & SiLU) ----
inline cudnnStatus_t cudnnOpTensor(cudnnHandle_t,const cudnnOpTensorDescriptor_t od,const void* a1,const cudnnTensorDescriptor_t ad,const void* ap,const void* a2,const cudnnTensorDescriptor_t bd,const void* bp,const void* beta,const cudnnTensorDescriptor_t cd,void* cp){
  float A1=*(const float*)a1,A2=*(const float*)a2,B=*(const float*)beta;
  const float* A=(const float*)ap; const float* Bp=(const float*)bp; float* C=(float*)cp;
  int64_t n=(int64_t)cd->n*cd->c*cd->h*cd->w;
#ifdef CUDNN_CPU_USE_EIGEN
  using namespace cudnn_cpu_detail; CArr Am(A,n),Bm(Bp,n); Arr Cm(C,n);
  Eigen::ArrayXf x=A1*Am, y=A2*Bm, r;
  switch(od->op){ case CUDNN_OP_TENSOR_MUL: r=x*y; break; case CUDNN_OP_TENSOR_MAX: r=x.max(y); break; default: r=x+y; }
  Cm = r + B*Cm;
#else
  for (int64_t i=0;i<n;++i){ float x=A1*A[i], y=A2*Bp[i], r;
    switch(od->op){ case CUDNN_OP_TENSOR_MUL: r=x*y; break; case CUDNN_OP_TENSOR_MAX: r=std::max(x,y); break; default: r=x+y; }
    C[i]=r + B*C[i]; }
#endif
  return CUDNN_STATUS_SUCCESS;
}

// ---- activation forward ----
inline cudnnStatus_t cudnnActivationForward(cudnnHandle_t,const cudnnActivationDescriptor_t d,const void* alpha,const cudnnTensorDescriptor_t xd,const void* xp,const void* beta,const cudnnTensorDescriptor_t,void* yp){
  float a=*(const float*)alpha,b=*(const float*)beta; const float* X=(const float*)xp; float* Y=(float*)yp;
  int64_t n=(int64_t)xd->n*xd->c*xd->h*xd->w;
#ifdef CUDNN_CPU_USE_EIGEN
  using namespace cudnn_cpu_detail; CArr Xm(X,n); Arr Ym(Y,n); Eigen::ArrayXf y;
  switch(d->mode){
    case CUDNN_ACTIVATION_SIGMOID: y=1.f/(1.f+(-Xm).exp()); break;
    case CUDNN_ACTIVATION_RELU: y=Xm.max(0.f); break;
    case CUDNN_ACTIVATION_TANH: y=Xm.tanh(); break;
    case CUDNN_ACTIVATION_CLIPPED_RELU: y=Xm.max(0.f).min((float)d->coef); break;
    default: y=Xm; }
  Ym = a*y + b*Ym;
#else
  for (int64_t i=0;i<n;++i){ float x=X[i],y;
    switch(d->mode){
      case CUDNN_ACTIVATION_SIGMOID: y=1.f/(1.f+std::exp(-x)); break;
      case CUDNN_ACTIVATION_RELU: y=x>0?x:0; break;
      case CUDNN_ACTIVATION_TANH: y=std::tanh(x); break;
      case CUDNN_ACTIVATION_CLIPPED_RELU: y=std::min((float)d->coef,std::max(0.f,x)); break;
      default: y=x; }
    Y[i]=a*y + b*Y[i]; }
#endif
  return CUDNN_STATUS_SUCCESS;
}

// ---- pooling forward (MAX / AVG) ----
inline cudnnStatus_t cudnnPoolingForward(cudnnHandle_t,const cudnnPoolingDescriptor_t d,const void* alpha,const cudnnTensorDescriptor_t xd,const void* xp,const void* beta,const cudnnTensorDescriptor_t yd,void* yp){
  float a=*(const float*)alpha,b=*(const float*)beta; const float* X=(const float*)xp; float* Y=(float*)yp;
  int N=xd->n,C=xd->c,H=xd->h,W=xd->w,OH=yd->h,OW=yd->w;
  for (int n=0;n<N;++n) for (int c=0;c<C;++c) for (int oh=0;oh<OH;++oh) for (int ow=0;ow<OW;++ow){
    float best=(d->mode==CUDNN_POOLING_MAX)?-1e30f:0.f; int cnt=0;
    for (int r=0;r<d->kh;++r){ int ih=oh*d->sh-d->ph+r; if(ih<0||ih>=H)continue;
      for (int s=0;s<d->kw;++s){ int iw=ow*d->sw-d->pw+s; if(iw<0||iw>=W)continue;
        float v=X[((n*C+c)*H+ih)*W+iw];
        if(d->mode==CUDNN_POOLING_MAX) best=std::max(best,v); else { best+=v; cnt++; } } }
    if(d->mode!=CUDNN_POOLING_MAX) best/= (d->mode==CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)?(d->kh*d->kw):std::max(cnt,1);
    float* o=&Y[((n*C+c)*OH+oh)*OW+ow]; *o=a*best + b*(*o);
  }
  return CUDNN_STATUS_SUCCESS;
}

// ---- softmax forward (per-channel: over C, for each n,h,w) ----
inline cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t mode,const void* alpha,const cudnnTensorDescriptor_t xd,const void* xp,const void* beta,const cudnnTensorDescriptor_t,void* yp){
  float a=*(const float*)alpha,b=*(const float*)beta; const float* X=(const float*)xp; float* Y=(float*)yp;
  int N=xd->n,C=xd->c,H=xd->h,W=xd->w;
  auto sm=[&](int base,int stride,int len){ float m=-1e30f; for(int i=0;i<len;++i)m=std::max(m,X[base+i*stride]);
    double s=0; for(int i=0;i<len;++i)s+=std::exp(X[base+i*stride]-m);
    for(int i=0;i<len;++i){ float v=std::exp(X[base+i*stride]-m)/(float)s; Y[base+i*stride]=a*v+b*Y[base+i*stride]; } };
  if(mode==CUDNN_SOFTMAX_MODE_CHANNEL) for(int n=0;n<N;++n)for(int h=0;h<H;++h)for(int w=0;w<W;++w) sm(((n*C)*H+h)*W+w, H*W, C);
  else for(int n=0;n<N;++n) sm(n*C*H*W, 1, C*H*W);
  return CUDNN_STATUS_SUCCESS;
}

// ---- batchnorm forward inference: y = scale*(x-mean)/sqrt(var+eps) + bias (SPATIAL: per-channel) ----
inline cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t,cudnnBatchNormMode_t,const void* alpha,const void* beta,const cudnnTensorDescriptor_t xd,const void* xp,const cudnnTensorDescriptor_t,void* yp,const cudnnTensorDescriptor_t,const void* scale,const void* bias,const void* mean,const void* var,double eps){
  float a=*(const float*)alpha,b=*(const float*)beta;
  const float* X=(const float*)xp; float* Y=(float*)yp;
  const float* g=(const float*)scale;const float* bt=(const float*)bias;const float* mu=(const float*)mean;const float* v=(const float*)var;
  int N=xd->n,C=xd->c,H=xd->h,W=xd->w, HW=H*W;
  for (int n=0;n<N;++n) for (int c=0;c<C;++c){ float inv=1.f/std::sqrt((float)(v[c]+eps)); int i0=((n*C+c)*H)*W;
#ifdef CUDNN_CPU_USE_EIGEN
    using namespace cudnn_cpu_detail; CArr Xm(X+i0,HW); Arr Ym(Y+i0,HW);
    Ym = a*(g[c]*(Xm-mu[c])*inv + bt[c]) + b*Ym;
#else
    for (int p=0;p<HW;++p){ int i=i0+p; float yv=g[c]*(X[i]-mu[c])*inv+bt[c]; Y[i]=a*yv+b*Y[i]; }
#endif
  }
  return CUDNN_STATUS_SUCCESS;
}
