// test_integration.cpp — a GPU-style pipeline written exactly as CUDA code, built &
// run entirely on the CPU via the four shims. Demonstrates "swap the includes only":
//   cudaMalloc/cudaMemcpy (cuda_runtime_cpu.h)
//   cudnnConvolutionForward + bias + sigmoid (cudnn_cpu.h)
//   cublasSgemm linear projection (cublas_cpu.h)
//   thrust::sort_by_key top-scoring boxes, NMS-style (thrust_cpu.h)
// Then verifies every stage against a plain-C++ reference.
//
//   g++ -std=c++17 -O2 -I. test_integration.cpp -o ti && ./ti
//   g++ -std=c++17 -O3 -I. -DCUDNN_CPU_USE_EIGEN -DCUBLAS_CPU_USE_EIGEN test_integration.cpp -o tie && ./tie
#include "cuda_runtime_cpu.h"
#include "cudnn_cpu.h"
#include "cublas_cpu.h"
#include "thrust_cpu.h"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

static int fails=0;
static void check(const char* n,float d,float tol=1e-4f){ printf("  %-32s max|diff|=%.2e  %s\n",n,d,d<tol?"OK":"FAIL"); if(d>=tol)++fails; }

int main(){
  std::mt19937 rng(11); std::normal_distribution<float> nd(0,1);
  const float one=1.f,zero=0.f;

  // ---- problem sizes ----
  const int N=1,C=3,H=16,W=16,K=8,ks=3,pad=1;   // conv: 3->8 ch, 3x3
  const int HW=H*W, featdim=K*HW;               // flattened conv output
  const int P=5;                                // linear projection: featdim -> P "scores"

  // ---- host data ----
  std::vector<float> hX(N*C*H*W), hWt(K*C*ks*ks), hBias(K), hLin(P*featdim);
  for(auto&v:hX)v=nd(rng); for(auto&v:hWt)v=nd(rng); for(auto&v:hBias)v=nd(rng); for(auto&v:hLin)v=nd(rng);

  // ==================== GPU-STYLE PIPELINE (all on CPU shims) ====================
  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0);

  // device buffers
  float *dX,*dWt,*dBias,*dConv,*dLin,*dScores;
  cudaMalloc((void**)&dX,   sizeof(float)*hX.size());
  cudaMalloc((void**)&dWt,  sizeof(float)*hWt.size());
  cudaMalloc((void**)&dBias,sizeof(float)*hBias.size());
  cudaMalloc((void**)&dConv,sizeof(float)*N*K*HW);
  cudaMalloc((void**)&dLin, sizeof(float)*hLin.size());
  cudaMalloc((void**)&dScores,sizeof(float)*P);
  cudaMemcpy(dX,  hX.data(),  sizeof(float)*hX.size(),  cudaMemcpyHostToDevice);
  cudaMemcpy(dWt, hWt.data(), sizeof(float)*hWt.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dBias,hBias.data(),sizeof(float)*hBias.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dLin,hLin.data(),sizeof(float)*hLin.size(),cudaMemcpyHostToDevice);

  cudnnHandle_t cud; cudnnCreate(&cud);
  cublasHandle_t cub; cublasCreate(&cub);

  // conv 3->8, 3x3 pad1
  cudnnTensorDescriptor_t xd,convd,biasd; cudnnFilterDescriptor_t wd; cudnnConvolutionDescriptor_t cd;
  cudnnCreateTensorDescriptor(&xd);cudnnCreateTensorDescriptor(&convd);cudnnCreateTensorDescriptor(&biasd);
  cudnnCreateFilterDescriptor(&wd);cudnnCreateConvolutionDescriptor(&cd);
  cudnnSetTensor4dDescriptor(xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,C,H,W);
  cudnnSetTensor4dDescriptor(convd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,N,K,H,W);
  cudnnSetTensor4dDescriptor(biasd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,K,1,1);
  cudnnSetFilter4dDescriptor(wd,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,K,C,ks,ks);
  cudnnSetConvolution2dDescriptor(cd,pad,pad,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
  cudnnConvolutionForward(cud,&one,xd,dX,wd,dWt,cd,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,nullptr,0,&zero,convd,dConv);
  cudnnAddTensor(cud,&one,biasd,dBias,&one,convd,dConv);            // + bias
  cudnnActivationDescriptor_t act; cudnnCreateActivationDescriptor(&act);
  cudnnSetActivationDescriptor(act,CUDNN_ACTIVATION_SIGMOID,CUDNN_NOT_PROPAGATE_NAN,0);
  cudnnActivationForward(cud,act,&one,convd,dConv,&zero,convd,dConv);  // sigmoid in place

  // linear projection: scores[P] = Lin[P x featdim] * feat[featdim]  (gemv, col-major)
  // store Lin column-major: element (p, f) at dLin[p + f*P]  -> build a col-major copy
  std::vector<float> hLinCM(P*featdim);
  for(int p=0;p<P;++p)for(int f=0;f<featdim;++f) hLinCM[p+(size_t)f*P]=hLin[p*featdim+f];
  cudaMemcpy(dLin,hLinCM.data(),sizeof(float)*hLinCM.size(),cudaMemcpyHostToDevice);
  cublasSgemv(cub,CUBLAS_OP_N,P,featdim,&one,dLin,P,dConv,1,&zero,dScores,1);

  // results back to host
  std::vector<float> hConv(N*K*HW), hScores(P);
  cudaMemcpy(hConv.data(),dConv,sizeof(float)*hConv.size(),cudaMemcpyDeviceToHost);
  cudaMemcpy(hScores.data(),dScores,sizeof(float)*P,cudaMemcpyDeviceToHost);

  // thrust NMS-style: sort candidate ids by descending score
  thrust::device_vector<float> sc(hScores.begin(),hScores.end());
  thrust::device_vector<int>   ids(P); thrust::sequence(ids.begin(),ids.end());
  thrust::sort_by_key(sc.begin(),sc.end(),ids.begin(),thrust::greater<float>());

  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float ms=0; cudaEventElapsedTime(&ms,t0,t1);

  // ==================== REFERENCE (plain C++) ====================
  std::vector<float> rConv(N*K*HW,0);
  for(int k=0;k<K;++k)for(int y=0;y<H;++y)for(int x=0;x<W;++x){ float a=hBias[k];
    for(int c=0;c<C;++c)for(int r=0;r<ks;++r)for(int s=0;s<ks;++s){int ih=y-pad+r,iw=x-pad+s;
      if(ih>=0&&ih<H&&iw>=0&&iw<W) a+=hX[(c*H+ih)*W+iw]*hWt[((k*C+c)*ks+r)*ks+s];}
    rConv[(k*H+y)*W+x]=1.f/(1.f+std::exp(-a)); }
  std::vector<float> rScores(P,0);
  for(int p=0;p<P;++p){ float a=0; for(int f=0;f<featdim;++f) a+=hLin[p*featdim+f]*rConv[f]; rScores[p]=a; }

  // ==================== VERIFY ====================
  float dc=0; for(size_t i=0;i<rConv.size();++i)dc=std::max(dc,std::fabs(rConv[i]-hConv[i]));
  check("conv+bias+sigmoid (cudnn)",dc);
  float dscore=0; for(int p=0;p<P;++p)dscore=std::max(dscore,std::fabs(rScores[p]-hScores[p]));
  check("linear projection (cublas)",dscore,1e-3f);
  // reference top order
  std::vector<int> ref(P); for(int i=0;i<P;++i)ref[i]=i;
  std::sort(ref.begin(),ref.end(),[&](int a,int b){return rScores[a]>rScores[b];});
  bool order_ok=true; for(int i=0;i<P;++i) order_ok &= (ids[i]==ref[i]);
  printf("  %-32s %s  (top id=%d score=%.3f)\n","sort_by_key top (thrust)",order_ok?"OK":"FAIL",ids[0],sc[0]);
  if(!order_ok)++fails;

  printf("\npipeline ran in %.3f ms (host clock via cudaEvent)\n", ms);

  // cleanup
  cudaFree(dX);cudaFree(dWt);cudaFree(dBias);cudaFree(dConv);cudaFree(dLin);cudaFree(dScores);
  cudnnDestroy(cud); cublasDestroy(cub); cudaEventDestroy(t0); cudaEventDestroy(t1);
  printf("\n%s\n", fails? "INTEGRATION FAILED":"INTEGRATION OK: full GPU-style pipeline on CPU");
  return fails?1:0;
}
