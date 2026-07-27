// test_cublas_cpu.cpp — verify cublas_cpu.h (column-major) against hand references.
//   g++ -std=c++17 -O2 -I. test_cublas_cpu.cpp -o tc && ./tc
//   g++ -std=c++17 -O3 -I. -DCUBLAS_CPU_USE_EIGEN test_cublas_cpu.cpp -o tce && ./tce
#include "cublas_cpu.h"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

static std::mt19937 rng(3);
static std::normal_distribution<float> nd(0,1);
static int fails=0;
static void check(const char* name,float d,float tol=1e-4f){
  printf("  %-30s max|diff| = %.3e  %s\n",name,d,d<tol?"OK":"FAIL"); if(d>=tol)++fails; }

// column-major getters for a logical (rows x cols) matrix with leading dim ld
static float cm(const std::vector<float>& M,int i,int j,int ld){ return M[i+(size_t)j*ld]; }

int main(){
  cublasHandle_t h; cublasCreate(&h); const float one=1.f,zero=0.f;

  // ---- SGEMM: all four transpose combinations, with beta ----
  {
    int m=5,n=4,k=3;
    for(int ta=0;ta<2;++ta) for(int tb=0;tb<2;++tb){
      cublasOperation_t TA=ta?CUBLAS_OP_T:CUBLAS_OP_N, TB=tb?CUBLAS_OP_T:CUBLAS_OP_N;
      // op(A) is m x k, op(B) is k x n. Physical A: N->m x k (lda=m), T->k x m (lda=k)
      int Ar=ta?k:m, Ac=ta?m:k, lda=Ar;
      int Br=tb?n:k, Bc=tb?k:n, ldb=Br;
      std::vector<float> A(Ar*Ac),B(Br*Bc),C(m*n),Cref;
      for(auto&v:A)v=nd(rng); for(auto&v:B)v=nd(rng); for(auto&v:C)v=nd(rng); Cref=C;
      float alpha=1.5f,beta=-0.7f;
      cublasSgemm(h,TA,TB,m,n,k,&alpha,A.data(),lda,B.data(),ldb,&beta,C.data(),m);
      // reference: C(i,j)=alpha*sum_l opA(i,l)*opB(l,j)+beta*C(i,j)
      auto oA=[&](int i,int l){ return ta? cm(A,l,i,lda) : cm(A,i,l,lda); };
      auto oB=[&](int l,int j){ return tb? cm(B,j,l,ldb) : cm(B,l,j,ldb); };
      float md=0; for(int j=0;j<n;++j)for(int i=0;i<m;++i){ float acc=0; for(int l=0;l<k;++l)acc+=oA(i,l)*oB(l,j);
        float ref=alpha*acc+beta*cm(Cref,i,j,m); md=std::max(md,std::fabs(ref-cm(C,i,j,m))); }
      char nm[32]; snprintf(nm,32,"sgemm %s%s",ta?"T":"N",tb?"T":"N"); check(nm,md);
    }
  }

  // ---- strided-batched SGEMM ----
  {
    int m=3,n=3,k=2,batch=4; long long sA=m*k,sB=k*n,sC=m*n;
    std::vector<float> A(sA*batch),B(sB*batch),C(sC*batch,0),Cref(sC*batch,0);
    for(auto&v:A)v=nd(rng); for(auto&v:B)v=nd(rng);
    cublasSgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&one,A.data(),m,sA,B.data(),k,sB,&zero,C.data(),m,sC,batch);
    float md=0; for(int b=0;b<batch;++b)for(int j=0;j<n;++j)for(int i=0;i<m;++i){ float acc=0;
      for(int l=0;l<k;++l)acc+=A[b*sA+i+l*m]*B[b*sB+l+j*k]; md=std::max(md,std::fabs(acc-C[b*sC+i+j*m])); }
    check("sgemm strided-batched",md);
  }

  // ---- SGEMV (N and T) ----
  {
    int m=4,n=3; std::vector<float> A(m*n),x,y,yref;
    for(auto&v:A)v=nd(rng);
    { x.assign(n,0);y.assign(m,0);for(auto&v:x)v=nd(rng); yref=y;
      cublasSgemv(h,CUBLAS_OP_N,m,n,&one,A.data(),m,x.data(),1,&zero,y.data(),1);
      float md=0;for(int i=0;i<m;++i){float acc=0;for(int j=0;j<n;++j)acc+=cm(A,i,j,m)*x[j];md=std::max(md,std::fabs(acc-y[i]));}
      check("sgemv N",md); }
    { x.assign(m,0);y.assign(n,0);for(auto&v:x)v=nd(rng);
      cublasSgemv(h,CUBLAS_OP_T,m,n,&one,A.data(),m,x.data(),1,&zero,y.data(),1);
      float md=0;for(int j=0;j<n;++j){float acc=0;for(int i=0;i<m;++i)acc+=cm(A,i,j,m)*x[i];md=std::max(md,std::fabs(acc-y[j]));}
      check("sgemv T",md); }
  }

  // ---- SGER rank-1 ----
  {
    int m=4,n=3; std::vector<float> A(m*n),Aref,x(m),y(n);
    for(auto&v:A)v=nd(rng); for(auto&v:x)v=nd(rng); for(auto&v:y)v=nd(rng); Aref=A; float al=0.9f;
    cublasSger(h,m,n,&al,x.data(),1,y.data(),1,A.data(),m);
    float md=0;for(int j=0;j<n;++j)for(int i=0;i<m;++i){float ref=cm(Aref,i,j,m)+al*x[i]*y[j];md=std::max(md,std::fabs(ref-cm(A,i,j,m)));}
    check("sger rank-1",md);
  }

  // ---- level-1 ----
  {
    int n=100; std::vector<float> x(n),y(n),yref(n); for(auto&v:x)v=nd(rng); for(auto&v:y)v=nd(rng); yref=y;
    float al=1.3f; cublasSaxpy(h,n,&al,x.data(),1,y.data(),1);
    float md=0;for(int i=0;i<n;++i)md=std::max(md,std::fabs((yref[i]+al*x[i])-y[i])); check("saxpy",md);
    std::vector<float> xs=x; cublasSscal(h,n,&al,xs.data(),1);
    md=0;for(int i=0;i<n;++i)md=std::max(md,std::fabs(al*x[i]-xs[i])); check("sscal",md);
    float dot; cublasSdot(h,n,x.data(),1,yref.data(),1,&dot);
    double dref=0;for(int i=0;i<n;++i)dref+=(double)x[i]*yref[i]; check("sdot",std::fabs((float)dref-dot),1e-3f);
    float asum; cublasSasum(h,n,x.data(),1,&asum); double aref=0;for(auto v:x)aref+=std::fabs(v);
    check("sasum",std::fabs((float)aref-asum),1e-3f);
    float nrm; cublasSnrm2(h,n,x.data(),1,&nrm); double nref=0;for(auto v:x)nref+=(double)v*v; nref=std::sqrt(nref);
    check("snrm2",std::fabs((float)nref-nrm),1e-3f);
    int idx; cublasIsamax(h,n,x.data(),1,&idx); int iref=0;float bv=-1;for(int i=0;i<n;++i)if(std::fabs(x[i])>bv){bv=std::fabs(x[i]);iref=i;}
    check("isamax (1-based)",std::fabs((float)((iref+1)-idx)));
  }

  cublasDestroy(h);
  printf("\n%s\n", fails? "SOME TESTS FAILED":"ALL CUBLAS TESTS PASSED");
  return fails?1:0;
}
