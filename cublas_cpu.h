// cublas_cpu.h — header-only, source-available CPU implementation of the cuBLAS
// (single-precision) subset that deep-learning / YOLO code uses. Same idea as
// cudnn_cpu.h: build GPU code on a GPU-less box, swap for <cublas_v2.h> on the GPU.
//
// cuBLAS is COLUMN-MAJOR: element (i,j) with leading dim ld lives at A[i + j*ld].
// These routines honor that plus op(A) transposes and lda/ldb/ldc exactly, so call
// sites are unchanged. Host-pointer alpha/beta (the default pointer mode).
//
// Optional Eigen backend: define CUBLAS_CPU_USE_EIGEN (Eigen is column-major too, so
// the mapping is direct). Without it the header is standalone (naive loops), zero deps.
#pragma once
#include <cstdint>
#include <cmath>
#include <cstddef>

typedef enum { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_INVALID_VALUE = 7 } cublasStatus_t;
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;
typedef enum { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 } cublasPointerMode_t;
typedef enum { CUBLAS_FILL_MODE_LOWER = 0, CUBLAS_FILL_MODE_UPPER = 1 } cublasFillMode_t;
typedef enum { CUBLAS_DEFAULT_MATH = 0, CUBLAS_TENSOR_OP_MATH = 1 } cublasMath_t;
struct cublasContext { int _ = 0; };
typedef cublasContext* cublasHandle_t;
typedef void* cudaStream_t;   // opaque; ignored on CPU

inline cublasStatus_t cublasCreate_v2(cublasHandle_t* h){ *h=new cublasContext; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasDestroy_v2(cublasHandle_t h){ delete h; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetStream_v2(cublasHandle_t,cudaStream_t){ return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t,cublasPointerMode_t){ return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetMathMode(cublasHandle_t,cublasMath_t){ return CUBLAS_STATUS_SUCCESS; }
// cuBLAS apps commonly use the un-suffixed names (mapped to _v2 by the real header)
#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasSetStream cublasSetStream_v2
#define cublasSetPointerMode cublasSetPointerMode_v2

#ifdef CUBLAS_CPU_USE_EIGEN
  #ifndef CUBLAS_CPU_EIGEN_HEADER
    #define CUBLAS_CPU_EIGEN_HEADER "third_party/eigen_flat/Eigen_Core.h"
  #endif
  #include CUBLAS_CPU_EIGEN_HEADER
namespace cublas_cpu_detail {
  using ColStride = Eigen::OuterStride<>;
  using CMap  = Eigen::Map<const Eigen::MatrixXf, 0, ColStride>;
  using MMap  = Eigen::Map<Eigen::MatrixXf, 0, ColStride>;
}
#endif

// ---- SGEMM: C = alpha*op(A)*op(B) + beta*C  (column-major) ----
//   op(A) is m×k, op(B) is k×n, C is m×n.
inline cublasStatus_t cublasSgemm_v2(cublasHandle_t,cublasOperation_t transa,cublasOperation_t transb,
    int m,int n,int k,const float* alpha,const float* A,int lda,const float* B,int ldb,
    const float* beta,float* C,int ldc){
  float al=*alpha, be=*beta;
#ifdef CUBLAS_CPU_USE_EIGEN
  using namespace cublas_cpu_detail;
  CMap Am(A, transa==CUBLAS_OP_N?m:k, transa==CUBLAS_OP_N?k:m, ColStride(lda));
  CMap Bm(B, transb==CUBLAS_OP_N?k:n, transb==CUBLAS_OP_N?n:k, ColStride(ldb));
  MMap Cm(C, m, n, ColStride(ldc));
  Eigen::MatrixXf OA = (transa==CUBLAS_OP_N)? Eigen::MatrixXf(Am) : Eigen::MatrixXf(Am.transpose());
  Eigen::MatrixXf OB = (transb==CUBLAS_OP_N)? Eigen::MatrixXf(Bm) : Eigen::MatrixXf(Bm.transpose());
  if (be==0.f) Cm.noalias() = al*(OA*OB);
  else         Cm = al*(OA*OB) + be*Cm;
#else
  auto gA=[&](int i,int l){ return transa==CUBLAS_OP_N ? A[i+(size_t)l*lda] : A[l+(size_t)i*lda]; };
  auto gB=[&](int l,int j){ return transb==CUBLAS_OP_N ? B[l+(size_t)j*ldb] : B[j+(size_t)l*ldb]; };
  for(int j=0;j<n;++j) for(int i=0;i<m;++i){ float acc=0; for(int l=0;l<k;++l) acc+=gA(i,l)*gB(l,j);
    float* c=&C[i+(size_t)j*ldc]; *c = al*acc + be*(*c); }
#endif
  return CUBLAS_STATUS_SUCCESS;
}
#define cublasSgemm cublasSgemm_v2

// ---- strided-batched SGEMM (attention / batched matmul) ----
inline cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,
    int m,int n,int k,const float* alpha,const float* A,int lda,long long strideA,
    const float* B,int ldb,long long strideB,const float* beta,float* C,int ldc,long long strideC,int batch){
  for(int b=0;b<batch;++b)
    cublasSgemm_v2(h,ta,tb,m,n,k,alpha,A+b*strideA,lda,B+b*strideB,ldb,beta,C+b*strideC,ldc);
  return CUBLAS_STATUS_SUCCESS;
}

// ---- SGEMV: y = alpha*op(A)*x + beta*y  (A is m×n column-major) ----
inline cublasStatus_t cublasSgemv_v2(cublasHandle_t,cublasOperation_t trans,int m,int n,
    const float* alpha,const float* A,int lda,const float* x,int incx,const float* beta,float* y,int incy){
  float al=*alpha, be=*beta;
  int leny = (trans==CUBLAS_OP_N)? m : n;
  int lenx = (trans==CUBLAS_OP_N)? n : m;
  for(int i=0;i<leny;++i){ float acc=0;
    for(int j=0;j<lenx;++j){ float aij = (trans==CUBLAS_OP_N)? A[i+(size_t)j*lda] : A[j+(size_t)i*lda];
      acc += aij * x[(size_t)j*incx]; }
    float* yi=&y[(size_t)i*incy]; *yi = al*acc + be*(*yi); }
  return CUBLAS_STATUS_SUCCESS;
}
#define cublasSgemv cublasSgemv_v2

// ---- SGER: rank-1 update A += alpha * x * y^T  (A is m×n column-major) ----
inline cublasStatus_t cublasSger_v2(cublasHandle_t,int m,int n,const float* alpha,
    const float* x,int incx,const float* y,int incy,float* A,int lda){
  float al=*alpha;
  for(int j=0;j<n;++j){ float ay=al*y[(size_t)j*incy]; for(int i=0;i<m;++i) A[i+(size_t)j*lda]+=x[(size_t)i*incx]*ay; }
  return CUBLAS_STATUS_SUCCESS;
}
#define cublasSger cublasSger_v2

// ---- level-1 BLAS ----
inline cublasStatus_t cublasSaxpy_v2(cublasHandle_t,int n,const float* alpha,const float* x,int incx,float* y,int incy){
  float a=*alpha; for(int i=0;i<n;++i) y[(size_t)i*incy]+=a*x[(size_t)i*incx]; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSscal_v2(cublasHandle_t,int n,const float* alpha,float* x,int incx){
  float a=*alpha; for(int i=0;i<n;++i) x[(size_t)i*incx]*=a; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasScopy_v2(cublasHandle_t,int n,const float* x,int incx,float* y,int incy){
  for(int i=0;i<n;++i) y[(size_t)i*incy]=x[(size_t)i*incx]; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSdot_v2(cublasHandle_t,int n,const float* x,int incx,const float* y,int incy,float* result){
  double s=0; for(int i=0;i<n;++i) s+=(double)x[(size_t)i*incx]*y[(size_t)i*incy]; *result=(float)s; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSasum_v2(cublasHandle_t,int n,const float* x,int incx,float* result){
  double s=0; for(int i=0;i<n;++i) s+=std::fabs(x[(size_t)i*incx]); *result=(float)s; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSnrm2_v2(cublasHandle_t,int n,const float* x,int incx,float* result){
  double s=0; for(int i=0;i<n;++i){ double v=x[(size_t)i*incx]; s+=v*v; } *result=(float)std::sqrt(s); return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasIsamax_v2(cublasHandle_t,int n,const float* x,int incx,int* result){
  int best=0; float bv=-1; for(int i=0;i<n;++i){ float v=std::fabs(x[(size_t)i*incx]); if(v>bv){bv=v;best=i;} }
  *result=best+1; return CUBLAS_STATUS_SUCCESS; }   // cuBLAS indices are 1-based
inline cublasStatus_t cublasSswap_v2(cublasHandle_t,int n,float* x,int incx,float* y,int incy){
  for(int i=0;i<n;++i){ float t=x[(size_t)i*incx]; x[(size_t)i*incx]=y[(size_t)i*incy]; y[(size_t)i*incy]=t; } return CUBLAS_STATUS_SUCCESS; }
#define cublasSaxpy cublasSaxpy_v2
#define cublasSscal cublasSscal_v2
#define cublasScopy cublasScopy_v2
#define cublasSdot  cublasSdot_v2
#define cublasSasum cublasSasum_v2
#define cublasSnrm2 cublasSnrm2_v2
#define cublasIsamax cublasIsamax_v2
#define cublasSswap cublasSswap_v2
