// thrust_cpu.h — header-only, source-available CPU implementation of the Thrust API
// subset that GPU / YOLO code uses (device_vector, the parallel algorithms, functors,
// counting iterator). Include this INSTEAD of <thrust/*.h> to build & correctness-test
// on a GPU-less box without pulling the full CCCL/CUB dependency; swap back to real
// Thrust on the GPU — call sites are unchanged.
//
// Everything runs serially on the host (delegates to <algorithm>/<numeric>). Execution
// policies (thrust::device / host / seq) are accepted and ignored.
#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <utility>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace thrust {

// ---- execution policies (accepted, ignored) ----
struct execution_policy_base {};
struct host_execution_policy   : execution_policy_base {};
struct device_execution_policy : execution_policy_base {};
inline host_execution_policy   host;
inline device_execution_policy device;
inline host_execution_policy   seq;
template<class T> using NotPolicy = std::enable_if_t<!std::is_base_of<execution_policy_base,std::decay_t<T>>::value,int>;
template<class T> using IsPolicy  = std::enable_if_t< std::is_base_of<execution_policy_base,std::decay_t<T>>::value,int>;
// generate a policy-stripping forwarder overload for a named algorithm
#define THRUST_POLICY_FWD(NAME) \
  template<class P,class...A,IsPolicy<P> =0> \
  auto NAME(P&&,A&&...a)->decltype(::thrust::NAME(std::forward<A>(a)...)){ return ::thrust::NAME(std::forward<A>(a)...); }

// ---- containers & pointers ----
template<class T> using host_vector   = std::vector<T>;
template<class T> using device_vector = std::vector<T>;   // host memory on CPU
template<class T> using device_ptr    = T*;
template<class T> T* raw_pointer_cast(T* p){ return p; }
template<class T> T* device_pointer_cast(T* p){ return p; }
template<class A,class B> using pair = std::pair<A,B>;
template<class A,class B> std::pair<A,B> make_pair(A a,B b){ return std::make_pair(a,b); }

// ---- functors ----
template<class T=void> struct plus       { T operator()(const T&a,const T&b)const{return a+b;} };
template<class T=void> struct minus      { T operator()(const T&a,const T&b)const{return a-b;} };
template<class T=void> struct multiplies { T operator()(const T&a,const T&b)const{return a*b;} };
template<class T=void> struct divides    { T operator()(const T&a,const T&b)const{return a/b;} };
template<class T=void> struct negate     { T operator()(const T&a)const{return -a;} };
template<class T=void> struct maximum    { T operator()(const T&a,const T&b)const{return a<b?b:a;} };
template<class T=void> struct minimum    { T operator()(const T&a,const T&b)const{return b<a?b:a;} };
template<class T=void> struct greater      { bool operator()(const T&a,const T&b)const{return a>b;} };
template<class T=void> struct less         { bool operator()(const T&a,const T&b)const{return a<b;} };
template<class T=void> struct greater_equal{ bool operator()(const T&a,const T&b)const{return a>=b;} };
template<class T=void> struct less_equal   { bool operator()(const T&a,const T&b)const{return a<=b;} };
template<class T=void> struct equal_to     { bool operator()(const T&a,const T&b)const{return a==b;} };
template<class T=void> struct not_equal_to { bool operator()(const T&a,const T&b)const{return a!=b;} };
template<class T=void> struct identity     { T operator()(const T&a)const{return a;} };

// ---- counting / constant iterators ----
template<class I> struct counting_iterator {
  using iterator_category=std::random_access_iterator_tag; using value_type=I;
  using difference_type=std::ptrdiff_t; using pointer=const I*; using reference=I;
  I v; counting_iterator(I x=I()):v(x){}
  I operator*()const{return v;} I operator[](std::ptrdiff_t n)const{return v+(I)n;}
  counting_iterator& operator++(){++v;return *this;} counting_iterator operator++(int){auto t=*this;++v;return t;}
  counting_iterator& operator--(){--v;return *this;}
  counting_iterator operator+(std::ptrdiff_t n)const{return counting_iterator(v+(I)n);}
  counting_iterator operator-(std::ptrdiff_t n)const{return counting_iterator(v-(I)n);}
  std::ptrdiff_t operator-(const counting_iterator&o)const{return (std::ptrdiff_t)(v-o.v);}
  bool operator==(const counting_iterator&o)const{return v==o.v;} bool operator!=(const counting_iterator&o)const{return v!=o.v;}
  bool operator<(const counting_iterator&o)const{return v<o.v;}
};
template<class I> counting_iterator<I> make_counting_iterator(I i){ return counting_iterator<I>(i); }
template<class T> struct constant_iterator {
  using iterator_category=std::random_access_iterator_tag; using value_type=T;
  using difference_type=std::ptrdiff_t; using pointer=const T*; using reference=const T&;
  T v; std::ptrdiff_t i; constant_iterator(T x,std::ptrdiff_t k=0):v(x),i(k){}
  T operator*()const{return v;} T operator[](std::ptrdiff_t)const{return v;}
  constant_iterator& operator++(){++i;return *this;}
  constant_iterator operator+(std::ptrdiff_t n)const{return constant_iterator(v,i+n);}
  std::ptrdiff_t operator-(const constant_iterator&o)const{return i-o.i;}
  bool operator==(const constant_iterator&o)const{return i==o.i;} bool operator!=(const constant_iterator&o)const{return i!=o.i;}
};
template<class T> constant_iterator<T> make_constant_iterator(T v){ return constant_iterator<T>(v); }

// ---- algorithms (core; a policy-first call routes through THRUST_POLICY_FWD) ----
template<class It,class F,NotPolicy<It> =0> F for_each(It f,It l,F fn){ for(;f!=l;++f)fn(*f); return fn; }
THRUST_POLICY_FWD(for_each)

template<class It,class Ot,class F,NotPolicy<It> =0> Ot transform(It f,It l,Ot o,F op){ for(;f!=l;++f,++o)*o=op(*f); return o; }
template<class I1,class I2,class Ot,class F,NotPolicy<I1> =0> Ot transform(I1 f,I1 l,I2 g,Ot o,F op){ for(;f!=l;++f,++g,++o)*o=op(*f,*g); return o; }
THRUST_POLICY_FWD(transform)

template<class It,class T,NotPolicy<It> =0> void fill(It f,It l,const T&v){ for(;f!=l;++f)*f=v; }
THRUST_POLICY_FWD(fill)
template<class It,class S,class T,NotPolicy<It> =0> It fill_n(It f,S n,const T&v){ for(S i=0;i<n;++i,++f)*f=v; return f; }
THRUST_POLICY_FWD(fill_n)

template<class It,class T,NotPolicy<It> =0> void sequence(It f,It l,T init,T step){ for(T v=init;f!=l;++f,v+=step)*f=v; }
template<class It,class T,NotPolicy<It> =0> void sequence(It f,It l,T init){ sequence(f,l,init,T(1)); }
template<class It,NotPolicy<It> =0> void sequence(It f,It l){ using T=typename std::iterator_traits<It>::value_type; sequence(f,l,T(0),T(1)); }
THRUST_POLICY_FWD(sequence)

template<class It,class Ot,NotPolicy<It> =0> Ot copy(It f,It l,Ot o){ for(;f!=l;++f,++o)*o=*f; return o; }
THRUST_POLICY_FWD(copy)
template<class It,class S,class Ot,NotPolicy<It> =0> Ot copy_n(It f,S n,Ot o){ for(S i=0;i<n;++i,++f,++o)*o=*f; return o; }
THRUST_POLICY_FWD(copy_n)
template<class It,class Ot,class P,NotPolicy<It> =0> Ot copy_if(It f,It l,Ot o,P pred){ for(;f!=l;++f)if(pred(*f)){*o=*f;++o;} return o; }
template<class It,class St,class Ot,class P,NotPolicy<It> =0> Ot copy_if(It f,It l,St s,Ot o,P pred){ for(;f!=l;++f,++s)if(pred(*s)){*o=*f;++o;} return o; }
THRUST_POLICY_FWD(copy_if)

template<class It,NotPolicy<It> =0> typename std::iterator_traits<It>::value_type reduce(It f,It l){ using T=typename std::iterator_traits<It>::value_type; return std::accumulate(f,l,T(0)); }
template<class It,class T,NotPolicy<It> =0> T reduce(It f,It l,T init){ return std::accumulate(f,l,init); }
template<class It,class T,class B,NotPolicy<It> =0> T reduce(It f,It l,T init,B op){ for(;f!=l;++f)init=op(init,*f); return init; }
THRUST_POLICY_FWD(reduce)

template<class It,class T,class U,NotPolicy<It> =0> T transform_reduce(It f,It l,U un,T init,std::plus<T>){ for(;f!=l;++f)init=init+un(*f); return init; }
template<class It,class U,class T,class B,NotPolicy<It> =0> T transform_reduce(It f,It l,U un,T init,B br){ for(;f!=l;++f)init=br(init,un(*f)); return init; }
THRUST_POLICY_FWD(transform_reduce)

template<class I1,class I2,class T,NotPolicy<I1> =0> T inner_product(I1 f,I1 l,I2 g,T init){ for(;f!=l;++f,++g)init=init+(*f)*(*g); return init; }
template<class I1,class I2,class T,class B1,class B2,NotPolicy<I1> =0> T inner_product(I1 f,I1 l,I2 g,T init,B1 red,B2 op){ for(;f!=l;++f,++g)init=red(init,op(*f,*g)); return init; }
THRUST_POLICY_FWD(inner_product)

template<class It,class T,NotPolicy<It> =0> std::ptrdiff_t count(It f,It l,const T&v){ return std::count(f,l,v); }
THRUST_POLICY_FWD(count)
template<class It,class P,NotPolicy<It> =0> std::ptrdiff_t count_if(It f,It l,P p){ return std::count_if(f,l,p); }
THRUST_POLICY_FWD(count_if)

template<class It,NotPolicy<It> =0> void sort(It f,It l){ std::sort(f,l); }
template<class It,class C,NotPolicy<It> =0> void sort(It f,It l,C c){ std::sort(f,l,c); }
THRUST_POLICY_FWD(sort)
template<class It,NotPolicy<It> =0> void stable_sort(It f,It l){ std::stable_sort(f,l); }
template<class It,class C,NotPolicy<It> =0> void stable_sort(It f,It l,C c){ std::stable_sort(f,l,c); }
THRUST_POLICY_FWD(stable_sort)

// sort_by_key: reorder [values...) to follow the key permutation (stable index sort)
namespace detail {
  template<class KIt,class VIt,class Cmp>
  void keysort(KIt kf,KIt kl,VIt vf,Cmp cmp,bool stable){
    std::ptrdiff_t n=kl-kf; std::vector<std::ptrdiff_t> idx(n); std::iota(idx.begin(),idx.end(),0);
    auto by=[&](std::ptrdiff_t a,std::ptrdiff_t b){ return cmp(*(kf+a),*(kf+b)); };
    if(stable) std::stable_sort(idx.begin(),idx.end(),by); else std::sort(idx.begin(),idx.end(),by);
    using K=typename std::iterator_traits<KIt>::value_type; using V=typename std::iterator_traits<VIt>::value_type;
    std::vector<K> ks(n); std::vector<V> vs(n);
    for(std::ptrdiff_t i=0;i<n;++i){ ks[i]=*(kf+idx[i]); vs[i]=*(vf+idx[i]); }
    for(std::ptrdiff_t i=0;i<n;++i){ *(kf+i)=ks[i]; *(vf+i)=vs[i]; }
  }
}
template<class KIt,class VIt,NotPolicy<KIt> =0> void sort_by_key(KIt kf,KIt kl,VIt vf){ detail::keysort(kf,kl,vf,std::less<typename std::iterator_traits<KIt>::value_type>(),false); }
template<class KIt,class VIt,class C,NotPolicy<KIt> =0> void sort_by_key(KIt kf,KIt kl,VIt vf,C c){ detail::keysort(kf,kl,vf,c,false); }
THRUST_POLICY_FWD(sort_by_key)
template<class KIt,class VIt,NotPolicy<KIt> =0> void stable_sort_by_key(KIt kf,KIt kl,VIt vf){ detail::keysort(kf,kl,vf,std::less<typename std::iterator_traits<KIt>::value_type>(),true); }
template<class KIt,class VIt,class C,NotPolicy<KIt> =0> void stable_sort_by_key(KIt kf,KIt kl,VIt vf,C c){ detail::keysort(kf,kl,vf,c,true); }
THRUST_POLICY_FWD(stable_sort_by_key)

template<class It,class Ot,NotPolicy<It> =0> Ot inclusive_scan(It f,It l,Ot o){ using T=typename std::iterator_traits<It>::value_type; T acc{}; bool first=true; for(;f!=l;++f,++o){ acc=first?*f:acc+*f; first=false; *o=acc; } return o; }
template<class It,class Ot,class B,NotPolicy<It> =0> Ot inclusive_scan(It f,It l,Ot o,B op){ using T=typename std::iterator_traits<It>::value_type; T acc{}; bool first=true; for(;f!=l;++f,++o){ acc=first?*f:op(acc,*f); first=false; *o=acc; } return o; }
THRUST_POLICY_FWD(inclusive_scan)
template<class It,class Ot,NotPolicy<It> =0> Ot exclusive_scan(It f,It l,Ot o){ using T=typename std::iterator_traits<It>::value_type; return exclusive_scan(f,l,o,T(0)); }
template<class It,class Ot,class T,NotPolicy<It> =0> Ot exclusive_scan(It f,It l,Ot o,T init){ T acc=init; for(;f!=l;++f,++o){ T x=*f; *o=acc; acc=acc+x; } return o; }
template<class It,class Ot,class T,class B,NotPolicy<It> =0> Ot exclusive_scan(It f,It l,Ot o,T init,B op){ T acc=init; for(;f!=l;++f,++o){ T x=*f; *o=acc; acc=op(acc,x); } return o; }
THRUST_POLICY_FWD(exclusive_scan)

template<class It,NotPolicy<It> =0> It max_element(It f,It l){ return std::max_element(f,l); }
template<class It,class C,NotPolicy<It> =0> It max_element(It f,It l,C c){ return std::max_element(f,l,c); }
THRUST_POLICY_FWD(max_element)
template<class It,NotPolicy<It> =0> It min_element(It f,It l){ return std::min_element(f,l); }
template<class It,class C,NotPolicy<It> =0> It min_element(It f,It l,C c){ return std::min_element(f,l,c); }
THRUST_POLICY_FWD(min_element)

template<class It,class P,NotPolicy<It> =0> It remove_if(It f,It l,P p){ return std::remove_if(f,l,p); }
THRUST_POLICY_FWD(remove_if)
template<class It,class T,NotPolicy<It> =0> It remove(It f,It l,const T&v){ return std::remove(f,l,v); }
THRUST_POLICY_FWD(remove)
template<class It,NotPolicy<It> =0> It unique(It f,It l){ return std::unique(f,l); }
template<class It,class P,NotPolicy<It> =0> It unique(It f,It l,P p){ return std::unique(f,l,p); }
THRUST_POLICY_FWD(unique)

template<class It,class C,NotPolicy<It> =0> It find(It f,It l,const C&v){ return std::find(f,l,v); }
THRUST_POLICY_FWD(find)
template<class It,class P,NotPolicy<It> =0> It find_if(It f,It l,P p){ return std::find_if(f,l,p); }
THRUST_POLICY_FWD(find_if)

// gather: result[i] = input[ map[i] ];  scatter: output[ map[i] ] = input[i]
template<class MIt,class IIt,class OIt,NotPolicy<MIt> =0> OIt gather(MIt mf,MIt ml,IIt in,OIt out){ for(;mf!=ml;++mf,++out)*out=*(in+*mf); return out; }
THRUST_POLICY_FWD(gather)
template<class IIt,class MIt,class OIt,NotPolicy<IIt> =0> void scatter(IIt f,IIt l,MIt m,OIt out){ for(;f!=l;++f,++m)*(out+*m)=*f; }
THRUST_POLICY_FWD(scatter)

// reduce_by_key: collapse consecutive equal keys, summing (or binary_op-ing) values
template<class KIt,class VIt,class KO,class VO,class BP,class BO,NotPolicy<KIt> =0>
std::pair<KO,VO> reduce_by_key(KIt kf,KIt kl,VIt vf,KO ko,VO vo,BP keq,BO op){
  while(kf!=kl){ auto key=*kf; auto acc=*vf; ++kf; ++vf;
    while(kf!=kl && keq(*kf,key)){ acc=op(acc,*vf); ++kf; ++vf; }
    *ko=key; *vo=acc; ++ko; ++vo; }
  return std::make_pair(ko,vo);
}
template<class KIt,class VIt,class KO,class VO,NotPolicy<KIt> =0>
std::pair<KO,VO> reduce_by_key(KIt kf,KIt kl,VIt vf,KO ko,VO vo){
  using V=typename std::iterator_traits<VIt>::value_type; using K=typename std::iterator_traits<KIt>::value_type;
  return reduce_by_key(kf,kl,vf,ko,vo,std::equal_to<K>(),std::plus<V>());
}
THRUST_POLICY_FWD(reduce_by_key)

template<class I1,class I2,NotPolicy<I1> =0> bool equal(I1 f,I1 l,I2 g){ return std::equal(f,l,g); }
THRUST_POLICY_FWD(equal)

using std::distance;
using std::swap;

} // namespace thrust
