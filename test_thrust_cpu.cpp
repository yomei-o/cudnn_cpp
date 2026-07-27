// test_thrust_cpu.cpp — verify thrust_cpu.h against hand references.
//   g++ -std=c++17 -O2 -I. test_thrust_cpu.cpp -o tt && ./tt
#include "thrust_cpu.h"
#include <cstdio>
#include <cmath>

static int fails=0;
static void ok(const char* name,bool cond){ printf("  %-34s %s\n",name,cond?"OK":"FAIL"); if(!cond)++fails; }

int main(){
  // device_vector + sequence + transform (x -> 2x+1)
  {
    thrust::device_vector<int> d(6); thrust::sequence(d.begin(),d.end());        // 0..5
    thrust::transform(d.begin(),d.end(),d.begin(),[](int x){return 2*x+1;});
    bool good=true; for(int i=0;i<6;++i) good &= (d[i]==2*i+1);
    ok("device_vector+sequence+transform",good);
  }
  // reduce & transform_reduce (sum of squares 1..4 = 30)
  {
    thrust::device_vector<int> d{1,2,3,4};
    int s=thrust::reduce(d.begin(),d.end(),0);
    int sq=thrust::transform_reduce(d.begin(),d.end(),[](int x){return x*x;},0,thrust::plus<int>());
    ok("reduce (sum=10)",s==10);
    ok("transform_reduce (sumsq=30)",sq==30);
  }
  // binary transform + inner_product
  {
    thrust::device_vector<int> a{1,2,3},b{4,5,6},c(3);
    thrust::transform(a.begin(),a.end(),b.begin(),c.begin(),thrust::multiplies<int>());
    ok("binary transform (4,10,18)",c[0]==4&&c[1]==10&&c[2]==18);
    int ip=thrust::inner_product(a.begin(),a.end(),b.begin(),0);
    ok("inner_product (32)",ip==32);
  }
  // sort_by_key — NMS pattern: sort indices by descending score
  {
    thrust::device_vector<float> score{0.2f,0.9f,0.5f,0.1f,0.7f};
    thrust::device_vector<int>   idx{0,1,2,3,4};
    thrust::sort_by_key(score.begin(),score.end(),idx.begin(),thrust::greater<float>());
    // scores now descending; idx follows: expect [1,4,2,0,3]
    bool good = idx[0]==1&&idx[1]==4&&idx[2]==2&&idx[3]==0&&idx[4]==3;
    good &= score[0]>score[1] && score[1]>score[2] && score[2]>score[3] && score[3]>score[4];
    ok("sort_by_key desc (NMS order)",good);
  }
  // copy_if (stream compaction: keep > 0)
  {
    thrust::device_vector<int> in{-1,3,-2,5,0,8}, out(6);
    auto e=thrust::copy_if(in.begin(),in.end(),out.begin(),[](int x){return x>0;});
    int n=(int)(e-out.begin());
    ok("copy_if (>0 -> 3,5,8)", n==3 && out[0]==3 && out[1]==5 && out[2]==8);
  }
  // exclusive_scan (compaction offsets)
  {
    thrust::device_vector<int> flags{1,0,1,1,0,1}, off(6);
    thrust::exclusive_scan(flags.begin(),flags.end(),off.begin(),0);
    // 0,1,1,2,3,3
    bool good = off[0]==0&&off[1]==1&&off[2]==1&&off[3]==2&&off[4]==3&&off[5]==3;
    ok("exclusive_scan",good);
  }
  // gather / scatter
  {
    thrust::device_vector<int> src{10,20,30,40}, map{3,0,2}, g(3);
    thrust::gather(map.begin(),map.end(),src.begin(),g.begin());
    ok("gather (40,10,30)",g[0]==40&&g[1]==10&&g[2]==30);
    thrust::device_vector<int> out(4,0), val{7,8}, m2{2,0};
    thrust::scatter(val.begin(),val.end(),m2.begin(),out.begin());
    ok("scatter",out[2]==7&&out[0]==8);
  }
  // counting_iterator + transform_reduce (sum 0..99 = 4950)
  {
    auto b=thrust::make_counting_iterator(0), e=thrust::make_counting_iterator(100);
    int s=thrust::reduce(b,e,0);
    ok("counting_iterator reduce (4950)",s==4950);
  }
  // count_if / max_element
  {
    thrust::device_vector<int> d{3,1,4,1,5,9,2,6};
    ok("count_if (>3 -> 4)",thrust::count_if(d.begin(),d.end(),[](int x){return x>3;})==4);
    ok("max_element (9)",*thrust::max_element(d.begin(),d.end())==9);
  }
  // reduce_by_key
  {
    thrust::device_vector<int> keys{1,1,2,2,2,3}, vals{10,20,30,40,50,60};
    thrust::device_vector<int> ko(6,-1), vo(6,-1);
    auto ends=thrust::reduce_by_key(keys.begin(),keys.end(),vals.begin(),ko.begin(),vo.begin());
    int n=(int)(ends.first-ko.begin());
    bool good = n==3 && ko[0]==1&&vo[0]==30 && ko[1]==2&&vo[1]==120 && ko[2]==3&&vo[2]==60;
    ok("reduce_by_key",good);
  }
  // execution policy accepted & ignored
  {
    thrust::device_vector<int> d{5,3,1,4,2};
    thrust::sort(thrust::device, d.begin(), d.end());
    bool good=true; for(int i=0;i<5;++i) good &= (d[i]==i+1);
    ok("thrust::device policy (sort)",good);
  }

  printf("\n%s\n", fails? "SOME TESTS FAILED":"ALL THRUST TESTS PASSED");
  return fails?1:0;
}
