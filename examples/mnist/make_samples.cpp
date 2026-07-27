// make_samples.cpp — dump a few MNIST test digits as upscaled black-on-white PNGs, so
// the samples look like something a user would draw/scan (and exercise auto-invert).
//   g++ -std=c++17 -O2 -I. examples/mnist/make_samples.cpp -o mk && ./mk
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
#include "examples/mnist/mnist_data.h"
#include <cstdio>
#include <string>

int main(int argc,char**argv){
  std::string dir = argc>1?argv[1]:"examples/mnist/data";
  std::string out = argc>2?argv[2]:"examples/mnist/samples";
  mnist::Dataset tr,te; if(!mnist::load(dir,tr,te)){ std::fprintf(stderr,"load failed\n"); return 1; }
  int want[10]; for(int&w:want)w=-1; int found=0;
  for(int i=0;i<te.n && found<10;++i){ int l=te.labels[i]; if(want[l]<0){ want[l]=i; ++found; } }
  const int UP=4, S=28*UP;
  std::vector<unsigned char> img(S*S);
  for(int d=0;d<10;++d){ int idx=want[d]; if(idx<0)continue;
    for(int y=0;y<S;++y)for(int x=0;x<S;++x){ float v=te.images[(size_t)idx*784 + (y/UP)*28 + (x/UP)];
      img[y*S+x]=(unsigned char)(255.f*(1.f-v)); }   // black-on-white
    char path[256]; std::snprintf(path,256,"%s/digit%d.png",out.c_str(),d);
    stbi_write_png(path,S,S,1,img.data(),S); std::printf("wrote %s (true=%d)\n",path,d);
  }
  return 0;
}
