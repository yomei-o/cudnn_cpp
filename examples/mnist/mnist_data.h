// mnist_data.h — MNIST IDX loader + PNG digit loader (via stb). Host-side, no CUDA.
#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace mnist {

struct Dataset { std::vector<float> images; std::vector<int> labels; int n=0, rows=28, cols=28; };

static uint32_t rd_be32(std::FILE* f){ unsigned char b[4]; if(std::fread(b,1,4,f)!=4)return 0; return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3]; }

// images normalized to [0,1], layout [n][rows*cols]
inline bool load_images(const std::string& path, Dataset& d){
  std::FILE* f=std::fopen(path.c_str(),"rb"); if(!f){ std::fprintf(stderr,"cannot open %s\n",path.c_str()); return false; }
  uint32_t magic=rd_be32(f), n=rd_be32(f), r=rd_be32(f), c=rd_be32(f);
  if(magic!=0x00000803){ std::fprintf(stderr,"bad image magic in %s\n",path.c_str()); std::fclose(f); return false; }
  d.n=(int)n; d.rows=(int)r; d.cols=(int)c; d.images.resize((size_t)n*r*c);
  std::vector<unsigned char> buf((size_t)n*r*c); if(std::fread(buf.data(),1,buf.size(),f)!=buf.size()){ std::fclose(f); return false; }
  for(size_t i=0;i<buf.size();++i) d.images[i]=buf[i]/255.f;
  std::fclose(f); return true;
}
inline bool load_labels(const std::string& path, Dataset& d){
  std::FILE* f=std::fopen(path.c_str(),"rb"); if(!f){ std::fprintf(stderr,"cannot open %s\n",path.c_str()); return false; }
  uint32_t magic=rd_be32(f), n=rd_be32(f);
  if(magic!=0x00000801){ std::fprintf(stderr,"bad label magic in %s\n",path.c_str()); std::fclose(f); return false; }
  d.labels.resize(n); std::vector<unsigned char> buf(n); if(std::fread(buf.data(),1,n,f)!=n){ std::fclose(f); return false; }
  for(uint32_t i=0;i<n;++i) d.labels[i]=(int)buf[i];
  std::fclose(f); return true;
}
inline bool load(const std::string& dir, Dataset& train, Dataset& test){
  return load_images(dir+"/train-images-idx3-ubyte",train) && load_labels(dir+"/train-labels-idx1-ubyte",train)
      && load_images(dir+"/t10k-images-idx3-ubyte", test)  && load_labels(dir+"/t10k-labels-idx1-ubyte", test);
}

} // namespace mnist

// PNG digit loader — needs stb_image.h (define STB_IMAGE_IMPLEMENTATION in ONE TU).
#ifdef MNIST_WITH_STB
#include "third_party/stb_image.h"
namespace mnist {
// Load any PNG/JPG, convert to a 28x28 [0,1] MNIST-style digit (white on black).
// Auto-inverts if the image looks like black-on-white (typical hand drawings/scans).
inline bool load_png_digit(const std::string& path, std::vector<float>& out28){
  int W,H,ch; unsigned char* im=stbi_load(path.c_str(),&W,&H,&ch,1); // force grayscale
  if(!im){ std::fprintf(stderr,"cannot load image %s\n",path.c_str()); return false; }
  out28.assign(28*28,0.f);
  // area-average resize to 28x28
  for(int y=0;y<28;++y)for(int x=0;x<28;++x){
    int x0=x*W/28, x1=std::max(x0+1,(x+1)*W/28), y0=y*H/28, y1=std::max(y0+1,(y+1)*H/28);
    double s=0; int cnt=0; for(int yy=y0;yy<y1;++yy)for(int xx=x0;xx<x1;++xx){ s+=im[yy*W+xx]; ++cnt; }
    out28[y*28+x]=(float)(s/std::max(cnt,1)/255.0);
  }
  stbi_image_free(im);
  double mean=0; for(float v:out28)mean+=v; mean/=out28.size();
  if(mean>0.5) for(auto&v:out28) v=1.f-v;      // invert black-on-white -> white-on-black
  return true;
}
} // namespace mnist
#endif
