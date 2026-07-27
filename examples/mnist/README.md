# MNIST サンプル — GPUを使わずにGPUの使い方を学ぶ

*(English version: [README.en.md](README.en.md))*

このディレクトリは、**GPUを持っていなくてもGPUプログラミングを学べる**教材です。
核になる考え方はひとつ：

> **同じC++ソースが、コンパイル方法を変えるだけでCPUでもGPUでも動く。**

CPUでは [cudnn_cpp](../../README.md) のヘッダオンリー・シム（cuDNN / cuBLAS / Thrust /
CUDA runtime のCPU実装）を使い、GPUでは本物のCUDAライブラリをリンクします。
**ソースコードは1文字も変えません。** 変わるのはコンパイルコマンドだけです。

```sh
# CPU（GPU不要・g++だけ）
g++  -std=c++17 -O3 -I. examples/mnist/mnist_mlp.cpp -o mnist_mlp

# GPU（同じソースを nvcc でビルドするだけ）
nvcc -std=c++17 -O3 -x cu -DMNIST_GPU -I. examples/mnist/mnist_mlp.cpp -o mnist_mlp -lcublas
```

## 2つのモデル

| ソース | 中身 | 使うライブラリ |
|---|---|---|
| [`mnist_mlp.cpp`](mnist_mlp.cpp) | 全結合2層 (784→128→10) | **cuBLAS**(行列積) + **Thrust**(要素演算) + CUDA runtime |
| [`mnist_cnn.cpp`](mnist_cnn.cpp) | CNN (conv8→conv16→FC) | **cuDNN**(畳み込み/プール/活性/softmax) + cuBLAS + Thrust |

どちらもカスタムカーネル (`<<<>>>`) を使わず、ライブラリ呼び出しとバッファ操作だけで
**学習（forward + backward + SGD）**まで完結します。だからCPUシムでそのまま動きます。

## 4つの Colab ノートブック

学習 → モデル保存 → PNG画像を読んで数字を推論、まで通しで実行できます。
**CPU版とGPU版を両方動かして、`images/sec`（1秒あたり学習枚数）を比べる**のがこの教材の要です。

| ノートブック | 内容 | Colab |
|---|---|---|
| [`mnist_thrust_cublas_cpu.ipynb`](colab/mnist_thrust_cublas_cpu.ipynb) | MLP / CPU | [開く](https://colab.research.google.com/github/yomei-o/cudnn_cpp/blob/main/examples/mnist/colab/mnist_thrust_cublas_cpu.ipynb) |
| [`mnist_thrust_cublas_gpu.ipynb`](colab/mnist_thrust_cublas_gpu.ipynb) | MLP / GPU | [開く](https://colab.research.google.com/github/yomei-o/cudnn_cpp/blob/main/examples/mnist/colab/mnist_thrust_cublas_gpu.ipynb) |
| [`mnist_cudnn_thrust_cublas_cpu.ipynb`](colab/mnist_cudnn_thrust_cublas_cpu.ipynb) | CNN / CPU | [開く](https://colab.research.google.com/github/yomei-o/cudnn_cpp/blob/main/examples/mnist/colab/mnist_cudnn_thrust_cublas_cpu.ipynb) |
| [`mnist_cudnn_thrust_cublas_gpu.ipynb`](colab/mnist_cudnn_thrust_cublas_gpu.ipynb) | CNN / GPU | [開く](https://colab.research.google.com/github/yomei-o/cudnn_cpp/blob/main/examples/mnist/colab/mnist_cudnn_thrust_cublas_gpu.ipynb) |

## 手元（CPU）で試す

```sh
# リポジトリ直下で。MNISTを examples/mnist/data/ に置く（各notebook参照）
g++ -std=c++17 -O3 -I. examples/mnist/mnist_mlp.cpp -o mnist_mlp
./mnist_mlp --epochs 5 --save mnist_mlp.bin           # 学習して保存
g++ -std=c++17 -O2 -I. examples/mnist/make_samples.cpp -o mk && ./mk   # 0-9のPNG生成
./mnist_mlp --load mnist_mlp.bin --infer examples/mnist/samples/digit7.png   # PNGを推論
```

## GPUを使うメリット（この教材で分かること）

`cudaEvent` で学習時間を計測し、`>> trained ... images/sec` を表示します。CPUとGPUで
**精度はほぼ同じ**（同じ計算だから）で、**違うのは速度だけ**——それがGPUを使う理由です。

- **MLPは小さい**ので、CPUでもそこそこ速い（＝小さいモデルにGPUは必須ではない）。
- **CNNはconvが重い**ので、CPUでは急に遅くなる。ここでGPU（cuDNN）が圧倒的に効きます。
- モデル／バッチを大きくするほど差が開きます。ノート末尾の「やってみよう」で体感できます。

参考（このリポジトリ作者のCPU実測、素のシム）：MLP ≈ 3,400 枚/秒、CNN ≈ 660 枚/秒。
GPU版ノートを回すと、この数字がどれだけ跳ね上がるかを自分の目で確認できます。

### CPUを速くする裏技（任意）

`-DCUDNN_CPU_USE_EIGEN -DCUBLAS_CPU_USE_EIGEN` を付けると、行列積が
[Eigen](../../third_party/eigen_flat) 経由になり、**大きな行列（例：MLPのFC）で数倍**速くなります。
ただし畳み込みが小さいMNISTのCNNでは効果が薄い（むしろ僅かに遅い）ことも。
「CPUも最適化ライブラリで速くできる。GPUだけが速さの手段ではない」という点も学べます。

## ファイル

- `mnist_mlp.cpp` / `mnist_cnn.cpp` — 学習＋推論プログラム（CPU/GPU共通ソース）
- `gpu_backend.h` — CPU/GPU を切り替える唯一の場所
- `mnist_data.h` — MNIST(IDX) と PNG のローダ
- `make_samples.cpp` — テスト画像から 0-9 のPNGを書き出す
- `colab/*.ipynb` — 4つの Colab ノートブック
