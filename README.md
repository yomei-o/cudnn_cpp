# cudnn_cpp

*(English version: [README.en.md](README.en.md))*

YOLO系の物体検出が使う **CUDAのGPUライブラリAPI** を、**ヘッダオンリー・ソース公開で
CPU実装**したものです — **cuDNN**・**cuBLAS**・**Thrust**。**GPUの無いマシン**（CUDA未インストール）で、
CUDA/GPU向けコードを開発し、正しさを検証できます。

| ヘッダ | 置き換え対象 | 用途 |
|---|---|---|
| [`cudnn_cpu.h`](cudnn_cpu.h) | `<cudnn.h>` | conv/act/pool/softmax/bn — **forward と backward** |
| [`cublas_cpu.h`](cublas_cpu.h) | `<cublas_v2.h>` | sgemm(+strided-batched)・gemv・ger・level-1 |
| [`thrust_cpu.h`](thrust_cpu.h) | `<thrust/*.h>` | device_vector・sort/sort_by_key・transform・reduce・scan・gather… |
| [`cuda_runtime_cpu.h`](cuda_runtime_cpu.h) | `<cuda_runtime.h>` | cudaMalloc/Memcpy/Memset・stream・event・device照会 |

ランタイムシムを併せて使えば、**ライブラリベースのCUDAプログラム**（バッファ操作 +
cuDNN/cuBLAS/Thrust、生の `<<<>>>` カーネルを使わないもの）は、GPU無しのマシンで**そのまま
（無改変で）**ビルド・実行できます。実例が [`test_integration.cpp`](test_integration.cpp) —
検出ヘッダ風の一連の処理（cudaMalloc → conv+bias+sigmoid → gemv → sort_by_key → cudaMemcpy）を
plain C++ のリファレンスと突き合わせて検証しています。cuDNN と cuBLAS は任意でEigenバックエンドを
共有します。以下、まず cuDNN を詳しく、その他は後半にまとめます
（[cuBLAS](#cublas_cpuh) · [Thrust](#thrust_cpuh) · [runtime](#cuda_runtime_cpuh)）。

> **📚 実践サンプル — [`examples/mnist`](examples/mnist/README.md):** MLP(cuBLAS+Thrust) と
> CNN(cuDNN+cuBLAS+Thrust) を MNIST で学習し、PNG画像を読んで数字を推論します。**同じソースが
> CPU(g++) でも GPU(nvcc) でもビルド可能**。4つの Colab ノートブックで `images/sec` を比較して、
> GPUで何が得られるかを体感できます。日本語/英語 両対応。

---

`cudnn_cpu.h` — YOLO系が使う **cuDNN APIのサブセットをCPU実装** したもの。GPUの無いマシンで
cuDNN/GPUコードを開発・検証できます。

狙いは開発ループにあります。ふつう cuDNN を使うコードは、テスト実行のたびにGPUが必要
（例えばColabの無料枠を消費）です。このヘッダを使えば、**同じソース**をCPU開発機でビルドして
即座に正しさを確認し、GPUは最終確認だけに温存できます：

```cpp
#ifdef __CUDACC__
  #include <cudnn.h>          // 本物のGPU: cuDNNをリンク
#else
  #include "cudnn_cpu.h"      // CPU開発機: このヘッダ、GPU不要
#endif
// ... 呼び出しは同一: cudnnConvolutionForward(...), cudnnActivationForward(...) ...
```

型・enum・関数シグネチャは cuDNN に合わせてあるので、呼び出し側は変更不要です。
ビット単位一致のクローンではありません（アルゴリズム選択やworkspaceは無視 —
`GetConvolutionForwardWorkspaceSize` は 0 を返す）が、計算結果は float 精度で cuDNN と一致します。
検出器が実際に必要とする部分だけを実装しています。

## カバー範囲（YOLOサブセット）

| cuDNN 関数 | 備考 |
|---|---|
| `cudnnConvolutionForward` | grouped / **depthwise**(nano)、cross-correlation、pad/stride/dilation |
| `cudnnAddTensor` | bias broadcast `(1,C,1,1)`、residual |
| `cudnnOpTensor` | ADD / MUL / MAX — MUL で SiLU=`x·sigmoid(x)` を構成 |
| `cudnnActivationForward` | SIGMOID, RELU, TANH, CLIPPED_RELU, IDENTITY |
| `cudnnPoolingForward` | MAX(SPPF)、AVG |
| `cudnnSoftmaxForward` | per-channel(DFL) / per-instance |
| `cudnnBatchNormalizationForwardInference` | spatial(チャンネル毎) |
| descriptor / handle | tensor4d, filter4d, conv2d(+group count), activation, pooling, opTensor |

**Backward / 学習**（推論だけでなくCPUで学習もできる）：

| cuDNN 関数 | 備考 |
|---|---|
| `cudnnConvolutionBackwardData` / `BackwardFilter` / `BackwardBias` | grouped/depthwise、Eigen gemm経路 |
| `cudnnActivationBackward` | sigmoid / relu / tanh / clipped-relu |
| `cudnnPoolingBackward` | MAX(argmaxへ配分) / AVG |
| `cudnnSoftmaxBackward` | channel / instance |
| `cudnnBatchNormalizationForwardTraining` + `BatchNormalizationBackward` | バッチ統計・running統計更新・統計を通した勾配 |

各backwardは `test_backward.cpp` で **有限差分VJP**（解析勾配 vs 中心差分、相対誤差 ~1e-4）検証済み。

## 任意のEigenバックエンド（高速）

`CUDNN_CPU_USE_EIGEN` を定義すると、重い計算がヘッダオンリーで同梱の
[`eigen_flat`](third_party/eigen_flat/)（1ディレクトリにフラット化したEigen。yolo*リポジトリ群と共通）
経由になります。畳み込みは **im2col + Eigen gemm** に、要素演算は `Eigen::Array` でベクトル化。
マクロ無しならヘッダ単体（素朴ループ）で動くので、常に依存ゼロで動作します。

```
conv 1x64x128x128 k3 -> 64 ch
  [naive]  860 ms/iter   (1.4 GFLOP/s)
  [EIGEN]   61 ms/iter  (19.9 GFLOP/s)   約14倍、出力は一致
```

## ビルド & テスト

```sh
# 単体（依存なし）
g++ -std=c++17 -O2 -I. test_cudnn_cpu.cpp -o t && ./t

# Eigenバックエンド
g++ -std=c++17 -O3 -I. -DCUDNN_CPU_USE_EIGEN test_cudnn_cpu.cpp -o te && ./te

# backward / 学習op（有限差分での勾配チェック）
g++ -std=c++17 -O2 -I. test_backward.cpp -o tb && ./tb

# conv 1層のベンチ
g++ -std=c++17 -O3 -I. -DCUDNN_CPU_USE_EIGEN bench_conv.cpp -o b && ./b
```

`test_cudnn_cpu.cpp` は各opを独立した素朴リファレンスと突き合わせます。両バックエンドの結果は
一致します（差はfloat丸めのみ）。

## cublas_cpu.h

cuBLAS(単精度)サブセットのCPU実装。cuBLASの **列優先(column-major)** レイアウト、
`op(A)` の転置、`lda/ldb/ldc` を忠実に扱います：

`cublasSgemm`(全N/T組合せ+beta)、`cublasSgemmStridedBatched`、`cublasSgemv`、`cublasSger`、
level-1 の `Saxpy/Sscal/Scopy/Sdot/Sasum/Snrm2/Isamax/Sswap`。`Isamax` は本物同様に1始まりの
インデックスを返します。任意で `CUBLAS_CPU_USE_EIGEN`（Eigenも列優先なので直接マップ）。

```sh
g++ -std=c++17 -O2 -I. test_cublas_cpu.cpp -o tc && ./tc                          # naive
g++ -std=c++17 -O3 -I. -DCUBLAS_CPU_USE_EIGEN test_cublas_cpu.cpp -o tce && ./tce # Eigen
```

`test_cublas_cpu.cpp` が全ルーチン（4通りの転置・batched・gemv N/T・rank-1・level-1）を
リファレンスと照合します。

## thrust_cpu.h

Thrust APIサブセットのCPU実装 — `<thrust/*.h>` の**代わりに**includeすれば、巨大な
CCCL/CUB依存なしでビルド・検証できます（すべてホスト上で逐次実行、`thrust::device`/`host`/`seq`
ポリシーは受理して無視）。

コンテナ: `device_vector` / `host_vector`(ホストメモリ)、`raw_pointer_cast`。
アルゴリズム: `transform`・`reduce`・`transform_reduce`・`inner_product`・`sort`・
`sort_by_key`/`stable_sort_by_key`(NMSパターン)・`inclusive/exclusive_scan`・`copy_if`・
`remove_if`・`unique`・`count_if`・`max/min_element`・`gather`・`scatter`・`reduce_by_key`・
`sequence`・`fill`。functor(`plus`,`multiplies`,`maximum`,`greater`…) と
`counting_iterator`/`constant_iterator`。

```sh
g++ -std=c++17 -O2 -I. test_thrust_cpu.cpp -o tt && ./tt
```

## cuda_runtime_cpu.h

CUDAランタイム・サブセットのCPUシム — `<cuda_runtime.h>` の**代わりに**include。
「デバイス」メモリはホストメモリ、streamはno-op、eventは `std::chrono` で計時、
`__host__`/`__device__` 注釈は除去され、device指定のヘルパもコンパイルできます。

`cudaMalloc` / `cudaFree` / `cudaMallocHost`、`cudaMemcpy` / `cudaMemcpyAsync` /
`cudaMemset`、`cudaStreamCreate/Destroy/Synchronize`、`cudaEventCreate/Record/ElapsedTime`、
`cudaDeviceSynchronize`、`cudaGetDeviceCount/Properties`、エラー処理ヘルパ。

**制約:** 生の `__global__` カーネル（`<<<>>>`）は起動できません（nvccが必要）。GPUを
cuDNN/cuBLAS/Thrust + 手動バッファ経由で駆動するコードはカーネル不要なので、CPUで完全に動きます
——それが狙いです。

```sh
g++ -std=c++17 -O2 -I. test_integration.cpp -o ti && ./ti   # 全シムを使う通しパイプライン
```

## ファイル
- `cudnn_cpu.h` — cuDNNサブセット（forward + backward）、単一ヘッダ。
- `cublas_cpu.h` — cuBLASサブセット、単一ヘッダ。
- `thrust_cpu.h` — Thrustサブセット、単一ヘッダ。
- `cuda_runtime_cpu.h` — CUDAランタイム・サブセット、単一ヘッダ。
- `test_cudnn_cpu.cpp` / `test_backward.cpp` — cuDNN forward / backward テスト。
- `test_cublas_cpu.cpp` / `test_thrust_cpu.cpp` — cuBLAS / Thrust テスト。
- `test_integration.cpp` — 全4シムを使うGPU風の通しパイプライン。
- `bench_conv.cpp` — naive vs Eigen のconv計時。
- `third_party/eigen_flat/` — 同梱のフラットEigen（`*_USE_EIGEN` 時に使用）。
- `examples/mnist/` — MNIST教材（GPU無しでGPUを学ぶ）。
