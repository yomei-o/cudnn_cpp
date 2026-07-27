# cudnn_cpp

Tiny, **header-only, source-available CPU implementations of the CUDA GPU-library APIs**
that YOLO-style detectors use — **cuDNN**, **cuBLAS**, (Thrust next) — so you can develop
and correctness-test CUDA/GPU code on a **machine with no GPU** and no CUDA install.

| header | replaces | for |
|---|---|---|
| [`cudnn_cpu.h`](cudnn_cpu.h) | `<cudnn.h>` | conv/act/pool/softmax/bn — forward **and backward** |
| [`cublas_cpu.h`](cublas_cpu.h) | `<cublas_v2.h>` | sgemm (+strided-batched), gemv, ger, level-1 |

Both share the optional Eigen backend and the same drop-in idea. The rest of this README
focuses on cuDNN; cuBLAS is summarized [below](#cublas_cpuh).

---

`cudnn_cpu.h` — a **CPU implementation of the cuDNN API subset**
that YOLO-style detectors use — so you can develop and correctness-test cuDNN/GPU code
on a **machine with no GPU** and no cuDNN install.

The point is the development loop: writing against cuDNN normally means every test run
needs a GPU (e.g. burning Colab free-tier quota). With this header you build the *same
source* on your CPU dev box, verify correctness instantly, and reserve the GPU only for
the final confirmation run:

```cpp
#ifdef __CUDACC__
  #include <cudnn.h>          // real GPU: link cuDNN
#else
  #include "cudnn_cpu.h"      // CPU dev box: this header, no GPU needed
#endif
// ... identical call sites: cudnnConvolutionForward(...), cudnnActivationForward(...) ...
```

The types, enums and function signatures mirror cuDNN's, so call sites don't change.
It is **not** a bit-exact cuDNN clone (algorithm selection & workspace are ignored —
`GetConvolutionForwardWorkspaceSize` returns 0); results match cuDNN's math to float
precision. Only the pieces detectors actually need are implemented.

## What's covered (the YOLO subset)

| cuDNN function | notes |
|---|---|
| `cudnnConvolutionForward` | grouped / **depthwise** (nano), cross-correlation, pads/strides/dilation |
| `cudnnAddTensor` | bias broadcast `(1,C,1,1)`, residual |
| `cudnnOpTensor` | ADD / MUL / MAX — MUL builds SiLU as `x·sigmoid(x)` |
| `cudnnActivationForward` | SIGMOID, RELU, TANH, CLIPPED_RELU, IDENTITY |
| `cudnnPoolingForward` | MAX (SPPF), AVG |
| `cudnnSoftmaxForward` | per-channel (DFL) / per-instance |
| `cudnnBatchNormalizationForwardInference` | spatial (per-channel) |
| descriptors / handle | tensor4d, filter4d, conv2d (+group count), activation, pooling, opTensor |

**Backward / training** (so you can train, not just infer, on CPU):

| cuDNN function | notes |
|---|---|
| `cudnnConvolutionBackwardData` / `BackwardFilter` / `BackwardBias` | grouped/depthwise; Eigen gemm path |
| `cudnnActivationBackward` | sigmoid / relu / tanh / clipped-relu |
| `cudnnPoolingBackward` | MAX (argmax routing) / AVG |
| `cudnnSoftmaxBackward` | channel / instance |
| `cudnnBatchNormalizationForwardTraining` + `BatchNormalizationBackward` | batch stats, running-stat update, grad through the statistics |

Every backward op is checked by finite-difference VJP in `test_backward.cpp`
(analytic grad vs central difference, rel diff ~1e-4).

## Optional Eigen backend (fast)

Define `CUDNN_CPU_USE_EIGEN` to route the heavy math through the header-only, vendored
[`eigen_flat`](third_party/eigen_flat/) (flattened single-directory Eigen — the same one
used across the yolo* repos). Convolution becomes **im2col + Eigen gemm**; elementwise
ops vectorize via `Eigen::Array`. Without the macro the header is fully standalone
(naive loops), so it always works with zero dependencies.

```
conv 1x64x128x128 k3 -> 64 ch
  [naive]  860 ms/iter   (1.4 GFLOP/s)
  [EIGEN]   61 ms/iter  (19.9 GFLOP/s)   ~14x, identical output
```

## Build & test

```sh
# standalone (no deps)
g++ -std=c++17 -O2 -I. test_cudnn_cpu.cpp -o t && ./t

# with the Eigen backend
g++ -std=c++17 -O3 -I. -DCUDNN_CPU_USE_EIGEN test_cudnn_cpu.cpp -o te && ./te

# backward / training ops (finite-difference gradient check)
g++ -std=c++17 -O2 -I. test_backward.cpp -o tb && ./tb

# benchmark one conv layer
g++ -std=c++17 -O3 -I. -DCUDNN_CPU_USE_EIGEN bench_conv.cpp -o b && ./b
```

`test_cudnn_cpu.cpp` checks every op against an independent naive reference; both
backends produce identical results (diffs are float rounding only).

## cublas_cpu.h

CPU implementation of the cuBLAS (single-precision) subset, honoring cuBLAS's
**column-major** layout, `op(A)` transposes and `lda/ldb/ldc` exactly:

`cublasSgemm` (all N/T combos, with beta), `cublasSgemmStridedBatched`,
`cublasSgemv`, `cublasSger`, and level-1 `Saxpy/Sscal/Scopy/Sdot/Sasum/Snrm2/Isamax/Sswap`.
`Isamax` returns a 1-based index like the real one. Optional Eigen backend via
`CUBLAS_CPU_USE_EIGEN` (Eigen is column-major too, so the map is direct).

```sh
g++ -std=c++17 -O2 -I. test_cublas_cpu.cpp -o tc && ./tc                       # naive
g++ -std=c++17 -O3 -I. -DCUBLAS_CPU_USE_EIGEN test_cublas_cpu.cpp -o tce && ./tce  # Eigen
```

`test_cublas_cpu.cpp` checks every routine (all four transpose combos, batched,
gemv N/T, rank-1, level-1) against hand references.

## Files
- `cudnn_cpu.h` — cuDNN subset (forward + backward), single header.
- `cublas_cpu.h` — cuBLAS subset, single header.
- `test_cudnn_cpu.cpp` / `test_backward.cpp` — cuDNN forward / backward tests.
- `test_cublas_cpu.cpp` — cuBLAS tests.
- `bench_conv.cpp` — naive-vs-Eigen conv timing.
- `third_party/eigen_flat/` — vendored flat Eigen (used when `*_USE_EIGEN`).
