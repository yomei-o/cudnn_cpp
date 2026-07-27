# MNIST samples — learn how to use a GPU *without* a GPU

*(日本語版: [README.md](README.md))*

This directory is a teaching resource for **learning GPU programming even if you don't own
a GPU**. One idea is at its heart:

> **The same C++ source runs on CPU and GPU — only the compile command changes.**

On CPU it uses the header-only shims from [cudnn_cpp](../../README.md) (CPU implementations
of cuDNN / cuBLAS / Thrust / the CUDA runtime); on GPU it links the real CUDA libraries.
**Not one character of the source changes** — only how you compile it.

```sh
# CPU (no GPU needed, just g++)
g++  -std=c++17 -O3 -I. examples/mnist/mnist_mlp.cpp -o mnist_mlp

# GPU (same source, built with nvcc)
nvcc -std=c++17 -O3 -x cu -DMNIST_GPU -I. examples/mnist/mnist_mlp.cpp -o mnist_mlp -lcublas
```

## Two models

| source | what | libraries |
|---|---|---|
| [`mnist_mlp.cpp`](mnist_mlp.cpp) | 2-layer MLP (784→128→10) | **cuBLAS** (matmul) + **Thrust** (elementwise) + CUDA runtime |
| [`mnist_cnn.cpp`](mnist_cnn.cpp) | CNN (conv8→conv16→FC) | **cuDNN** (conv/pool/act/softmax) + cuBLAS + Thrust |

Both do full **training (forward + backward + SGD)** with only library calls and buffer
management — no custom kernels (`<<<>>>`) — which is exactly why the CPU shims can run them.

## Four Colab notebooks

Each runs the whole thing: train → save the model → load a PNG → predict the digit.
The point is to run **both the CPU and GPU version and compare `images/sec`**.

| notebook | what | Colab |
|---|---|---|
| [`mnist_thrust_cublas_cpu.ipynb`](colab/mnist_thrust_cublas_cpu.ipynb) | MLP / CPU | [open](https://colab.research.google.com/github/yomei-o/cudnn_cpp/blob/main/examples/mnist/colab/mnist_thrust_cublas_cpu.ipynb) |
| [`mnist_thrust_cublas_gpu.ipynb`](colab/mnist_thrust_cublas_gpu.ipynb) | MLP / GPU | [open](https://colab.research.google.com/github/yomei-o/cudnn_cpp/blob/main/examples/mnist/colab/mnist_thrust_cublas_gpu.ipynb) |
| [`mnist_cudnn_thrust_cublas_cpu.ipynb`](colab/mnist_cudnn_thrust_cublas_cpu.ipynb) | CNN / CPU | [open](https://colab.research.google.com/github/yomei-o/cudnn_cpp/blob/main/examples/mnist/colab/mnist_cudnn_thrust_cublas_cpu.ipynb) |
| [`mnist_cudnn_thrust_cublas_gpu.ipynb`](colab/mnist_cudnn_thrust_cublas_gpu.ipynb) | CNN / GPU | [open](https://colab.research.google.com/github/yomei-o/cudnn_cpp/blob/main/examples/mnist/colab/mnist_cudnn_thrust_cublas_gpu.ipynb) |

## Try it locally (CPU)

```sh
# from the repo root; put MNIST in examples/mnist/data/ (see any notebook)
g++ -std=c++17 -O3 -I. examples/mnist/mnist_mlp.cpp -o mnist_mlp
./mnist_mlp --epochs 5 --save mnist_mlp.bin
g++ -std=c++17 -O2 -I. examples/mnist/make_samples.cpp -o mk && ./mk   # make 0-9 PNGs
./mnist_mlp --load mnist_mlp.bin --infer examples/mnist/samples/digit7.png
```

## Why use a GPU (what this teaches)

Training time is measured with `cudaEvent` and printed as `>> trained ... images/sec`.
Accuracy is ~identical on CPU and GPU (same math); **only the speed differs** — that's the
reason to use a GPU.

- The **MLP is small**, so even the CPU is fairly fast (a GPU isn't essential for tiny models).
- The **CNN's convolutions are heavy**, so the CPU slows down sharply — this is where the GPU
  (cuDNN) wins big.
- The gap widens as the model / batch grows. See "Try it" at the end of each notebook.

Reference (author's CPU, plain shims): MLP ≈ 3,400 img/s, CNN ≈ 660 img/s. Run the GPU
notebooks to see how far these numbers jump.

### Making the CPU faster (optional)

Add `-DCUDNN_CPU_USE_EIGEN -DCUBLAS_CPU_USE_EIGEN` to route matmuls through
[Eigen](../../third_party/eigen_flat): **several× faster on large matrices** (e.g. the MLP's
FC layer). For MNIST's tiny CNN convolutions the effect is small (sometimes slightly slower).
A useful lesson in itself: the CPU can be optimized too — a GPU isn't the only way to go fast.

## Files

- `mnist_mlp.cpp` / `mnist_cnn.cpp` — train + infer (one source for CPU and GPU)
- `gpu_backend.h` — the single place that switches CPU ⇄ GPU
- `mnist_data.h` — MNIST (IDX) and PNG loaders
- `make_samples.cpp` — write 0-9 PNGs from the test set
- `colab/*.ipynb` — the four Colab notebooks
