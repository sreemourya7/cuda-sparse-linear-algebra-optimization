# Results — CUDA CSR SpMM Baseline

## Test Configuration

- Platform: Windows
- GPU: NVIDIA GeForce GTX 1650
- Language standard: C++17
- Compiler: NVIDIA CUDA Compiler (`nvcc`)
- Dense matrix layout: Row-major
- Thread mapping: One CUDA thread per output element
- Thread-block dimensions: \(16 \times 16\)

## Matrix Dimensions

```text
Sparse matrix A: 4 x 5
Dense matrix B:  5 x 3
Output matrix C: 4 x 3