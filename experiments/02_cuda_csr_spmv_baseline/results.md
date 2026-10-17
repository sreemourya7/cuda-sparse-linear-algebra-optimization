# Results — CUDA CSR SpMV Baseline

## Test Configuration

- Platform: Windows
- Language standard: C++17
- Compiler: NVIDIA CUDA Compiler (`nvcc`)
- Kernel mapping: One CUDA thread per CSR row
- Threads per block: 256

## Test Matrix

- Dimensions: \(4 \times 5\)
- Total elements: 20
- Nonzero elements: 7
- Density: 35%
- Sparsity: 65%

Input vector:

```text
x = [1, 2, 3, 4, 5]