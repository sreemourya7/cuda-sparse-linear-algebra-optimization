# Results — CSR Matrix Representation

## Test Configuration

- Platform: Windows
- Language standard: C++17
- Build tool: NVIDIA CUDA Compiler (`nvcc`)
- Host compiler: Microsoft Visual C++ x64

## Test Matrix

- Dimensions: \(4 \times 5\)
- Total elements: 20
- Nonzero elements: 7
- Density: 35%
- Sparsity: 65%

## Build and Run

```cmd
nvcc -std=c++17 main.cpp csr_matrix.cpp -o csr_test.exe
csr_test.exe