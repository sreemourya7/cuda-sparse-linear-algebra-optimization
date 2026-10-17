# Results — CUDA CSR SpMV Baseline Benchmark

## Hardware and Configuration

- GPU: NVIDIA GeForce GTX 1650
- Matrix dimensions: \(16{,}384 \times 16{,}384\)
- Kernel mapping: One CUDA thread per CSR row
- Threads per block: 256
- Warm-up iterations: 10
- Timed iterations: 100
- Timing method: CUDA events
- Timed region: Kernel execution only

## Build and Run

```cmd
nvcc -O3 -std=c++17 benchmark.cu ..\00_csr_representation\csr_matrix.cpp ..\01_cpu_csr_spmv\csr_spmv.cpp -o benchmark.exe
benchmark.exe