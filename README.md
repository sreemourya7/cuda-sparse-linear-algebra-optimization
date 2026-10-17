# Profiling-Driven Sparse Linear Algebra on CUDA

A from-scratch, measurement-driven study of sparse matrix computation on GPUs.

This project explores how sparse storage formats, irregular memory access, load imbalance, and indexing overhead affect CUDA kernel performance. The main focus is progressing from simple sparse operations to optimized sparse matrix–dense matrix multiplication (SpMM), with profiling and comparison against NVIDIA cuSPARSE.

## Project Goals

- Understand CSR, COO, and BSR sparse matrix formats
- Build correct CPU reference implementations
- Implement sparse CUDA kernels from scratch
- Develop and optimize CSR SpMV and SpMM
- Study irregular memory access and warp-level load imbalance
- Profile kernels using NVIDIA Nsight Compute
- Compare custom kernels against cuSPARSE
- Document every optimization with measurements and explanations

## Planned Progression

1. Sparse matrix fundamentals and storage formats
2. CPU reference implementations
3. CSR sparse matrix–vector multiplication (SpMV)
4. Naive CSR sparse matrix–dense matrix multiplication (SpMM)
5. Memory-access and load-balancing optimizations
6. Block-sparse matrix multiplication using BSR
7. Nsight Compute bottleneck analysis
8. cuSPARSE performance comparison

## Methodology

Each stage will follow the same performance-engineering cycle:

1. Implement a correct baseline
2. Validate numerical correctness
3. Benchmark representative matrix shapes and sparsity patterns
4. Profile the kernel
5. Identify the primary bottleneck
6. Apply one optimization
7. Measure and explain the result

## Current Status

### Completed

- [x] Documented sparse-computation foundations
- [x] Implemented and validated CSR storage
- [x] Implemented CPU and CUDA CSR SpMV
- [x] Benchmarked CSR SpMV across multiple row lengths
- [x] Implemented CPU and CUDA CSR SpMM baselines
- [x] Benchmarked 12 CSR SpMM workloads
- [x] Implemented width-aware CSR SpMM dispatch
- [x] Eliminated narrow-width thread underutilization
- [x] Implemented a scalar warp-aligned mapping
- [x] Implemented warp broadcast and register reuse
- [x] Improved all 12 tested SpMM workloads
- [x] Verified every result against CPU references

### Best Measured Results

- CSR SpMV: 6.541 GNNZ/s and 13.082 GFLOP/s
- Baseline CSR SpMM: 123.007 GFLOP/s
- Optimized CSR SpMM: 163.301 GFLOP/s
- Maximum CSR SpMM speedup: 1.897×

The largest speedup reduced kernel time from 5.303499 ms to 2.795092 ms for 128 nonzeros per row and \(N=128\).

### Next — cuSPARSE and Nsight Compute Comparison

The final core experiment will compare the custom baseline and optimized kernels against NVIDIA cuSPARSE and use Nsight Compute metrics to explain the observed performance behavior.