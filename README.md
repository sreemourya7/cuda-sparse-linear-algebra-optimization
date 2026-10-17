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
- [x] Selected and analyzed the CSR storage format
- [x] Implemented dense-to-CSR and CSR-to-dense conversion
- [x] Implemented CSR structural validation
- [x] Implemented and validated a CPU CSR SpMV reference
- [x] Implemented a one-thread-per-row CUDA CSR SpMV kernel
- [x] Added CUDA memory management and runtime error checking
- [x] Verified that GPU, CPU, and manually calculated results match
- [x] Documented reproducible build procedures and results

### Next — CUDA CSR SpMV Benchmarking

The next stage will scale the baseline to larger matrices and measure performance across different dimensions, sparsity levels, and row-length distributions. These measurements will establish the bottlenecks that motivate the first GPU optimization.

After the SpMV baseline is characterized, the project will progress to CSR SpMM.