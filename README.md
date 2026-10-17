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


### Phase 1 — Sparse Computation Foundations

- [x] Analyzed the execution differences between dense and sparse GEMM
- [x] Selected CSR as the initial storage format
- [x] Defined the progression from SpMV to SpMM
- [x] Identified key optimization challenges: irregular memory access, indexing overhead, and workload imbalance
- [x] Established the correctness, benchmarking, profiling, and cuSPARSE comparison methodology
- [ ] Implement CSR matrix construction and validation
- [ ] Build the CPU reference implementation
- [ ] Implement the first CUDA CSR kernel

The project is currently transitioning from architectural analysis to baseline implementation.