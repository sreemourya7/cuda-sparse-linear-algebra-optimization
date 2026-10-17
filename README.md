# CUDA Sparse Linear Algebra Optimization

A correctness-first study of CSR sparse matrix operations, CUDA kernel mapping, workload-aware SpMM optimization, cuSPARSE comparison, and Nsight Compute profiling.

This repository develops sparse GPU kernels incrementally—from CSR representation and CPU references to optimized CUDA execution and profiler-backed performance analysis.

## Project Highlights

- Implemented CSR storage, validation, and dense-to-CSR conversion
- Built independent CPU references for SpMV and SpMM
- Implemented CUDA CSR SpMV and SpMM baselines
- Benchmarked multiple sparse-row lengths and dense output widths
- Designed width-aware CUDA SpMM dispatch
- Used warp broadcasts and register reuse for wide outputs
- Compared custom kernels against NVIDIA cuSPARSE
- Captured and analyzed full Nsight Compute profiles
- Verified every GPU result against a CPU reference

## Best Measured Results

Test system:

- GPU: NVIDIA GeForce GTX 1650
- Compute capability: 7.5
- Sparse matrix dimensions: 8192 × 8192
- Data type: FP32
- Sparse format: CSR
- Dense matrices: Row-major

Key results:

| Result | Measurement |
|---|---:|
| Maximum speedup over fixed-block SpMM | **2.063×** |
| Peak optimized custom throughput | **169.932 GFLOP/s** |
| Peak cuSPARSE throughput | **102.247 GFLOP/s** |
| Profiled custom memory throughput | **183.48 GB/s** |
| Profiled custom DRAM utilization | **94.15%** |
| Optimized kernel achieved occupancy | **98.92%** |
| Local-memory spilling | **0** |
| Correct benchmark configurations | **12 of 12** |

For the representative workload with 128 nonzeros per row and `N = 128`:

| Implementation | Time | Throughput |
|---|---:|---:|
| Fixed-block baseline | 6.200369 ms | 43.293 GFLOP/s |
| Width-aware custom | 3.004826 ms | 89.335 GFLOP/s |
| cuSPARSE | 3.007715 ms | 89.249 GFLOP/s |

The custom implementation reached essentially the same performance and memory-bandwidth limit as cuSPARSE for this workload.

## Operations

### Sparse Matrix–Vector Multiplication

SpMV computes:

`y = A_sparse × x`

where:

- `A` has dimensions `M × K`
- `x` has length `K`
- `y` has length `M`

### Sparse Matrix–Dense Matrix Multiplication

SpMM computes:

`C = A_sparse × B_dense`

where:

- `A` has dimensions `M × K`
- `B` has dimensions `K × N`
- `C` has dimensions `M × N`

SpMV is the special case of SpMM where `N = 1`.

## CSR Representation

A sparse matrix is stored using:

- `values`: Nonzero values
- `columnIndices`: Column index for each nonzero
- `rowOffsets`: Start and end offsets for each row

CSR reduces storage and avoids arithmetic on explicit zero values. It also introduces irregular row lengths, indirect addressing, and workload-dependent parallelism.

## Experiment Progression

| Experiment | Topic | Status |
|---:|---|:---:|
| 00 | CSR representation and validation | Complete |
| 01 | CPU CSR SpMV reference | Complete |
| 02 | CUDA CSR SpMV baseline | Complete |
| 03 | CUDA CSR SpMV benchmark | Complete |
| 04 | CPU CSR SpMM reference | Complete |
| 05 | CUDA CSR SpMM baseline | Complete |
| 06 | CUDA CSR SpMM benchmark | Complete |
| 07 | Width-aware CSR SpMM optimization | Complete |
| 08 | cuSPARSE comparison and Nsight profiling | Complete |

Each experiment contains its own explanation, implementation, correctness checks, and results.

## Optimization Evolution

### Fixed-Block Baseline

The original SpMM kernel uses:

- Block dimensions: `16 × 16`
- One thread per output element
- One independent CSR-row traversal per output column

This mapping is simple and correct but repeats CSR metadata loads and provides poor utilization for narrow output widths.

### Narrow-Width Mapping

For `N < 32`, the block shape is adapted to the number of output columns.

This avoids inactive threads when the dense output contains only one or a few columns.

### Warp-Aligned N = 32 Mapping

For `N = 32`, each warp processes one sparse row and each lane computes one output column.

An initial shuffle-based version regressed performance. Measurements showed that broadcast overhead was not worthwhile when every lane computed only one result.

The final `N = 32` path therefore uses warp-aligned scalar execution without explicit shuffles.

### Warp Broadcast and Register Reuse

For `N > 32`, one warp processes one sparse row.

For `N = 128`, each lane computes four columns:

- `lane`
- `lane + 32`
- `lane + 64`
- `lane + 96`

Lane 0 loads each CSR value and column index. Warp shuffle instructions broadcast the metadata to the remaining lanes.

Each lane maintains four register accumulators, allowing one CSR traversal to update four output values.

## Width-Aware Dispatch

| Dense width | Selected strategy |
|---:|---|
| `N < 32` | Adaptive narrow scalar block |
| `N = 32` | Warp-aligned scalar mapping |
| `N > 32` | Warp broadcast and register reuse |

The benchmark results demonstrate that sparse performance depends on both sparse-row length and dense output width. No single mapping was optimal for every tested workload.

## Benchmark Methodology

The main benchmark matrix uses:

- Dimensions: 8192 × 8192
- Nonzeros per row: 8, 32, and 128
- Dense widths: 1, 8, 32, and 128
- Warm-up iterations: 10
- Timed iterations: 100

Timing uses CUDA events and excludes:

- Memory allocation
- Host-to-device transfers
- Descriptor creation
- cuSPARSE workspace setup

Throughput is calculated as:

`GFLOP/s = 2 × NNZ × N / execution time`

Every implementation is validated against the CPU CSR SpMM reference before its performance is reported.

## cuSPARSE Comparison

The library reference uses:

- Generic cuSPARSE descriptors
- FP32 CSR input
- Row-major dense matrices
- `CUSPARSE_SPMM_CSR_ALG2`

The custom implementation exceeded cuSPARSE throughput for every tested `N = 8` and `N = 32` workload.

cuSPARSE was stronger for longer-row SpMV cases and some wide-output configurations.

At 128 nonzeros per row and `N = 128`, the custom and cuSPARSE implementations were effectively tied.

These results show why library comparisons must cover multiple shapes instead of relying on one favorable workload.

## Nsight Compute Findings

The representative profiling workload uses:

- Matrix dimensions: 8192 × 8192
- Nonzeros per row: 128
- Dense width: 128

Primary profiler measurements:

| Metric | Fixed Baseline | Width-Aware Custom | cuSPARSE Main Kernel |
|---|---:|---:|---:|
| Profiled duration | 6.22 ms | 2.94 ms | 2.99 ms |
| DRAM utilization | 46.20% | 94.15% | 94.79% |
| Memory throughput | 88.63 GB/s | 183.48 GB/s | 181.72 GB/s |
| Compute utilization | 21.32% | 29.26% | 23.87% |
| Registers per thread | 54 | 36 | 48 |
| Achieved occupancy | 94.81% | 98.92% | 98.40% |
| Warp cycles per issued instruction | 125.87 | 72.33 | 44.04 |

The profiler confirms that the optimization:

- Reduces launched threads by four
- Reduces repeated CSR-row traversal
- Removes the baseline LG-throttle bottleneck
- More than doubles measured memory throughput
- Introduces no register spilling
- Converts the workload into a DRAM-bandwidth-bound kernel

The optimized custom kernel and cuSPARSE both reach approximately 94–95% DRAM utilization, explaining their similar final performance.

## Repository Structure

- `experiments/00_csr_representation`: CSR data structure, conversion, and validation
- `experiments/01_cpu_csr_spmv`: CPU SpMV correctness reference
- `experiments/02_cuda_csr_spmv_baseline`: Initial CUDA SpMV kernel
- `experiments/03_cuda_csr_spmv_benchmark`: SpMV scaling measurements
- `experiments/04_cpu_csr_spmm`: CPU SpMM correctness reference
- `experiments/05_cuda_csr_spmm_baseline`: Fixed-block CUDA SpMM
- `experiments/06_cuda_csr_spmm_benchmark`: Width and sparsity sweep
- `experiments/07_width_aware_csr_spmm_optimization`: Adaptive custom kernel
- `experiments/08_cusparse_nsight_comparison`: Library comparison and profiler analysis

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- NVCC
- cuSPARSE
- C++17 compiler
- Nsight Compute for profiler reproduction

The experiments were developed and tested on Windows using the Visual Studio x64 Native Tools Command Prompt.

## Build the Final Comparison

From `experiments\08_cusparse_nsight_comparison`:

    nvcc -O3 -lineinfo -std=c++17 benchmark.cu ..\00_csr_representation\csr_matrix.cpp ..\04_cpu_csr_spmm\csr_spmm.cpp -lcusparse -o benchmark.exe

Run:

    benchmark.exe

Profile:

    ncu --set full --export profiles\baseline benchmark.exe --profile baseline

    ncu --set full --export profiles\optimized benchmark.exe --profile optimized

    ncu --set full --export profiles\cusparse benchmark.exe --profile cusparse

On Windows, profiling hardware counters may require an administrator command prompt.

<!-- ## Limitations

The benchmark matrices use a controlled uniform number of nonzeros per row.

Real sparse matrices may contain:

- Highly irregular row lengths
- Empty rows
- Long-tail row distributions
- Structured sparsity
- Repeated or clustered column indices
- Different cache and load-balance behavior

The reported results are specific to the tested GPU, CUDA environment, matrix dimensions, data layout, and cuSPARSE algorithm.

This repository is an optimization study and learning project, not a replacement for a production sparse linear algebra library.

## Possible Future Work

- Irregular and real-world sparse matrices
- Row-length bucketing
- Vectorized memory access
- Adaptive kernel selection using matrix statistics
- Block Sparse Row storage
- Sparse matrix–sparse matrix multiplication
- Mixed-precision sparse kernels
- Multi-GPU sparse computation

## Project Status

The core CSR SpMV and SpMM learning path is complete:

- [x] Sparse representation
- [x] CPU correctness references
- [x] CUDA baselines
- [x] Controlled benchmarking
- [x] Workload-aware optimization
- [x] cuSPARSE comparison
- [x] Nsight Compute profiling
- [x] Profiler-backed performance explanation -->

