# Experiment 08 Results — cuSPARSE Comparison

## Test Configuration

- GPU: NVIDIA GeForce GTX 1650
- Sparse matrix dimensions: 8192 × 8192
- Sparse distribution: Uniform nonzeros per row
- Nonzeros per row: 8, 32, and 128
- Dense output widths: 1, 8, 32, and 128
- Warm-up iterations: 10
- Timed iterations: 100
- cuSPARSE algorithm: `CUSPARSE_SPMM_CSR_ALG2`
- Dense matrix layout: Row-major

Memory allocation, host-to-device transfers, descriptor creation, and workspace setup were excluded from kernel timing.

## Correctness

The baseline CUDA kernel, width-aware CUDA kernel, and cuSPARSE implementation were compared against the CPU CSR SpMM reference.

All 12 workload configurations passed correctness verification for all three GPU implementations.

## Benchmark Results

| NNZ/row | N | Baseline ms | Optimized ms | cuSPARSE ms | Baseline GFLOP/s | Optimized GFLOP/s | cuSPARSE GFLOP/s | Custom Speedup | Custom as % of cuSPARSE | Correct |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 8 | 1 | 0.018944 | 0.014843 | 0.039793 | 6.919 | 8.831 | 3.294 | 1.276× | 268.092% | PASS |
| 8 | 8 | 0.021430 | 0.016638 | 0.084661 | 48.930 | 63.021 | 12.386 | 1.288× | 508.830% | PASS |
| 8 | 32 | 0.050422 | 0.046162 | 0.111686 | 83.184 | 90.861 | 37.555 | 1.092× | 241.943% | PASS |
| 8 | 128 | 0.273357 | 0.223966 | 0.191361 | 61.375 | 74.910 | 87.673 | 1.221× | 85.442% | PASS |
| 32 | 1 | 0.063964 | 0.063428 | 0.026869 | 8.197 | 8.266 | 19.512 | 1.008× | 42.362% | PASS |
| 32 | 8 | 0.064041 | 0.060191 | 0.202835 | 65.494 | 69.684 | 20.678 | 1.064× | 336.987% | PASS |
| 32 | 32 | 0.146816 | 0.120218 | 0.302123 | 114.274 | 139.557 | 55.531 | 1.221× | 251.313% | PASS |
| 32 | 128 | 0.748660 | 0.765908 | 0.656338 | 89.639 | 87.620 | 102.247 | 0.977× | 85.694% | PASS |
| 128 | 1 | 0.237202 | 0.240783 | 0.060600 | 8.841 | 8.710 | 34.606 | 0.985× | 25.168% | PASS |
| 128 | 8 | 0.235404 | 0.222925 | 0.742307 | 71.270 | 75.259 | 22.601 | 1.056× | 332.985% | PASS |
| 128 | 32 | 0.541144 | 0.394916 | 1.014984 | 124.013 | 169.932 | 66.118 | 1.370× | 257.013% | PASS |
| 128 | 128 | 6.200369 | 3.004826 | 3.007715 | 43.293 | 89.335 | 89.249 | 2.063× | 100.096% | PASS |

The `% of cuSPARSE` column represents:

`optimized custom throughput / cuSPARSE throughput × 100`

A value above 100% means that the custom kernel achieved higher throughput.

## Best Results

- Maximum custom speedup over the fixed-block baseline: **2.063×**
- Workload producing the maximum speedup: **128 nonzeros per row, N = 128**
- Peak optimized custom throughput: **169.932 GFLOP/s**
- Workload producing peak custom throughput: **128 nonzeros per row, N = 32**
- Peak cuSPARSE throughput: **102.247 GFLOP/s**
- Workload producing peak cuSPARSE throughput: **32 nonzeros per row, N = 128**

## Key Observations

### 1. No single implementation wins every workload

The best implementation changes with sparse-row length and dense output width. This demonstrates why sparse kernel performance must be evaluated across multiple matrix shapes rather than using only one benchmark.

### 2. Custom kernels perform strongly at N = 8 and N = 32

The optimized custom kernel exceeded cuSPARSE throughput for every tested `N = 8` and `N = 32` workload.

The largest measured advantage was at 8 nonzeros per row and `N = 8`, where the custom kernel reached approximately 5.09 times the cuSPARSE throughput.

### 3. cuSPARSE is strong for longer SpMV rows

For `N = 1`, cuSPARSE substantially outperformed the custom kernel at 32 and 128 nonzeros per row.

At 128 nonzeros per row, cuSPARSE achieved 34.606 GFLOP/s compared with 8.710 GFLOP/s for the optimized custom kernel.

This indicates that the current custom SpMV mapping does not distribute work within a long sparse row as effectively as the library implementation.

### 4. Wide-output register reuse is effective

At 128 nonzeros per row and `N = 128`, the width-aware kernel improved performance from 43.293 GFLOP/s to 89.335 GFLOP/s.

This corresponds to a 2.063× speedup over the fixed-block baseline and essentially matches cuSPARSE throughput for this workload.

### 5. Width awareness is beneficial but not universally faster

The optimized kernel outperformed the fixed-block baseline in 10 of the 12 configurations.

Small regressions occurred for:

- 32 nonzeros per row with `N = 128`
- 128 nonzeros per row with `N = 1`

This reinforces the need for benchmark-driven dispatch decisions instead of assuming that one mapping is optimal everywhere.

### 6. Small workloads are sensitive to overhead

For short sparse rows and narrow output widths, the custom kernels can outperform a general-purpose library implementation because they use simpler workload-specific execution paths.

These results are specific to the tested GPU, matrix dimensions, sparsity pattern, data layout, and cuSPARSE algorithm. They should not be interpreted as a universal comparison against cuSPARSE.

## Nsight Compute Profiling Target

The representative profiling workload is:

- Matrix dimensions: 8192 × 8192
- Nonzeros per row: 128
- Dense output width: 128

This workload was selected because it exercises warp broadcasts, four register accumulators per lane, CSR metadata reuse, and wide-output memory access.

The following implementations will be profiled:

- Fixed 16 × 16 baseline
- Width-aware custom kernel
- cuSPARSE CSR SpMM

Profiling will examine:

- Memory throughput
- Compute throughput
- Warp-stall reasons
- Achieved occupancy
- Register usage
- Kernel launch behavior

## Current Status

- [x] Compared the fixed-block and width-aware custom kernels
- [x] Added a cuSPARSE CSR SpMM reference
- [x] Excluded setup and transfer costs from kernel timing
- [x] Tested all 12 workload configurations
- [x] Verified every result against the CPU reference
- [x] Recorded benchmark results
- [x] Analyzed workload-dependent performance
- [ ] Capture Nsight Compute profiles
- [ ] Record profiling observations

## Update

## Nsight Compute Profiling Results

### Profiled Workload

- Matrix dimensions: 8192 × 8192
- Nonzeros per row: 128
- Dense output width: 128
- GPU: NVIDIA GeForce GTX 1650
- Compute capability: 7.5
- Nsight Compute version: 2025.4.1
- Collection set: Full

This workload exercises warp broadcasts, four output accumulators per lane, CSR metadata reuse, and wide-output memory access.

### Primary Kernel Comparison

| Metric | Fixed Baseline | Width-Aware Custom | cuSPARSE Main Kernel |
|---|---:|---:|---:|
| Kernel duration | 6.22 ms | 2.94 ms | 2.99 ms |
| DRAM throughput utilization | 46.20% | 94.15% | 94.79% |
| Measured memory throughput | 88.63 GB/s | 183.48 GB/s | 181.72 GB/s |
| Compute throughput utilization | 21.32% | 29.26% | 23.87% |
| Registers per thread | 54 | 36 | 48 |
| Achieved occupancy | 94.81% | 98.92% | 98.40% |
| Warp cycles per issued instruction | 125.87 | 72.33 | 44.04 |
| Local-memory spills | 0 | 0 | 0 |
| Primary stall | Long scoreboard and LG throttle | Long scoreboard | Long scoreboard |

The profiler durations are collected under profiling and therefore differ slightly from the CUDA-event benchmark timings.

### Launch Configuration

| Implementation | Grid | Block | Threads Launched | Waves per SM |
|---|---:|---:|---:|---:|
| Fixed baseline | 4096 blocks | 256 threads | 1,048,576 | 73.14 |
| Width-aware custom | 1024 blocks | 256 threads | 262,144 | 18.29 |
| cuSPARSE main kernel | 10,923 blocks | 128 threads | 1,398,144 | 97.53 |

The width-aware mapping launches four times fewer threads than the fixed baseline because each lane computes four output columns.

### cuSPARSE Kernel Sequence

A single `cusparseSpMM` operation launched three GPU kernels:

| Kernel | Purpose | Duration |
|---|---|---:|
| `matrix_scalar_multiply_kernel` | Output scaling or initialization | 22.85 µs |
| `csr_partition_kernel` | CSR workload partitioning | 13.73 µs |
| `csrmm_alg2_kernel` | Main CSR SpMM computation | 2.99 ms |

The two auxiliary kernels contribute approximately 36.58 µs in addition to the main SpMM kernel.

The custom optimized implementation performs the operation using one kernel launch.

## Profiling Analysis

### 1. The optimized kernel approximately doubles memory bandwidth

Measured memory throughput increased from:

- Baseline: 88.63 GB/s
- Optimized: 183.48 GB/s

This is a 2.07× increase in effective memory throughput.

At the same time, profiled kernel duration decreased from 6.22 ms to 2.94 ms, corresponding to approximately a 2.12× reduction.

This agrees with the 2.063× speedup measured using CUDA events.

### 2. The kernel becomes DRAM-bandwidth bound

The fixed baseline used only 46.20% of peak DRAM throughput.

The optimized kernel reached 94.15%, indicating that it is operating close to the available device-memory bandwidth for this workload.

Further improvement is therefore unlikely to come from simply increasing occupancy or launching more threads. It would require reducing memory traffic, improving reuse, compressing metadata, or changing the algorithm.

### 3. Register reuse does not cause spilling

Despite maintaining four output accumulators per lane, the optimized kernel used:

- 36 registers per thread
- No local-memory spills
- No shared-memory spills
- 98.92% achieved occupancy

The additional per-thread work therefore did not introduce a register-spilling penalty.

### 4. The optimization reduces redundant execution

The baseline launches one thread for every output element:

`8192 × 128 = 1,048,576 threads`

The optimized mapping launches:

`1024 blocks × 256 threads = 262,144 threads`

Each optimized thread computes four output columns, reducing the number of launched threads and CSR-row traversals by a factor of four.

### 5. Warp-stall behavior changes after optimization

The baseline spent approximately:

- 48.6% of issue cycles on long-scoreboard stalls
- 47.2% on LG-throttle stalls

LG-throttle stalls indicate pressure from frequent local or global memory instructions.

In the optimized kernel, the LG-throttle bottleneck is removed from the primary findings. Long-scoreboard stalls account for 91.8% of the remaining issue interval.

This higher percentage does not mean the optimized kernel is slower. Warp cycles per issued instruction decreased from 125.87 to 72.33. The optimization removed other overheads and exposed global-memory latency as the dominant remaining limitation.

### 6. Occupancy was not the main source of speedup

Both implementations already achieved high occupancy:

- Baseline: 94.81%
- Optimized: 98.92%

The performance improvement primarily comes from better work mapping, fewer CSR traversals, coalesced wide-output accesses, and higher memory-bandwidth utilization—not merely from increasing occupancy.

### 7. The custom kernel and cuSPARSE reach similar hardware limits

For the main computation:

- Custom DRAM utilization: 94.15%
- cuSPARSE DRAM utilization: 94.79%
- Custom memory throughput: 183.48 GB/s
- cuSPARSE memory throughput: 181.72 GB/s
- Custom kernel duration: 2.94 ms
- cuSPARSE main-kernel duration: 2.99 ms

Both kernels reach approximately the same DRAM-bandwidth ceiling.

This explains why their benchmark throughput at 128 nonzeros per row and `N = 128` was nearly identical:

- Custom: 89.335 GFLOP/s
- cuSPARSE: 89.249 GFLOP/s

### 8. cuSPARSE includes additional dispatch work

cuSPARSE uses separate scaling, partitioning, and computation kernels. Its auxiliary kernels add approximately 36.58 µs.

The custom implementation avoids this general-purpose setup by using one workload-specific kernel. This reduces launch complexity but provides less generality than the library implementation.

## Overall Conclusion

The width-aware optimization successfully transformed the representative CSR SpMM workload from a partially utilized kernel into a bandwidth-saturating implementation.

The main improvements came from:

- Reducing the number of launched threads by four
- Traversing each CSR row once for four output columns
- Broadcasting CSR metadata within each warp
- Reusing sparse values and column indices
- Maintaining multiple results in registers
- Eliminating the baseline LG-throttle bottleneck
- Increasing measured memory throughput from 88.63 GB/s to 183.48 GB/s

For the profiled workload, the custom kernel reached essentially the same hardware bandwidth limit and end-to-end performance as cuSPARSE.

These conclusions apply to the tested GPU, uniform sparsity pattern, matrix dimensions, dense width, and cuSPARSE algorithm. Irregular real-world matrices may produce different load-balance and cache behavior.

## Final Status

- [x] Compared fixed-block, width-aware, and cuSPARSE implementations
- [x] Tested all 12 workload configurations
- [x] Verified every result against the CPU reference
- [x] Captured full Nsight Compute reports
- [x] Examined throughput and memory behavior
- [x] Examined launch configuration and register usage
- [x] Examined occupancy and warp stalls
- [x] Connected profiler evidence to benchmark results
- [x] Documented limitations of the comparison

**Experiment status: Complete**