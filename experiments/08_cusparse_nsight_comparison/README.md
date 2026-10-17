# Experiment 08 — cuSPARSE and Nsight Compute Comparison

## Objective

Here our main objective is to compare the custom CUDA CSR SpMM implementations against NVIDIA cuSPARSE and use Nsight Compute to explain the measured performance.

This experiment answers three questions:

1. How close does the custom implementation come to cuSPARSE?
2. Which implementation performs best for different sparse-row lengths and dense widths?
3. Which hardware bottlenecks explain the performance of the representative wide-output workload?

## Implementations

### Fixed-Block Baseline

The baseline uses a fixed `16 × 16` block.

Each thread computes one output element and independently traverses the corresponding CSR row.

### Width-Aware Custom Kernel

The optimized implementation selects a mapping based on the dense output width:

- `N < 32`: narrow scalar mapping
- `N = 32`: warp-aligned scalar mapping
- `N > 32`: warp broadcast with register reuse

For `N = 128`, each lane computes four output columns while CSR values and column indices are broadcast from lane 0.

### cuSPARSE Reference

The library comparison uses:

- Generic cuSPARSE descriptors
- CSR sparse-matrix format
- Row-major dense matrices
- FP32 values
- `CUSPARSE_SPMM_CSR_ALG2`

Descriptor creation, workspace allocation, and memory transfers are performed outside the timed region.

## Benchmark Workloads

All benchmarks use an `8192 × 8192` sparse matrix.

The tested nonzeros per row are:

- 8
- 32
- 128

The tested dense output widths are:

- 1
- 8
- 32
- 128

This produces 12 workload configurations.

## Correctness

Every GPU result is compared against the CPU CSR SpMM reference.

Correctness is checked independently for:

- Fixed-block baseline
- Width-aware custom kernel
- cuSPARSE

All 12 workload configurations passed for all three implementations.

## Timing Methodology

- 10 warm-up iterations
- 100 timed iterations
- CUDA events for GPU timing
- Allocation excluded
- Host-to-device transfers excluded
- Descriptor creation excluded
- Workspace setup excluded
- Identical CSR and dense input data for every implementation

Reported throughput is calculated as:

`2 × NNZ × N / execution time`

## Build

From this experiment directory:

    nvcc -O3 -lineinfo -std=c++17 benchmark.cu ..\00_csr_representation\csr_matrix.cpp ..\04_cpu_csr_spmm\csr_spmm.cpp -lcusparse -o benchmark.exe

## Run the Benchmark

    benchmark.exe

## Profile Modes

The executable provides three dedicated profiling modes:

    benchmark.exe --profile baseline

    benchmark.exe --profile optimized

    benchmark.exe --profile cusparse

Each mode runs the representative workload:

- Matrix dimensions: 8192 × 8192
- Nonzeros per row: 128
- Dense output width: 128

## Nsight Compute Commands

    ncu --set full --force-overwrite --export profiles\baseline benchmark.exe --profile baseline

    ncu --set full --force-overwrite --export profiles\optimized benchmark.exe --profile optimized

    ncu --set full --force-overwrite --export profiles\cusparse benchmark.exe --profile cusparse

On Windows, access to GPU performance counters may require running the command prompt as administrator.

## Benchmark Highlights

- Maximum custom speedup over the fixed baseline: **2.063×**
- Peak optimized custom throughput: **169.932 GFLOP/s**
- Peak cuSPARSE throughput: **102.247 GFLOP/s**
- Optimized custom throughput at the profiled workload: **89.335 GFLOP/s**
- cuSPARSE throughput at the profiled workload: **89.249 GFLOP/s**
- All correctness checks passed

The custom kernel performed especially well at `N = 8` and `N = 32`.

cuSPARSE was stronger for longer-row SpMV workloads and some wide-output configurations.

No single implementation was optimal for every tested shape.

## Nsight Compute Highlights

For 128 nonzeros per row and `N = 128`:

| Metric | Fixed Baseline | Width-Aware Custom | cuSPARSE Main Kernel |
|---|---:|---:|---:|
| Profiled duration | 6.22 ms | 2.94 ms | 2.99 ms |
| DRAM utilization | 46.20% | 94.15% | 94.79% |
| Memory throughput | 88.63 GB/s | 183.48 GB/s | 181.72 GB/s |
| Registers per thread | 54 | 36 | 48 |
| Achieved occupancy | 94.81% | 98.92% | 98.40% |

The optimized kernel:

- Launches four times fewer threads than the baseline
- Reuses CSR metadata across four output columns
- More than doubles measured memory throughput
- Does not spill registers into local memory
- Removes the baseline LG-throttle bottleneck
- Reaches approximately 94% of peak DRAM throughput
- Reaches essentially the same memory-bandwidth limit as cuSPARSE

cuSPARSE launches three kernels for the profiled operation:

1. Output scaling or initialization
2. CSR workload partitioning
3. Main CSR SpMM computation

The custom implementation performs the workload using one specialized kernel.

## Files

- `benchmark.cu`: Custom kernels, cuSPARSE comparison, correctness checks, timing, and profiling modes
- `results.md`: Complete benchmark tables and Nsight Compute analysis
- `profiles/`: Locally generated Nsight Compute reports

Binary profiler reports are hardware- and tool-version-specific. The important measurements and observations are recorded in `results.md`.

## Key Conclusion

The width-aware mapping transforms the representative workload from a partially utilized implementation into a DRAM-bandwidth-bound kernel.

For the profiled workload, the custom implementation reaches essentially the same hardware limit and performance as cuSPARSE while using a single specialized kernel.

The results also demonstrate that sparse performance is workload-dependent. Dense width, sparse-row length, launch overhead, load distribution, and memory-access behavior all influence which implementation performs best.

## Current Status

- [x] Added fixed-block baseline
- [x] Added width-aware custom implementation
- [x] Added cuSPARSE CSR SpMM
- [x] Tested 12 workload configurations
- [x] Verified every implementation against the CPU reference
- [x] Recorded benchmark results
- [x] Captured full Nsight Compute profiles
- [x] Compared throughput and memory behavior
- [x] Compared launch configuration and occupancy
- [x] Analyzed warp-stall behavior
- [x] Documented conclusions and limitations

**Experiment status: Complete**