# Results — Width-Aware CUDA CSR SpMM Optimization

## Hardware and Configuration

- GPU: NVIDIA GeForce GTX 1650
- Sparse matrix dimensions: \(8192 \times 8192\)
- Nonzeros per row: 8, 32, and 128
- Dense widths: 1, 8, 32, and 128
- Warm-up iterations: 10
- Timed iterations: 100
- Timing method: CUDA events
- Timed region: Kernel execution only

## Build and Run

    nvcc -O3 -std=c++17 benchmark.cu ..\00_csr_representation\csr_matrix.cpp ..\04_cpu_csr_spmm\csr_spmm.cpp -o benchmark.exe
    benchmark.exe

## Final Results

| NNZ/row | \(N\) | Baseline (ms) | Optimized (ms) | Baseline GFLOP/s | Optimized GFLOP/s | Speedup | Correct |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 8 | 1 | 0.018817 | 0.011221 | 6.966 | 11.680 | 1.677× | PASS |
| 8 | 8 | 0.028738 | 0.023985 | 36.487 | 43.718 | 1.198× | PASS |
| 8 | 32 | 0.050503 | 0.046940 | 83.051 | 89.354 | 1.076× | PASS |
| 8 | 128 | 0.264462 | 0.222761 | 63.439 | 75.315 | 1.187× | PASS |
| 32 | 1 | 0.079364 | 0.073631 | 6.606 | 7.121 | 1.078× | PASS |
| 32 | 8 | 0.080979 | 0.074711 | 51.795 | 56.140 | 1.084× | PASS |
| 32 | 32 | 0.179931 | 0.138916 | 93.243 | 120.773 | 1.295× | PASS |
| 32 | 128 | 0.946981 | 0.640556 | 70.866 | 104.767 | 1.478× | PASS |
| 128 | 1 | 0.231167 | 0.213502 | 9.072 | 9.823 | 1.083× | PASS |
| 128 | 8 | 0.232509 | 0.217313 | 72.157 | 77.203 | 1.070× | PASS |
| 128 | 32 | 0.545571 | 0.410952 | 123.007 | 163.301 | 1.328× | PASS |
| 128 | 128 | 5.303499 | 2.795092 | 50.615 | 96.038 | 1.897× | PASS |

## Key Results

### All 12 workloads improved

The final width-aware dispatch produced a speedup in every tested configuration.

The measured speedups ranged from:

- Minimum: 1.070×
- Maximum: 1.897×

### Highest optimized throughput

The highest throughput was:

- 163.301 GFLOP/s
- 128 nonzeros per row
- Dense width \(N=32\)
- Optimized time: 0.410952 ms
- Speedup: 1.328×

### Largest speedup

The largest improvement occurred for:

- 128 nonzeros per row
- Dense width \(N=128\)
- Baseline time: 5.303499 ms
- Optimized time: 2.795092 ms
- Speedup: 1.897×

This corresponds to an execution-time reduction of approximately 47.3%.

## Narrow-Width Behavior

For \(N=1\) and \(N=8\), the adaptive scalar mapping removes inactive thread columns from the fixed \(16 \times 16\) baseline block.

The largest narrow-width improvement was:

- 8 nonzeros per row
- \(N=1\)
- Speedup: 1.677×

The improvement becomes smaller for longer rows because CSR traversal represents a larger fraction of the total execution time.

Very short kernel measurements are also more sensitive to clock and system-level timing variation.

## The \(N=32\) Design Iteration

The first optimized implementation used warp shuffles at \(N=32\). That version caused regressions:

| NNZ/row | Initial speedup |
|---:|---:|
| 8 | 0.680× |
| 32 | 0.693× |
| 128 | 0.688× |

At \(N=32\), each lane computes only one output column. The shuffle instructions introduced overhead without providing enough per-lane reuse.

The final implementation therefore uses a scalar \(32 \times 8\) block:

- One warp maps naturally to one sparse row
- Each lane computes one output column
- No explicit shuffle instructions are required
- Dense-matrix accesses remain contiguous across the warp

The refined mapping improved all three \(N=32\) workloads:

| NNZ/row | Final speedup |
|---:|---:|
| 8 | 1.076× |
| 32 | 1.295× |
| 128 | 1.328× |

This demonstrates that warp-level operations are not automatically beneficial. Their overhead must be justified by sufficient data reuse.

## Wide-Width Behavior

For \(N=128\), each warp processes one CSR row and each lane maintains four output accumulators.

The sparse value and column index are loaded by lane 0 and broadcast across the warp. Each sparse value is then reused across four output columns per lane.

Measured \(N=128\) speedups increased with row length:

| NNZ/row | Speedup |
|---:|---:|
| 8 | 1.187× |
| 32 | 1.478× |
| 128 | 1.897× |

Longer rows provide more opportunities to amortize warp-broadcast overhead and reuse CSR metadata.

## Correctness

All baseline and optimized GPU outputs matched the CPU CSR SpMM reference.

- Baseline results: 12 of 12 PASS
- Optimized results: 12 of 12 PASS
- Baseline versus optimized: 12 of 12 PASS

## Conclusion

A single CUDA CSR SpMM mapping was not optimal across all dense widths.

The final implementation uses:

- Adaptive scalar blocks for \(N<32\)
- A scalar warp-aligned mapping for \(N=32\)
- Warp broadcast and register reuse for \(N>32\)

This width-aware dispatch improved every tested workload, increased peak throughput from 123.007 GFLOP/s to 163.301 GFLOP/s in the direct comparison, and achieved a maximum speedup of 1.897×.