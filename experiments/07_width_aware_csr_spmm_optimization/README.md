# Experiment 07 — Width-Aware CUDA CSR SpMM Optimization

## Objective

Optimize CUDA CSR SpMM using the workload behavior measured in Experiment 06.

The baseline used a fixed \(16 \times 16\) block for every dense width. The measurements showed:

- Poor utilization for \(N=1\)
- Increasing throughput through \(N=32\)
- A throughput decrease at \(N=128\)
- Repeated traversal of the same CSR row for different output columns

This experiment introduces a width-aware execution strategy.

## Baseline Kernel

The baseline mapping uses one thread for each output element with a fixed \(16 \times 16\) block.

This mapping is correct but does not adapt to the number of dense output columns.

For \(N=1\), only one of the 16 thread columns performs useful work.

For \(N=128\), multiple thread blocks independently traverse the same sparse row.

## Optimized Strategy

The optimized implementation selects one of three mappings according to the dense width.

### Strategy 1 — Adaptive Scalar Mapping

Used when:

\[
N < 32
\]

The block width matches the dense width.

| Dense width | Block dimensions |
|---:|---:|
| 1 | \(1 \times 256\) |
| 8 | \(8 \times 32\) |

This eliminates the inactive thread columns created by the fixed \(16 \times 16\) block.

Each active thread computes one output element.

### Strategy 2 — Scalar Warp-Aligned Mapping

Used when:

\[
N = 32
\]

The block dimensions are:

\[
32 \times 8
\]

Each warp processes one sparse row, and each lane computes one output column.

No explicit warp shuffles are used because the initial measurements showed that shuffle overhead caused a performance regression when every lane computed only one output.

### Strategy 3 — Warp Broadcast and Register Reuse

Used when:

\[
N > 32
\]

Each warp processes one sparse row.

For \(N=128\), each lane computes four output columns:

- `lane`
- `lane + 32`
- `lane + 64`
- `lane + 96`

Lane 0 loads each CSR value and column index. Warp shuffle instructions broadcast the metadata to the remaining lanes.

Each lane maintains four register accumulators, allowing one CSR traversal to update multiple output columns.

## Design Iteration at \(N=32\)

The first optimized implementation also used warp shuffles at \(N=32\).

That version produced regressions of approximately 31% to 32% because every lane computed only one output column. The shuffle overhead was not offset by sufficient data reuse.

The \(N=32\) path was therefore changed to the scalar \(32 \times 8\) mapping.

This refined mapping improved all three tested \(N=32\) workloads.

## Benchmark Methodology

The baseline and optimized kernels use the same 12 workloads from Experiment 06:

- Matrix dimensions: \(8192 \times 8192\)
- Nonzeros per row: 8, 32, and 128
- Dense widths: 1, 8, 32, and 128
- Warm-up iterations: 10
- Timed iterations: 100
- Timing method: CUDA events
- Timed region: Kernel execution only

Both kernels use identical input matrices.

## Reported Metrics

For every workload, the benchmark reports:

- Baseline kernel time
- Optimized kernel time
- Baseline GFLOP/s
- Optimized GFLOP/s
- Speedup
- Correctness

Speedup is calculated as:

\[
\text{Speedup}
=
\frac{\text{baseline time}}
{\text{optimized time}}
\]

## Correctness Requirements

For all 12 workloads:

- The baseline result must match the CPU reference
- The optimized result must match the CPU reference
- Baseline and optimized outputs must match each other
- CUDA runtime and kernel-launch errors must be checked
- Data transfers must remain outside the timed regions

## Measured Outcome

The final width-aware dispatch improved all 12 tested workloads.

The best measured results were:

- Peak optimized throughput: 163.301 GFLOP/s
- Maximum speedup: 1.897×
- Largest time reduction: 5.303499 ms to 2.795092 ms

Detailed measurements and analysis are available in `results.md`.

## Limitations

The optimized warp kernel currently targets dense widths up to 128 columns.

The benchmark uses uniform row lengths. Irregular row distributions remain a separate load-balancing problem.

## Files

- `README.md` — Experiment design and methodology
- `benchmark.cu` — Baseline and optimized CUDA kernels
- `results.md` — Measurements and analysis

## Current Status

- [x] Identified baseline utilization problems
- [x] Defined an adaptive narrow-width mapping
- [x] Evaluated a warp-shuffle mapping at \(N=32\)
- [x] Identified and removed the \(N=32\) shuffle regression
- [x] Implemented a scalar warp-aligned \(N=32\) mapping
- [x] Implemented warp broadcast for wide outputs
- [x] Implemented register reuse across output columns
- [x] Ran all 12 baseline-versus-optimized comparisons
- [x] Verified all outputs against the CPU reference
- [x] Recorded and analyzed the results

**Experiment status: Complete**