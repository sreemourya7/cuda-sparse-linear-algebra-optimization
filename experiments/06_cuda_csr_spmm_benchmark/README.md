# Experiment 06 — CUDA CSR SpMM Workload Benchmark

## Objective

Here our Objective is to benchmark the one-thread-per-output-element CUDA CSR SpMM kernel across different sparse row lengths and dense output widths.

This experiment studies how the kernel changes as:

- Each sparse row contains more nonzero elements
- Each sparse value contributes to more output columns
- The total amount of arithmetic per matrix multiplication increases

## Fixed Matrix Dimensions

The benchmark uses:

\[
A \in \mathbb{R}^{8192 \times 8192}
\]

\[
B \in \mathbb{R}^{8192 \times N}
\]

\[
C \in \mathbb{R}^{8192 \times N}
\]

The dense output width \(N\) varies between benchmark cases.

## Sparse Row-Length Cases

The CSR matrix uses uniform row lengths:

| Nonzeros per row | Total nonzeros | Density | Sparsity |
|---:|---:|---:|---:|
| 8 | 65,536 | 0.0977% | 99.9023% |
| 32 | 262,144 | 0.3906% | 99.6094% |
| 128 | 1,048,576 | 1.5625% | 98.4375% |

Uniform row lengths remove load imbalance from this experiment, allowing row length and dense width to be studied independently.

## Dense-Width Cases

The benchmark evaluates:

```text
N = 1, 8, 32, and 128
```

## Current Status

- [x] Defined the matrix dimensions
- [x] Defined three sparse row-length cases
- [x] Defined four dense-width cases
- [x] Defined the timing methodology
- [x] Defined the performance metric
- [x] Documented the initial hypotheses
- [x] Implemented the benchmark
- [x] Executed all 12 cases
- [x] Recorded and analyzed the results

**Experiment status: Complete**