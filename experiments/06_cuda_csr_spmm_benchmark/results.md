# Results — CUDA CSR SpMM Workload Benchmark

## Hardware and Configuration

- GPU: NVIDIA GeForce GTX 1650
- Sparse matrix dimensions: \(8192 \times 8192\)
- Kernel mapping: One thread per output element
- Thread-block dimensions: \(16 \times 16\)
- Warm-up iterations: 10
- Timed iterations: 100
- Timing method: CUDA events
- Timed region: Kernel execution only

## Build and Run

```cmd
nvcc -O3 -std=c++17 benchmark.cu ..\00_csr_representation\csr_matrix.cpp ..\04_cpu_csr_spmm\csr_spmm.cpp -o benchmark.exe
benchmark.exe

## Benchmark Results

| NNZ per row | Dense width \(N\) | Total NNZ | Average time (ms) | GFLOP/s | Correctness |
|---:|---:|---:|---:|---:|:---:|
| 8 | 1 | 65,536 | 0.018869 | 6.946 | PASS |
| 8 | 8 | 65,536 | 0.023122 | 45.350 | PASS |
| 8 | 32 | 65,536 | 0.050985 | 82.265 | PASS |
| 8 | 128 | 65,536 | 0.265523 | 63.186 | PASS |
| 32 | 1 | 262,144 | 0.079591 | 6.587 | PASS |
| 32 | 8 | 262,144 | 0.079524 | 52.743 | PASS |
| 32 | 32 | 262,144 | 0.176519 | 95.045 | PASS |
| 32 | 128 | 262,144 | 0.944080 | 71.084 | PASS |
| 128 | 1 | 1,048,576 | 0.235297 | 8.913 | PASS |
| 128 | 8 | 1,048,576 | 0.234132 | 71.657 | PASS |
| 128 | 32 | 1,048,576 | 0.546663 | 122.761 | PASS |
| 128 | 128 | 1,048,576 | 5.308345 | 50.569 | PASS |

## Key Observations

### 1. Every row-length case achieved its highest throughput at \(N=32\)

| NNZ per row | Peak GFLOP/s | Dense width |
|---:|---:|---:|
| 8 | 82.265 | 32 |
| 32 | 95.045 | 32 |
| 128 | 122.761 | 32 |

Increasing the dense width initially improves output-column parallelism and amortizes the cost of traversing the CSR structure.

### 2. Small dense widths underutilize the two-dimensional block

For \(N=1\), only one of the 16 thread columns in each block performs useful work.

Measured throughput remained below 9 GFLOP/s:

```text
8 NNZ/row   → 6.946 GFLOP/s
32 NNZ/row  → 6.587 GFLOP/s
128 NNZ/row → 8.913 GFLOP/s
```

This confirms that the fixed \(16 \times 16\) launch configuration is inefficient for narrow dense matrices.

### 3. Increasing \(N\) from 1 to 8 added useful work with little additional latency

For 32 and 128 nonzeros per row, the measured time was nearly unchanged:

```text
32 NNZ/row:
N=1 → 0.079591 ms
N=8 → 0.079524 ms

128 NNZ/row:
N=1 → 0.235297 ms
N=8 → 0.234132 ms
```

The additional output-column work uses lanes that were inactive in the \(N=1\) configuration.

### 4. Throughput decreased at \(N=128\)

Compared with each \(N=32\) result, throughput at \(N=128\) decreased:

```text
8 NNZ/row:   82.265 → 63.186 GFLOP/s
32 NNZ/row:  95.045 → 71.084 GFLOP/s
128 NNZ/row: 122.761 → 50.569 GFLOP/s
```

The baseline independently traverses the same CSR row for every output column. Larger dense widths also increase the dense input and output working sets.

These measurements are consistent with increasing memory-system pressure and redundant CSR traversal, but Nsight Compute profiling is required before identifying the limiting hardware resource conclusively.

### 5. Longer rows improved peak arithmetic throughput up to \(N=32\)

At \(N=32\):

```text
8 NNZ/row   → 82.265 GFLOP/s
32 NNZ/row  → 95.045 GFLOP/s
128 NNZ/row → 122.761 GFLOP/s
```

Longer rows provide more arithmetic work per output thread and better amortize fixed kernel-launch and indexing costs.

## Correctness

All 12 GPU outputs matched their corresponding CPU CSR SpMM references.

```text
12 of 12 configurations: PASS
```

## Conclusion

The baseline CUDA CSR SpMM kernel benefits from increasing dense width up to \(N=32\), where it reaches a maximum of:

```text
122.761 GFLOP/s
```

Performance decreases at \(N=128\), especially for the longest sparse rows. The next stage will use profiling and controlled kernel changes to determine whether the primary causes are redundant CSR loads, cache behavior, memory latency, or execution inefficiency.