# Experiment 03 — CUDA CSR SpMV Baseline Benchmark

## Objective

Measure the performance of the one-thread-per-row CUDA CSR SpMV kernel using larger synthetic sparse matrices.

This experiment establishes a repeatable performance baseline before introducing load-balancing or warp-cooperative optimizations.

## Benchmark Scope

The benchmark uses a square CSR matrix with:

```text
Rows    = 16,384
Columns = 16,384

## Current Status

- [x] Defined benchmark dimensions and sparsity cases
- [x] Defined deterministic CSR generation
- [x] Defined warm-up and timed iteration counts
- [x] Defined performance metrics
- [x] Defined correctness requirements
- [x] Implemented CUDA-event kernel timing
- [x] Ran all four benchmark cases
- [x] Validated every GPU result against the CPU reference
- [x] Recorded and analyzed the results

**Experiment status: Complete**