# Experiment 02 — CUDA CSR SpMV Baseline

## Objective

Here our Objective is to implement the 1st CUDA sparse matrix kernel using the CSR representation developed in Experiment 00 and the CPU SpMV reference developed in Experiment 01.

The initial kernel prioritizes correctness and simplicity. Performance optimization will begin only after the GPU result matches the CPU reference.

## Thread Mapping

The baseline uses:

```text
One CUDA thread → One sparse matrix row

## Current Status

- [x] Defined the one-thread-per-row mapping
- [x] Documented the baseline kernel algorithm
- [x] Defined the initial launch configuration
- [x] Identified expected performance limitations
- [x] Defined correctness requirements
- [x] Implemented the CUDA CSR SpMV kernel
- [x] Added device-memory allocation and transfers
- [x] Added CUDA launch and runtime error checking
- [x] Compared GPU, CPU, and manually calculated results
- [x] Recorded reproducible build instructions and results

**Experiment status: Correctness baseline complete**