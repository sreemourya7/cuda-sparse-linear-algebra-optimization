# Experiment 01 — CPU CSR SpMV Reference

## Objective

Implement a correct CPU reference for sparse matrix–vector multiplication using the CSR representation validated in Experiment 00.

This implementation will become the correctness baseline for future CUDA SpMV kernels.

## Operation

Sparse matrix–vector multiplication computes:

\[
y = Ax
\]

where:

- \(A\) is an \(M \times K\) sparse matrix
- \(x\) is a dense vector containing \(K\) elements
- \(y\) is a dense vector containing \(M\) elements

For every matrix row, only the stored nonzero values are processed.

## CSR Algorithm

For row `i`, the nonzero elements are located between:

```text
rowOffsets[i]

## Current Status

- [x] Defined the CSR SpMV operation
- [x] Documented the row-based traversal algorithm
- [x] Derived the expected output for the validation matrix
- [x] Defined correctness requirements
- [x] Implemented CPU CSR SpMV
- [x] Implemented the dense CPU reference
- [x] Compared CSR, dense, and manually calculated results
- [x] Recorded reproducible build instructions and results

**Experiment status: Complete**