# Experiment 04 — CPU CSR SpMM Reference

## Objective

Here our Objective is to implement a correct CPU reference for sparse matrix–dense matrix multiplication using CSR storage.

This implementation will provide the correctness baseline for all future CUDA CSR SpMM kernels.

## Operation

CSR SpMM computes:

\[
C = A_{\text{sparse}}B_{\text{dense}}
\]

where:

\[
A \in \mathbb{R}^{M \times K}
\]

\[
B \in \mathbb{R}^{K \times N}
\]

\[
C \in \mathbb{R}^{M \times N}
\]

The output matrix is dense.

SpMV can be considered a special case of SpMM where:

\[
N = 1
\]

## Data Layout

The CSR matrix \(A\) uses:

```text
values
columnIndices
rowOffsets

## Current Status

- [x] Defined the CSR SpMM operation
- [x] Defined row-major indexing for the dense matrices
- [x] Documented the CPU traversal algorithm
- [x] Derived the expected validation output
- [x] Defined correctness requirements
- [x] Implemented CPU CSR SpMM
- [x] Implemented the dense reference
- [x] Ran correctness validation
- [x] Recorded the results

**Experiment status: Complete**