# Experiment 05 — CUDA CSR SpMM Baseline

## Objective

Implement and validate the first CUDA kernel for CSR sparse matrix–dense matrix multiplication.

The CPU CSR SpMM implementation from Experiment 04 will serve as the correctness reference.

## Operation

The kernel computes:

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

The CSR matrix \(A\) is sparse, while \(B\) and \(C\) are dense row-major matrices.

## Thread Mapping

The baseline uses:

```text
One CUDA thread → One output element C[row, column]

## Status :

- [x] Implemented the CUDA CSR SpMM kernel
- [x] Validated against the CPU reference
- [x] Recorded the results

**Experiment status: Correctness baseline complete**