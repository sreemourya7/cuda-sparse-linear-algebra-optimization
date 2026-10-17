# Experiment 00 — CSR Matrix Representation

## Objective

Implement and validate the Compressed Sparse Row (CSR) format before developing CUDA sparse matrix kernels.

This experiment establishes the data representation that will be reused by the CSR SpMV and SpMM implementations.

## Dense Matrix Example

Consider the following \(4 \times 5\) matrix:

\[
A =
\begin{bmatrix}
1 & 0 & 0 & 2 & 0 \\
0 & 0 & 3 & 0 & 0 \\
4 & 0 & 5 & 0 & 6 \\
0 & 0 & 0 & 7 & 0
\end{bmatrix}
\]

The matrix contains seven nonzero elements.

## CSR Representation

CSR stores the matrix using three arrays.

### Values

Stores the nonzero values in row-major order:

```text
values = [1, 2, 3, 4, 5, 6, 7]

## Current Status

- [x] Selected CSR as the baseline sparse format
- [x] Defined the three-array representation
- [x] Documented CSR row traversal
- [x] Defined representation validation requirements
- [x] Implemented dense-to-CSR conversion
- [x] Implemented CSR-to-dense conversion
- [x] Implemented structural CSR validation
- [x] Validated the implementation using a known test matrix
- [x] Recorded the build procedure and test results

**Experiment status: Complete**