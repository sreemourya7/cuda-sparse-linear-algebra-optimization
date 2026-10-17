# Sparse Computation Foundations

## 1. Motivation

Dense GEMM assumes that every matrix element participates in computation:

\[
C = A \times B
\]

For matrices

\[
A \in \mathbb{R}^{M \times K}, \qquad
B \in \mathbb{R}^{K \times N},
\]

dense GEMM performs approximately:

\[
2MKN
\]

floating-point operations.

When most elements of a matrix are zero, storing and multiplying every element wastes memory capacity, memory bandwidth, and computation. Sparse matrix algorithms store only the nonzero values and the metadata required to locate them.

## 2. Sparsity Terminology

Let `nnz` represent the number of nonzero elements in a matrix.

\[
\text{Density} =
\frac{\text{nnz}}{\text{number of matrix elements}}
\]

\[
\text{Sparsity} = 1 - \text{Density}
\]

For example, if a matrix contains 1,000,000 elements but only 100,000 are nonzero:

\[
\text{Density} = 0.10
\]

\[
\text{Sparsity} = 0.90
\]

The matrix is therefore 90% sparse.

## 3. Sparse Matrix Operations

### SpMV

Sparse matrix–vector multiplication:

\[
y = A_{\text{sparse}}x
\]

SpMV performs approximately:

\[
2 \times \text{nnz}(A)
\]

floating-point operations.

It is a useful starting point for understanding sparse storage, indirect memory access, and workload imbalance.

### SpMM

Sparse matrix–dense matrix multiplication:

\[
C = A_{\text{sparse}}B_{\text{dense}}
\]

If `A` has `nnz(A)` nonzero values and `B` has `N` columns, SpMM performs approximately:

\[
2 \times \text{nnz}(A) \times N
\]

floating-point operations.

SpMM provides more data reuse than SpMV and is the primary optimization target of this project.

### SpGEMM

Sparse matrix–sparse matrix multiplication:

\[
C_{\text{sparse}} =
A_{\text{sparse}}B_{\text{sparse}}
\]

SpGEMM is more difficult because the number and positions of nonzero output elements are not generally known before multiplication.

It will be considered only after establishing strong SpMV and SpMM baselines.

## 4. How Sparse Computation Differs from Dense GEMM

Dense GEMM has regular memory access and predictable work distribution. Sparse computation introduces:

- Storage and processing of row or column indices
- Indirect memory accesses
- Different numbers of nonzeros in different rows
- Unequal work across threads and warps
- Reduced opportunities for regular memory coalescing
- Possible warp divergence
- Lower and input-dependent arithmetic intensity
- Performance dependence on the sparsity pattern, not only matrix dimensions

Therefore, removing arithmetic operations does not automatically produce proportional speedup.

## 5. Initial Design Decision

CSR will be the first storage format used in this project because it:

- Stores matrix rows compactly
- Provides efficient row traversal
- Is widely used for SpMV and SpMM
- Makes row-length imbalance directly observable
- Provides a clear baseline for later load-balancing experiments

The initial implementation path is:

1. Construct and validate a CSR matrix
2. Implement a CPU CSR reference
3. Implement CSR SpMV
4. Implement naive CSR SpMM
5. Profile memory behavior and workload imbalance
6. Apply optimizations based on measured bottlenecks

## 6. Initial Performance Hypothesis

Dense GEMM optimization primarily improves reuse and computational throughput through tiling.

For sparse kernels, performance is expected to depend more heavily on:

- Memory bandwidth
- Index-loading overhead
- Access patterns into the dense operand
- Nonzeros per row
- Warp utilization
- Work distribution between thread blocks

This hypothesis will be evaluated using controlled benchmarks and NVIDIA Nsight Compute.