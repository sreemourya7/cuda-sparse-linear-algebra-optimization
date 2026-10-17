#include "../00_csr_representation/csr_matrix.h"
#include "../01_cpu_csr_spmv/csr_spmv.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                  \
    do                                                    \
    {                                                     \
        cudaError_t error = call;                         \
        if (error != cudaSuccess)                         \
        {                                                 \
            std::cerr << "CUDA error: "                   \
                      << cudaGetErrorString(error)        \
                      << " at line " << __LINE__ << '\n'; \
            return 1;                                     \
        }                                                 \
    } while (0)

__global__ void csrSpmvKernel(
    int rows,
    const int *rowOffsets,
    const int *columnIndices,
    const float *values,
    const float *x,
    float *y)
{
    int row =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows)
    {
        return;
    }

    float sum = 0.0f;

    for (
        int position = rowOffsets[row];
        position < rowOffsets[row + 1];
        ++position)
    {
        int col = columnIndices[position];
        sum += values[position] * x[col];
    }

    y[row] = sum;
}

void printVector(
    const char *name,
    const std::vector<float> &values)
{
    std::cout << name << " = [";

    for (size_t i = 0; i < values.size(); ++i)
    {
        std::cout << values[i];

        if (i + 1 < values.size())
        {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";
}

int main()
{
    const int rows = 4;
    const int cols = 5;

    const std::vector<float> denseMatrix = {
        1, 0, 0, 2, 0,
        0, 0, 3, 0, 0,
        4, 0, 5, 0, 6,
        0, 0, 0, 7, 0};

    const std::vector<float> x = {
        1, 2, 3, 4, 5};

    const std::vector<float> expected = {
        9, 9, 49, 28};

    CSRMatrix csr =
        CSRMatrix::fromDense(denseMatrix, rows, cols);

    std::vector<float> cpuResult =
        csrSpmv(csr, x);

    std::vector<float> gpuResult(rows, 0.0f);

    float *deviceValues = nullptr;
    int *deviceColumnIndices = nullptr;
    int *deviceRowOffsets = nullptr;
    float *deviceX = nullptr;
    float *deviceY = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceValues),
        csr.values.size() * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceColumnIndices),
        csr.columnIndices.size() * sizeof(int)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceRowOffsets),
        csr.rowOffsets.size() * sizeof(int)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceX),
        x.size() * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceY),
        gpuResult.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        deviceValues,
        csr.values.data(),
        csr.values.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceColumnIndices,
        csr.columnIndices.data(),
        csr.columnIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceRowOffsets,
        csr.rowOffsets.data(),
        csr.rowOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceX,
        x.data(),
        x.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocks =
        (rows + threadsPerBlock - 1) / threadsPerBlock;

    csrSpmvKernel<<<blocks, threadsPerBlock>>>(
        rows,
        deviceRowOffsets,
        deviceColumnIndices,
        deviceValues,
        deviceX,
        deviceY);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        gpuResult.data(),
        deviceY,
        gpuResult.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(deviceValues));
    CUDA_CHECK(cudaFree(deviceColumnIndices));
    CUDA_CHECK(cudaFree(deviceRowOffsets));
    CUDA_CHECK(cudaFree(deviceX));
    CUDA_CHECK(cudaFree(deviceY));

    printVector("CPU result", cpuResult);
    printVector("GPU result", gpuResult);
    printVector("Expected result", expected);

    bool passed =
        vectorsNearlyEqual(gpuResult, cpuResult) &&
        vectorsNearlyEqual(gpuResult, expected);

    if (!passed)
    {
        std::cerr << "\nCUDA CSR SpMV validation: FAILED\n";
        return 1;
    }

    std::cout << "\nCUDA CSR SpMV validation: PASSED\n";
    return 0;
}