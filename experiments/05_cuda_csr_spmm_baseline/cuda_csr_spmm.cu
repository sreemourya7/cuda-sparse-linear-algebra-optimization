#include "../00_csr_representation/csr_matrix.h"
#include "../04_cpu_csr_spmm/csr_spmm.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

void checkCuda(
    cudaError_t error,
    const char *operation,
    int line)
{
    if (error != cudaSuccess)
    {
        std::cerr
            << "CUDA error at line " << line
            << " during " << operation
            << ": " << cudaGetErrorString(error)
            << '\n';

        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) \
    checkCuda((call), #call, __LINE__)

__global__ void csrSpmmKernel(
    int rows,
    int denseColumns,
    const int *rowOffsets,
    const int *columnIndices,
    const float *sparseValues,
    const float *denseMatrix,
    float *output)
{
    int outputColumn =
        blockIdx.x * blockDim.x + threadIdx.x;

    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (
        row >= rows ||
        outputColumn >= denseColumns)
    {
        return;
    }

    float sum = 0.0f;

    for (
        int position = rowOffsets[row];
        position < rowOffsets[row + 1];
        ++position)
    {
        int sharedIndex =
            columnIndices[position];

        sum +=
            sparseValues[position] *
            denseMatrix[sharedIndex * denseColumns +
                        outputColumn];
    }

    output[row * denseColumns + outputColumn] = sum;
}

void printMatrix(
    const char *name,
    const std::vector<float> &matrix,
    int rows,
    int cols)
{
    std::cout << name << ":\n";

    for (int row = 0; row < rows; ++row)
    {
        std::cout << "[ ";

        for (int col = 0; col < cols; ++col)
        {
            std::cout
                << std::setw(6)
                << matrix[row * cols + col];

            if (col + 1 < cols)
            {
                std::cout << " ";
            }
        }

        std::cout << " ]\n";
    }
}

int main()
{
    const int rowsA = 4;
    const int sharedDimension = 5;
    const int columnsB = 3;

    const std::vector<float> denseA = {
        1, 0, 0, 2, 0,
        0, 0, 3, 0, 0,
        4, 0, 5, 0, 6,
        0, 0, 0, 7, 0};

    const std::vector<float> denseB = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15};

    const std::vector<float> expected = {
        21, 24, 27,
        21, 24, 27,
        117, 132, 147,
        70, 77, 84};

    CSRMatrix csrA =
        CSRMatrix::fromDense(
            denseA,
            rowsA,
            sharedDimension);

    if (!csrA.isValid())
    {
        std::cerr << "The CSR matrix is invalid.\n";
        return 1;
    }

    std::vector<float> cpuResult =
        csrSpmm(
            csrA,
            denseB,
            columnsB);

    std::vector<float> gpuResult(
        static_cast<size_t>(rowsA) *
            columnsB,
        0.0f);

    float *deviceSparseValues = nullptr;
    int *deviceColumnIndices = nullptr;
    int *deviceRowOffsets = nullptr;
    float *deviceDenseMatrix = nullptr;
    float *deviceOutput = nullptr;

    const size_t sparseValuesBytes =
        csrA.values.size() * sizeof(float);

    const size_t columnIndicesBytes =
        csrA.columnIndices.size() * sizeof(int);

    const size_t rowOffsetsBytes =
        csrA.rowOffsets.size() * sizeof(int);

    const size_t denseMatrixBytes =
        denseB.size() * sizeof(float);

    const size_t outputBytes =
        gpuResult.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceSparseValues),
        sparseValuesBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceColumnIndices),
        columnIndicesBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceRowOffsets),
        rowOffsetsBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceDenseMatrix),
        denseMatrixBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceOutput),
        outputBytes));

    CUDA_CHECK(cudaMemcpy(
        deviceSparseValues,
        csrA.values.data(),
        sparseValuesBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceColumnIndices,
        csrA.columnIndices.data(),
        columnIndicesBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceRowOffsets,
        csrA.rowOffsets.data(),
        rowOffsetsBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceDenseMatrix,
        denseB.data(),
        denseMatrixBytes,
        cudaMemcpyHostToDevice));

    const dim3 blockDimensions(16, 16);

    const dim3 gridDimensions(
        (columnsB + blockDimensions.x - 1) /
            blockDimensions.x,
        (rowsA + blockDimensions.y - 1) /
            blockDimensions.y);

    csrSpmmKernel<<<
        gridDimensions,
        blockDimensions>>>(
        rowsA,
        columnsB,
        deviceRowOffsets,
        deviceColumnIndices,
        deviceSparseValues,
        deviceDenseMatrix,
        deviceOutput);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        gpuResult.data(),
        deviceOutput,
        outputBytes,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(deviceSparseValues));
    CUDA_CHECK(cudaFree(deviceColumnIndices));
    CUDA_CHECK(cudaFree(deviceRowOffsets));
    CUDA_CHECK(cudaFree(deviceDenseMatrix));
    CUDA_CHECK(cudaFree(deviceOutput));

    printMatrix(
        "CPU CSR SpMM result",
        cpuResult,
        rowsA,
        columnsB);

    std::cout << '\n';

    printMatrix(
        "GPU CSR SpMM result",
        gpuResult,
        rowsA,
        columnsB);

    std::cout << '\n';

    printMatrix(
        "Expected result",
        expected,
        rowsA,
        columnsB);

    bool passed =
        matricesNearlyEqual(gpuResult, cpuResult) &&
        matricesNearlyEqual(gpuResult, expected);

    if (!passed)
    {
        std::cerr
            << "\nCUDA CSR SpMM validation: FAILED\n";

        return 1;
    }

    std::cout
        << "\nCUDA CSR SpMM validation: PASSED\n";

    return 0;
}