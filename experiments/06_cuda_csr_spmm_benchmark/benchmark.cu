#include "../00_csr_representation/csr_matrix.h"
#include "../04_cpu_csr_spmm/csr_spmm.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
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

CSRMatrix generateUniformCsr(
    int rows,
    int cols,
    int nonzerosPerRow)
{
    if (
        rows <= 0 ||
        cols <= 0 ||
        nonzerosPerRow <= 0 ||
        nonzerosPerRow > cols)
    {
        throw std::invalid_argument(
            "Invalid CSR generation parameters.");
    }

    CSRMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    const size_t totalNonzeros =
        static_cast<size_t>(rows) *
        nonzerosPerRow;

    matrix.values.reserve(totalNonzeros);
    matrix.columnIndices.reserve(totalNonzeros);
    matrix.rowOffsets.resize(rows + 1);

    for (int row = 0; row < rows; ++row)
    {
        matrix.rowOffsets[row] =
            static_cast<int>(matrix.values.size());

        int startingColumn =
            (row * 17) % cols;

        for (
            int item = 0;
            item < nonzerosPerRow;
            ++item)
        {
            int col =
                (startingColumn + item * 131) % cols;

            float value =
                1.0f +
                static_cast<float>((row + item) % 5) *
                    0.1f;

            matrix.columnIndices.push_back(col);
            matrix.values.push_back(value);
        }
    }

    matrix.rowOffsets[rows] =
        static_cast<int>(matrix.values.size());

    return matrix;
}

std::vector<float> generateDenseMatrix(
    int rows,
    int cols)
{
    std::vector<float> matrix(
        static_cast<size_t>(rows) * cols);

    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            matrix[row * cols + col] =
                static_cast<float>(
                    ((row * 3 + col) % 17) + 1) *
                0.01f;
        }
    }

    return matrix;
}

struct BenchmarkResult
{
    float averageMilliseconds;
    double gigaFlops;
    bool correct;
};

BenchmarkResult runBenchmark(
    const CSRMatrix &sparseMatrix,
    const std::vector<float> &denseMatrix,
    int denseColumns,
    const std::vector<float> &cpuResult,
    int warmupIterations,
    int timedIterations)
{
    float *deviceSparseValues = nullptr;
    int *deviceColumnIndices = nullptr;
    int *deviceRowOffsets = nullptr;
    float *deviceDenseMatrix = nullptr;
    float *deviceOutput = nullptr;

    const size_t sparseValuesBytes =
        sparseMatrix.values.size() * sizeof(float);

    const size_t columnIndicesBytes =
        sparseMatrix.columnIndices.size() *
        sizeof(int);

    const size_t rowOffsetsBytes =
        sparseMatrix.rowOffsets.size() * sizeof(int);

    const size_t denseMatrixBytes =
        denseMatrix.size() * sizeof(float);

    const size_t outputElements =
        static_cast<size_t>(sparseMatrix.rows) *
        denseColumns;

    const size_t outputBytes =
        outputElements * sizeof(float);

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
        sparseMatrix.values.data(),
        sparseValuesBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceColumnIndices,
        sparseMatrix.columnIndices.data(),
        columnIndicesBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceRowOffsets,
        sparseMatrix.rowOffsets.data(),
        rowOffsetsBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceDenseMatrix,
        denseMatrix.data(),
        denseMatrixBytes,
        cudaMemcpyHostToDevice));

    const dim3 blockDimensions(16, 16);

    const dim3 gridDimensions(
        (denseColumns + blockDimensions.x - 1) /
            blockDimensions.x,
        (sparseMatrix.rows + blockDimensions.y - 1) /
            blockDimensions.y);

    csrSpmmKernel<<<
        gridDimensions,
        blockDimensions>>>(
        sparseMatrix.rows,
        denseColumns,
        deviceRowOffsets,
        deviceColumnIndices,
        deviceSparseValues,
        deviceDenseMatrix,
        deviceOutput);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpuResult(outputElements);

    CUDA_CHECK(cudaMemcpy(
        gpuResult.data(),
        deviceOutput,
        outputBytes,
        cudaMemcpyDeviceToHost));

    bool correct =
        matricesNearlyEqual(
            gpuResult,
            cpuResult,
            1e-3f);

    for (
        int iteration = 0;
        iteration < warmupIterations;
        ++iteration)
    {
        csrSpmmKernel<<<
            gridDimensions,
            blockDimensions>>>(
            sparseMatrix.rows,
            denseColumns,
            deviceRowOffsets,
            deviceColumnIndices,
            deviceSparseValues,
            deviceDenseMatrix,
            deviceOutput);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (
        int iteration = 0;
        iteration < timedIterations;
        ++iteration)
    {
        csrSpmmKernel<<<
            gridDimensions,
            blockDimensions>>>(
            sparseMatrix.rows,
            denseColumns,
            deviceRowOffsets,
            deviceColumnIndices,
            deviceSparseValues,
            deviceDenseMatrix,
            deviceOutput);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMilliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMilliseconds,
        start,
        stop));

    float averageMilliseconds =
        totalMilliseconds / timedIterations;

    double totalOperations =
        2.0 *
        static_cast<double>(
            sparseMatrix.values.size()) *
        denseColumns;

    double gigaFlops =
        totalOperations /
        (averageMilliseconds * 1.0e6);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(deviceSparseValues));
    CUDA_CHECK(cudaFree(deviceColumnIndices));
    CUDA_CHECK(cudaFree(deviceRowOffsets));
    CUDA_CHECK(cudaFree(deviceDenseMatrix));
    CUDA_CHECK(cudaFree(deviceOutput));

    return {
        averageMilliseconds,
        gigaFlops,
        correct};
}

int main()
{
    const int rows = 8192;
    const int sharedDimension = 8192;
    const int warmupIterations = 10;
    const int timedIterations = 100;

    const std::vector<int> nonzerosPerRowCases = {
        8, 32, 128};

    const std::vector<int> denseColumnCases = {
        1, 8, 32, 128};

    cudaDeviceProp deviceProperties;

    CUDA_CHECK(cudaGetDeviceProperties(
        &deviceProperties,
        0));

    std::cout
        << "GPU: " << deviceProperties.name << '\n'
        << "Sparse matrix dimensions: "
        << rows << " x "
        << sharedDimension << '\n'
        << "Warm-up iterations: "
        << warmupIterations << '\n'
        << "Timed iterations: "
        << timedIterations << "\n\n";

    std::cout
        << std::left
        << std::setw(12) << "NNZ/row"
        << std::setw(10) << "N"
        << std::setw(14) << "Total NNZ"
        << std::setw(16) << "Average ms"
        << std::setw(14) << "GFLOP/s"
        << "Correct\n";

    for (
        int nonzerosPerRow :
        nonzerosPerRowCases)
    {
        CSRMatrix sparseMatrix =
            generateUniformCsr(
                rows,
                sharedDimension,
                nonzerosPerRow);

        if (!sparseMatrix.isValid())
        {
            std::cerr
                << "Generated CSR matrix is invalid.\n";

            return 1;
        }

        for (
            int denseColumns :
            denseColumnCases)
        {
            std::vector<float> denseMatrix =
                generateDenseMatrix(
                    sharedDimension,
                    denseColumns);

            std::vector<float> cpuResult =
                csrSpmm(
                    sparseMatrix,
                    denseMatrix,
                    denseColumns);

            BenchmarkResult result =
                runBenchmark(
                    sparseMatrix,
                    denseMatrix,
                    denseColumns,
                    cpuResult,
                    warmupIterations,
                    timedIterations);

            std::cout
                << std::left
                << std::setw(12)
                << nonzerosPerRow
                << std::setw(10)
                << denseColumns
                << std::setw(14)
                << sparseMatrix.values.size()
                << std::setw(16)
                << std::fixed
                << std::setprecision(6)
                << result.averageMilliseconds
                << std::setw(14)
                << std::setprecision(3)
                << result.gigaFlops
                << (result.correct ? "PASS" : "FAIL")
                << '\n';

            if (!result.correct)
            {
                return 1;
            }
        }
    }

    return 0;
}