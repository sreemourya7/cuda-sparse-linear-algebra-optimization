#include "../00_csr_representation/csr_matrix.h"
#include "../01_cpu_csr_spmv/csr_spmv.h"

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
            "Invalid matrix-generation parameters.");
    }

    CSRMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    const size_t totalNonzeros =
        static_cast<size_t>(rows) * nonzerosPerRow;

    matrix.values.reserve(totalNonzeros);
    matrix.columnIndices.reserve(totalNonzeros);
    matrix.rowOffsets.resize(rows + 1);

    for (int row = 0; row < rows; ++row)
    {
        matrix.rowOffsets[row] =
            static_cast<int>(matrix.values.size());

        int startingColumn = (row * 17) % cols;

        for (int item = 0; item < nonzerosPerRow; ++item)
        {
            int col =
                (startingColumn + item * 131) % cols;

            float value =
                1.0f +
                static_cast<float>((row + item) % 5) * 0.1f;

            matrix.columnIndices.push_back(col);
            matrix.values.push_back(value);
        }
    }

    matrix.rowOffsets[rows] =
        static_cast<int>(matrix.values.size());

    return matrix;
}

struct BenchmarkResult
{
    float averageMilliseconds;
    double gigaNonzerosPerSecond;
    double gigaFlops;
    bool correct;
};

BenchmarkResult runBenchmark(
    const CSRMatrix &matrix,
    const std::vector<float> &x,
    const std::vector<float> &cpuResult,
    int warmupIterations,
    int timedIterations)
{
    float *deviceValues = nullptr;
    int *deviceColumnIndices = nullptr;
    int *deviceRowOffsets = nullptr;
    float *deviceX = nullptr;
    float *deviceY = nullptr;

    const size_t valuesBytes =
        matrix.values.size() * sizeof(float);

    const size_t columnBytes =
        matrix.columnIndices.size() * sizeof(int);

    const size_t rowOffsetBytes =
        matrix.rowOffsets.size() * sizeof(int);

    const size_t xBytes =
        x.size() * sizeof(float);

    const size_t yBytes =
        static_cast<size_t>(matrix.rows) * sizeof(float);

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceValues),
        valuesBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceColumnIndices),
        columnBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceRowOffsets),
        rowOffsetBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceX),
        xBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceY),
        yBytes));

    CUDA_CHECK(cudaMemcpy(
        deviceValues,
        matrix.values.data(),
        valuesBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceColumnIndices,
        matrix.columnIndices.data(),
        columnBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceRowOffsets,
        matrix.rowOffsets.data(),
        rowOffsetBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceX,
        x.data(),
        xBytes,
        cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocks =
        (matrix.rows + threadsPerBlock - 1) /
        threadsPerBlock;

    csrSpmvKernel<<<blocks, threadsPerBlock>>>(
        matrix.rows,
        deviceRowOffsets,
        deviceColumnIndices,
        deviceValues,
        deviceX,
        deviceY);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpuResult(matrix.rows);

    CUDA_CHECK(cudaMemcpy(
        gpuResult.data(),
        deviceY,
        yBytes,
        cudaMemcpyDeviceToHost));

    bool correct =
        vectorsNearlyEqual(gpuResult, cpuResult, 1e-4f);

    for (int iteration = 0;
         iteration < warmupIterations;
         ++iteration)
    {
        csrSpmvKernel<<<blocks, threadsPerBlock>>>(
            matrix.rows,
            deviceRowOffsets,
            deviceColumnIndices,
            deviceValues,
            deviceX,
            deviceY);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int iteration = 0;
         iteration < timedIterations;
         ++iteration)
    {
        csrSpmvKernel<<<blocks, threadsPerBlock>>>(
            matrix.rows,
            deviceRowOffsets,
            deviceColumnIndices,
            deviceValues,
            deviceX,
            deviceY);
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

    double gigaNonzerosPerSecond =
        static_cast<double>(matrix.values.size()) /
        (averageMilliseconds * 1.0e6);

    double gigaFlops =
        2.0 * static_cast<double>(matrix.values.size()) /
        (averageMilliseconds * 1.0e6);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(deviceValues));
    CUDA_CHECK(cudaFree(deviceColumnIndices));
    CUDA_CHECK(cudaFree(deviceRowOffsets));
    CUDA_CHECK(cudaFree(deviceX));
    CUDA_CHECK(cudaFree(deviceY));

    return {
        averageMilliseconds,
        gigaNonzerosPerSecond,
        gigaFlops,
        correct};
}

int main()
{
    const int rows = 16384;
    const int cols = 16384;
    const int warmupIterations = 10;
    const int timedIterations = 100;

    const std::vector<int> nonzerosPerRowCases = {
        4, 16, 64, 256};

    std::vector<float> x(cols);

    for (int col = 0; col < cols; ++col)
    {
        x[col] =
            static_cast<float>((col % 13) + 1) * 0.05f;
    }

    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(
        &deviceProperties,
        0));

    std::cout
        << "GPU: " << deviceProperties.name << "\n"
        << "Matrix dimensions: "
        << rows << " x " << cols << "\n"
        << "Warm-up iterations: "
        << warmupIterations << "\n"
        << "Timed iterations: "
        << timedIterations << "\n\n";

    std::cout
        << std::left
        << std::setw(12) << "NNZ/row"
        << std::setw(14) << "Total NNZ"
        << std::setw(14) << "Avg ms"
        << std::setw(14) << "GNNZ/s"
        << std::setw(14) << "GFLOP/s"
        << "Correct\n";

    for (int nonzerosPerRow : nonzerosPerRowCases)
    {
        CSRMatrix matrix = generateUniformCsr(
            rows,
            cols,
            nonzerosPerRow);

        if (!matrix.isValid())
        {
            std::cerr << "Generated CSR matrix is invalid.\n";
            return 1;
        }

        std::vector<float> cpuResult =
            csrSpmv(matrix, x);

        BenchmarkResult result = runBenchmark(
            matrix,
            x,
            cpuResult,
            warmupIterations,
            timedIterations);

        std::cout
            << std::left
            << std::setw(12) << nonzerosPerRow
            << std::setw(14) << matrix.values.size()
            << std::setw(14) << std::fixed
            << std::setprecision(6)
            << result.averageMilliseconds
            << std::setw(14) << std::setprecision(3)
            << result.gigaNonzerosPerSecond
            << std::setw(14) << result.gigaFlops
            << (result.correct ? "PASS" : "FAIL")
            << '\n';

        if (!result.correct)
        {
            return 1;
        }
    }

    return 0;
}