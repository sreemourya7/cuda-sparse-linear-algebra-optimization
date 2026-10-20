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

__global__ void scalarCsrSpmmKernel(
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

__global__ void warpRowCsrSpmmKernel(
    int rows,
    int denseColumns,
    const int *rowOffsets,
    const int *columnIndices,
    const float *sparseValues,
    const float *denseMatrix,
    float *output)
{
    const int globalThread =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int lane =
        threadIdx.x & 31;

    const int warpInGrid =
        globalThread >> 5;

    const int row = warpInGrid;

    if (row >= rows)
    {
        return;
    }

    int rowStart = 0;
    int rowEnd = 0;

    if (lane == 0)
    {
        rowStart = rowOffsets[row];
        rowEnd = rowOffsets[row + 1];
    }

    const unsigned int fullMask = 0xffffffffu;

    rowStart =
        __shfl_sync(fullMask, rowStart, 0);

    rowEnd =
        __shfl_sync(fullMask, rowEnd, 0);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (
        int position = rowStart;
        position < rowEnd;
        ++position)
    {
        int sharedIndex = 0;
        float sparseValue = 0.0f;

        if (lane == 0)
        {
            sharedIndex =
                columnIndices[position];

            sparseValue =
                sparseValues[position];
        }

        sharedIndex =
            __shfl_sync(
                fullMask,
                sharedIndex,
                0);

        sparseValue =
            __shfl_sync(
                fullMask,
                sparseValue,
                0);

        const int column0 = lane;
        const int column1 = lane + 32;
        const int column2 = lane + 64;
        const int column3 = lane + 96;

        if (column0 < denseColumns)
        {
            sum0 +=
                sparseValue *
                denseMatrix[sharedIndex * denseColumns +
                            column0];
        }

        if (column1 < denseColumns)
        {
            sum1 +=
                sparseValue *
                denseMatrix[sharedIndex * denseColumns +
                            column1];
        }

        if (column2 < denseColumns)
        {
            sum2 +=
                sparseValue *
                denseMatrix[sharedIndex * denseColumns +
                            column2];
        }

        if (column3 < denseColumns)
        {
            sum3 +=
                sparseValue *
                denseMatrix[sharedIndex * denseColumns +
                            column3];
        }
    }

    const int column0 = lane;
    const int column1 = lane + 32;
    const int column2 = lane + 64;
    const int column3 = lane + 96;

    if (column0 < denseColumns)
    {
        output[row * denseColumns + column0] = sum0;
    }

    if (column1 < denseColumns)
    {
        output[row * denseColumns + column1] = sum1;
    }

    if (column2 < denseColumns)
    {
        output[row * denseColumns + column2] = sum2;
    }

    if (column3 < denseColumns)
    {
        output[row * denseColumns + column3] = sum3;
    }
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

void launchBaseline(
    int rows,
    int denseColumns,
    const int *rowOffsets,
    const int *columnIndices,
    const float *sparseValues,
    const float *denseMatrix,
    float *output)
{
    const dim3 blockDimensions(16, 16);

    const dim3 gridDimensions(
        (denseColumns + blockDimensions.x - 1) /
            blockDimensions.x,
        (rows + blockDimensions.y - 1) /
            blockDimensions.y);

    scalarCsrSpmmKernel<<<
        gridDimensions,
        blockDimensions>>>(
        rows,
        denseColumns,
        rowOffsets,
        columnIndices,
        sparseValues,
        denseMatrix,
        output);
}

void launchOptimized(
    int rows,
    int denseColumns,
    const int *rowOffsets,
    const int *columnIndices,
    const float *sparseValues,
    const float *denseMatrix,
    float *output)
{
    if (denseColumns < 32)
    {
        const int blockWidth = denseColumns;
        const int blockHeight = 256 / blockWidth;

        const dim3 blockDimensions(
            blockWidth,
            blockHeight);

        const dim3 gridDimensions(
            1,
            (rows + blockHeight - 1) /
                blockHeight);

        scalarCsrSpmmKernel<<<
            gridDimensions,
            blockDimensions>>>(
            rows,
            denseColumns,
            rowOffsets,
            columnIndices,
            sparseValues,
            denseMatrix,
            output);

        return;
    }

    if (denseColumns == 32)
    {
        const dim3 blockDimensions(32, 8);

        const dim3 gridDimensions(
            1,
            (rows + blockDimensions.y - 1) /
                blockDimensions.y);

        scalarCsrSpmmKernel<<<
            gridDimensions,
            blockDimensions>>>(
            rows,
            denseColumns,
            rowOffsets,
            columnIndices,
            sparseValues,
            denseMatrix,
            output);

        return;
    }

    const int threadsPerBlock = 256;
    const int warpsPerBlock =
        threadsPerBlock / 32;

    const int blocks =
        (rows + warpsPerBlock - 1) /
        warpsPerBlock;

    warpRowCsrSpmmKernel<<<
        blocks,
        threadsPerBlock>>>(
        rows,
        denseColumns,
        rowOffsets,
        columnIndices,
        sparseValues,
        denseMatrix,
        output);
}

float timeKernel(
    bool optimized,
    int rows,
    int denseColumns,
    const int *rowOffsets,
    const int *columnIndices,
    const float *sparseValues,
    const float *denseMatrix,
    float *output,
    int warmupIterations,
    int timedIterations)
{
    for (
        int iteration = 0;
        iteration < warmupIterations;
        ++iteration)
    {
        if (optimized)
        {
            launchOptimized(
                rows,
                denseColumns,
                rowOffsets,
                columnIndices,
                sparseValues,
                denseMatrix,
                output);
        }
        else
        {
            launchBaseline(
                rows,
                denseColumns,
                rowOffsets,
                columnIndices,
                sparseValues,
                denseMatrix,
                output);
        }
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
        if (optimized)
        {
            launchOptimized(
                rows,
                denseColumns,
                rowOffsets,
                columnIndices,
                sparseValues,
                denseMatrix,
                output);
        }
        else
        {
            launchBaseline(
                rows,
                denseColumns,
                rowOffsets,
                columnIndices,
                sparseValues,
                denseMatrix,
                output);
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMilliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMilliseconds,
        start,
        stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMilliseconds / timedIterations;
}

struct ComparisonResult
{
    float baselineMilliseconds;
    float optimizedMilliseconds;
    double baselineGigaFlops;
    double optimizedGigaFlops;
    double speedup;
    bool correct;
};

ComparisonResult runComparison(
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

    std::vector<float> baselineResult(
        outputElements);

    std::vector<float> optimizedResult(
        outputElements);

    launchBaseline(
        sparseMatrix.rows,
        denseColumns,
        deviceRowOffsets,
        deviceColumnIndices,
        deviceSparseValues,
        deviceDenseMatrix,
        deviceOutput);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        baselineResult.data(),
        deviceOutput,
        outputBytes,
        cudaMemcpyDeviceToHost));

    launchOptimized(
        sparseMatrix.rows,
        denseColumns,
        deviceRowOffsets,
        deviceColumnIndices,
        deviceSparseValues,
        deviceDenseMatrix,
        deviceOutput);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        optimizedResult.data(),
        deviceOutput,
        outputBytes,
        cudaMemcpyDeviceToHost));

    bool correct =
        matricesNearlyEqual(
            baselineResult,
            cpuResult,
            1e-3f) &&
        matricesNearlyEqual(
            optimizedResult,
            cpuResult,
            1e-3f) &&
        matricesNearlyEqual(
            baselineResult,
            optimizedResult,
            1e-3f);

    float baselineMilliseconds =
        timeKernel(
            false,
            sparseMatrix.rows,
            denseColumns,
            deviceRowOffsets,
            deviceColumnIndices,
            deviceSparseValues,
            deviceDenseMatrix,
            deviceOutput,
            warmupIterations,
            timedIterations);

    float optimizedMilliseconds =
        timeKernel(
            true,
            sparseMatrix.rows,
            denseColumns,
            deviceRowOffsets,
            deviceColumnIndices,
            deviceSparseValues,
            deviceDenseMatrix,
            deviceOutput,
            warmupIterations,
            timedIterations);

    double totalOperations =
        2.0 *
        static_cast<double>(
            sparseMatrix.values.size()) *
        denseColumns;

    double baselineGigaFlops =
        totalOperations /
        (baselineMilliseconds * 1.0e6);

    double optimizedGigaFlops =
        totalOperations /
        (optimizedMilliseconds * 1.0e6);

    double speedup =
        baselineMilliseconds /
        optimizedMilliseconds;

    CUDA_CHECK(cudaFree(deviceSparseValues));
    CUDA_CHECK(cudaFree(deviceColumnIndices));
    CUDA_CHECK(cudaFree(deviceRowOffsets));
    CUDA_CHECK(cudaFree(deviceDenseMatrix));
    CUDA_CHECK(cudaFree(deviceOutput));

    return {
        baselineMilliseconds,
        optimizedMilliseconds,
        baselineGigaFlops,
        optimizedGigaFlops,
        speedup,
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
        << std::setw(9) << "NNZ/row"
        << std::setw(6) << "N"
        << std::setw(12) << "Base ms"
        << std::setw(12) << "Opt ms"
        << std::setw(12) << "Base GF"
        << std::setw(12) << "Opt GF"
        << std::setw(10) << "Speedup"
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

            ComparisonResult result =
                runComparison(
                    sparseMatrix,
                    denseMatrix,
                    denseColumns,
                    cpuResult,
                    warmupIterations,
                    timedIterations);

            std::cout
                << std::left
                << std::setw(9)
                << nonzerosPerRow
                << std::setw(6)
                << denseColumns
                << std::setw(12)
                << std::fixed
                << std::setprecision(6)
                << result.baselineMilliseconds
                << std::setw(12)
                << result.optimizedMilliseconds
                << std::setw(12)
                << std::setprecision(3)
                << result.baselineGigaFlops
                << std::setw(12)
                << result.optimizedGigaFlops
                << std::setw(10)
                << result.speedup
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
