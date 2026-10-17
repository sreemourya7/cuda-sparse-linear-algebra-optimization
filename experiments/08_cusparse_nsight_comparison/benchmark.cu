#include "../00_csr_representation/csr_matrix.h"
#include "../04_cpu_csr_spmm/csr_spmm.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void checkCuda(
    cudaError_t status,
    const char *expression,
    const char *file,
    int line)
{
    if (status != cudaSuccess)
    {
        std::cerr
            << "CUDA error at " << file << ":" << line
            << " for " << expression << ": "
            << cudaGetErrorString(status) << '\n';

        std::exit(EXIT_FAILURE);
    }
}

void checkCusparse(
    cusparseStatus_t status,
    const char *expression,
    const char *file,
    int line)
{
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cerr
            << "cuSPARSE error at " << file << ":" << line
            << " for " << expression
            << ". Status code: "
            << static_cast<int>(status) << '\n';

        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) \
    checkCuda((expression), #expression, __FILE__, __LINE__)

#define CUSPARSE_CHECK(expression) \
    checkCusparse((expression), #expression, __FILE__, __LINE__)

__global__ void csrSpmmBaselineKernel(
    int rows,
    int denseColumns,
    const int *rowOffsets,
    const int *columnIndices,
    const float *values,
    const float *denseMatrix,
    float *output)
{
    const int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    const int outputColumn =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || outputColumn >= denseColumns)
    {
        return;
    }

    float sum = 0.0f;

    const int rowStart = rowOffsets[row];
    const int rowEnd = rowOffsets[row + 1];

    for (int index = rowStart; index < rowEnd; ++index)
    {
        const int sparseColumn = columnIndices[index];

        sum +=
            values[index] *
            denseMatrix[sparseColumn * denseColumns +
                        outputColumn];
    }

    output[row * denseColumns + outputColumn] = sum;
}

__global__ void csrSpmmWarpRowKernel(
    int rows,
    int denseColumns,
    const int *rowOffsets,
    const int *columnIndices,
    const float *values,
    const float *denseMatrix,
    float *output)
{
    const int lane = threadIdx.x;
    const int warpInBlock = threadIdx.y;

    const int row =
        blockIdx.x * blockDim.y + warpInBlock;

    if (row >= rows)
    {
        return;
    }

    const int rowStart = rowOffsets[row];
    const int rowEnd = rowOffsets[row + 1];

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const int outputColumn0 = lane;
    const int outputColumn1 = lane + 32;
    const int outputColumn2 = lane + 64;
    const int outputColumn3 = lane + 96;

    for (int index = rowStart; index < rowEnd; ++index)
    {
        float sparseValue = 0.0f;
        int sparseColumn = 0;

        if (lane == 0)
        {
            sparseValue = values[index];
            sparseColumn = columnIndices[index];
        }

        sparseValue =
            __shfl_sync(0xffffffff, sparseValue, 0);

        sparseColumn =
            __shfl_sync(0xffffffff, sparseColumn, 0);

        const int denseRowOffset =
            sparseColumn * denseColumns;

        if (outputColumn0 < denseColumns)
        {
            sum0 +=
                sparseValue *
                denseMatrix[denseRowOffset + outputColumn0];
        }

        if (outputColumn1 < denseColumns)
        {
            sum1 +=
                sparseValue *
                denseMatrix[denseRowOffset + outputColumn1];
        }

        if (outputColumn2 < denseColumns)
        {
            sum2 +=
                sparseValue *
                denseMatrix[denseRowOffset + outputColumn2];
        }

        if (outputColumn3 < denseColumns)
        {
            sum3 +=
                sparseValue *
                denseMatrix[denseRowOffset + outputColumn3];
        }
    }

    const int outputRowOffset = row * denseColumns;

    if (outputColumn0 < denseColumns)
    {
        output[outputRowOffset + outputColumn0] = sum0;
    }

    if (outputColumn1 < denseColumns)
    {
        output[outputRowOffset + outputColumn1] = sum1;
    }

    if (outputColumn2 < denseColumns)
    {
        output[outputRowOffset + outputColumn2] = sum2;
    }

    if (outputColumn3 < denseColumns)
    {
        output[outputRowOffset + outputColumn3] = sum3;
    }
}

void launchBaseline(
    int rows,
    int denseColumns,
    const int *deviceRowOffsets,
    const int *deviceColumnIndices,
    const float *deviceValues,
    const float *deviceDenseMatrix,
    float *deviceOutput)
{
    const dim3 block(16, 16);

    const dim3 grid(
        (denseColumns + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y);

    csrSpmmBaselineKernel<<<grid, block>>>(
        rows,
        denseColumns,
        deviceRowOffsets,
        deviceColumnIndices,
        deviceValues,
        deviceDenseMatrix,
        deviceOutput);
}

void launchOptimized(
    int rows,
    int denseColumns,
    const int *deviceRowOffsets,
    const int *deviceColumnIndices,
    const float *deviceValues,
    const float *deviceDenseMatrix,
    float *deviceOutput)
{
    if (denseColumns < 32)
    {
        const int blockWidth = denseColumns;
        const int blockHeight = 256 / blockWidth;

        const dim3 block(
            blockWidth,
            blockHeight);

        const dim3 grid(
            1,
            (rows + blockHeight - 1) / blockHeight);

        csrSpmmBaselineKernel<<<grid, block>>>(
            rows,
            denseColumns,
            deviceRowOffsets,
            deviceColumnIndices,
            deviceValues,
            deviceDenseMatrix,
            deviceOutput);

        return;
    }

    if (denseColumns == 32)
    {
        const dim3 block(32, 8);

        const dim3 grid(
            1,
            (rows + block.y - 1) / block.y);

        csrSpmmBaselineKernel<<<grid, block>>>(
            rows,
            denseColumns,
            deviceRowOffsets,
            deviceColumnIndices,
            deviceValues,
            deviceDenseMatrix,
            deviceOutput);

        return;
    }

    const dim3 block(32, 8);

    const dim3 grid(
        (rows + block.y - 1) / block.y);

    csrSpmmWarpRowKernel<<<grid, block>>>(
        rows,
        denseColumns,
        deviceRowOffsets,
        deviceColumnIndices,
        deviceValues,
        deviceDenseMatrix,
        deviceOutput);
}

CSRMatrix generateUniformCsrMatrix(
    int rows,
    int columns,
    int nonzerosPerRow)
{
    CSRMatrix matrix;

    matrix.rows = rows;
    matrix.cols = columns;

    matrix.rowOffsets.resize(rows + 1, 0);

    matrix.values.reserve(
        static_cast<std::size_t>(rows) *
        nonzerosPerRow);

    matrix.columnIndices.reserve(
        static_cast<std::size_t>(rows) *
        nonzerosPerRow);

    for (int row = 0; row < rows; ++row)
    {
        for (int element = 0;
             element < nonzerosPerRow;
             ++element)
        {
            const int column =
                (row * 131 + element * 17) % columns;

            matrix.columnIndices.push_back(column);

            const float value =
                0.25f +
                0.05f *
                    static_cast<float>(
                        (row + element) % 11);

            matrix.values.push_back(value);
        }

        matrix.rowOffsets[row + 1] =
            static_cast<int>(matrix.values.size());
    }

    return matrix;
}

std::vector<float> generateDenseMatrix(
    int rows,
    int columns)
{
    std::vector<float> matrix(
        static_cast<std::size_t>(rows) * columns);

    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0;
             column < columns;
             ++column)
        {
            matrix[static_cast<std::size_t>(row) *
                       columns +
                   column] =
                0.01f *
                static_cast<float>(
                    1 + ((row * 7 + column * 3) % 17));
        }
    }

    return matrix;
}

bool matricesAreClose(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    float absoluteTolerance,
    float relativeTolerance)
{
    if (expected.size() != actual.size())
    {
        return false;
    }

    for (std::size_t index = 0;
         index < expected.size();
         ++index)
    {
        const float difference =
            std::fabs(expected[index] - actual[index]);

        const float allowedDifference =
            absoluteTolerance +
            relativeTolerance *
                std::fabs(expected[index]);

        if (difference > allowedDifference)
        {
            std::cerr
                << "Mismatch at index " << index
                << ": expected " << expected[index]
                << ", received " << actual[index]
                << ", difference " << difference
                << '\n';

            return false;
        }
    }

    return true;
}

float timeCustomKernel(
    bool optimized,
    int rows,
    int denseColumns,
    const int *deviceRowOffsets,
    const int *deviceColumnIndices,
    const float *deviceValues,
    const float *deviceDenseMatrix,
    float *deviceOutput,
    int warmupIterations,
    int timedIterations)
{
    for (int iteration = 0;
         iteration < warmupIterations;
         ++iteration)
    {
        if (optimized)
        {
            launchOptimized(
                rows,
                denseColumns,
                deviceRowOffsets,
                deviceColumnIndices,
                deviceValues,
                deviceDenseMatrix,
                deviceOutput);
        }
        else
        {
            launchBaseline(
                rows,
                denseColumns,
                deviceRowOffsets,
                deviceColumnIndices,
                deviceValues,
                deviceDenseMatrix,
                deviceOutput);
        }
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
        if (optimized)
        {
            launchOptimized(
                rows,
                denseColumns,
                deviceRowOffsets,
                deviceColumnIndices,
                deviceValues,
                deviceDenseMatrix,
                deviceOutput);
        }
        else
        {
            launchBaseline(
                rows,
                denseColumns,
                deviceRowOffsets,
                deviceColumnIndices,
                deviceValues,
                deviceDenseMatrix,
                deviceOutput);
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

    return totalMilliseconds /
           static_cast<float>(timedIterations);
}

void runCusparseSpmm(
    cusparseHandle_t handle,
    cusparseSpMatDescr_t sparseMatrix,
    cusparseDnMatDescr_t denseMatrix,
    cusparseDnMatDescr_t outputMatrix,
    void *externalBuffer)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUSPARSE_CHECK(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        sparseMatrix,
        denseMatrix,
        &beta,
        outputMatrix,
        CUDA_R_32F,
        CUSPARSE_SPMM_CSR_ALG2,
        externalBuffer));
}

float timeCusparse(
    cusparseHandle_t handle,
    cusparseSpMatDescr_t sparseMatrix,
    cusparseDnMatDescr_t denseMatrix,
    cusparseDnMatDescr_t outputMatrix,
    void *externalBuffer,
    int warmupIterations,
    int timedIterations)
{
    for (int iteration = 0;
         iteration < warmupIterations;
         ++iteration)
    {
        runCusparseSpmm(
            handle,
            sparseMatrix,
            denseMatrix,
            outputMatrix,
            externalBuffer);
    }

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
        runCusparseSpmm(
            handle,
            sparseMatrix,
            denseMatrix,
            outputMatrix,
            externalBuffer);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMilliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMilliseconds,
        start,
        stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMilliseconds /
           static_cast<float>(timedIterations);
}

struct ComparisonResult
{
    float baselineMilliseconds = 0.0f;
    float optimizedMilliseconds = 0.0f;
    float cusparseMilliseconds = 0.0f;

    double baselineGflops = 0.0;
    double optimizedGflops = 0.0;
    double cusparseGflops = 0.0;

    double customSpeedup = 0.0;
    double cusparsePercentage = 0.0;

    bool baselineCorrect = false;
    bool optimizedCorrect = false;
    bool cusparseCorrect = false;
};

ComparisonResult runComparison(
    const CSRMatrix &sparseMatrix,
    const std::vector<float> &denseMatrix,
    int denseColumns,
    const std::vector<float> &cpuReference,
    int warmupIterations,
    int timedIterations,
    const std::string &profileMode)
{
    const std::size_t rowOffsetBytes =
        sparseMatrix.rowOffsets.size() *
        sizeof(int);

    const std::size_t columnIndexBytes =
        sparseMatrix.columnIndices.size() *
        sizeof(int);

    const std::size_t valueBytes =
        sparseMatrix.values.size() *
        sizeof(float);

    const std::size_t denseBytes =
        denseMatrix.size() *
        sizeof(float);

    const std::size_t outputElementCount =
        static_cast<std::size_t>(
            sparseMatrix.rows) *
        denseColumns;

    const std::size_t outputBytes =
        outputElementCount *
        sizeof(float);

    int *deviceRowOffsets = nullptr;
    int *deviceColumnIndices = nullptr;

    float *deviceValues = nullptr;
    float *deviceDenseMatrix = nullptr;
    float *deviceOutput = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceRowOffsets),
        rowOffsetBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceColumnIndices),
        columnIndexBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceValues),
        valueBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceDenseMatrix),
        denseBytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&deviceOutput),
        outputBytes));

    CUDA_CHECK(cudaMemcpy(
        deviceRowOffsets,
        sparseMatrix.rowOffsets.data(),
        rowOffsetBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceColumnIndices,
        sparseMatrix.columnIndices.data(),
        columnIndexBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceValues,
        sparseMatrix.values.data(),
        valueBytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        deviceDenseMatrix,
        denseMatrix.data(),
        denseBytes,
        cudaMemcpyHostToDevice));

    cusparseHandle_t cusparseHandle = nullptr;
    cusparseSpMatDescr_t sparseDescriptor = nullptr;
    cusparseDnMatDescr_t denseDescriptor = nullptr;
    cusparseDnMatDescr_t outputDescriptor = nullptr;

    CUSPARSE_CHECK(cusparseCreate(
        &cusparseHandle));

    CUSPARSE_CHECK(cusparseCreateCsr(
        &sparseDescriptor,
        static_cast<int64_t>(sparseMatrix.rows),
        static_cast<int64_t>(sparseMatrix.cols),
        static_cast<int64_t>(sparseMatrix.values.size()),
        deviceRowOffsets,
        deviceColumnIndices,
        deviceValues,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &denseDescriptor,
        static_cast<int64_t>(sparseMatrix.cols),
        static_cast<int64_t>(denseColumns),
        static_cast<int64_t>(denseColumns),
        deviceDenseMatrix,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &outputDescriptor,
        static_cast<int64_t>(sparseMatrix.rows),
        static_cast<int64_t>(denseColumns),
        static_cast<int64_t>(denseColumns),
        deviceOutput,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::size_t workspaceBytes = 0;

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        sparseDescriptor,
        denseDescriptor,
        &beta,
        outputDescriptor,
        CUDA_R_32F,
        CUSPARSE_SPMM_CSR_ALG2,
        &workspaceBytes));

    void *deviceWorkspace = nullptr;

    if (workspaceBytes > 0)
    {
        CUDA_CHECK(cudaMalloc(
            &deviceWorkspace,
            workspaceBytes));
    }

    auto cleanup = [&]()
    {
        if (deviceWorkspace != nullptr)
        {
            CUDA_CHECK(cudaFree(deviceWorkspace));
        }

        CUSPARSE_CHECK(cusparseDestroyDnMat(
            outputDescriptor));

        CUSPARSE_CHECK(cusparseDestroyDnMat(
            denseDescriptor));

        CUSPARSE_CHECK(cusparseDestroySpMat(
            sparseDescriptor));

        CUSPARSE_CHECK(cusparseDestroy(
            cusparseHandle));

        CUDA_CHECK(cudaFree(deviceOutput));
        CUDA_CHECK(cudaFree(deviceDenseMatrix));
        CUDA_CHECK(cudaFree(deviceValues));
        CUDA_CHECK(cudaFree(deviceColumnIndices));
        CUDA_CHECK(cudaFree(deviceRowOffsets));
    };

    if (!profileMode.empty())
    {
        if (profileMode == "baseline")
        {
            launchBaseline(
                sparseMatrix.rows,
                denseColumns,
                deviceRowOffsets,
                deviceColumnIndices,
                deviceValues,
                deviceDenseMatrix,
                deviceOutput);
        }
        else if (profileMode == "optimized")
        {
            launchOptimized(
                sparseMatrix.rows,
                denseColumns,
                deviceRowOffsets,
                deviceColumnIndices,
                deviceValues,
                deviceDenseMatrix,
                deviceOutput);
        }
        else
        {
            runCusparseSpmm(
                cusparseHandle,
                sparseDescriptor,
                denseDescriptor,
                outputDescriptor,
                deviceWorkspace);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cleanup();

        return {};
    }

    std::vector<float> gpuOutput(
        outputElementCount);

    ComparisonResult result;

    launchBaseline(
        sparseMatrix.rows,
        denseColumns,
        deviceRowOffsets,
        deviceColumnIndices,
        deviceValues,
        deviceDenseMatrix,
        deviceOutput);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        gpuOutput.data(),
        deviceOutput,
        outputBytes,
        cudaMemcpyDeviceToHost));

    result.baselineCorrect =
        matricesAreClose(
            cpuReference,
            gpuOutput,
            1.0e-3f,
            1.0e-4f);

    launchOptimized(
        sparseMatrix.rows,
        denseColumns,
        deviceRowOffsets,
        deviceColumnIndices,
        deviceValues,
        deviceDenseMatrix,
        deviceOutput);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        gpuOutput.data(),
        deviceOutput,
        outputBytes,
        cudaMemcpyDeviceToHost));

    result.optimizedCorrect =
        matricesAreClose(
            cpuReference,
            gpuOutput,
            1.0e-3f,
            1.0e-4f);

    runCusparseSpmm(
        cusparseHandle,
        sparseDescriptor,
        denseDescriptor,
        outputDescriptor,
        deviceWorkspace);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        gpuOutput.data(),
        deviceOutput,
        outputBytes,
        cudaMemcpyDeviceToHost));

    result.cusparseCorrect =
        matricesAreClose(
            cpuReference,
            gpuOutput,
            1.0e-3f,
            1.0e-4f);

    result.baselineMilliseconds =
        timeCustomKernel(
            false,
            sparseMatrix.rows,
            denseColumns,
            deviceRowOffsets,
            deviceColumnIndices,
            deviceValues,
            deviceDenseMatrix,
            deviceOutput,
            warmupIterations,
            timedIterations);

    result.optimizedMilliseconds =
        timeCustomKernel(
            true,
            sparseMatrix.rows,
            denseColumns,
            deviceRowOffsets,
            deviceColumnIndices,
            deviceValues,
            deviceDenseMatrix,
            deviceOutput,
            warmupIterations,
            timedIterations);

    result.cusparseMilliseconds =
        timeCusparse(
            cusparseHandle,
            sparseDescriptor,
            denseDescriptor,
            outputDescriptor,
            deviceWorkspace,
            warmupIterations,
            timedIterations);

    const double floatingPointOperations =
        2.0 *
        static_cast<double>(
            sparseMatrix.values.size()) *
        static_cast<double>(denseColumns);

    result.baselineGflops =
        floatingPointOperations /
        (static_cast<double>(
             result.baselineMilliseconds) *
         1.0e6);

    result.optimizedGflops =
        floatingPointOperations /
        (static_cast<double>(
             result.optimizedMilliseconds) *
         1.0e6);

    result.cusparseGflops =
        floatingPointOperations /
        (static_cast<double>(
             result.cusparseMilliseconds) *
         1.0e6);

    result.customSpeedup =
        static_cast<double>(
            result.baselineMilliseconds) /
        static_cast<double>(
            result.optimizedMilliseconds);

    result.cusparsePercentage =
        100.0 *
        result.optimizedGflops /
        result.cusparseGflops;

    cleanup();

    return result;
}

int main(int argc, char **argv)
{
    const int rows = 8192;
    const int sparseColumns = 8192;

    const int warmupIterations = 10;
    const int timedIterations = 100;

    cudaDeviceProp deviceProperties;

    CUDA_CHECK(cudaGetDeviceProperties(
        &deviceProperties,
        0));

    std::cout
        << "GPU: "
        << deviceProperties.name
        << '\n';

    if (argc == 3 &&
        std::string(argv[1]) == "--profile")
    {
        const std::string profileMode = argv[2];

        if (profileMode != "baseline" &&
            profileMode != "optimized" &&
            profileMode != "cusparse")
        {
            std::cerr
                << "Profile mode must be baseline, "
                << "optimized, or cusparse.\n";

            return EXIT_FAILURE;
        }

        const int nonzerosPerRow = 128;
        const int denseColumns = 128;

        const CSRMatrix sparseMatrix =
            generateUniformCsrMatrix(
                rows,
                sparseColumns,
                nonzerosPerRow);

        const std::vector<float> denseMatrix =
            generateDenseMatrix(
                sparseColumns,
                denseColumns);

        runComparison(
            sparseMatrix,
            denseMatrix,
            denseColumns,
            {},
            0,
            0,
            profileMode);

        std::cout
            << "Profile workload completed for: "
            << profileMode
            << '\n';

        return EXIT_SUCCESS;
    }

    const std::vector<int> nonzerosPerRowCases =
        {
            8,
            32,
            128};

    const std::vector<int> denseColumnCases =
        {
            1,
            8,
            32,
            128};

    std::cout
        << "Sparse matrix dimensions: "
        << rows << " x " << sparseColumns
        << '\n';

    std::cout
        << "Warm-up iterations: "
        << warmupIterations
        << '\n';

    std::cout
        << "Timed iterations: "
        << timedIterations
        << "\n\n";

    std::cout
        << std::left
        << std::setw(8) << "NNZ/row"
        << std::setw(6) << "N"
        << std::setw(11) << "Base ms"
        << std::setw(11) << "Opt ms"
        << std::setw(11) << "cuSP ms"
        << std::setw(10) << "Base GF"
        << std::setw(10) << "Opt GF"
        << std::setw(10) << "cuSP GF"
        << std::setw(9) << "Speedup"
        << std::setw(9) << "%cuSP"
        << "Correct"
        << '\n';

    for (const int nonzerosPerRow :
         nonzerosPerRowCases)
    {
        const CSRMatrix sparseMatrix =
            generateUniformCsrMatrix(
                rows,
                sparseColumns,
                nonzerosPerRow);

        for (const int denseColumns :
             denseColumnCases)
        {
            const std::vector<float> denseMatrix =
                generateDenseMatrix(
                    sparseColumns,
                    denseColumns);

            const std::vector<float> cpuReference =
                csrSpmm(
                    sparseMatrix,
                    denseMatrix,
                    denseColumns);

            const ComparisonResult result =
                runComparison(
                    sparseMatrix,
                    denseMatrix,
                    denseColumns,
                    cpuReference,
                    warmupIterations,
                    timedIterations,
                    "");

            const bool allCorrect =
                result.baselineCorrect &&
                result.optimizedCorrect &&
                result.cusparseCorrect;

            std::cout
                << std::left
                << std::setw(8) << nonzerosPerRow
                << std::setw(6) << denseColumns
                << std::fixed
                << std::setprecision(6)
                << std::setw(11)
                << result.baselineMilliseconds
                << std::setw(11)
                << result.optimizedMilliseconds
                << std::setw(11)
                << result.cusparseMilliseconds
                << std::setprecision(3)
                << std::setw(10)
                << result.baselineGflops
                << std::setw(10)
                << result.optimizedGflops
                << std::setw(10)
                << result.cusparseGflops
                << std::setw(9)
                << result.customSpeedup
                << std::setw(9)
                << result.cusparsePercentage
                << (allCorrect ? "PASS" : "FAIL")
                << '\n';
        }
    }

    return EXIT_SUCCESS;
}