#include "csr_spmm.h"

#include <cmath>
#include <stdexcept>

std::vector<float> csrSpmm(
    const CSRMatrix &sparseMatrix,
    const std::vector<float> &denseMatrix,
    int denseColumns)
{
    if (!sparseMatrix.isValid())
    {
        throw std::invalid_argument(
            "The CSR matrix is invalid.");
    }

    if (denseColumns <= 0)
    {
        throw std::invalid_argument(
            "The dense matrix must have at least one column.");
    }

    const size_t expectedDenseSize =
        static_cast<size_t>(sparseMatrix.cols) *
        denseColumns;

    if (denseMatrix.size() != expectedDenseSize)
    {
        throw std::invalid_argument(
            "Dense matrix size is incompatible with the CSR matrix.");
    }

    std::vector<float> output(
        static_cast<size_t>(sparseMatrix.rows) *
            denseColumns,
        0.0f);

    for (int row = 0; row < sparseMatrix.rows; ++row)
    {
        for (
            int position = sparseMatrix.rowOffsets[row];
            position < sparseMatrix.rowOffsets[row + 1];
            ++position)
        {
            int sharedIndex =
                sparseMatrix.columnIndices[position];

            float sparseValue =
                sparseMatrix.values[position];

            for (
                int outputColumn = 0;
                outputColumn < denseColumns;
                ++outputColumn)
            {
                output[row * denseColumns + outputColumn] +=
                    sparseValue *
                    denseMatrix[sharedIndex * denseColumns +
                                outputColumn];
            }
        }
    }

    return output;
}

std::vector<float> denseGemmReference(
    const std::vector<float> &leftMatrix,
    int leftRows,
    int sharedDimension,
    const std::vector<float> &rightMatrix,
    int rightColumns)
{
    if (
        leftRows <= 0 ||
        sharedDimension <= 0 ||
        rightColumns <= 0)
    {
        throw std::invalid_argument(
            "Matrix dimensions must be positive.");
    }

    const size_t expectedLeftSize =
        static_cast<size_t>(leftRows) *
        sharedDimension;

    const size_t expectedRightSize =
        static_cast<size_t>(sharedDimension) *
        rightColumns;

    if (
        leftMatrix.size() != expectedLeftSize ||
        rightMatrix.size() != expectedRightSize)
    {
        throw std::invalid_argument(
            "Dense matrix size does not match its dimensions.");
    }

    std::vector<float> output(
        static_cast<size_t>(leftRows) *
            rightColumns,
        0.0f);

    for (int row = 0; row < leftRows; ++row)
    {
        for (
            int sharedIndex = 0;
            sharedIndex < sharedDimension;
            ++sharedIndex)
        {
            float leftValue =
                leftMatrix[row * sharedDimension + sharedIndex];

            for (
                int outputColumn = 0;
                outputColumn < rightColumns;
                ++outputColumn)
            {
                output[row * rightColumns + outputColumn] +=
                    leftValue *
                    rightMatrix[sharedIndex * rightColumns +
                                outputColumn];
            }
        }
    }

    return output;
}

bool matricesNearlyEqual(
    const std::vector<float> &first,
    const std::vector<float> &second,
    float tolerance)
{
    if (first.size() != second.size())
    {
        return false;
    }

    for (size_t index = 0; index < first.size(); ++index)
    {
        if (
            std::fabs(first[index] - second[index]) >
            tolerance)
        {
            return false;
        }
    }

    return true;
}