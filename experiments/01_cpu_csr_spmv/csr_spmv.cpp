#include "csr_spmv.h"
// #include<bits/stdc++.h>
#include <cmath>
#include <stdexcept>

std::vector<float> csrSpmv(
    const CSRMatrix &matrix,
    const std::vector<float> &x)
{
    if (!matrix.isValid())
    {
        throw std::invalid_argument("The CSR matrix is invalid.");
    }

    if (x.size() != static_cast<size_t>(matrix.cols))
    {
        throw std::invalid_argument(
            "Vector size does not match the matrix column count.");
    }

    std::vector<float> y(matrix.rows, 0.0f);

    for (int row = 0; row < matrix.rows; ++row)
    {
        float sum = 0.0f;

        for (
            int position = matrix.rowOffsets[row];
            position < matrix.rowOffsets[row + 1];
            ++position)
        {
            int col = matrix.columnIndices[position];
            sum += matrix.values[position] * x[col];
        }

        y[row] = sum;
    }

    return y;
}

std::vector<float> denseSpmv(
    const std::vector<float> &matrix,
    int rows,
    int cols,
    const std::vector<float> &x)
{
    if (matrix.size() != static_cast<size_t>(rows * cols))
    {
        throw std::invalid_argument(
            "Dense matrix size does not match its dimensions.");
    }

    if (x.size() != static_cast<size_t>(cols))
    {
        throw std::invalid_argument(
            "Vector size does not match the matrix column count.");
    }

    std::vector<float> y(rows, 0.0f);

    for (int row = 0; row < rows; ++row)
    {
        float sum = 0.0f;

        for (int col = 0; col < cols; ++col)
        {
            sum += matrix[row * cols + col] * x[col];
        }

        y[row] = sum;
    }

    return y;
}

bool vectorsNearlyEqual(
    const std::vector<float> &first,
    const std::vector<float> &second,
    float tolerance)
{
    if (first.size() != second.size())
    {
        return false;
    }

    for (size_t i = 0; i < first.size(); ++i)
    {
        if (std::fabs(first[i] - second[i]) > tolerance)
        {
            return false;
        }
    }

    return true;
}
