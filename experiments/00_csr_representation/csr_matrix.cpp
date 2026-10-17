#include "csr_matrix.h"

#include <stdexcept>

CSRMatrix CSRMatrix::fromDense(
    const std::vector<float> &dense,
    int rows,
    int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }

    if (dense.size() != static_cast<size_t>(rows * cols))
    {
        throw std::invalid_argument(
            "Dense data size does not match the matrix dimensions.");
    }

    CSRMatrix csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.rowOffsets.push_back(0);

    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            float value = dense[row * cols + col];

            if (value != 0.0f)
            {
                csr.values.push_back(value);
                csr.columnIndices.push_back(col);
            }
        }

        csr.rowOffsets.push_back(
            static_cast<int>(csr.values.size()));
    }

    return csr;
}

std::vector<float> CSRMatrix::toDense() const
{
    if (!isValid())
    {
        throw std::runtime_error("Invalid CSR matrix.");
    }

    std::vector<float> dense(rows * cols, 0.0f);

    for (int row = 0; row < rows; ++row)
    {
        for (
            int position = rowOffsets[row];
            position < rowOffsets[row + 1];
            ++position)
        {
            int col = columnIndices[position];
            dense[row * cols + col] = values[position];
        }
    }

    return dense;
}

bool CSRMatrix::isValid() const
{
    if (rows <= 0 || cols <= 0)
    {
        return false;
    }

    if (values.size() != columnIndices.size())
    {
        return false;
    }

    if (rowOffsets.size() != static_cast<size_t>(rows + 1))
    {
        return false;
    }

    if (rowOffsets.front() != 0)
    {
        return false;
    }

    if (rowOffsets.back() != static_cast<int>(values.size()))
    {
        return false;
    }

    for (int row = 0; row < rows; ++row)
    {
        if (rowOffsets[row] > rowOffsets[row + 1])
        {
            return false;
        }
    }

    for (int col : columnIndices)
    {
        if (col < 0 || col >= cols)
        {
            return false;
        }
    }

    return true;
}