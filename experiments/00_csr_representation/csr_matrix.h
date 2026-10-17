#pragma once

#include <vector>

struct CSRMatrix
{
    int rows;
    int cols;

    std::vector<float> values;
    std::vector<int> columnIndices;
    std::vector<int> rowOffsets;

    static CSRMatrix fromDense(
        const std::vector<float> &dense,
        int rows,
        int cols);

    std::vector<float> toDense() const;

    bool isValid() const;
};