#include "csr_matrix.h"

#include <iostream>
#include <vector>

template <typename T>
void printVector(const char *name, const std::vector<T> &data)
{
    std::cout << name << " = [";

    for (size_t i = 0; i < data.size(); ++i)
    {
        std::cout << data[i];

        if (i + 1 < data.size())
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

    const std::vector<float> dense = {
        1, 0, 0, 2, 0,
        0, 0, 3, 0, 0,
        4, 0, 5, 0, 6,
        0, 0, 0, 7, 0};

    const std::vector<float> expectedValues = {
        1, 2, 3, 4, 5, 6, 7};

    const std::vector<int> expectedColumns = {
        0, 3, 2, 0, 2, 4, 3};

    const std::vector<int> expectedRowOffsets = {
        0, 2, 3, 6, 7};

    CSRMatrix csr = CSRMatrix::fromDense(dense, rows, cols);

    printVector("values", csr.values);
    printVector("column_indices", csr.columnIndices);
    printVector("row_offsets", csr.rowOffsets);

    bool passed =
        csr.isValid() &&
        csr.values == expectedValues &&
        csr.columnIndices == expectedColumns &&
        csr.rowOffsets == expectedRowOffsets &&
        csr.toDense() == dense;

    if (!passed)
    {
        std::cerr << "\nCSR validation: FAILED\n";
        return 1;
    }

    std::cout << "\nCSR validation: PASSED\n";
    return 0;
}