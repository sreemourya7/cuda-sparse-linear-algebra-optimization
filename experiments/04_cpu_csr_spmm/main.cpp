#include "csr_spmm.h"

#include <iomanip>
#include <iostream>
#include <vector>

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

    std::vector<float> csrResult =
        csrSpmm(
            csrA,
            denseB,
            columnsB);

    std::vector<float> denseResult =
        denseGemmReference(
            denseA,
            rowsA,
            sharedDimension,
            denseB,
            columnsB);

    printMatrix(
        "CSR SpMM result",
        csrResult,
        rowsA,
        columnsB);

    std::cout << '\n';

    printMatrix(
        "Dense reference",
        denseResult,
        rowsA,
        columnsB);

    std::cout << '\n';

    printMatrix(
        "Expected result",
        expected,
        rowsA,
        columnsB);

    bool passed =
        matricesNearlyEqual(csrResult, denseResult) &&
        matricesNearlyEqual(csrResult, expected);

    if (!passed)
    {
        std::cerr
            << "\nCPU CSR SpMM validation: FAILED\n";

        return 1;
    }

    std::cout
        << "\nCPU CSR SpMM validation: PASSED\n";

    return 0;
}