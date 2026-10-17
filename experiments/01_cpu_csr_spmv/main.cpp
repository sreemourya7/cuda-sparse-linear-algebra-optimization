#include "csr_spmv.h"

#include <iostream>
#include <vector>

void printVector(
    const char *name,
    const std::vector<float> &values)
{
    std::cout << name << " = [";

    for (size_t i = 0; i < values.size(); ++i)
    {
        std::cout << values[i];

        if (i + 1 < values.size())
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

    const std::vector<float> denseMatrix = {
        1, 0, 0, 2, 0,
        0, 0, 3, 0, 0,
        4, 0, 5, 0, 6,
        0, 0, 0, 7, 0};

    const std::vector<float> x = {
        1, 2, 3, 4, 5};

    const std::vector<float> expected = {
        9, 9, 49, 28};

    CSRMatrix csr =
        CSRMatrix::fromDense(denseMatrix, rows, cols);

    std::vector<float> csrResult =
        csrSpmv(csr, x);

    std::vector<float> denseResult =
        denseSpmv(denseMatrix, rows, cols, x);

    printVector("Input vector", x);
    printVector("CSR result", csrResult);
    printVector("Dense result", denseResult);
    printVector("Expected result", expected);

    bool passed =
        vectorsNearlyEqual(csrResult, denseResult) &&
        vectorsNearlyEqual(csrResult, expected);

    if (!passed)
    {
        std::cerr << "\nCPU CSR SpMV validation: FAILED\n";
        return 1;
    }

    std::cout << "\nCPU CSR SpMV validation: PASSED\n";
    return 0;
}