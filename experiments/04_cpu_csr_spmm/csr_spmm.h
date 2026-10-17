#pragma once

#include "../00_csr_representation/csr_matrix.h"

#include <vector>

std::vector<float> csrSpmm(
    const CSRMatrix &sparseMatrix,
    const std::vector<float> &denseMatrix,
    int denseColumns);

std::vector<float> denseGemmReference(
    const std::vector<float> &leftMatrix,
    int leftRows,
    int sharedDimension,
    const std::vector<float> &rightMatrix,
    int rightColumns);

bool matricesNearlyEqual(
    const std::vector<float> &first,
    const std::vector<float> &second,
    float tolerance = 1e-5f);
