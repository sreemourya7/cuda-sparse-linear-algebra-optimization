#pragma once

#include "../00_csr_representation/csr_matrix.h"

#include <vector>

std::vector<float> csrSpmv(
    const CSRMatrix &matrix,
    const std::vector<float> &x);

std::vector<float> denseSpmv(
    const std::vector<float> &matrix,
    int rows,
    int cols,
    const std::vector<float> &x);

bool vectorsNearlyEqual(
    const std::vector<float> &first,
    const std::vector<float> &second,
    float tolerance = 1e-5f);