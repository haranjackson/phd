#ifndef WENO_MATRICES_H
#define WENO_MATRICES_H

#include <vector>

#include "../../etc/types.h"
#include "../../scipy/poly.h"

std::vector<MatN_N> coefficient_matrices(const std::vector<poly> &basis);

MatN_N oscillation_indicator(const std::vector<poly> &basis);

#endif // WENO_MATRICES_H
