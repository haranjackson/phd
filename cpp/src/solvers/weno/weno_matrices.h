#ifndef WENO_MATRICES_H
#define WENO_MATRICES_H


#include <vector>

#include "../../etc/types.h"
#include "../../scipy/poly.h"


std::vector<Matn_n> coefficient_matrices(const std::vector<poly> & basis);

Matn_n oscillation_indicator(const std::vector<poly> & basis);


#endif // WENO_MATRICES_H
