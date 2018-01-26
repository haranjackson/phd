#ifndef BASIS_H
#define BASIS_H

#include "../etc/types.h"
#include "../scipy/poly.h"
#include <vector>

VecN scaled_nodes();

VecN scaled_weights();

std::vector<poly> basis_polys();

poly lagrange(Vecr x, int i);

Mat2_N end_values(const std::vector<poly> &basis);

MatN_N derivative_values(const std::vector<poly> &basis, const VecN nodes);

#endif // BASIS_H
