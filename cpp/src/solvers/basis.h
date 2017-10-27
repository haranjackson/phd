#ifndef BASIS_H
#define BASIS_H

#include <vector>
#include "../etc/types.h"
#include "../scipy/poly.h"


Vecn scaled_nodes();

Vecn scaled_weights();

std::vector<poly> basis_polys();

poly lagrange(Vecr x, int i);

Mat2_n end_values(const std::vector<poly> & basis);

Matn_n derivative_values(const std::vector<poly> & basis, const Vecn nodes);


#endif // BASIS_H
