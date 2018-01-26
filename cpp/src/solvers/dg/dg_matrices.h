#ifndef DG_MATRICES_H
#define DG_MATRICES_H

#include "../../etc/types.h"

Mat kron(std::vector<Mat> &mats);

MatN_N end_value_products(const std::vector<poly> &basis);

MatN_N derivative_products(const std::vector<poly> &basis, const VecN nodes,
                           const VecN wghts);

#endif // DG_MATRICES_H
