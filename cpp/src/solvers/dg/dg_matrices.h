#ifndef DG_MATRICES_H
#define DG_MATRICES_H

#include "../../etc/types.h"


Mat kron(std::vector<Mat> & mats);

Matn_n end_value_products(const std::vector<poly> & basis);

Matn_n derivative_products(const std::vector<poly> & basis,
                           const Vecn nodes, const Vecn wghts);


#endif // DG_MATRICES_H
