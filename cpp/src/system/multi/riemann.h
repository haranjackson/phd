#ifndef RIEMANN_H
#define RIEMANN_H

#include "../../etc/types.h"
#include "../objects/gpr_objects.h"
#include "eigenvecs.h"

MatBV riemann_constraints(VecBVr Q, double sgn, Par &MP);

std::vector<VecV> star_states(VecVr QL, VecVr QR, Par &MPL, Par &MPR,
                              double dt);

#endif // RIEMANN_H
