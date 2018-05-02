#ifndef RIEMANN_H
#define RIEMANN_H

#include "../../etc/types.h"
#include "../objects/gpr_objects.h"
#include "eigenvecs.h"

MatV_V riemann_constraints(VecVr Q, double sgn, Par &MP);

std::vector<VecV> star_states(VecV QL_, VecV QR_, Par &MPL, Par &MPR,
                              double dt);

#endif // RIEMANN_H
