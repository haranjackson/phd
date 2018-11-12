#ifndef RIEMANN_H
#define RIEMANN_H

#include "../../etc/types.h"
#include "../objects.h"
#include "eigenvecs.h"

MatV_V riemann_constraints(VecVr Q, double sgn, Par &MP);

void star_stepper(VecVr QL, VecVr QR, Par &MPL, Par &MPR);

VecV left_star_state(VecV QL_, VecV QR_, Par &MPL, Par &MPR, double dt, Vecr n);

#endif // RIEMANN_H
