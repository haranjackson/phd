#ifndef RIEMANN_H
#define RIEMANN_H

#include "../../etc/types.h"
#include "../objects/gpr_objects.h"
#include "eigenvecs.h"

struct StarStates {
  VecV QL_;
  VecV QR_;
};

MatV_V riemann_constraints(VecVr Q, double sgn, Par &MP);

void star_stepper(VecVr QL, VecVr QR, Par &MPL, Par &MPR);

StarStates star_states(VecV QL_, VecV QR_, Par &MPL, Par &MPR, double dt,
                       Vecr n);

#endif // RIEMANN_H
