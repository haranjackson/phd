#ifndef JACOBIANS_H
#define JACOBIANS_H

#include "../etc/types.h"
#include "objects/gpr_objects.h"

MatV_V dFdP(VecVr Q, int d, Par &MP);
MatV_V dPdQ(VecVr Q, Par &MP);

#endif // JACOBIANS_H
