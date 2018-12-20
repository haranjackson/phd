#ifndef JACOBIANS_H
#define JACOBIANS_H

#include "../etc/types.h"
#include "objects.h"

MatV_V dPdQ(VecVr Q, Par &MP);
MatV_V dFdP(VecVr Q, int d, Par &MP);

#endif // JACOBIANS_H
