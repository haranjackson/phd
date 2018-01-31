#ifndef DG_H
#define DG_H

#include "../../etc/globals.h"
#include "../../system/objects/gpr_objects.h"

MatN2_V rhs1(MatN2_Vr q, MatN2_Vr Ww, double dt, double dx, Par &MP);

Vec obj1(Vecr q, MatN2_Vr Ww, double dt, double dx, Par &MP);

void predictor(Vecr qh, Vecr wh, int ndim, double dt, double dx, double dy,
               double dz, bool STIFF, bool HIDALGO, Par &MP);

#endif // DG_H
