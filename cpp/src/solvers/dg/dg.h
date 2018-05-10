#ifndef DG_H
#define DG_H

#include "../../etc/globals.h"
#include "../../system/objects/gpr_objects.h"

MatN2_V rhs1(MatN2_Vr q, MatN2_Vr Ww, double dt, double dx, Par &MP);

MatN3_V rhs2(MatN3_Vr q, MatN3_Vr Ww, double dt, double dx, double dy, Par &MP);

Vec obj1(Vecr q, MatN2_Vr Ww, double dt, double dx, Par &MP);

Vec obj2(Vecr q, MatN3_Vr Ww, double dt, double dx, double dy, Par &MP);

void predictor(Vecr qh, Vecr wh, int ndim, double dt, Vec3r dX, bool STIFF,
               bool HIDALGO, Par &MP, bVecr mask);

#endif // DG_H
