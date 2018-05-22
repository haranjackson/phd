#ifndef DG2_H
#define DG2_H

#include "../../etc/globals.h"
#include "../../system/objects/gpr_objects.h"

MatN3_V rhs2(MatN3_Vr q, MatN3_Vr Ww, double dt, double dx, double dy, Par &MP);

Vec obj2(Vecr q, MatN3_Vr Ww, double dt, double dx, double dy, Par &MP);

void predictor2(Vecr qh, Vecr wh, double dt, double dx, double dy, bool STIFF,
                bool STIFF_IG, Par &MP, bVecr mask);

#endif // DG2_H
