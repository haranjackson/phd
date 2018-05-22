#ifndef DG1_H
#define DG1_H

#include "../../etc/globals.h"
#include "../../system/objects/gpr_objects.h"

MatN2_V rhs1(MatN2_Vr q, MatN2_Vr Ww, double dt, double dx, Par &MP);

Vec obj1(Vecr q, MatN2_Vr Ww, double dt, double dx, Par &MP);

void predictor1(Vecr qh, Vecr wh, double dt, double dx, bool STIFF,
                bool STIFF_IG, Par &MP, bVecr mask);

#endif // DG1_H
