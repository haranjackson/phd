#ifndef DG_H
#define DG_H

#include "../../etc/types.h"
#include "../../system/objects.h"

void predictor(Vecr qh, Vecr wh, double dt, Vecr dX, bool STIFF, bool STIFF_IG,
               Par &MP, bVecr mask);

#endif // DG_H
