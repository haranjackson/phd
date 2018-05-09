#ifndef SOLVERS_H
#define SOLVERS_H

#include "../etc/globals.h"
#include "../system/objects/gpr_objects.h"

void ader_stepper(Vecr u, Vecr ub, Vecr wh, Vecr qh, int ndim, iVec3r nX,
                  double dt, Vec3r dX, bool STIFF, int FLUX, Par &MP,
                  bVecr mask);

void split_stepper(Vecr u, Vecr ub, Vecr wh, int ndim, iVec3r nX, double dt,
                   Vec3r dX, bool HALF_STEP, int FLUX, Par &MP, bVecr mask);

#endif // SOLVERS_H
