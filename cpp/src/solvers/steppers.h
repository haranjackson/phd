#ifndef SOLVERS_H
#define SOLVERS_H

#include "../etc/globals.h"
#include "../system/objects/gpr_objects.h"

void ader_stepper(Vecr u, Vecr ub, Vecr wh, Vecr qh, int ndim, Veci3r nX,
                  double dt, Vec3r dX, bool STIFF, int FLUX, bool PERR_FROB,
                  Par &MP);

void split_stepper(Vecr u, Vecr ub, Vecr wh, int ndim, Veci3r nX, double dt,
                   Vec3r dX, bool STRANG, bool HALF_STEP, int FLUX,
                   bool PERR_FROB, Par &MP);

#endif // SOLVERS_H
