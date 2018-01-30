#ifndef SOLVERS_H
#define SOLVERS_H

#include "../etc/globals.h"
#include "../system/objects/gpr_objects.h"

void ader_stepper(Vecr u, Vecr ub, Vecr wh, Vecr qh, int ndim, int nx, int ny,
                  int nz, double dt, double dx, double dy, double dz,
                  bool STIFF, bool OSHER, bool PERR_FROB, Par &MP);

void split_stepper(Vecr u, Vecr ub, Vecr wh, int ndim, int nx, int ny, int nz,
                   double dt, double dx, double dy, double dz, bool STRANG,
                   bool HALF_STEP, bool OSHER, bool PERR_FROB, Par &MP);

#endif // SOLVERS_H
