#include "../etc/globals.h"
#include "../etc/grid.h"
#include "dg/dg.h"
#include "fv/fv.h"
#include "split/homogeneous.h"
#include "split/ode.h"
#include "weno/weno.h"

void ader_stepper(Vecr u, Vecr ub, Vecr wh, Vecr qh, int ndim, iVec3r nX,
                  double dt, Vec3r dX, bool STIFF, int FLUX, Par &MP,
                  bVecr mask) {

  weno_launcher(wh, ub, ndim, nX);

  predictor(qh, wh, ndim, dt, dX, STIFF, false, MP);

  fv_launcher(u, qh, ndim, nX, dt, dX, true, true, FLUX, MP, mask);
}

void split_stepper(Vecr u, Vecr ub, Vecr wh, int ndim, iVec3r nX, double dt,
                   Vec3r dX, bool HALF_STEP, int FLUX, Par &MP, bVecr mask) {

  ode_launcher(u, dt / 2, MP);
  ode_launcher(ub, dt / 2, MP);

  weno_launcher(wh, ub, ndim, nX);

  if (HALF_STEP)
    midstepper(wh, ndim, dt, dX, MP, mask);

  fv_launcher(u, wh, ndim, nX, dt, dX, false, false, FLUX, MP, mask);

  ode_launcher(u, dt / 2, MP);
}
