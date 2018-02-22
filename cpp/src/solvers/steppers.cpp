#include "../etc/globals.h"
#include "../etc/grid.h"
#include "dg/dg.h"
#include "fv/fv.h"
#include "split/homogeneous.h"
#include "split/ode.h"
#include "weno/weno.h"

void ader_stepper(Vecr u, Vecr ub, Vecr wh, Vecr qh, int ndim, Veci3r nX,
                  double dt, Vec3r dX, bool STIFF, int FLUX, bool PERR_FROB,
                  Par &MP) {

  weno_launcher(wh, ub, ndim, nX);

  predictor(qh, wh, ndim, dt, dX, STIFF, false, MP);

  fv_launcher(u, qh, ndim, nX, dt, dX, true, true, FLUX, PERR_FROB, MP);
}

void split_stepper(Vecr u, Vecr ub, Vecr wh, int ndim, Veci3r nX, double dt,
                   Vec3r dX, bool STRANG, bool HALF_STEP, int FLUX,
                   bool PERR_FROB, Par &MP) {

  double Dt = STRANG ? dt / 2 : dt;

  ode_launcher(ub, Dt, MP);

  weno_launcher(wh, ub, ndim, nX);

  if (HALF_STEP)
    midstepper(wh, ndim, dt, dX, MP);

  fv_launcher(u, wh, ndim, nX, dt, dX, false, false, FLUX, PERR_FROB, MP);

  if (STRANG)
    ode_launcher(u, Dt, MP);
}
