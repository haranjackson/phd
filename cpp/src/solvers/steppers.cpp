#include "../etc/globals.h"
#include "../etc/grid.h"
#include "dg/dg.h"
#include "fv/fv.h"
#include "split/homogeneous.h"
#include "split/ode.h"
#include "weno/weno.h"

void ader_stepper(Vecr u, Vecr ub, Vecr wh, Vecr qh, int ndim, int nx, int ny,
                  int nz, double dt, double dx, double dy, double dz,
                  bool STIFF, bool OSHER, bool PERR_FROB, Par &MP) {

  weno_launcher(wh, ub, ndim, nx, ny, nz);

  predictor(qh, wh, ndim, dt, dx, dy, dz, STIFF, false, MP);

  fv_launcher(u, qh, ndim, nx, ny, nz, dt, dx, dy, dz, true, true, OSHER,
              PERR_FROB, MP);
}

void split_stepper(Vecr u, Vecr ub, Vecr wh, int ndim, int nx, int ny, int nz,
                   double dt, double dx, double dy, double dz, bool STRANG,
                   bool HALF_STEP, bool OSHER, bool PERR_FROB, Par &MP) {

  double Dt = STRANG ? dt / 2 : dt;

  ode_launcher(ub, Dt, MP);

  weno_launcher(wh, ub, ndim, nx, ny, nz);

  if (HALF_STEP)
    midstepper(wh, ndim, dt, dx, dy, dz, MP);

  fv_launcher(u, wh, ndim, nx, ny, nz, dt, dx, dy, dz, false, false, OSHER,
              PERR_FROB, MP);

  if (STRANG)
    ode_launcher(u, Dt, MP);
}
