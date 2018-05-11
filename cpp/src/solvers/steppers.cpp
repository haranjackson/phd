#include "../etc/globals.h"
#include "../etc/grid.h"
#include "dg/dg.h"
#include "fv/fv.h"
#include "split/homogeneous.h"
#include "split/ode.h"
#include "weno/weno.h"

void ader_stepper(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX, bool STIFF,
                  int FLUX, Par &MP, bVecr mask) {

  int ndim = nX.size();
  Vec wh(extended_dimensions(nX, 1) * int(pow(N, ndim)) * V);
  Vec qh(extended_dimensions(nX, 1) * int(pow(N, ndim + 1)) * V);

  weno_launcher(wh, ub, nX);

  predictor(qh, wh, dt, dX, STIFF, false, MP, mask);

  fv_launcher(u, qh, nX, dt, dX, true, true, FLUX, MP, mask);
}

void split_stepper(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX,
                   bool HALF_STEP, int FLUX, Par &MP, bVecr mask) {

  int ndim = nX.size();
  ode_launcher(u, dt / 2, MP);
  ode_launcher(ub, dt / 2, MP);

  Vec wh(extended_dimensions(nX, 1) * int(pow(N, ndim)) * V);
  weno_launcher(wh, ub, nX);

  if (HALF_STEP)
    midstepper(wh, dt, dX, MP, mask);

  fv_launcher(u, wh, nX, dt, dX, false, false, FLUX, MP, mask);

  ode_launcher(u, dt / 2, MP);
}
