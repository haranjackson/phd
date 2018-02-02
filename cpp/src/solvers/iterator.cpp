#include "../etc/debug.h"
#include "../etc/globals.h"
#include "../etc/grid.h"
#include "../system/eig.h"
#include "steppers.h"
#include <iostream>

double timestep(Vecr u, double dX[3], int ndim, double CFL, double t, double tf,
                int count, bool PERR_FROB, Par &MP) {
  double MIN = 1e5;
  int ncell = u.size() / V;
  VecV q;

  for (int ind = 0; ind < ncell; ind++) {
    q = u.segment<V>(ind * V);
    for (int d = 0; d < ndim; d++)
      MIN = std::min(MIN, dX[d] / max_abs_eigs(q, d, PERR_FROB, MP));
  }

  double dt = CFL * MIN;

  if (count < 5)
    dt *= 0.2;

  if (t + dt > tf)
    return tf - t;
  else
    return dt;
}

void iterator(Vecr u, double tf, int nx, int ny, int nz, double dx, double dy,
              double dz, double CFL, bool PERIODIC, bool SPLIT, bool STRANG,
              bool HALF_STEP, bool STIFF, bool OSHER, bool PERR_FROB, Par &MP) {

  int ndim = int(nx > 1) + int(ny > 1) + int(nz > 1);
  int extDims = extended_dimensions(nx, ny, nz);
  double dX[3] = {dx, dy, dz};

  Vec ub(extDims * V);
  Vec wh(extDims * int(pow(N, ndim)) * V);
  Vec qh(extDims * int(pow(N, ndim + 1)) * V);

  double t = 0.;
  long count = 0;

  while (t < tf) {

    double dt = timestep(u, dX, ndim, CFL, t, tf, count, PERR_FROB, MP);
    boundaries(u, ub, ndim, nx, ny, nz, PERIODIC);

    if (SPLIT)
      split_stepper(u, ub, wh, ndim, nx, ny, nz, dt, dx, dy, dz, STRANG,
                    HALF_STEP, OSHER, PERR_FROB, MP);
    else
      ader_stepper(u, ub, wh, qh, ndim, nx, ny, nz, dt, dx, dy, dz, STIFF,
                   OSHER, PERR_FROB, MP);
    t += dt;
    count += 1;

    print(int(t / tf * 100.));
  }
}
