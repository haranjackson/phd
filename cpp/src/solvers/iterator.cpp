#ifdef BINDINGS
#include "../etc/debug.h"
#endif

#include "../etc/globals.h"
#include "../etc/grid.h"
#include "../multi/fill.h"
#include "../system/eig.h"
#include "steppers.h"
#include <iostream>

double timestep(std::vector<Vec> &grids, std::vector<bVec> &masks, Vec3r dX,
                int ndim, double CFL, double t, double tf, int count,
                std::vector<Par> &MPs, int nmat) {
  double MIN = 1e5;
  int ncell = grids[0].size() / V;
  VecV Q;

  for (int i = 0; i < nmat; i++)
    for (int ind = 0; ind < ncell; ind++)
      if (masks[i](ind)) {
        Q = grids[i].segment<V>(ind * V);
        for (int d = 0; d < ndim; d++)
          MIN = std::min(MIN, dX(d) / max_abs_eigs(Q, d, MPs[i]));
      }

  double dt = CFL * MIN;

  if (count < 5)
    dt *= 0.2;

  if (t + dt > tf)
    return tf - t;
  else
    return dt;
}

void iterator(Vecr u, double tf, iVec3r nX, Vec3r dX, double CFL, bool PERIODIC,
              bool SPLIT, bool HALF_STEP, bool STIFF, int FLUX,
              std::vector<Par> &MPs) {

  int nmat = MPs.size();
  int nx = nX(0);
  int ny = nX(1);
  int nz = nX(2);
  int ndim = int(nx > 1) + int(ny > 1) + int(nz > 1);

  Vec ub(extended_dimensions(nX, N) * V);
  Vec wh(extended_dimensions(nX, 1) * int(pow(N, ndim)) * V);
  Vec qh(extended_dimensions(nX, 1) * int(pow(N, ndim + 1)) * V);

  std::vector<Vec> grids(nmat);
  std::vector<bVec> masks(nmat);

  double t = 0.;
  long count = 0;

  double dt = 0.;

  while (t < tf) {

    fill_ghost_cells(grids, masks, u, ndim, nX, dX, dt, MPs);

    dt = timestep(grids, masks, dX, ndim, CFL, t, tf, count, MPs, nmat);

    for (int i = 0; i < nmat; i++) {
      boundaries(grids[i], ub, ndim, nX, PERIODIC);
      if (SPLIT)
        split_stepper(grids[i], ub, wh, ndim, nX, dt, dX, HALF_STEP, FLUX,
                      MPs[i], masks[i]);
      else
        ader_stepper(grids[i], ub, wh, qh, ndim, nX, dt, dX, STIFF, FLUX,
                     MPs[i], masks[i]);
    }
    t += dt;
    count += 1;

#ifdef BINDINGS
    print(int(t / tf * 100.));
#else
    std::cout << "\n" << int(t / tf * 100.);
#endif
  }
}
