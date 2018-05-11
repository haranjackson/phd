#ifdef BINDINGS
#include "../etc/debug.h"
#endif

#include "../etc/globals.h"
#include "../etc/grid.h"
#include "../multi/fill.h"
#include "../system/eig.h"
#include "steppers.h"
#include <iostream>

int get_material_index(VecVr Q) {
  int ret = 0;
  for (int i = V - LSET; i < V; i++)
    if (Q(i) >= 0.)
      ret += 1;
  return ret;
}

void make_u(Vecr u, std::vector<Vec> &grids, iVecr nX) {
  // Builds u across the domain, from the different material grids

  int ndim = nX.size();
  int ncell = u.size() / V;
  int nmat = grids.size();

  Vec av = grids[0];
  for (int i = 1; i < grids.size(); i++)
    av += grids[i];
  av /= nmat;

  MatMap avMap(av.data(), ncell, V, OuterStride(V));

  int nx = nX(0);
  switch (ndim) {

  case 1:
    for (int i = 0; i < nx; i++) {
      int ind = get_material_index(avMap.row(i));
      u(i) = grids[ind](i);
    }
    break;

  case 2:
    int ny = nX(1);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        int idx = i * ny + j;
        int ind = get_material_index(avMap.row(idx));
        u(idx) = grids[ind](idx);
      }
    break;
  }
}

double timestep(std::vector<Vec> &grids, std::vector<bVec> &masks, Vecr dX,
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

void iterator(Vecr u, double tf, iVecr nX, Vecr dX, double CFL, bool PERIODIC,
              bool SPLIT, bool HALF_STEP, bool STIFF, int FLUX,
              std::vector<Par> &MPs) {

  int nmat = MPs.size();
  int ndim = nX.size();

  Vec ub(extended_dimensions(nX, N) * V);

  std::vector<Vec> grids(nmat);
  std::vector<bVec> masks(nmat);

  double t = 0.;
  long count = 0;

  double dt = 0.;

  while (t < tf) {

    fill_ghost_cells(grids, masks, u, nX, dX, dt, MPs);

    dt = timestep(grids, masks, dX, ndim, CFL, t, tf, count, MPs, nmat);

    for (int i = 0; i < nmat; i++) {

      boundaries(grids[i], ub, nX, PERIODIC);

      if (SPLIT)
        split_stepper(grids[i], ub, nX, dt, dX, HALF_STEP, FLUX, MPs[i],
                      masks[i]);
      else
        ader_stepper(grids[i], ub, nX, dt, dX, STIFF, FLUX, MPs[i], masks[i]);
    }
    make_u(u, grids, nX);
    t += dt;
    count += 1;

#ifdef BINDINGS
    print(int(t / tf * 100.));
#else
    std::cout << "\n" << int(t / tf * 100.);
#endif
  }
}
