#ifdef BINDINGS
#include "../etc/debug.h"
#endif

#include "../etc/globals.h"
#include "../etc/grid.h"
#include "../multi/fill.h"
#include "../options.h"
#include "../system/eig.h"
#include "../system/functions/vectors.h"
#include "steppers.h"
#include <iostream>

void make_u(Vecr u, std::vector<Vec> &grids, iVecr nX, std::vector<Par> &MPs) {
  // Builds u across the domain, from the different material grids

  int ndim = nX.size();
  int ncell = u.size() / V;
  int nmat = grids.size();

  Vec av = Vec::Zero(u.size());
  double count = 0.;
  for (int mat = 1; mat < nmat; mat++) {
    if (MPs[mat].EOS > -1) {
      av += grids[mat];
      count += 1.;
    }
  }
  av /= count;

  MatMap avMap(av.data(), ncell, V, OuterStride(V));

  int nx = nX(0);
  switch (ndim) {

  case 1:
    for (int idx = 0; idx < nx; idx++) {
      int ind = get_material_index(avMap.row(idx));
      if (MPs[ind].EOS > -1)
        u.segment<V>(idx * V) = grids[ind].segment<V>(idx * V);
      else
        u.segment<V>(idx * V) = av.segment<V>(idx * V);
    }
    break;

  case 2:
    int ny = nX(1);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        int idx = i * ny + j;
        int ind = get_material_index(avMap.row(idx));
        if (MPs[ind].EOS > -1)
          u.segment<V>(idx * V) = grids[ind].segment<V>(idx * V);
        else
          u.segment<V>(idx * V) = av.segment<V>(idx * V);
      }
    break;
  }
}

double timestep(std::vector<Vec> &grids, std::vector<bVec> &masks, aVecr dX,
                double CFL, double t, double tf, int count,
                std::vector<Par> &MPs, int nmat) {

  double MAX = 0.;

  int ndim = dX.size();
  int ncell = grids[0].size() / V;

  VecV Q;
  for (int mat = 0; mat < nmat; mat++)
    if (MPs[mat].EOS > -1) {

      for (int ind = 0; ind < ncell; ind++)
        if (masks[mat](ind)) {

          Q = grids[mat].segment<V>(ind * V);
          for (int d = 0; d < ndim; d++)
            MAX = std::max(MAX, max_abs_eigs(Q, d, MPs[mat]) / dX(d));
        }
    }

  double dt = CFL / MAX;

  if (count <= 5)
    dt *= 0.2;

  if (t + dt > tf)
    return tf - t;
  else
    return dt;
}

void iterator(Vecr u, double tf, iVecr nX, aVecr dX, double CFL, bool PERIODIC,
              bool SPLIT, bool HALF_STEP, bool STIFF, int FLUX,
              std::vector<Par> &MPs) {

  int nmat = MPs.size();
  Vec ub(extended_dimensions(nX, N) * V);
  bVec maskb(extended_dimensions(nX, 1));

  int ncell = u.size() / V;
  std::vector<Vec> grids(nmat);
  std::vector<bVec> masks(nmat);
  for (int i = 0; i < nmat; i++) {
    grids[i] = Vec(ncell);
    masks[i] = bVec(ncell);
  }

  double t = 0.;
  long count = 0;

  double dt = 0.;

  while (t < tf) {

    fill_ghost_cells(grids, masks, u, nX, dX, dt, MPs);

    dt = timestep(grids, masks, dX, CFL, t, tf, count, MPs, nmat);

    for (int mat = 0; mat < nmat; mat++) {
      if (MPs[mat].EOS > -1) {

        boundaries(grids[mat], ub, nX, PERIODIC);
        extend_mask(masks[mat], maskb, nX);

        if (SPLIT)
          split_stepper(grids[mat], ub, nX, dt, dX, HALF_STEP, FLUX, MPs[mat],
                        maskb);
        else
          ader_stepper(grids[mat], ub, nX, dt, dX, STIFF, FLUX, MPs[mat],
                       maskb);
      }
    }
    make_u(u, grids, nX, MPs);

    t += dt;
    count += 1;

#ifdef BINDINGS
    print(int(t / tf * 100.));
#else
    std::cout << "\n" << int(t / tf * 100.);
#endif
  }
}
