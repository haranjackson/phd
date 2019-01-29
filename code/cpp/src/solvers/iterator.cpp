#include "../etc/debug.h"

#include <iostream>

#include "../etc/grid.h"
#include "../multi/fill.h"
#include "../system/eig.h"
#include "../system/functions/vectors.h"
#include "steppers.h"
#include "utils.h"

void make_u(Vecr u, std::vector<Vec> &grids, std::vector<bVec> &masks,
            std::vector<Par> &MPs) {
  // Builds u across the domain, from the different material grids

  int ncell = u.size() / V;
  int nmat = grids.size();
  Vec matSum(LSET);

  for (int i = 0; i < ncell; i++) {

    // take average value of level sets in all cells that have been updated.
    // if cell hasn't been updated, keep value from previous timestep
    matSum.setZero(LSET);
    double matCnt = 0.;
    for (int mat = 0; mat < nmat; mat++) {
      if (masks[mat](i)) {
        matSum += grids[mat].segment<LSET>(i * V + V - LSET);
        matCnt += 1.;
      }
    }
    if (matCnt > 0.)
      u.segment<LSET>(i * V + V - LSET) = matSum / matCnt;

    int mi = get_material_index(u.segment<V>(i * V));

    if (MPs[mi].EOS > -1)
      u.segment<V - LSET>(i * V) = grids[mi].segment<V - LSET>(i * V);
    else
      u.segment<V - LSET>(i * V).setZero();
  }
}

void limit_concentration(Vecr u) {
  if (MULTI) {
    int ncell = u.size() / V;
    for (int i = 0; i < ncell; i++) {
      u(i * V + mV) = std::min(u(i * V + mV), u(i * V));
      u(i * V + mV) = std::max(u(i * V + mV), 0.);
    }
  }
}

double timestep(std::vector<Vec> &grids, std::vector<bVec> &masks, aVecr dX,
                double CFL, double t, double tf, int count,
                std::vector<Par> &MPs, int nmat) {

  double MAX = 0.;

  int ndim = dX.size();
  int ncell = grids[0].size() / V;

  for (int mat = 0; mat < nmat; mat++)
    if (MPs[mat].EOS > -1)
      for (int ind = 0; ind < ncell; ind++)
        if (masks[mat](ind)) {
          double tmp = 0.;
          for (int d = 0; d < ndim; d++)
            tmp += max_abs_eigs(grids[mat].segment<V>(ind * V), d, MPs[mat]) /
                   dX(d);
          MAX = std::max(MAX, tmp);
        }

  double dt = CFL / MAX;

  if (count <= 5)
    dt *= 0.2;

  if (t + dt > tf)
    return tf - t;
  else
    return dt;
}

std::vector<Vec> iterator(Vecr u, double tf, iVecr nX, aVecr dX, double CFL,
                          iVecr boundaryTypes, bool SPLIT, bool HALF_STEP,
                          bool STIFF, int FLUX, std::vector<Par> &MPs,
                          double contorted_tol) {

  Vec uprev(u.size());
  std::vector<Vec> ret(100);
  int nmat = MPs.size();
  Vec ub(extended_dimensions(nX, N) * V);
  bVec maskb(extended_dimensions(nX, 1));

  int ncell = u.size() / V;
  std::vector<Vec> grids(nmat);
  std::vector<bVec> masks(nmat);
  for (int i = 0; i < nmat; i++) {
    grids[i] = u;
    masks[i] = bVec::Ones(ncell);
  }

  double t = 0.;
  long count = 0;
  int pushCount = 0;

  double dt = 0.;

  while (t < tf) {

    uprev = u;

    if (LSET > 0)
      fill_ghost_cells(grids, masks, u, nX, dX, dt, MPs);
    else
      grids[0] = u;

    dt = timestep(grids, masks, dX, CFL, t, tf, count, MPs, nmat);

    for (int mat = 0; mat < nmat; mat++) {
      if (MPs[mat].EOS > -1) {

        boundaries(grids[mat], ub, nX, boundaryTypes);
        extend_mask(masks[mat], maskb, nX);

        if (SPLIT)
          split_stepper_para(grids[mat], ub, nX, dt, dX, HALF_STEP, FLUX,
                             MPs[mat], maskb);
        else
          ader_stepper_para(grids[mat], ub, nX, dt, dX, STIFF, FLUX, MPs[mat],
                            maskb);
      }
    }
    if (LSET > 0)
      make_u(u, grids, masks, MPs);
    else
      u = grids[0];

    limit_concentration(u);

    t += dt;
    count += 1;

    renorm_distortion(u, MPs);

    if (count % 10 == 0 and ROTATE_DISTORTION)
      rotate_distortion(u, MPs);

    if (contorted(u, contorted_tol))
      reset_distortion(u, MPs);

    if (t >= double(pushCount + 1) / 100. * tf) {
      ret[pushCount] = u;
      pushCount += 1;
    }

    print(int(t / tf * 100.));
    print(dt);

    if (u.array().isNaN().any()) {
      ret[pushCount] = uprev;
      ret[pushCount + 1] = u;
      return ret;
    }
  }

  ret[99] = u;
  return ret;
}
