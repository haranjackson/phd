#include "../etc/types.h"
#include "../system/objects/gpr_objects.h"
#include "pfmm.h"
#include <cmath>

int sign(double x) {
  if (x <= 0)
    return -1;
  else
    return 1;
}

Vec normal(Vecr Δφ) { return Δφ / Δφ.norm(); }

Mat finite_difference(Vecr φ, aVecr dX, iVec nX) {
  // ret[i,j,..][d] is the derivative in the dth direction in cell (i,j,...)

  int ndim = nX.size();
  int ncell = φ.size();
  Mat ret(ncell, ndim);

  switch (ndim) {

  case 1:
    ret.block(1, 0, ncell - 2, 1) =
        φ.segment(2, ncell - 2) - φ.segment(0, ncell - 2);
    ret.row(0) = ret.row(1);
    ret.row(ncell - 1) = ret.row(ncell - 2);
    ret /= 2 * dX(0);
    break;

  case 2:
    int nx = nX(0);
    int ny = nX(1);
    MatMap φMap(φ.data(), nx, ny, OuterStride(ny));
    Mat tmp(nx, ny);
    MatMap tmpCol(tmp.data(), ncell, 1, OuterStride(1));

    tmp.block(1, 0, nx - 2, ny) =
        (φMap.block(2, 0, nx - 2, ny) - φMap.block(0, 0, nx - 2, ny)) /
        (2 * dX(0));
    tmp.row(0) = tmp.row(1);
    tmp.row(nx - 1) = tmp.row(nx - 2);
    ret.col(0) = tmpCol;

    tmp.block(0, 1, nx, ny - 2) =
        (φMap.block(0, 2, nx, ny - 2) - φMap.block(0, 0, nx, ny - 2)) /
        (2 * dX(1));
    tmp.col(0) = tmp.col(1);
    tmp.col(ny - 1) = tmp.col(ny - 2);
    ret.col(1) = tmpCol;
    break;
  }
  return ret;
}

void renormalize_levelsets(MatMap uMap, int nmat, aVecr dX, iVecr nX) {
  for (int i = 0; i < nmat - 1; i++) {
    Vec phi = uMap.col(V - (nmat - 1) + i);
    Vec φ = distance(phi, dX, nX);
    uMap.col(V - (nmat - 1) + i) = φ;
  }
}

Vec material_indicator(MatMap uMap, int mat, int nmat, aVecr dX, iVecr nX) {

  Vec φ;
  if (mat == 0)
    φ = uMap.col(V - (nmat - 1));
  else
    φ = -uMap.col(V - (nmat - 1));

  if (nmat > 2) {

    for (int i = 1; i < mat; i++)
      φ = φ.array().max(-uMap.col(V + i - (nmat - 1)).array());
    for (int i = mat; i < nmat - 1; i++)
      φ = φ.array().max(uMap.col(V + i - (nmat - 1)).array());

    return distance(φ, dX, nX);

  } else {
    return φ;
  }
}
