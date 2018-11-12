#include <cmath>

#include "../etc/types.h"
#include "../system/objects.h"
#include "pfmm.h"

int sign(double x) {
  if (x <= 0)
    return -1;
  else
    return 1;
}

Vec normal(Vecr Δφ) { return Δφ / Δφ.norm(); }

void finite_difference(Matr Δφ, Vecr φ, aVecr dX, iVec nX) {
  // Δφ[i,j,..][d] is the derivative in the dth direction in cell (i,j,...)

  int ndim = nX.size();
  int ncell = φ.size();

  switch (ndim) {

  case 1:
    Δφ.block(1, 0, ncell - 2, 1) =
        φ.segment(2, ncell - 2) - φ.segment(0, ncell - 2);
    Δφ.row(0) = Δφ.row(1);
    Δφ.row(ncell - 1) = Δφ.row(ncell - 2);
    Δφ /= 2 * dX(0);
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
    Δφ.col(0) = tmpCol;

    tmp.block(0, 1, nx, ny - 2) =
        (φMap.block(0, 2, nx, ny - 2) - φMap.block(0, 0, nx, ny - 2)) /
        (2 * dX(1));
    tmp.col(0) = tmp.col(1);
    tmp.col(ny - 1) = tmp.col(ny - 2);
    Δφ.col(1) = tmpCol;
    break;
  }
}

void renormalize_levelsets(Matr u, aVecr dX, iVecr nX) {

  for (int i = 0; i < LSET; i++) {
    Vec phi = u.col(V - LSET + i);
    Vec φ = distance(phi, dX, nX);
    u.col(V - LSET + i) = φ;
  }
}

void material_indicator(Vecr φ, Matr u, int mat, int nmat, aVecr dX, iVecr nX) {

  if (mat == 0)
    φ = u.col(V - LSET);
  else
    φ = -u.col(V - LSET);

  if (nmat > 2) {

    for (int i = 1; i < mat; i++)
      φ = φ.array().max(-u.col(V - LSET + i).array());
    for (int i = mat; i < LSET; i++)
      φ = φ.array().max(u.col(V - LSET + i).array());

    φ = distance(φ, dX, nX);
  }
}
