#include "../etc/types.h"
#include "../system/objects/gpr_objects.h"
#include <cmath>

int sign(double x) {
  if (x <= 0)
    return -1;
  else
    return 1;
}

Vec normal(Vecr Δφ) { return Δφ / Δφ.norm(); }

Vec finite_difference(Vecr φ, Vecr dX, int NDIM, iVec nX) {
  // ret[i,j,..][d] is the derivative in the dth direction in cell (i,j,...)
  int ncell = φ.size();

  Vec ret(ncell);
  if (NDIM == 1) {
    ret.segment(ncell - 2, 1) =
        φ.segment(ncell - 2, 2) - φ.segment(ncell - 2, 0);
    ret(0) = ret(1);
    ret(ncell - 1) = ret(ncell - 2);
    ret /= dX(0);
  }
  // TODO NDIM == 2
  return ret;
}
