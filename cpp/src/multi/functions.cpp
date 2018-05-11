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

Mat finite_difference(Vecr φ, Vecr dX, iVec nX) {
  // ret[i,j,..][d] is the derivative in the dth direction in cell (i,j,...)

  int ndim = nX.size();
  int ncell = φ.size();
  Mat ret(ncell, ndim);

  if (ndim == 1) {
    ret.block(1, 0, ncell - 2, 1) =
        φ.segment(2, ncell - 2) - φ.segment(0, ncell - 2);
    ret.row(0) = ret.row(1);
    ret.row(ncell - 1) = ret.row(ncell - 2);
    ret /= dX(0);
  }
  // TODO ndim == 2
  return ret;
}
