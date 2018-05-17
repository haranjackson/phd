#include "../../include/skfmm/distance_marcher.h"
#include "../etc/types.h"

Vec distance(Vecr phi, aVecr dX, iVecr nX) {

  int ndim = nX.size();
  int ncell = nX.prod();
  Vec dist(ncell);
  dist.setZero(ncell);

  lVec FLAG(ncell);
  FLAG.setZero(ncell);

  int TEST = 0;
  int ORDER = 2;
  double NARROW = 0.;
  int PERIODIC = 0;

  baseMarcher *marcher = 0;
  marcher = new distanceMarcher(phi.data(), dX.data(), FLAG.data(), dist.data(),
                                ndim, nX.data(), TEST, ORDER, NARROW, PERIODIC);
  marcher->march();
  delete marcher;

  return dist;
}
