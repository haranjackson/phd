#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "../etc/types.h"
#include "../options.h"

int iVec_to_ind(iVecr v) { return v(0) * V; }

int iVec_to_ind(iVecr v, int ny) { return (v(0) * ny + v(1)) * V; }

struct BoundaryInds {
  int ii, iL, iR, ind;
};

int sign(double x);

Vec normal(Vecr Δφ);

Mat finite_difference(Vecr φ, aVecr dX, iVec nX);

void renormalize_levelsets(MatMap uMap, int nmat, aVecr dX, iVecr nX);

Vec material_indicator(MatMap uMap, int mat, int nmat, aVecr dX, iVecr nX);

#endif // FUNCTIONS_H
