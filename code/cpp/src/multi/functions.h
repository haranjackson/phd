#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "../etc/types.h"
#include "../options.h"

struct BoundaryInds {
  int interf, L, R, ind;
};

int sign(double x);

Vec normal(Vecr Δφ);

void finite_difference(Matr Δφ, Vecr φ, aVecr dX, iVec nX);

void renormalize_levelsets(Matr u, aVecr dX, iVecr nX);

void material_indicator(Vecr φ, Matr u, int mat, int nmat, aVecr dX, iVecr nX);

#endif // FUNCTIONS_H
