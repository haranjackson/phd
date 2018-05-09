#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "../etc/types.h"

int sign(double x);

Vec normal(Vecr Δφ);

int iVec_to_ind(iVecr v) { return v(0) * V; }

int iVec_to_ind(iVecr v, int ny) { return (v(0) * ny + v(1)) * V; }

Vec finite_difference(Vecr φ, Vecr dX, int NDIM, iVec nX);

#endif // FUNCTIONS_H
