#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "../etc/types.h"

int sign(double x);

Vec normal(Vecr Δφ);

Mat finite_difference(Vecr φ, Vecr dX, iVec nX);

#endif // FUNCTIONS_H
