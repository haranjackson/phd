#ifndef GRID_H
#define GRID_H

#include "types.h"

void boundaries(Vecr u, Vecr ub, int ndim, iVec3r nX, bool PERIODIC);

int extended_dimensions(iVec3r nX, int ext);

#endif // GRID_H
