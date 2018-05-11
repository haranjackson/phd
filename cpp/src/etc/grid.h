#ifndef GRID_H
#define GRID_H

#include "types.h"

void boundaries(Vecr u, Vecr ub, iVecr nX, bool PERIODIC);

int extended_dimensions(iVecr nX, int ext);

#endif // GRID_H
