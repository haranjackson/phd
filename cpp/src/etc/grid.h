#ifndef GRID_H
#define GRID_H

#include "types.h"

void boundaries(Vecr u, Vecr ub, iVecr nX, iVecr boundaryTypes);

int extended_dimensions(iVecr nX, int ext);

void extend_mask(bVecr mask, bVecr maskb, iVecr nX);

#endif // GRID_H
