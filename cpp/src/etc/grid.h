#ifndef GRID_H
#define GRID_H

#include "types.h"

void boundaries(Vecr u, Vecr ub, int ndim, int nx, int ny, int nz,
                bool PERIODIC);

int extended_dimensions(int nx, int ny, int nz);


#endif // GRID_H
