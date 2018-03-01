#ifndef GRID_H
#define GRID_H

#include "types.h"

void boundaries(Vecr u, Vecr ub, int ndim, Veci3r nX, bool PERIODIC);

int extended_dimensions(Veci3r nX, int ext);

#endif // GRID_H
