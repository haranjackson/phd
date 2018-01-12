#ifndef WENO_H
#define WENO_H

#include "../../etc/globals.h"

void weno_launcher(Vecr ret, Vecr u, int ndim, int nx, int ny, int nz);

#endif // WENO_H
