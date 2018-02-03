#ifndef ITERATOR_H
#define ITERATOR_H

#include "../system/objects/gpr_objects.h"

void iterator(Vecr u, double tf, int nx, int ny, int nz, double dx, double dy,
              double dz, double CFL, bool PERIODIC, bool SPLIT, bool STRANG,
              bool HALF_STEP, bool STIFF, int FLUX, bool PERR_FROB, Par &MP);

#endif // ITERATOR_H
