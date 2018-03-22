#ifndef ITERATOR_H
#define ITERATOR_H

#include "../etc/types.h"
#include "../system/objects/gpr_objects.h"

void iterator(Vecr u, double tf, Veci3r nX, Vec3r dX, double CFL, bool PERIODIC,
              bool SPLIT, bool STRANG, bool HALF_STEP, bool STIFF, int FLUX,
               Par &MP);

#endif // ITERATOR_H
