#ifndef ITERATOR_H
#define ITERATOR_H

#include "../etc/types.h"
#include "../system/objects/gpr_objects.h"

void iterator(Vecr u, double tf, iVec3r nX, Vec3r dX, double CFL, bool PERIODIC,
              bool SPLIT, bool HALF_STEP, bool STIFF, int FLUX,
              std::vector<Par> &MPs);

#endif // ITERATOR_H
