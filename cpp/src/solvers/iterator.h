#ifndef ITERATOR_H
#define ITERATOR_H

#include "../etc/types.h"
#include "../system/objects/gpr_objects.h"

void iterator(Vecr u, double tf, iVecr nX, Vecr dX, double CFL, bool PERIODIC,
              bool SPLIT, bool HALF_STEP, bool STIFF, int FLUX,
              std::vector<Par> &MPs);

#endif // ITERATOR_H
