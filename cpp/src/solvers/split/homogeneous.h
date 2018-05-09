#ifndef HOMOGENEOUS_H
#define HOMOGENEOUS_H

#include "../../etc/globals.h"
#include "../../system/objects/gpr_objects.h"

void midstepper(Vecr wh, int ndim, double dt, Vec3r dX, Par &MP, bVecr mask);

#endif // HOMOGENEOUS_H
