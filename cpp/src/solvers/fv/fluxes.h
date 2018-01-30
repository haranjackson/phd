#ifndef FLUXES_H
#define FLUXES_H

#include "../../etc/globals.h"
#include "../../system/objects/gpr_objects.h"

VecV Aint(VecVr qL, VecVr qR, int d, Par &MP);
VecV Bint(VecVr qL, VecVr qR, int d, Par &MP);
VecV Smax(VecVr qL, VecVr qR, int d, bool PERR_FROB, Par &MP);

#endif // FLUXES_H
