#ifndef FLUXES_H
#define FLUXES_H

#include "../../etc/globals.h"
#include "../../system/objects/gpr_objects.h"

const int RUSANOV = 0;
const int ROE = 1;
const int OSHER = 2;

VecV Bint(VecVr qL, VecVr qR, int d, Par &MP);
VecV D_OSH(VecVr qL, VecVr qR, int d, Par &MP);
VecV D_ROE(VecVr qL, VecVr qR, int d, Par &MP);
VecV D_RUS(VecVr qL, VecVr qR, int d, bool PERR_FROB, Par &MP);

#endif // FLUXES_H
