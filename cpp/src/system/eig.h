#ifndef EIG_H
#define EIG_H

#include "../etc/globals.h"
#include "objects/gpr_objects.h"

Mat4_4 thermo_acoustic_tensor(VecVr Q, int d, Par &MP);

double max_abs_eigs(VecVr Q, int d, bool PERR_FROB, Par &MP);

#endif // EIG_H
