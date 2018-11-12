#ifndef EIG_H
#define EIG_H

#include "../etc/globals.h"
#include "objects.h"

Mat Xi1(double ρ, double p, VecVr Q, Par &MP, int d);

Mat Xi2(double ρ, double p, VecVr Q, Par &MP, int d);

Mat thermo_acoustic_tensor(VecVr Q, int d, Par &MP);

double max_abs_eigs(VecVr Q, int d, Par &MP);

#endif // EIG_H
