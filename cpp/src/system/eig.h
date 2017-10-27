#ifndef EIG_H
#define EIG_H

#include "../etc/globals.h"
#include "objects/gpr_objects.h"


Mat4_4 thermo_acoustic_tensor(double r, Mat3_3r A, double p, double T, int d,
                              Par & MP);
double max_abs_eigs(VecVr Q, int d, bool PERRON_FROBENIUS,
                    Par & MP);


#endif // EIG_H
