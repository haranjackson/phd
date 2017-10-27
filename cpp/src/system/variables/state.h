#ifndef STATE_H
#define STATE_H

#include "../../etc/types.h"
#include "../objects/gpr_objects.h"


double pressure(VecVr Q, Par & MP);

Mat3_3 sigma(VecVr Q, Par & MP);

Vec3 sigma(VecVr Q, Par & MP, int d);

double temperature(double r, double p, Par & MP);

Vec3 heat_flux(double T, Vec3r J, Par & MP);


#endif // STATE_H

