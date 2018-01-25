#ifndef SHEAR_H
#define SHEAR_H

#include "../objects/gpr_objects.h"

double c_s2(double ρ, Par &MP);
double dc_s2dρ(double ρ, Par &MP);
double C_0(double ρ, Par &MP);
double dC_0dρ(double ρ, Par &MP);

#endif // SHEAR_H
