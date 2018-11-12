#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include "../../etc/types.h"
#include "../objects.h"

double dEdρ(double ρ, double p, Mat3_3r A, Par &MP);
double dEdp(double ρ, Par &MP);
Mat3_3 dEdA(VecVr Q, Par &MP);
Mat3_3 dEdA_s(VecVr Q, Par &MP);
Vec3 dEdJ(VecVr Q, Par &MP);

#endif // DERIVATIVES_H