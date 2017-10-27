#ifndef EOS_H
#define EOS_H

#include "../../etc/globals.h"
#include "../objects/gpr_objects.h"


double E_1(double r, double p, Par & MP);
double E_2A(VecVr Q, Par & MP);
double E_2J(VecVr Q, Par & MP);
double E_3(VecVr Q);
Mat3_3 dEdA(VecVr Q, Par & MP);
Vec3 dEdJ(VecVr Q, Par & MP);


#endif // EOS_H
