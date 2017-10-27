#ifndef VECTORS_H
#define VECTORS_H

#include "../../etc/types.h"
#include "../objects/gpr_objects.h"


Vec3Map get_rv(VecVr Q);
Mat3_3Map get_A(VecVr Q);
Vec3Map get_rJ(VecVr Q);

VecV Qvec(double r, double p, Vec3r v, Mat3_3r A, Vec3r J, Par & MP);

#endif
