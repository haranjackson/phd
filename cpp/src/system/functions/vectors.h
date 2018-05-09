#ifndef VECTORS_H
#define VECTORS_H

#include "../../etc/types.h"
#include "../objects/gpr_objects.h"

Vec3Map get_ρv(VecVr Q);
Mat3_3Map get_A(VecVr Q);
Vec3Map get_ρJ(VecVr Q);

VecV Cvec_to_Pvec(VecV Q, Par &MP);
VecV Pvec_to_Cvec(VecV P, Par &MP);

#endif
