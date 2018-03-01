#ifndef MATERIAL_FUNCTIONS_H
#define MATERIAL_FUNCTIONS_H

#include "../objects/gpr_objects.h"

double theta1inv(VecVr Q, Par &MP);
double theta2inv(VecVr Q, Par &MP);

void f_body(Vec3r x, Par &MP);

#endif // MATERIAL_FUNCTIONS_H
