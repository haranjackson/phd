#ifndef SOURCES_H
#define SOURCES_H

#include "../objects.h"

double theta1inv(VecVr Q, Par &MP);
double theta2inv(VecVr Q, Par &MP);

void f_body(Vec3r x, Par &MP);

double reaction_rate(VecVr Q, Par &MP);

#endif // SOURCES_H
