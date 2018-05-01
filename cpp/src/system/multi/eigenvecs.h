#ifndef EIGENVALUES_H
#define EIGENVALUES_H

#include "../../options.h"
#include "../objects/gpr_objects.h"

const int n1 = 3 + int(THERMAL);
const int n2 = 6 + 2 * int(THERMAL);
const int n3 = 8 + int(THERMAL);
const int n4 = 11 + int(THERMAL);
const int n5 = 14 + int(THERMAL);

MatBV eigen(VecBVr Q, int d, Par &MP);

#endif // EIGENVALUES_H
