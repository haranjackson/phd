#ifndef EIGENVALUES_H
#define EIGENVALUES_H

#include "../../options.h"
#include "../objects.h"

const int n1 = 3 + int(THERMAL);
const int n2 = 6 + 2 * int(THERMAL);
const int n3 = 8 + int(THERMAL);
const int n4 = 11 + int(THERMAL);

MatV_V eigen(VecVr Q, int d, Par &MP);

#endif // EIGENVALUES_H
