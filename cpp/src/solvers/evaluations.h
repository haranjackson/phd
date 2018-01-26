#ifndef EVALUATIONS_H
#define EVALUATIONS_H

#include "../etc/globals.h"

void derivs2d(MatN2_Vr ret, MatN2_Vr qh, int d);

void endpts2d(MatN_Vr ret, MatN2_Vr qh, int d, int e);

#endif // EVALUATIONS_H
