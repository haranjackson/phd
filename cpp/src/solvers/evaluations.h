#ifndef EVALUATIONS_H
#define EVALUATIONS_H

#include "../etc/globals.h"

void derivs2d(Matn2_Vr ret, Matn2_Vr qh, int d);

void endpts2d(Matn_Vr ret, Matn2_Vr qh, int d, int e);

#endif // EVALUATIONS_H
