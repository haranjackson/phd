#ifndef EQUATIONS_H
#define EQUATIONS_H

#include "../etc/globals.h"
#include "objects/gpr_objects.h"


void flux(VecVr ret, VecVr Q, int d,
          Par & MP);

void source(VecVr ret, VecVr Q,
            Par & MP);

void block(MatV_Vr ret, VecVr Q, int d);

void Bdot(VecVr ret, VecVr Q, VecVr x, int d);


#endif
