#ifndef CONDITIONS_H
#define CONDITIONS_H

#include "../../etc/types.h"

void stick_bcs(Vecr x_, MatV_Vr RL, MatV_Vr RR, VecVr QL, VecVr QR, Vecr xL, Vecr xR);

void slip_bcs(Vecr x_, MatV_Vr RL, MatV_Vr RR, VecVr QL, VecVr QR, Vecr xL, Vecr xR);

#endif // CONDITIONS_H
