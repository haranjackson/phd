#ifndef SOLVERS_H
#define SOLVERS_H

#include "../etc/globals.h"
#include "../system/objects.h"

void ader_stepper(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX, bool STIFF,
                  int FLUX, Par &MP, bVecr mask);

void ader_stepper_para(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX,
                       bool STIFF, int FLUX, Par &MP, bVecr mask);

void split_stepper(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX,
                   bool HALF_STEP, int FLUX, Par &MP, bVecr mask);

void split_stepper_para(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX,
                        bool STIFF, int FLUX, Par &MP, bVecr mask);

#endif // SOLVERS_H
