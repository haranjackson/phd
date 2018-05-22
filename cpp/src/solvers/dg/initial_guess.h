#ifndef INITIAL_GUESS_H
#define INITIAL_GUESS_H

#include "../../etc/types.h"
#include "../../system/objects/gpr_objects.h"

void standard_initial_guess1(Matr q, Matr w);

void standard_initial_guess2(Matr q, Matr w);

void stiff_initial_guess(Matr q, Matr w, int NT, double dt, Par &MP);

#endif // INITIAL_GUESS_H
