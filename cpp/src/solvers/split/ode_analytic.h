#ifndef ODE_ANALYTIC_H
#define ODE_ANALYTIC_H

#include "../../etc/globals.h"
#include "../../system/objects/gpr_objects.h"

void analyticSolver_distortion(VecVr Q, double dt, Par &MP);
void analyticSolver_thermal(VecVr Q, double dt, Par &MP);
void ode_stepper_analytic(VecVr Q, double dt, Par &MP);

#endif // ODE_ANALYTIC_H
