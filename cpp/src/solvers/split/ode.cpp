#include "../../etc/globals.h"
#include "../../system/relaxation/analytic.h"
#include "../../system/relaxation/numerical.h"

void ode_launcher(Vecr u, double dt, Par &MP) {
  int ncell = u.size() / V;
  for (int ind = 0; ind < ncell; ind++) {
    ode_stepper_analytic(u.segment<V>(ind * V), dt, MP);
    ode_stepper_numerical(u.segment<V>(ind * V), dt, MP);
  }
}
