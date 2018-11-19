#include "../../solvers/split/numeric.h"
#include "../objects.h"
#include "../variables/sources.h"

double func(double λ, VecVr Q, Par &MP) {

  double ρ = Q(0);
  VecV Q0 = Q;
  Q0(mV) = ρ * λ;
  return -reaction_rate(Q0, MP);
}

void ode_stepper_numerical(VecVr Q, double dt, Par &MP) {

  if (MP.REACTION > -1) {

    double ρ = Q(0);
    double λ = Q(mV) / ρ;

    using std::placeholders::_1;
    std::function<double(double)> f = std::bind(func, _1, Q, MP);

    λ = stiff_ode_solve(λ, dt, f);
    Q(mV) = ρ * λ;
  }
}