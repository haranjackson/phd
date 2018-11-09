#include "../../etc/globals.h"
#include "../../scipy/newton_krylov.h"
#include "../../system/equations.h"
#include "../../system/objects/gpr_objects.h"
#include "../evaluations.h"
#include "initial_guess.h"

VecN obj0(VecNr x, double x0, double dt, std::function<double(double)> f) {

  VecN ret = DG_MAT * x - x0 * ENDVALS.row(0).transpose();

  for (int i = 0; i < N; i++)
    ret(i) -= dt * WGHTS(i) * f(x(i));

  return ret;
}

double stiff_ode_solve(double x0, double dt, std::function<double(double)> f) {

  // solves dx/dt = f(x), x(0)=x0 for x(dt)

  using std::placeholders::_1;
  VecFunc obj_bound = std::bind(obj0, _1, x0, dt, f);

  VecN xin = x0 * VecN::Ones();
  VecN res = nonlin_solve(obj_bound, xin, 1e-7);
  return res.dot(ENDVALS.row(1));
}
