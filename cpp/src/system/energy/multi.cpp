#include "../../scipy/newton_krylov.h"
#include "../objects.h"
#include "mg.h"

Vec obj(double ρ, double e, double λ, Vec4r x, Par &MP1, Params &MP2) {

  Vec4 ret;

  double ρ1 = x(0);
  double ρ2 = x(1);
  double e1 = x(2);
  double e2 = x(3);

  double p1 = pressure_mg(ρ1, e1, MP1);
  double p2 = pressure_mg(ρ2, e2, MP2);
  double T1 = temperature_mg(ρ1, p1, MP1);
  double T2 = temperature_mg(ρ2, p2, MP2);

  ret(0) = 1 / ρ - λ / ρ1 - (1 - λ) / ρ2;
  ret(1) = e - λ * e1 - (1 - λ) * e2;
  ret(2) = p1 - p2;
  ret(3) = T1 - T2;

  return ret;
}

Vec4 solve_multi(VecVr Q, double e, Par &MP) {

  double ρ = Q(0);
  double λ = Q(mV) / ρ;

  using std::placeholders::_1;
  VecFunc obj_bound = std::bind(obj, ρ, e, λ, _1, MP, MP.MP2);

  Vec4 x0;
  x0 << ρ, ρ, e, e;
  return nonlin_solve(obj_bound, x0, DG_TOL);
}