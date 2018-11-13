#include "../../etc/debug.h"
#include "../../scipy/newton_krylov.h"
#include "../objects.h"
#include "eos.h"
#include "mg.h"

Vec obj(double ρ, double e, double λ, Vec3r x, Par &MP1, Params &MP2) {

  Vec3 ret;

  double p = x(0);
  double ρ1 = x(1);
  double ρ2 = x(2);

  double e1 = E_1(ρ1, p, MP1);
  double e2 = E_1(ρ2, p, MP2);
  double T1 = temperature_mg(ρ1, p, MP1);
  double T2 = temperature_mg(ρ2, p, MP2);

  ret(0) = 1 / ρ - λ / ρ1 - (1 - λ) / ρ2;
  ret(1) = e - λ * e1 - (1 - λ) * e2;
  ret(2) = T1 - T2;

  return ret;
}

double average_pressure(double ρ, double e, double λ, Par &MP) {
  double p1 = pressure_mg(ρ, e, MP);
  double p2 = pressure_mg(ρ, e, MP.MP2);
  return std::max(λ * p1 + (1 - λ) * p2, 0.);
}

Vec3 solve_multi(VecVr Q, double e, Par &MP) {

  double ρ = Q(0);
  double λ = Q(mV) / ρ;

  using std::placeholders::_1;
  VecFunc obj_bound = std::bind(obj, ρ, e, λ, _1, MP, MP.MP2);

  Vec3 x0;
  double p0 = average_pressure(ρ, e, λ, MP);
  x0 << p0, ρ, ρ;
  return nonlin_solve(obj_bound, x0, 6e-6);
}