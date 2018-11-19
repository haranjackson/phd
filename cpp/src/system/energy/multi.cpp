#include "../../etc/debug.h"
#include "../../scipy/newton_krylov.h"
#include "../objects.h"
#include "eos.h"
#include "mg.h"

Vec obj(double ρ, double e, double λ, Vec2r x, Par &MP1, Params &MP2) {

  Vec2 ret;

  double p = x(0);
  double ρ1 = x(1);
  double ρ2 = (1 - λ) / (1 / ρ - λ / ρ1);

  double e1 = E_1(ρ1, p, MP1);
  double e2 = E_1(ρ2, p, MP2);
  double T1 = temperature_mg(ρ1, p, MP1);
  double T2 = temperature_mg(ρ2, p, MP2);

  ret(0) = e - λ * e1 - (1 - λ) * e2;
  ret(1) = T1 - T2;

  return ret;
}

double average_pressure(double ρ, double e, double λ, Par &MP) {
  double p1 = pressure_mg(ρ, e, MP);
  double p2 = pressure_mg(ρ, e, MP.MP2);
  return std::max(λ * p1 + (1 - λ) * p2, 0.);
}

Vec3 solve_multi(VecVr Q, Par &MP) {

  double ρ = Q(0);
  double e = internal_energy(Q, MP);
  double λ = Q(mV) / ρ;

  Vec3 ret;

  if (λ == 1.) {
    double p = pressure_mg(ρ, e, MP);
    p = std::max(p, 0.);
    ret << p, ρ, ρ;
  } else if (λ == 0.) {
    double p = pressure_mg(ρ, e, MP.MP2);
    p = std::max(p, 0.);
    ret << p, ρ, ρ;
  } else {
    using std::placeholders::_1;
    VecFunc obj_bound = std::bind(obj, ρ, e, λ, _1, MP, MP.MP2);

    Vec2 x0;
    double p0 = average_pressure(ρ, e, λ, MP);
    x0 << p0, ρ;

    Vec2 res = nonlin_solve(obj_bound, x0);
    double p = res(0);
    double ρ1 = res(1);
    double ρ2 = (1 - λ) / (1 / ρ - λ / ρ1);
    ret << p, ρ1, ρ2;
  }
  return ret;
}