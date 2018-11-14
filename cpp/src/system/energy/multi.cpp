#include "../../etc/debug.h"
#include "../../scipy/newton_krylov.h"
#include "../objects.h"
#include "eos.h"
#include "mg.h"

Vec obj3(double ρ, double e, double λ, Vec3r x, Par &MP1, Params &MP2) {

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

Vec obj2(double ρ, double e, double λ, Vec2r x, Par &MP1, Params &MP2) {

  Vec2 ret;

  double ρ1 = x(0);
  double ρ2 = x(1);

  double e1 = E_1(ρ1, 0., MP1);
  double e2 = E_1(ρ2, 0., MP2);

  ret(0) = 1 / ρ - λ / ρ1 - (1 - λ) / ρ2;
  ret(1) = e - λ * e1 - (1 - λ) * e2;

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
    VecFunc obj3_bound = std::bind(obj3, ρ, e, λ, _1, MP, MP.MP2);

    Vec3 x0;
    double p0 = average_pressure(ρ, e, λ, MP);
    x0 << p0, ρ, ρ;

    ret = nonlin_solve(obj3_bound, x0);

    if (ret(0) < 0.) {

      Vec2 x1;
      x1 << ρ, ρ;

      using std::placeholders::_1;
      VecFunc obj2_bound = std::bind(obj2, ρ, e, λ, _1, MP, MP.MP2);

      Vec2 res2 = nonlin_solve(obj2_bound, x1);
      ret << 0., res2(0), res2(1);
    }
  }
  return ret;
}