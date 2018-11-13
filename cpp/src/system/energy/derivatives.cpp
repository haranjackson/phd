#include "../../etc/types.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects.h"
#include "../variables/state.h"
#include "../waves/shear.h"
#include "eos.h"
#include "mg.h"
#include "multi.h"

double dedρ_coeff(double ρ, double ρi, double λi) {
  double tmp = λi / ((1 - λi) / ρi * ρ - 1);
  return tmp * tmp;
}

double dEdρ(VecVr Q, Par &MP) {
  // Returns the partial derivative of E by ρ (holding p,A constant)

  double ρ = Q(0);
  Mat3_3Map A = get_A(Q);
  double dC0dρ = dC_0dρ(ρ, MP);
  double tmp = dC0dρ / 4 * devGsq(A);

  if (MULTI) {

    double λ = Q(mV) / ρ;
    double e = internal_energy(Q, MP);

    Vec3 sol = solve_multi(Q, e, MP);
    double p = sol(0);
    double ρ1 = sol(1);
    double ρ2 = sol(2);

    double dT1 = dTdρ(ρ1, p, MP);
    double dT2 = dTdρ(ρ2, p, MP.MP2);

    double c = λ / (ρ1 * ρ1) * dT2 + (1 - λ) / (ρ2 * ρ2) * dT1;

    double dρ1 = 1 / (ρ * ρ * c) * dT2;
    double dρ2 = 1 / (ρ * ρ * c) * dT1;

    double tmp1 = λ * dedρ(ρ1, p, MP) * dρ1;
    double tmp2 = (1 - λ) * dedρ(ρ2, p, MP.MP2) * dρ2;

    return tmp1 + tmp2 + tmp;
  }

  else {
    double p = pressure(Q, MP);
    return dedρ(ρ, p, MP) + tmp;
  }
}

double dEdp(VecVr Q, Par &MP) {
  // Returns the partial derivative of E by p (holding ρ constant)

  double ρ = Q(0);

  if (MULTI) {

    double λ = Q(mV) / ρ;
    double e = internal_energy(Q, MP);

    Vec3 sol = solve_multi(Q, e, MP);
    double ρ1 = sol(1);
    double ρ2 = sol(2);

    return λ * dedp(ρ1, MP) + (1 - λ) * dedp(ρ2, MP);
  } else {
    return dedp(ρ, MP);
  }
}

Mat3_3 dEdA(VecVr Q, Par &MP) {
  // Returns the partial derivative of E by A (holding ρ,s constant)
  double ρ = Q(0);
  Mat3_3Map A = get_A(Q);
  double C0 = C_0(ρ, MP);
  return C0 * AdevG(A);
}

Mat3_3 dEdA_s(VecVr Q, Par &MP) {
  // Returns the partial derivative of E by A, holding s constant
  double ρ = Q(0);
  Mat3_3Map A = get_A(Q);
  double cs2 = c_s2(ρ, MP);
  return cs2 * AdevG(A);
}

Vec3 dEdJ(VecVr Q, Par &MP) {
  // Returns the partial derivative of E by J
  double ρ = Q(0);
  Vec3Map ρJ = get_ρJ(Q);
  return MP.cα2 * ρJ / ρ;
}
