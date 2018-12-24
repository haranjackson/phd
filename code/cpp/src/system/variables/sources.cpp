#include <cmath>

#include "../../etc/globals.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../variables/state.h"
#include "../waves/shear.h"

const int DISCRETE = 0;
const int ARRHENIUS = 1;
const int IGNITION_GROWTH = 2;

double theta1inv(VecVr Q, Par &MP) {
  // Returns the relaxation parameter for the distortion tensor

  // TODO: implement mixing of μ

  double ρ = Q(0);
  Mat3_3Map A = get_A(Q);

  double A53 = pow(A.determinant(), 5. / 3.);
  double cs2 = c_s2(ρ, MP);
  double n = MP.n;

  double sn;
  if (MP.SOLID) {

    if (MP.POWER_LAW) {

      Mat3_3 σ = sigma(Q, MP);
      sn = sigma_norm(σ);
      sn = std::min(sn, 1e8); // Hacky fix

      double σY = MP.σY;
      return 3 * A53 / (cs2 * MP.τ0) * pow((sn / σY), n);
    } else
      return 0.;
  } else if (MP.BINGHAM) {

    Mat3_3 σ = sigma(Q, MP);
    sn = σ.norm() / sqrt(2.);
    double σY = MP.σY;
    if (sn > σY)
      return A53 * MP.ρ0 / (2 * MP.μ) * (1. - σY / sn);
    else
      return 0.;
    /*
    double τf = MP.τf;
    double τs = MP.τs;
    double bf2 = MP.bf2;
    double bs2 = MP.bs2;
    double ε = 1e-9;

    Mat3_3 σ = sigma(Q, MP);
    double Γ = sigma_norm(σ) / MP.σY;
    // double Γ = σ.norm() / sqrt(2.) / MP.σY;
    double h = 0.5 * (1. + Γ - sqrt(1. + (4. * ε - 2.) * Γ + Γ * Γ));

    double λ;
    if (Γ < ε)
      λ = 1.;
    else
      λ = h / Γ;
    λ = std::min(λ, 1.);

    double τ = τf * τs / (λ * τf + (1 - λ) * τs);
    double Cs = λ * sqrt(bs2) + (1 - λ) * sqrt(bf2);
    return 3. * A53 / (Cs * Cs * τ);
    */
  } else {
    double c = A53 * MP.ρ0 / (2 * std::pow(MP.μ, 1 / n));

    if (MP.POWER_LAW) {
      Mat3_3 σ = sigma(Q, MP);
      sn = σ.norm() / sqrt(2.);

      return c * pow(sn, (1 - n) / n);
    } else
      return c;
  }
}

double theta2inv(VecVr Q, Par &MP) {
  // Returns the relaxation parameter for the thermal impulse vector

  // TODO: implement mixing of κ

  double ρ = Q(0);
  double T = temperature(Q, MP);
  return T / (MP.κ * ρ);
}

void f_body(Vec3r x, Par &MP) { x = MP.δp; }

double reaction_rate(VecVr Q, Par &MP) {

  double λ = Q(mV) / Q(0);
  λ = std::max(0., λ);

  switch (MP.REACTION) {

  case DISCRETE: {
    if (temperature(Q, MP) > MP.Ti)
      return MP.Kc * λ;
    else
      return 0.;
  }
  case ARRHENIUS: {
    double T = temperature(Q, MP);
    return MP.Bc * exp(-MP.Ea / (MP.Rc * T)) * λ;
  }
  case IGNITION_GROWTH: {
    double ρ = Q(0);
    double p = pressure(Q, MP);
    p = std::max(p, 0.);
    λ = std::min(λ, 1.);
    double φ = 1 - λ;

    double ret = 0.;

    if (φ < MP.φI)
      ret += MP.I * std::pow(λ, MP.b) * std::pow(ρ / MP.ρ0 - 1 - MP.a, MP.x);

    if (φ < MP.φG1)
      ret += MP.G1 * std::pow(λ, MP.c) * std::pow(φ, MP.d) * std::pow(p, MP.y);

    if (φ > MP.φG2)
      ret += MP.G2 * std::pow(λ, MP.e) * std::pow(φ, MP.g) * std::pow(p, MP.z);

    return std::max(ret, 0.);
  }
  default:
    throw "Reaction type not recognized";
  }
}
