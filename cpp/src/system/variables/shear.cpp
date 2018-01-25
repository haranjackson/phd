#include "../objects/gpr_objects.h"
#include "mg.h"
#include <cmath>

double c_s2(double ρ, Par &MP) {
  // Returns the square of the characteristic velocity of propagation of
  // transverse perturbations
  double B0 = MP.B0;
  double ρ0 = MP.ρ0;
  double β = MP.β;
  return B0 * pow(ρ / ρ0, β);
}

double dc_s2dρ(double ρ, Par &MP) {
  // Returns the derivative of cs^2 with respect to ρ
  double cs2 = c_s2(ρ, MP);
  double β = MP.β;
  return β / ρ * cs2;
}

double C_0(double ρ, Par &MP) {
  // Returns the coefficient of |dev(G)|^2 in E(ρ,p,A,J,v)
  double β = MP.β;
  double cs2 = c_s2(ρ, MP);
  double Γ = Γ_MG(ρ, MP);
  return (1 - β / Γ) * cs2;
}

double dC_0dρ(double ρ, Par &MP) {
  // Returns the derivative of C0 with respect to ρ
  double β = MP.β;

  double cs2 = c_s2(ρ, MP);
  double dcs2dρ = dc_s2dρ(ρ, MP);

  double Γ = Γ_MG(ρ, MP);
  double dΓdρ = dΓ_MG(ρ, MP);

  return (1 - β / Γ) * dcs2dρ + β / (Γ * Γ) * dΓdρ * cs2;
}
