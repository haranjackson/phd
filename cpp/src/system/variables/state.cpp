#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"
#include "eos.h"
#include "mg.h"

double pressure(VecVr Q, Par &MP) {
  // Returns the pressure under the Mie-Gruneisen EOS
  double ρ = Q(0);
  double E = Q(1) / ρ;
  double E1 = E - E_2A(Q, MP) - E_2J(Q, MP) - E_3(Q);

  double Γ = Γ_MG(ρ, MP);
  double pr = p_ref(ρ, MP);
  double er = e_ref(ρ, MP);

  return (E1 - er) * ρ * Γ + pr;
}

Mat3_3 sigma(VecVr Q, Par &MP) {
  // Returns the symmetric viscous shear stress tensor
  double ρ = Q(0);
  Mat3_3Map A = get_A(Q);
  return -ρ * A.transpose() * dEdA(Q, MP);
}

Vec3 sigma(VecVr Q, Par &MP, int d) {
  // Returns the dth column of the symmetric  viscous shear stress tensor
  double ρ = Q(0);
  Mat3_3Map A = get_A(Q);
  Mat3_3 E_A = dEdA(Q, MP);
  return -ρ * E_A.transpose() * A.col(d);
}

double temperature(double ρ, double p, Par &MP) {
  // Returns the temperature for an stiffened gas
  double cv = MP.cv;
  double Γ = Γ_MG(ρ, MP);
  double pr = p_ref(ρ, MP);
  return (p - pr) / (ρ * Γ * cv);
}

Vec3 heat_flux(double T, Vec3r J, Par &MP) {
  // Returns the heat flux vector
  return MP.cα2 * T * J;
}
