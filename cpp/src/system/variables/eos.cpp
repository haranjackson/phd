#include "../../etc/globals.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"
#include "mg.h"
#include "shear.h"
#include <cmath>

double E_1(double ρ, double p, Par &MP) {
  // Returns the microscale energy under the MG EOS
  double Γ = Γ_MG(ρ, MP);
  double pr = p_ref(ρ, MP);
  double er = e_ref(ρ, MP);
  return er + (p - pr) / (ρ * Γ);
}

double E_2A(double ρ, Mat3_3r A, Par &MP) {
  // Returns the mesoscale energy dependent on the distortion
  double C0 = C_0(ρ, MP);
  return C0 / 4 * devGsq(A);
}

double E_2J(Vec3r J, Par &MP) {
  // Returns the mesoscale energy dependent on the thermal impulse
  return MP.cα2 * J.squaredNorm() / 2;
}

double E_3(Vec3r v) {
  // Returns the macroscale kinetic energy
  return v.squaredNorm() / 2;
}

double E_R(double λ, Par &MP) {
  // Returns the microscale energy corresponding to the chemical energy in a
  // reactive material
  return MP.Qc * (λ - 1);
}

double total_energy(double ρ, double p, Vec3r v, Par &MP) {
  double E = E_1(ρ, p, MP) + E_3(v);
  return E;
}

double total_energy(double ρ, double p, Mat3_3r A, Vec3r v, Par &MP) {
  double E = E_1(ρ, p, MP) + E_3(v);
  E += E_2A(ρ, A, MP);
  return E;
}

double total_energy(double ρ, double p, Mat3_3r A, Vec3r J, Vec3r v, Par &MP) {
  double E = E_1(ρ, p, MP) + E_3(v);
  E += E_2A(ρ, A, MP);
  E += E_2J(J, MP);
  return E;
}

double total_energy(double ρ, double p, Mat3_3r A, Vec3r J, Vec3r v, double λ,
                    Par &MP) {
  double E = E_1(ρ, p, MP) + E_3(v);
  E += E_2A(ρ, A, MP);
  E += E_2J(J, MP);
  E += E_R(λ, MP);
  return E;
}
