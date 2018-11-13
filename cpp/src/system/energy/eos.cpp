#include <cmath>

#include "../../etc/globals.h"
#include "../energy/mg.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects.h"
#include "../waves/shear.h"

double E_1(double ρ, double p, Params &MP) {
  // Returns the microscale energy under the MG EOS

  // TODO: update for multi (required for rgfm)

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

double internal_energy(VecVr Q, Par &MP) {
  double ρ = Q(0);
  double E = Q(1) / ρ;
  Vec3 v = get_ρv(Q) / ρ;
  double E1 = E - E_3(v);

  if (VISCOUS) {
    Mat3_3Map A = get_A(Q);
    E1 -= E_2A(ρ, A, MP);
  }

  if (THERMAL) {
    Vec3 J = get_ρJ(Q) / ρ;
    E1 -= E_2J(J, MP);
  }

  if (MP.REACTION > -1) {
    double λ = Q(mV) / ρ;
    E1 -= E_R(λ, MP);
  }

  return E1;
}
