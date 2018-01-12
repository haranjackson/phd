#include <cmath>

#include "../../etc/globals.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"
#include "mg.h"

double E_1(double ρ, double p, Par &MP) {
  // Returns the microscale energy corresponding to a stiffened gas
  // NB The ideal gas equation is obtained if pINF=0
  double Γ = Γ_MG(ρ, MP);
  double pr = p_ref(ρ, MP);
  double er = e_ref(ρ, MP);
  return er + (p - pr) / (ρ * Γ);
}

double E_2A(VecVr Q, Par &MP) {
  // Returns the mesoscale energy dependent on the distortion
  Mat3_3Map A = get_A(Q);
  double G00 = dot(A.row(0), A.row(0));
  double G11 = dot(A.row(1), A.row(1));
  double G22 = dot(A.row(2), A.row(2));
  double G01 = dot(A.row(0), A.row(1));
  double G02 = dot(A.row(0), A.row(2));
  double G12 = dot(A.row(1), A.row(2));
  double t = (G00 + G11 + G22) / 3;
  return MP.B0 / 4 *
         ((G00 - t) * (G00 - t) + (G11 - t) * (G11 - t) +
          (G22 - t) * (G22 - t) + 2 * (G01 * G01 + G02 * G02 + G12 * G12));
}

double E_2J(VecVr Q, Par &MP) {
  // Returns the mesoscale energy dependent on the thermal impulse
  double ρ = Q(0);
  Vec3Map rJ = get_rJ(Q);
  return MP.cα2 * L2_1D(rJ) / (2 * ρ * ρ);
}

double E_3(VecVr Q) {
  // Returns the macroscale kinetic energy
  double ρ = Q(0);
  Vec3Map rv = get_rv(Q);
  return L2_1D(rv) / (2 * ρ * ρ);
}

Mat3_3 dEdA(VecVr Q, Par &MP) {
  // Returns the partial derivative of E by A
  Mat3_3Map A = get_A(Q);
  return MP.B0 * AdevG(A);
}

Vec3 dEdJ(VecVr Q, Par &MP) {
  // Returns the partial derivative of E by J
  double ρ = Q(0);
  Vec3Map rJ = get_rJ(Q);
  return MP.cα2 * rJ / ρ;
}
