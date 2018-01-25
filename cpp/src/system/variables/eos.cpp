#include "../../etc/globals.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"
#include "mg.h"
#include <cmath>

double E_1(double ρ, double p, Par &MP) {
  // Returns the microscale energy under the MG EOS

  double Γ = Γ_MG(ρ, MP);
  double pr = p_ref(ρ, MP);
  double er = e_ref(ρ, MP);
  return er + (p - pr) / (ρ * Γ);
}

double E_2A(VecVr Q, Par &MP) {
  // Returns the mesoscale energy dependent on the distortion

  Mat3_3Map A = get_A(Q);
  return MP.B0 / 4 * devGsq(A);
}

double E_2J(VecVr Q, Par &MP) {
  // Returns the mesoscale energy dependent on the thermal impulse
  double ρ = Q(0);
  Vec3Map ρJ = get_ρJ(Q);
  return MP.cα2 * ρJ.squaredNorm() / (2 * ρ * ρ);
}

double E_3(VecVr Q) {
  // Returns the macroscale kinetic energy
  double ρ = Q(0);
  Vec3Map ρv = get_ρv(Q);
  return ρv.squaredNorm() / (2 * ρ * ρ);
}
