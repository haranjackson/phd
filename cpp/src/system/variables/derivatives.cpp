#include "../../etc/types.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"
#include "mg.h"
#include "shear.h"

double dEdρ(double ρ, double p, Mat3_3r A, Par &MP) {
  // Returns the partial derivative of E by ρ (holding p,A constant)
  double dC0dρ = dC_0dρ(ρ, MP);
  return dedρ(ρ, p, MP) + dC0dρ / 4 * devGsq(A);
}

double dEdp(double ρ, Par &MP) {
  // Returns the partial derivative of E by p (holding ρ constant)
  return dedp(ρ, MP);
}

Mat3_3 dEdA(VecVr Q, Par &MP) {
  // Returns the partial derivative of E by A
  Mat3_3Map A = get_A(Q);
  return MP.B0 * AdevG(A);
}

Vec3 dEdJ(VecVr Q, Par &MP) {
  // Returns the partial derivative of E by J
  double ρ = Q(0);
  Vec3Map ρJ = get_ρJ(Q);
  return MP.cα2 * ρJ / ρ;
}
