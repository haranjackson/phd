#include "../../etc/types.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects.h"
#include "../waves/shear.h"
#include "mg.h"

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
