#include "../../etc/types.h"
#include "../objects/gpr_objects.h"
#include "../variables/eos.h"

Vec3Map get_rv(VecVr Q) {
  // Returns the momentum vector
  return Vec3Map(Q.data() + 2);
}

Mat3_3Map get_A(VecVr Q) {
  // Returns the distortion matrix.
  return Mat3_3Map(Q.data() + 5);
}

Vec3Map get_rJ(VecVr Q) {
  // Returns the density times the thermal impulse vector
  return Vec3Map(Q.data() + 14);
}

VecV Qvec(double ρ, double p, Vec3r v, Mat3_3r A, Vec3r J, Par &MP) {
  // Returns the vector of conserved variables
  VecV Q = VecV::Zero();
  Q(0) = ρ;
  Q.segment<3>(2) = ρ * v;
  Q.segment<9>(5) = VecMap(A.data(), 9);
  Q.segment<3>(14) = ρ * J;
  double E = E_1(ρ, p, MP);
  E += E_2A(Q, MP);
  E += E_2J(Q, MP);
  E += E_3(Q);
  Q(1) = ρ * E;
  return Q;
}
