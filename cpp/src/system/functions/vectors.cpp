#include "../../etc/types.h"
#include "../objects/gpr_objects.h"
#include "../variables/eos.h"
#include "../variables/state.h"

Vec3Map get_ρv(VecVr Q) {
  // Returns the momentum vector
  return Vec3Map(Q.data() + 2);
}

Mat3_3Map get_A(VecVr Q) {
  // Returns the distortion matrix.
  return Mat3_3Map(Q.data() + 5);
}

Vec3Map get_ρJ(VecVr Q) {
  // Returns the density times the thermal impulse vector
  return Vec3Map(Q.data() + 14);
}

VecV Cvec_to_Pvec(VecV Q, Par &MP) {
  // Returns vector of primitive variables (atypical ordering), given a vector
  // of conserved variables (typical ordering)
  double ρ = Q(0);
  double p = pressure(Q, MP);
  Vec3 ρv = get_ρv(Q);
  Mat3_3 A = get_A(Q);

  Q(1) = p;
  Q.segment<3>(2) = A.col(0);
  Q.segment<3>(5) = A.col(1);
  Q.segment<3>(8) = A.col(2);
  Q.segment<3>(11) = ρv / ρ;

  if (THERMAL)
    Q.segment<3>(14) /= ρ;

  return Q;
}

VecV Pvec_to_Cvec(VecV P, Par &MP) {
  // Returns vector of conserved variables (typical ordering), given a vector of
  // primitive variables (atypical ordering)
  double ρ = P(0);
  double p = P(1);

  Mat3_3 A;
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
      A(i, j) = P(2 + 3 * j + i);

  Vec3 v = P.segment<3>(11);

  P.segment<3>(2) = ρ * v;
  P.segment<9>(5) = VecMap(A.data(), 9);

  if (THERMAL) {
    Vec3 J = P.segment<3>(14);
    P(1) = ρ * total_energy(ρ, p, A, J, v, MP);
    P.segment<3>(14) *= ρ;
  } else
    P(1) = ρ * total_energy(ρ, p, A, v, MP);

  return P;
}
