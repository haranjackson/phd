#include "../../etc/globals.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../variables/shear.h"
#include "../variables/state.h"
#include <cmath>

double theta1inv(VecVr Q, Par &MP) {
  // Returns the relaxation parameter for the distortion tensor
  double ρ = Q(0);
  Mat3_3Map A = get_A(Q);

  double A53 = pow(A.determinant(), 5. / 3.);
  double τ1 = MP.τ1;
  double cs2 = c_s2(ρ, MP);

  if (MP.PLASTIC) {
    double σY = MP.σY;
    double n = MP.n;
    Mat3_3 σ = sigma(Q, MP);
    double sn = sigma_norm(σ);
    sn = std::min(sn, 1e8); // Hacky fix
    return 3 * A53 / (cs2 * τ1) * pow((sn / σY), n);
  } else
    return 3 * A53 / (cs2 * τ1);
}

double theta2inv(VecVr Q, Par &MP) {
  // Returns the relaxation parameter for the thermal impulse vector
  double cα2 = MP.cα2;
  double τ2 = MP.τ2;
  double ρ0 = MP.ρ0;
  double T0 = MP.T0;

  double ρ = Q(0);
  double p = pressure(Q, MP);
  double T = temperature(ρ, p, MP);
  return (ρ0 / ρ) * (T / T0) / (cα2 * τ2);
}

void f_body(Vec3r x, Par &MP) { x = MP.δp; }
