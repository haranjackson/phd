#include "../etc/debug.h"
#include "functions/matrices.h"
#include "functions/vectors.h"
#include "objects/gpr_objects.h"
#include "variables/mg.h"
#include "variables/state.h"
#include "variables/wavespeeds.h"

Mat4_4 thermo_acoustic_tensor(VecVr Q, int d, Par &MP) {
  double ρ = Q(0);
  Mat3_3Map A = get_A(Q);

  double p = pressure(Q, MP);
  double T = temperature(ρ, p, MP);

  Mat Xi1 = Mat::Zero(4, 5);
  Mat Xi2 = Mat::Zero(5, 4);

  double c0 = c_0(ρ, p, A, MP);
  Xi1(0, 1) = 1 / ρ;
  Xi2(0, 0) = ρ;
  Xi2(1, d) = ρ * c0 * c0;

  if (MP.VISCOUS) {
    Vec3 σ = sigma(Q, MP, d);
    Vec3 dσdρ = dsigmadρ(Q, MP, d);
    Mat3_3 dσdA = dsigmadA(Q, MP, d);
    Xi1.topLeftCorner(3, 1) = -1 / ρ * dσdρ;
    Xi1.topRightCorner(3, 3) = -1 / ρ * dσdA;
    Xi2.block<1, 3>(1, 0) += σ - ρ * dσdρ;
    Xi2.bottomLeftCorner(3, 3) = A;
  }
  if (MP.THERMAL) {
    double ch = c_h(ρ, T, MP);
    double dT_dρ = dTdρ(ρ, p, MP);
    double dT_dp = dTdp(ρ, MP);
    Xi1(3, 0) = dT_dρ / ρ;
    Xi1(3, 1) = dT_dp / ρ;
    Xi2(1, 3) = ρ * ch * ch / dT_dp;
  }
  return Xi1 * Xi2;
}

double max_abs_eigs(VecVr Q, int d, bool PERRON_FROBENIUS, Par &MP) {
  // Returns the maximum of the absolute values of  the eigenvalues of the GPR
  // system
  Mat4_4 O = thermo_acoustic_tensor(Q, d, MP);
  double vd = Q(2 + d) / Q(0);

  double lam;
  if (PERRON_FROBENIUS) {
    double r01 = std::max(O.row(0).sum(), O.row(1).sum());
    double r23 = std::max(O.row(2).sum(), O.row(3).sum());
    double c01 = std::max(O.col(0).sum(), O.col(1).sum());
    double c23 = std::max(O.col(2).sum(), O.col(3).sum());
    double r = std::max(r01, r23);
    double c = std::max(c01, c23);
    lam = sqrt(std::min(r, c));
  } else {
    lam = sqrt(O.eigenvalues().array().abs().maxCoeff());
  }
  if (vd > 0)
    return vd + lam;
  else
    return lam - vd;
}
