#include "energy/mg.h"
#include "functions/matrices.h"
#include "functions/vectors.h"
#include "objects.h"
#include "variables/state.h"
#include "waves/speeds.h"

Mat Xi1(double ρ, double p, VecVr Q, Par &MP, int d) {

  Mat Ξ1 = Mat::Zero(4, 5);
  Ξ1(0, 1) = 1 / ρ;

  if (VISCOUS) {
    Vec3 dσdρ = dsigmadρ(Q, MP, d);
    Mat3_3 dσdA = dsigmadA(Q, MP, d);
    Ξ1.topLeftCorner(3, 1) = -1 / ρ * dσdρ;
    Ξ1.topRightCorner(3, 3) = -1 / ρ * dσdA;
  }
  if (THERMAL) {
    Ξ1(3, 0) = dTdρ(ρ, p, MP) / ρ;
    Ξ1(3, 1) = dTdp(ρ, MP) / ρ;
  }
  if (THERMAL)
    return Ξ1;
  else
    return Ξ1.topLeftCorner<3, 5>();
}

Mat Xi2(double ρ, double p, VecVr Q, Par &MP, int d) {

  Mat3_3Map A = get_A(Q);

  Mat Ξ2 = Mat::Zero(5, 4);
  double c0 = c_0(Q, MP);
  Ξ2(0, 0) = ρ;
  Ξ2(1, d) = ρ * c0 * c0;

  if (VISCOUS) {
    Vec3 σ = sigma(Q, MP, d);
    Vec3 dσdρ = dsigmadρ(Q, MP, d);
    Ξ2.block<1, 3>(1, 0) += σ - ρ * dσdρ;
    Ξ2.bottomLeftCorner(3, 3) = A;
  }
  if (THERMAL) {
    double T = temperature_prim(ρ, p, MP);
    double ch = c_h(ρ, T, MP);
    Ξ2(1, 3) = ρ * ch * ch / dTdp(ρ, MP);
  }
  if (THERMAL)
    return Ξ2;
  else
    return Ξ2.topLeftCorner<5, 3>();
}

Mat thermo_acoustic_tensor(VecVr Q, int d, Par &MP) {

  double ρ = Q(0);
  double p = pressure(Q, MP);

  Mat Ξ1 = Xi1(ρ, p, Q, MP, d);
  Mat Ξ2 = Xi2(ρ, p, Q, MP, d);

  return Ξ1 * Ξ2;
}

double max_abs_eigs(VecVr Q, int d, Par &MP) {
  // Returns the maximum of the absolute values of  the eigenvalues of the GPR
  // system
  Mat Ξ = thermo_acoustic_tensor(Q, d, MP);
  double vd = Q(2 + d) / Q(0);

  double lam = sqrt(Ξ.eigenvalues().array().abs().maxCoeff());
  if (vd > 0)
    return vd + lam;
  else
    return lam - vd;
}
