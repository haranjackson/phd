#include "../etc/types.h"
#include "functions/matrices.h"
#include "functions/vectors.h"
#include "objects/gpr_objects.h"
#include "variables/derivatives.h"
#include "variables/mg.h"
#include "variables/shear.h"
#include "variables/state.h"

MatV_V dFdP(VecVr Q, int d, Par &MP) {
  // Returns Jacobian of flux vector with respect to the primitive variables
  MatV_V ret = MatV_V::Zero();

  double ρ = Q(0);
  double E = Q(1) / ρ;
  double p = pressure(Q, MP);
  Vec3 v = get_ρv(Q) / ρ;
  Mat3_3Map A = get_A(Q);

  double Eρ = dEdρ(ρ, p, A, MP);
  double Ep = dEdp(ρ, MP);
  double vd = v(d);
  double ρvd = ρ * vd;
  Mat3_3 G = A.transpose() * A;
  Mat3_3 A_devG = AdevG(A);

  ret(0, 0) = vd;
  ret(0, 2 + d) = ρ;
  ret(1, 0) = (E + ρ * Eρ) * vd;
  ret(1, 1) = (ρ * Ep + 1) * vd;
  ret.block<1, 3>(1, 2) = ρ * vd * v;
  ret(1, 2 + d) += ρ * E + p;
  ret.block<3, 1>(2, 0) = vd * v;
  for (int i = 2; i < 5; i++)
    ret(i, i) = ρvd;
  ret.block<3, 1>(2, 2 + d) += ρ * v;
  ret(2 + d, 1) = 1.;

  if (MP.VISCOUS) {
    Vec3 σ = sigma(Q, MP, d);
    Vec3 σρ = dsigmadρ(Q, MP, d);
    Mat3_3 ψ_ = dEdA(Q, MP);

    double cs2 = c_s2(ρ, MP);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        int ind = i * 3 + j;
        double tmp = 0.;
        for (int k = 0; k < 3; k++) {
          double dsA = dsigmadA(ρ, cs2, A, G, A_devG, d, k, i, j);
          ret(2 + k, 5 + ind) = -dsA;
          tmp -= v(k) * dsA;
        }
        ret(1, 5 + ind) = tmp;
      }
    ret(1, 0) -= σρ.dot(v);
    ret.block<1, 3>(1, 2) -= σ;
    ret.block<1, 9>(1, 5) += ρ * vd * Vec9Map(ψ_.data());
    ret.block<3, 1>(2, 0) -= σρ;
    ret.block<1, 3>(5 + d, 2) = A.row(0);
    ret.block<1, 3>(8 + d, 2) = A.row(1);
    ret.block<1, 3>(11 + d, 2) = A.row(2);
    ret.block<1, 3>(5 + d, 5) = v;
    ret.block<1, 3>(8 + d, 8) = v;
    ret.block<1, 3>(11 + d, 11) = v;
  }
  if (MP.THERMAL) {

    double cα2 = MP.cα2;
    double T = temperature(ρ, p, MP);
    double Tρ = dTdρ(ρ, p, MP);
    double Tp = dTdp(ρ, MP);
    Vec3 J = get_ρJ(Q) / ρ;
    Vec3 H = dEdJ(Q, MP);

    ret(1, 0) += Tρ * H(d);
    ret(1, 1) += Tp * H(d);
    ret.block<1, 3>(1, 14) = ρvd * H;
    ret(1, 14 + d) += cα2 * T;
    ret.block<3, 1>(14, 0) = v(d) * J;
    ret(14 + d, 0) += Tρ;
    ret(14 + d, 1) = Tp;
    ret.block<3, 1>(14, 2 + d) = ρ * J;
    for (int i = 14; i < 17; i++)
      ret(i, i) = ρvd;
  }
  /*
  if (MP.REACTIVE) {
    double λ = P.λ;
    double Qc = MP.Qc;
    ret(1, 17) += Qc * ρvd;
    ret(17, 0) = v(d) * λ;
    ret(17, 2 + d) = ρ * λ;
    ret(17, 17) = ρvd;
  }
  */
  return ret;
}
MatV_V dPdQ(VecVr Q, Par &MP) {
  // Returns Jacobian of primitive vars with respect to conserved vars
  MatV_V ret = MatV_V::Identity();

  double ρ = Q(0);
  double E = Q(1) / ρ;
  double p = pressure(Q, MP);
  Vec3 v = get_ρv(Q) / ρ;
  Mat3_3Map A = get_A(Q);
  Vec3 J = get_ρJ(Q) / ρ;

  double Eρ = dEdρ(ρ, p, A, MP);
  double Ep = dEdp(ρ, MP);
  double Γ_ = 1 / (ρ * Ep);

  double tmp = v.squaredNorm() - (E + ρ * Eρ);
  if (MP.THERMAL) {
    double cα2 = MP.cα2;
    tmp += cα2 * J.squaredNorm();
  }
  double Υ = Γ_ * tmp;

  ret(1, 0) = Υ;
  ret(1, 1) = Γ_;
  ret.block<1, 3>(1, 2) = -Γ_ * v;
  ret.block<3, 1>(2, 0) = -v / ρ;

  for (int i = 2; i < 5; i++)
    ret(i, i) = 1 / ρ;

  if (MP.VISCOUS) {
    Mat3_3 ψ_ = dEdA(Q, MP);
    ret.block<1, 9>(1, 5) = -Γ_ * ρ * Vec9Map(ψ_.data());
  }
  if (MP.THERMAL) {
    Vec3 H = dEdJ(Q, MP);
    ret.block<1, 3>(1, 14) = -Γ_ * H;
    ret.block<3, 1>(14, 0) = -J / ρ;
    for (int i = 14; i < 17; i++)
      ret(i, i) = 1 / ρ;
  }
  /*
  if (MP.REACTIVE) {
    double λ = Q(17) / Q(0);
    double Qc = MP.Qc;
    ret(17, 0) = -λ / ρ;
    ret(17, 17) /= ρ;
    ret(1, 0) += Γ_ * Qc * λ;
    ret(1, 17) -= Γ_ * Qc;
  }
  */
  return ret;
}
