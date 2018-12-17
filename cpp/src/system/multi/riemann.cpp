#include <cmath>

#include "../../etc/types.h"
#include "../eig.h"
#include "../energy/eos.h"
#include "../energy/mg.h"
#include "../functions/vectors.h"
#include "../objects.h"
#include "../relaxation/analytic.h"
#include "../variables/state.h"
#include "../waves/speeds.h"
#include "conditions.h"
#include "eigenvecs.h"
#include "riemann.h"
#include "rotations.h"

double q_dims(Par &MP) {
  // Returns characteristic dimensions of heat flux
  double α2 = MP.cα2;
  double ρ0 = MP.ρ0;
  double T0 = MP.T0;
  double cv = MP.cv;
  return α2 / ρ0 * T0 * sqrt(T0 / cv);
}

bool check_star_convergence(VecVr QL_, VecVr QR_, Par &MPL, Par &MPR) {

  Vec3 ΣL_ = Sigma(QL_, MPL, 0);

  bool cond;

  if (MPR.EOS > -1) {
    double ρ0 = std::min(MPL.ρ0, MPR.ρ0);
    double b02 = std::min(MPL.b02, MPR.b02);

    Vec3 ΣR_ = Sigma(QR_, MPR, 0);
    double vL_0 = QL_(2) / QL_(0);
    double vR_0 = QR_(2) / QR_(0);

    cond = (ΣL_ - ΣR_).cwiseAbs().maxCoeff() / (b02 * ρ0) < STAR_TOL;
    cond &= std::abs(vL_0 - vR_0) / sqrt(b02) < STAR_TOL;
  } else {
    double ρ0 = MPL.ρ0;
    double b02 = MPL.b02;

    cond = ΣL_.cwiseAbs().maxCoeff() / (b02 * ρ0) < STAR_TOL;
  }

  if (THERMAL) {

    double TL_ = temperature(QL_, MPL);
    Vec3 JL_ = get_ρJ(QL_) / QL_(0);
    Vec3 qL_ = heat_flux(TL_, JL_, MPL);

    if (MPR.EOS > -1) {
      double q0 = std::min(q_dims(MPL), q_dims(MPR));
      double T0 = std::min(MPL.T0, MPR.T0);

      double TR_ = temperature(QR_, MPR);
      Vec3 JR_ = get_ρJ(QR_) / QR_(0);
      Vec3 qR_ = heat_flux(TR_, JR_, MPR);

      cond &= std::abs(qL_(0) - qR_(0)) / q0 < STAR_TOL;
      cond &= std::abs(TL_ - TR_) / T0 < STAR_TOL;
    } else {
      double q0 = q_dims(MPL);

      cond &= std::abs(qL_(0)) / q0 < STAR_TOL;
    }
  }
  return cond;
}

MatV_V riemann_constraints(VecVr Q, double sgn, Par &MP) {
  /* K=R: sgn = -1
   * K=L: sgn = 1
   * NOTE: Uses atypical ordering
   *
   * Extra constraints are:
   * dΣ = dΣ/dρ * dρ + dΣ/dp * dp + dΣ/dA * dA
   * dT = dT/dρ * dρ + dT/dp * dp
   * v*L = v*R
   * J*L = J*R
   */
  MatV_V Rhat = eigen(Q, 0, MP);

  double ρ = Q(0);
  double p = pressure(Q, MP);
  Mat3_3Map A = get_A(Q);
  Vec3 σρ0 = dsigmadρ(Q, MP, 0);

  Mat3_3 Ainv = A.inverse();
  Vec3 e0;
  e0 << 1., 0., 0.;
  Mat3_3 Π1 = dsigmadA(Q, MP, 0);

  Mat tmp = Mat::Zero(5, 5);
  tmp.topLeftCorner<3, 1>() = -σρ0;

  tmp(0, 1) = 1.;
  tmp.block<3, 3>(0, 2) = -Π1;

  if (THERMAL) {
    tmp(3, 0) = dTdρ(ρ, p, MP);
    tmp(3, 1) = dTdp(ρ, MP);
    tmp(4, 0) = -1. / ρ;
    tmp.block<1, 3>(4, 2) = Ainv.row(0);
  } else {
    double c0 = c_0(Q, MP);

    Mat B = Mat::Zero(2, 3);
    B(0, 0) = ρ;
    B.row(1) = sigma(Q, MP, 0) - ρ * σρ0;
    B(1, 0) += ρ * c0 * c0;

    Mat rhs(3, 2);
    rhs << -σρ0, e0;
    Mat3_3 Π1_1 = Π1.inverse();
    Mat C = Π1_1 * rhs;

    Mat BA_1 = B * Ainv;
    Mat Z = Mat::Identity(2, 2);
    Z -= BA_1 * C;
    Mat W(2, 5);
    W << Mat::Identity(2, 2), -BA_1;
    tmp.bottomRows(2) = Z.inverse() * W;
  }

  Mat b = Mat::Identity(5, n1);
  Mat X = tmp.colPivHouseholderQr().solve(b);

  Mat Ξ1 = Xi1(ρ, p, Q, MP, 0);
  Mat Ξ2 = Xi2(ρ, p, Q, MP, 0);
  Mat Ξ = Ξ1 * Ξ2;

  Eigen::EigenSolver<Mat> es(Ξ);
  Mat Q_1 = es.eigenvectors().real();
  Mat D_1 = es.eigenvalues().real().cwiseInverse().cwiseSqrt().asDiagonal();
  Mat Q1 = Q_1.inverse();

  Mat Y0 = Q_1 * D_1 * Q1;

  Rhat.topLeftCorner<5, n1>() = X;
  Rhat.block<n1, n1>(11, 0) = -sgn * Y0 * Ξ1 * X;
  Rhat.block<11, n1>(0, n1).setZero();
  Rhat.block<n1, n1>(11, n1) = sgn * Q_1 * D_1;

  return Rhat;
}

void star_stepper(VecVr QL, VecVr QR, Par &MPL, Par &MPR) {
  // Iterates to the next approximation of the star states.
  // NOTE: the material on the right may be a vacuum.

  MatV_V RL = riemann_constraints(QL, 1, MPL);
  VecV cL = VecV::Zero();

  Vec xL(n1);
  xL.head<3>() = Sigma(QL, MPL, 0);
  if (THERMAL)
    xL(3) = temperature(QL, MPL);

  if (MPR.EOS > -1) { // not a vacuum

    MatV_V RR = riemann_constraints(QR, -1, MPR);
    Vec xR(n1);
    xR.head<3>() = Sigma(QR, MPR, 0);
    if (THERMAL)
      xR(3) = temperature(QR, MPR);

    Vec x_(n1);

    if (RIEMANN_STICK)
      stick_bcs(x_, RL, RR, QL, QR, xL, xR);
    else
      slip_bcs(x_, RL, RR, QL, QR, xL, xR);

    VecV cR = VecV::Zero();
    cL.head<n1>() = x_ - xL;
    cR.head<n1>() = x_ - xR;

    VecV PRvec = Cvec_to_Pvec(QR, MPR);
    VecV PR_vec = RR * cR + PRvec;
    QR = Pvec_to_Cvec(PR_vec, MPR);
  } else {
    cL.head<n1>() = -xL;
    if (THERMAL) {
      double J0 = QL(14) / QL(0);
      cL(3) = (RL.block<1, 3>(14, 0) * xL.head<3>() - J0) / RL(14, 3);
    }
    QR.setZero();
  }
  VecV PLvec = Cvec_to_Pvec(QL, MPL);
  VecV PL_vec = RL * cL + PLvec;
  QL = Pvec_to_Cvec(PL_vec, MPL);
}

VecV left_star_state(VecV QL_, VecV QR_, Par &MPL, Par &MPR, double dt,
                     Vecr n) {

  VecV QL0 = QL_;

  Mat3_3 R = rotation_matrix(n);
  rotate_tensors(QL_, R);
  rotate_tensors(QR_, R);

  int count = 0;

  while (!check_star_convergence(QL_, QR_, MPL, MPR)) {

    if (RIEMANN_RELAXATION) {
      ode_stepper_analytic(QL_, dt / 2, MPL);
      if (MPR.EOS > -1)
        ode_stepper_analytic(QR_, dt / 2, MPR);
    }
    star_stepper(QL_, QR_, MPL, MPR);

    count += 1;
    if (count > 50)
      return QL0;
  }
  Mat3_3 RT = R.transpose();
  rotate_tensors(QL_, RT);
  return QL_;
}
