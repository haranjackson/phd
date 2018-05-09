#include <cmath>

#include "../../etc/types.h"
#include "../analytic.h"
#include "../eig.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"
#include "../variables/eos.h"
#include "../variables/mg.h"
#include "../variables/state.h"
#include "../variables/wavespeeds.h"
#include "eigenvecs.h"
#include "riemann.h"
#include "rotations.h"

bool STICK = true;
bool RELAXATION = true;
double STAR_TOL = 1e-6;

bool check_star_convergence(VecVr QL_, VecVr QR_, Par &MPL, Par &MPR) {

  Vec3 ΣL_ = Sigma(QL_, MPL, 0);
  Vec3 ΣR_ = Sigma(QR_, MPR, 0);

  bool cond = (ΣL_ - ΣR_).cwiseAbs().maxCoeff() < STAR_TOL;

  if (THERMAL) {
    double TL_ = temperature(QL_, MPL);
    double TR_ = temperature(QR_, MPR);
    cond = cond && (abs(TL_ - TR_) < STAR_TOL);
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
    double c0 = c_0(ρ, p, A, MP);

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

  MatV_V RL = riemann_constraints(QL, 1, MPL);
  MatV_V RR = riemann_constraints(QR, -1, MPR);

  Vec xL(n1);
  Vec xR(n1);
  xL.head<3>() = Sigma(QL, MPL, 0);
  xR.head<3>() = Sigma(QR, MPR, 0);

  if (THERMAL) {
    xL(3) = temperature(QL, MPL);
    xR(3) = temperature(QR, MPR);
  }

  Vec x_(n1);

  if (STICK) {
    Mat YL = RL.block<n1, n1>(11, 0);
    Mat YR = RR.block<n1, n1>(11, 0);

    Vec3 vL = get_ρv(QL) / QL(0);
    Vec3 vR = get_ρv(QR) / QR(0);

    Vec yL(n1);
    Vec yR(n1);

    yL.head<3>() = vL;
    yR.head<3>() = vR;

    if (THERMAL) {
      yL(3) = QL(14) / QL(0);
      yR(3) = QR(14) / QR(0);
    }
    x_ = (YL - YR).inverse() * (yR - yL + YL * xL - YR * xR);

  } else {

    if (THERMAL) {
      Mat YL(2, n1);
      Mat YR(2, n1);
      YL << RL.block<1, n1>(11, 0), RL.block<1, n1>(14, 0);
      YL << RR.block<1, n1>(11, 0), RR.block<1, n1>(14, 0);

      Vec yL(2);
      Vec yR(2);
      yL << QL(2) / QL(0), QL(14) / QL(0);
      yR << QR(2) / QR(0), QR(14) / QR(0);

      Mat M(2, 2);
      M << YL.col(0) - YR.col(0), YL.col(n1 - 1) - YR.col(n1 - 1);
      Vec tmp = M.inverse() * (yR - yL + YL * xL - YR * xR);
      x_ << tmp(0), 0., 0., tmp(1);

    } else {
      Vec YL = RL.block<1, n1>(11, 0);
      Vec YR = RR.block<1, n1>(11, 0);

      double yL = QL(2) / QL(0);
      double yR = QR(2) / QR(0);

      double tmp = (yR - yL + YL.transpose() * xL - YR.transpose() * xR) /
                   (YL(0) - YR(0));
      x_ << tmp, 0., 0.;
    }
  }
  VecV cL = VecV::Zero();
  VecV cR = VecV::Zero();
  cL.head<n1>() = x_ - xL;
  cR.head<n1>() = x_ - xR;

  VecV PLvec = Cvec_to_Pvec(QL, MPL);
  VecV PRvec = Cvec_to_Pvec(QR, MPR);
  VecV PL_vec = RL * cL + PLvec;
  VecV PR_vec = RR * cR + PRvec;

  QL = Pvec_to_Cvec(PL_vec, MPL);
  QR = Pvec_to_Cvec(PR_vec, MPR);
}

StarStates star_states(VecV QL_, VecV QR_, Par &MPL, Par &MPR, double dt,
                       Vecr n) {

  Mat3_3 R = rotation_matrix(n);
  rotate_tensors(QL_, R);
  rotate_tensors(QR_, R);

  while (!check_star_convergence(QL_, QR_, MPL, MPR)) {

    if (RELAXATION) {
      ode_stepper_analytic(QL_, dt / 2, MPL);
      ode_stepper_analytic(QR_, dt / 2, MPR);
    }
    star_stepper(QL_, QR_, MPL, MPR);
  }
  Mat3_3 RT = R.transpose();
  rotate_tensors(QL_, RT);
  rotate_tensors(QR_, RT);

  StarStates ret = {QL_, QR_};
  return ret;
}
