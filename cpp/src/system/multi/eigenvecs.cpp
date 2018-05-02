#include "eigenvecs.h"
#include "../../etc/types.h"
#include "../eig.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"
#include "../variables/mg.h"
#include "../variables/state.h"

MatV_V eigen(VecVr Q, int d, Par &MP) {

  MatV_V R = MatV_V::Zero();

  double ρ = Q(0);
  double p = pressure(Q, MP);
  Vec3 σρd = dsigmadρ(Q, MP, d);

  Mat3_3 Π1, Π2, Π3;

  Mat3_3Map A = get_A(Q);
  Mat3_3 G = A.transpose() * A;
  Mat3_3 A_devG = AdevG(A);
  double B0 = MP.B0;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      Π1(i, j) = dsigmadA(ρ, B0, A, G, A_devG, d, i, j, 0);
      Π2(i, j) = dsigmadA(ρ, B0, A, G, A_devG, d, i, j, 1);
      Π3(i, j) = dsigmadA(ρ, B0, A, G, A_devG, d, i, j, 2);
    }
  Mat Ξ1 = Xi1(ρ, p, Q, MP, d);
  Mat Ξ2 = Xi2(ρ, p, Q, MP, d);
  Mat Ξ = Ξ1 * Ξ2;

  Eigen::EigenSolver<Mat> es(Ξ);
  Mat Q_1 = es.eigenvectors().real();
  Mat D_2 = es.eigenvalues().real().cwiseInverse().asDiagonal();
  Mat D_1 = D_2.array().sqrt();

  Mat tmp1 = 0.5 * Ξ2 * Q_1 * D_2;
  Mat tmp2 = 0.5 * Q_1 * D_1;

  R.topLeftCorner<5, n1>() = tmp1;
  R.block<5, n2 - n1>(0, n1) = tmp1;
  R.block<n1, n1>(11, 0) = tmp2;
  R.block<n1, n1>(11, n1) = -tmp2;

  if (THERMAL) {
    double Tρ = dTdρ(ρ, p, MP);
    double Tp = dTdp(ρ, MP);

    Vec3 b = Tp * σρd;
    b(0) += Tρ;
    Mat3_3 Π1A_1 = (Π1 * A).inverse();
    double c = 1 / (Π1A_1.row(d) * b + Tp / ρ);

    R(0, 8) = -c * Tp;
    R(1, 8) = c * Tρ;
    R.block<3, 1>(2, 8) = c * Π1.inverse() * b;

    for (int i = 15; i < V; i++)
      R(i, i) = 1.;
  } else {
    Mat3_3 Π1_1 = Π1.inverse();
    R(0, 6) = 1.;
    R(1, 7) = 1.;
    R.block<3, 1>(2, 6) = -Π1_1 * σρd;
    R.block<3, 1>(2, 7) = Π1_1.col(0);

    R.block<3, 3>(2, n3) = -Π1_1 * Π2;
    R.block<3, 3>(2, n4) = -Π1_1 * Π3;
    for (int i = 0; i < 6; i++)
      R(5 + i, n3 + i) = 1.;
    for (int i = 14; i < V; i++)
      R(i, i) = 1.;
  }
  return R;
}
