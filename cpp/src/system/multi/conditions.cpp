#include "../../etc/types.h"
#include "../../options.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"
#include "eigenvecs.h"

void stick_bcs(Vecr x_, MatV_Vr RL, MatV_Vr RR, VecVr QL, VecVr QR, Vecr xL,
               Vecr xR) {

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
}

void slip_bcs(Vecr x_, MatV_Vr RL, MatV_Vr RR, VecVr QL, VecVr QR, Vecr xL,
              Vecr xR) {

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

    double tmp =
        (yR - yL + YL.transpose() * xL - YR.transpose() * xR) / (YL(0) - YR(0));
    x_ << tmp, 0., 0.;
  }
}
