#include "../../etc/types.h"
#include "../objects/gpr_objects.h"

Mat2_2 inv2(Mat2_2r X) {
  double detX = X.determinant();
  Mat2_2 ret;
  ret(0, 0) = X(1, 1);
  ret(0, 1) = -X(0, 1);
  ret(1, 0) = -X(1, 0);
  ret(1, 1) = X(0, 0);
  return ret / detX;
}

Mat3_3 inv3(Mat3_3r X) {
  double detX = X.determinant();
  Mat3_3 ret;
  ret(0, 0) = (X(1, 1) * X(2, 2) - X(2, 1) * X(1, 2));
  ret(0, 1) = (X(0, 2) * X(2, 1) - X(0, 1) * X(2, 2));
  ret(0, 2) = (X(0, 1) * X(1, 2) - X(0, 2) * X(1, 1));
  ret(1, 0) = (X(1, 2) * X(2, 0) - X(1, 0) * X(2, 2));
  ret(1, 1) = (X(0, 0) * X(2, 2) - X(0, 2) * X(2, 0));
  ret(1, 2) = (X(1, 0) * X(0, 2) - X(0, 0) * X(1, 2));
  ret(2, 0) = (X(1, 0) * X(2, 1) - X(2, 0) * X(1, 1));
  ret(2, 1) = (X(2, 0) * X(0, 1) - X(0, 0) * X(2, 1));
  ret(2, 2) = (X(0, 0) * X(1, 1) - X(1, 0) * X(0, 1));
  return ret / detX;
}

Mat3_3 AdevG(Mat3_3r A) {
  Mat3_3 G = A.transpose() * A;
  double x = (G(0, 0) + G(1, 1) + G(2, 2)) / 3;
  G(0, 0) -= x;
  G(1, 1) -= x;
  G(2, 2) -= x;
  return A * G;
}

double devGsq(Mat3_3r A) {
  double G00 = A.row(0).dot(A.row(0));
  double G11 = A.row(1).dot(A.row(1));
  double G22 = A.row(2).dot(A.row(2));
  double G01 = A.row(0).dot(A.row(1));
  double G02 = A.row(0).dot(A.row(2));
  double G12 = A.row(1).dot(A.row(2));
  double t = (G00 + G11 + G22) / 3;
  return ((G00 - t) * (G00 - t) + (G11 - t) * (G11 - t) +
          (G22 - t) * (G22 - t) + 2 * (G01 * G01 + G02 * G02 + G12 * G12));
}
