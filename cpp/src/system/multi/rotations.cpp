#include "../../options.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"

Mat3_3 rotation_matrix(Vecr n) {
  // returns the matrix that rotates vector quantities into a coordinate
  // system defined by e1=n, e2,e3 ⟂ n
  Vec3 e1 = Vec3::Zero();
  for (int i = 0; i < n.size(); i++)
    e1(i) = n(i);

  double den;
  Vec3 e2, e3;
  double Sq = e1.squaredNorm();
  if (abs(e1(1) + e1(2)) <= abs(e1(1) - e1(2))) {
    den = sqrt(2 * (1 - e1(0) * e1(1) - e1(1) * e1(2) - e1(2) * e1(0)));
    e2 << e1(1) - e1(2), e1(2) - e1(0), e1(0) - e1(1);
    double Sum = e1.sum();
    e3 = e1.array() * Sum - Sq;
  } else {
    den = sqrt(2 * (1 + e1(0) * e1(1) + e1(1) * e1(2) - e1(2) * e1(0)));
    e2 << e1(1) + e1(2), e1(2) - e1(0), -e1(0) - e1(1);
    double Sum = e1(0) - e1(1) + e1(2);
    e3 << e1(0) * Sum - Sq, e1(1) * Sum + Sq, e1(2) * Sum - Sq;
  }
  e2.array() /= den;
  e3.array() /= den;
  Mat3_3 ret;
  ret << e1, e2, e3;
  return ret.transpose();
}

void rotate_tensors(VecVr Q, Mat3_3r R) {

  Q.segment<3>(2) = R * Q.segment<3>(2);

  Mat3_3Map A = get_A(Q);
  A = R * A * R.transpose();

  if (THERMAL)
    Q.segment<3>(14) = R * Q.segment<3>(14);
}
