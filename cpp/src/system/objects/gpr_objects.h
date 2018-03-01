#ifndef GPR_OBJECTS_H
#define GPR_OBJECTS_H

#include "../../../include/eigen3/Eigen"

#include "../../etc/types.h"

typedef Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Mat2_2;
typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Mat3_3;
typedef Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Mat4_4;

typedef Eigen::Ref<Mat2_2> Mat2_2r;
typedef Eigen::Ref<Mat3_3> Mat3_3r;
typedef Eigen::Ref<Mat4_4> Mat4_4r;

typedef Eigen::Matrix<double, 9, 1> Vec9;
typedef Eigen::Ref<Vec9> Vec9r;

typedef Eigen::Map<Mat3_3, 0, Eigen::OuterStride<3>> Mat3_3Map;
typedef Eigen::Map<Vec3, 0, Eigen::InnerStride<1>> Vec3Map;
typedef Eigen::Map<Vec9, 0, Eigen::InnerStride<1>> Vec9Map;

struct Par {
  double Rc;
  int EOS;

  bool VISCOUS;
  bool THERMAL;
  bool REACTIVE;
  bool MULTI;

  double ρ0;
  double p0;
  double T0;
  double Tref;
  double cv;

  double pINF;

  double Γ0;
  double c02;

  double s;
  double e0;

  double α;
  double β;
  double γ;

  double A;
  double B;
  double R1;
  double R2;

  double B0;
  double τ1;
  bool PLASTIC;
  double σY;
  double n;

  double cα2;
  double τ2;

  Vec3 δp;
};

#endif // GPR_OBJECTS_H
