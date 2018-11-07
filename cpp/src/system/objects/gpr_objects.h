#ifndef GPR_OBJECTS_H
#define GPR_OBJECTS_H

#include "eigen3/Eigen"

#include "../../etc/types.h"

typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Mat3_3;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 9, 1> Vec9;

typedef Eigen::Ref<Mat3_3> Mat3_3r;
typedef Eigen::Ref<Vec3> Vec3r;
typedef Eigen::Ref<Vec9> Vec9r;

typedef Eigen::Map<Mat3_3, 0, Eigen::OuterStride<3>> Mat3_3Map;
typedef Eigen::Map<Vec3, 0, Eigen::InnerStride<1>> Vec3Map;
typedef Eigen::Map<Vec9, 0, Eigen::InnerStride<1>> Vec9Map;

struct Par {

  double Rc;
  bool POWER_LAW;
  bool SOLID;
  double ρ0;
  double T0;

  int EOS;

  double Tref;
  double cv;

  double pINF;

  double Γ0;
  double c02;
  double s;

  double α;
  double β;
  double γ;

  double A;
  double B;
  double R1;
  double R2;

  double b02;
  double μ;
  double τ0;
  double σY;
  double n;

  double cα2;
  double κ;

  double Qc;

  Vec3 δp;
};

#endif // GPR_OBJECTS_H
