#ifndef GPR_OBJECTS_H
#define GPR_OBJECTS_H

#include "eigen3/Eigen"

#include "../etc/types.h"

typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Mat3_3;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<double, 9, 1> Vec9;

typedef Eigen::Ref<Mat3_3> Mat3_3r;
typedef Eigen::Ref<Vec3> Vec3r;
typedef Eigen::Ref<Vec4> Vec4r;
typedef Eigen::Ref<Vec9> Vec9r;

typedef Eigen::Map<Mat3_3, 0, Eigen::OuterStride<3>> Mat3_3Map;
typedef Eigen::Map<Vec3, 0, Eigen::InnerStride<1>> Vec3Map;
typedef Eigen::Map<Vec4, 0, Eigen::InnerStride<1>> Vec4Map;
typedef Eigen::Map<Vec9, 0, Eigen::InnerStride<1>> Vec9Map;

struct Params {

  int EOS;
  double ρ0;

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
};

struct Par : Params {

  bool SOLID;
  bool POWER_LAW;
  bool MULTI;

  double T0;

  Vec3 δp;

  Params MP2;

  int REACTION;
  double Qc;

  double K0;
  double Ti;

  double Bc;
  double Ea;
  double Rc;

  double G1;
  double c;
  double d;
  double y;
  double λ0;
};

#endif // GPR_OBJECTS_H
