#ifndef GPR_OBJECTS_H
#define GPR_OBJECTS_H

#include "eigen3/Eigen"

#include "../etc/types.h"

typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Mat3_3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 9, 1> Vec9;

typedef Eigen::Ref<Mat3_3> Mat3_3r;
typedef Eigen::Ref<Vec2> Vec2r;
typedef Eigen::Ref<Vec3> Vec3r;
typedef Eigen::Ref<Vec9> Vec9r;

typedef Eigen::Map<Mat3_3, 0, Eigen::OuterStride<3>> Mat3_3Map;
typedef Eigen::Map<Vec3, 0, Eigen::InnerStride<1>> Vec3Map;
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

  double bf2;
  double bs2;
  double τf;
  double τs;

  double cα2;
  double κ;
};

struct Par : Params {

  bool SOLID;
  bool POWER_LAW;
  bool MULTI;
  bool BINGHAM;

  double T0;

  Vec3 δp;

  Params MP2;

  int REACTION;
  double Qc;

  double Kc;
  double Ti;

  double Bc;
  double Ea;
  double Rc;

  double I;
  double G1;
  double G2;
  double a;
  double b;
  double c;
  double d;
  double e;
  double g;
  double x;
  double y;
  double z;
  double φI;
  double φG1;
  double φG2;
};

#endif // GPR_OBJECTS_H
