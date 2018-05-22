#include "../etc/types.h"
#include "../options.h"
#include "../system/functions/vectors.h"
#include "../system/objects/gpr_objects.h"
#include "../system/variables/eos.h"
#include "params.h"
#include <iostream>

VecV Cvec(double ρ, double p, Vec3r v, Mat3_3r A, Vec3r J, Par &MP) {
  // Returns the vector of conserved variables
  VecV Q;
  Q(0) = ρ;
  Q.segment<3>(2) = ρ * v;
  Q.segment<9>(5) = VecMap(A.data(), 9);

  if (THERMAL) {
    Q.segment<3>(14) = ρ * J;
    Q(1) = ρ * total_energy(ρ, p, A, J, v, MP);
  } else {
    Q(1) = ρ * total_energy(ρ, p, A, v, MP);
  }

  return Q;
}

iVec heat_conduction_dims() {
  iVec nX(1);
  nX << 200;
  return nX;
}

aVec heat_conduction_spacing() {
  iVec nX = heat_conduction_dims();
  aVec dX(1);
  dX << 1. / nX(0);
  return dX;
}

Vec heat_conduction_IC() {

  Par MP = air_params();
  iVec nX = heat_conduction_dims();
  int nx = nX(0);

  double ρL = 2;
  double ρR = 0.5;
  Mat AL = pow(ρL, 1 / 3.) * Mat::Identity(3, 3);
  Mat AR = pow(ρR, 1 / 3.) * Mat::Identity(3, 3);
  Vec v = Vec::Zero(3);
  Vec J = Vec::Zero(3);

  VecV QL = Cvec(ρL, 1., v, AL, J, MP);
  VecV QR = Cvec(ρR, 1., v, AR, J, MP);

  Vec u(nx * V);
  double dx = 1. / nx;
  for (int i = 0; i < nx; i++) {
    if (i * dx < 0.5)
      u.segment<V>(i * V) = QL;
    else
      u.segment<V>(i * V) = QR;
  }
  return u;
}

iVec aluminium_plate_impact_dims() {
  iVec nX(2);
  nX << 120, 160;
  return nX;
}

aVec aluminium_plate_impact_spacing() {

  double Lx = 0.03;
  double Ly = 0.04;
  iVec nX = aluminium_plate_impact_dims();
  aVec dX(2);
  dX << Lx / nX(0), Ly / nX(1);
  return dX;
}

Vec aluminium_plate_impact_IC() {

  Par MP = aluminium_params();
  iVec nX = aluminium_plate_impact_dims();
  int nx = nX(0);
  int ny = nX(1);

  double Lx = 0.03;
  double Ly = 0.04;

  double ρ = MP.ρ0;
  double p = MP.p0;
  Vec v1 = Vec::Zero(3);
  Vec v0 = v1;
  v0(0) = 400;
  Mat A = Mat::Identity(3, 3);
  Vec J = Vec::Zero(3);

  VecV Q0 = Cvec(ρ, p, v0, A, J, MP);
  VecV Q1 = Cvec(ρ, p, v1, A, J, MP);

  Vec u = Vec::Zero(nx * ny * V);

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      double x = (double(i) + 0.5) * Lx / nx;
      double y = (double(j) + 0.5) * Ly / ny;

      int ind = (i * ny + j) * V;

      if (0.001 <= x <= 0.006 and 0.014 <= y <= 0.026) { // projectile
        u.segment<V>(ind) = Q0;
        u(ind + V - 2) = 1;
        u(ind + V - 1) = -1;
      } else if (0.006 <= x <= 0.028 and 0.003 <= y <= 0.037) { // plate
        u.segment<V>(ind) = Q1;
        u(ind + V - 2) = 1;
        u(ind + V - 1) = 1;
      } else { // vacuum
        u(ind + V - 2) = -1;
        u(ind + V - 1) = -1;
      }
    }
  return u;
}
