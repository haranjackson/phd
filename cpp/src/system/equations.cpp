#include "../etc/globals.h"
#include "energy/derivatives.h"
#include "energy/eos.h"
#include "functions/matrices.h"
#include "functions/vectors.h"
#include "jacobians.h"
#include "objects.h"
#include "variables/sources.h"
#include "variables/state.h"

void flux(VecVr ret, VecVr Q, int d, Par &MP) {
  // Adds the flux vector in the dth direction, given  conserved variables Q
  // NOTE It may be necessary to initialize ret to 0 first

  double ρ = Q(0);
  double ρE = Q(1);
  Vec3Map ρv = get_ρv(Q);
  double p = pressure(Q, MP);
  double vd = ρv(d) / ρ;

  ret(0) += ρv(d);
  ret(1) += vd * (ρE + p);
  ret.segment<3>(2) += vd * ρv;
  ret(2 + d) += p;

  if (VISCOUS) {
    Mat3_3Map A = get_A(Q);
    Vec3 σd = sigma(Q, MP, d);

    ret(1) -= σd.dot(ρv) / ρ;
    ret.segment<3>(2) -= σd;
    Vec3 Av = A * ρv / ρ;
    ret(5 + d) += Av(0);
    ret(8 + d) += Av(1);
    ret(11 + d) += Av(2);
  }
  if (THERMAL) {
    Vec3Map ρJ = get_ρJ(Q);
    double T = temperature(ρ, p, MP);

    ret(1) += MP.cα2 * T * ρJ(d) / ρ;
    ret.segment<3>(14) += vd * ρJ;
    ret(14 + d) += T;
  }
  if (MULTI) {
    ret(mV) = Q(mV) * vd;
  }
}

void source(VecVr ret, VecVr Q, Par &MP) {

  ret.setZero();

  f_body(ret.segment<3>(2), MP);

  if (VISCOUS) {
    Mat3_3 Asource = -dEdA_s(Q, MP) * theta1inv(Q, MP);
    ret.segment<9>(5) = Vec9Map(Asource.data());
  }
  if (THERMAL) {
    double ρ = Q(0);
    ret.segment<3>(14) = -ρ * dEdJ(Q, MP) * theta2inv(Q, MP);
  }
  if (MP.REACTION > -1) {
    ret(mV) = -Q(0) * reaction_rate(Q, MP);
  }
}

void block(MatV_Vr ret, VecVr Q, int d) {

  double ρ = Q(0);
  Vec3Map ρv = get_ρv(Q);
  double vd = ρv(d) / ρ;

  if (VISCOUS) {
    for (int i = 5; i < 14; i++)
      ret(i, i) = vd;
    for (int i = 0; i < 3; i++) {
      double vi = ρv(i) / ρ;
      ret(5 + d, 5 + i) -= vi;
      ret(8 + d, 8 + i) -= vi;
      ret(11 + d, 11 + i) -= vi;
    }
  }
  for (int i = 1; i < LSET + 1; i++)
    ret(V - i, V - i) = vd;
}

void B0dot(VecVr ret, VecVr x, Vec3r v) {
  double v0 = v(0);
  double v1 = v(1);
  double v2 = v(2);
  ret.head<5>().setZero();
  ret(5) = -v1 * x(6) - v2 * x(7);
  ret(6) = v0 * x(6);
  ret(7) = v0 * x(7);
  ret(8) = -v1 * x(9) - v2 * x(10);
  ret(9) = v0 * x(9);
  ret(10) = v0 * x(10);
  ret(11) = -v1 * x(12) - v2 * x(13);
  ret(12) = v0 * x(12);
  ret(13) = v0 * x(13);
  ret.tail<V - 14>().setZero();

  for (int i = 1; i < LSET + 1; i++)
    ret(V - i) = v0 * x(V - i);
}

void B1dot(VecVr ret, VecVr x, Vec3r v) {
  double v0 = v(0);
  double v1 = v(1);
  double v2 = v(2);
  ret.head<5>().setZero();
  ret(5) = v1 * x(5);
  ret(6) = -v0 * x(5) - v2 * x(7);
  ret(7) = v1 * x(7);
  ret(8) = v1 * x(8);
  ret(9) = -v0 * x(8) - v2 * x(10);
  ret(10) = v1 * x(10);
  ret(11) = v1 * x(11);
  ret(12) = -v0 * x(11) - v2 * x(13);
  ret(13) = v1 * x(13);
  ret.tail<V - 14>().setZero();

  for (int i = 1; i < LSET + 1; i++)
    ret(V - i) = v1 * x(V - i);
}

void B2dot(VecVr ret, VecVr x, Vec3r v) {
  double v0 = v(0);
  double v1 = v(1);
  double v2 = v(2);
  ret.head<5>().setZero();
  ret(5) = v2 * x(5);
  ret(6) = v2 * x(6);
  ret(7) = -v0 * x(5) - v1 * x(6);
  ret(8) = v2 * x(8);
  ret(9) = v2 * x(9);
  ret(10) = -v0 * x(8) - v1 * x(9);
  ret(11) = v2 * x(11);
  ret(12) = v2 * x(12);
  ret(13) = -v0 * x(11) - v1 * x(12);
  ret.tail<V - 14>().setZero();

  for (int i = 1; i < LSET + 1; i++)
    ret(V - i) = v2 * x(V - i);
}

void Bdot(VecVr ret, VecVr Q, VecVr x, int d, Par &MP) {

  if (VISCOUS) {
    double ρ = Q(0);
    Vec3 v = get_ρv(Q) / ρ;

    switch (d) {
    case 0:
      B0dot(ret, x, v);
      break;
    case 1:
      B1dot(ret, x, v);
      break;
    case 2:
      B2dot(ret, x, v);
      break;
    }
  }
}

MatV_V system_matrix(VecVr Q, int d, Par &MP) {
  // Returns the Jacobian in the dth direction
  MatV_V DFDP = dFdP(Q, d, MP);
  MatV_V DPDQ = dPdQ(Q, MP);
  MatV_V B = MatV_V::Zero();
  block(B, Q, d);
  return DFDP * DPDQ + B;
}
