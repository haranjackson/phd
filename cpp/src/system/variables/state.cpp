#include "state.h"
#include "../../scipy/newton_krylov.h"
#include "../energy/derivatives.h"
#include "../energy/eos.h"
#include "../energy/mg.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects.h"
#include "../waves/shear.h"

Vec pobj(double ρ, double e, double λ, Vec4r x, Par &MP1, Params &MP2) {

  Vec4 ret;

  double ρ1 = x(0);
  double ρ2 = x(1);
  double e1 = x(2);
  double e2 = x(3);

  double p1 = pressure_mg(ρ1, e1, MP1);
  double p2 = pressure_mg(ρ2, e2, MP2);
  double T1 = temperature(ρ1, p1, MP1);
  double T2 = temperature(ρ2, p2, MP2);

  ret(0) = 1 / ρ - λ / ρ1 - (1 - λ) / ρ2;
  ret(1) = e - λ * e1 - (1 - λ) * e2;
  ret(2) = p1 - p2;
  ret(3) = T1 - T2;

  return ret;
}

double pressure_double(VecVr Q, double e, Par &MP) {

  double ρ = Q(0);
  double λ = Q(mV) / ρ;

  using std::placeholders::_1;
  VecFunc obj_bound = std::bind(pobj, ρ, e, λ, _1, MP, MP.MP2);

  Vec4 x0;
  x0 << ρ, ρ, e, e;
  Vec4 ret = nonlin_solve(obj_bound, x0, DG_TOL);

  double ρ1 = ret(0);
  double e1 = ret(2);
  return pressure_mg(ρ1, e1, MP);
}

double pressure(VecVr Q, Par &MP) {
  // Returns the pressure under the Mie-Gruneisen EOS

  double e = internal_energy(Q, MP);

  if (MULTI) {
    return pressure_double(Q, e, MP);
  } else {
    double ρ = Q(0);
    return pressure_mg(ρ, e, MP);
  }
}

Mat3_3 sigma(VecVr Q, Par &MP) {
  // Returns the symmetric  viscous shear stress tensor
  double ρ = Q(0);
  Mat3_3Map A = get_A(Q);
  Mat3_3 E_A = dEdA_s(Q, MP);
  return -ρ * E_A.transpose() * A;
}

Vec3 sigma(VecVr Q, Par &MP, int d) {
  // Returns the dth column of the symmetric  viscous shear stress tensor
  double ρ = Q(0);
  Mat3_3Map A = get_A(Q);
  Mat3_3 E_A = dEdA_s(Q, MP);
  return -ρ * E_A.transpose() * A.col(d);
}

Vec3 Sigma(VecVr Q, Par &MP, int d) {
  // Returns the dth column of the total stress tensor
  double p = pressure(Q, MP);
  Vec3 Sig = sigma(Q, MP, d);
  Sig *= -1;
  Sig(d) += p;
  return Sig;
}

Vec3 dsigmadρ(VecVr Q, Par &MP, int d) {
  // Returns dσ_di / dρ
  double ρ = Q(0);
  double cs2 = c_s2(ρ, MP);
  double dcs2dρ = dc_s2dρ(ρ, MP);
  return (1 / ρ + dcs2dρ / cs2) * sigma(Q, MP, d);
}

Mat3_3 dsigmadA(VecVr Q, Par &MP, int d) {
  // Returns Mij = dσ_di / dA_jd, holding ρ constant.
  // NOTE: Only valid for EOS with E_2A = cs^2/4 * |devG|^2
  double ρ = Q(0);
  double cs2 = c_s2(ρ, MP);
  Mat3_3Map A = get_A(Q);
  Mat3_3 G = A.transpose() * A;

  Mat3_3 ret = AdevG(A);
  ret.col(d) *= 2.;
  ret += 1. / 3. * A.col(d) * G.row(d);
  ret += G(d, d) * A;
  return -ρ * cs2 * ret.transpose();
}

double dsigmadA(double ρ, double cs2, Mat3_3r A, Mat3_3r G, Mat3_3r AdevG,
                int i, int j, int m, int n) {
  // Returns dσ_ij / dA_mn, holding ρ constant.
  // NOTE: Only valid for EOS with E_2A = cs^2/4 * |devG|^2

  double ret =
      A(m, i) * G(j, n) + A(m, j) * G(i, n) - 2. / 3. * G(i, j) * A(m, n);
  if (i == n)
    ret += AdevG(m, j);
  if (j == n)
    ret += AdevG(m, i);

  return -ρ * cs2 * ret;
}

double temperature(double ρ, double p, Params &MP) {
  // Returns the temperature for a Mie-Gruneisen material
  double cv = MP.cv;
  double Γ = Γ_MG(ρ, MP);
  double pr = p_ref(ρ, MP);
  return φ(ρ, MP) * MP.Tref + (p - pr) / (ρ * Γ * cv);
}

double temperature(VecVr Q, Par &MP) {
  double ρ = Q(0);
  double p = pressure(Q, MP);
  return temperature(ρ, p, MP);
}

Vec3 heat_flux(double T, Vec3r J, Par &MP) {
  // Returns the heat flux vector
  return MP.cα2 * T * J;
}
