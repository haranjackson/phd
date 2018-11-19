#include "state.h"
#include "../energy/derivatives.h"
#include "../energy/eos.h"
#include "../energy/mg.h"
#include "../energy/multi.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects.h"
#include "../waves/shear.h"

double pressure(VecVr Q, Par &MP) {
  // Returns the pressure under the Mie-Gruneisen EOS

  if (MULTI) {
    return solve_multi(Q, MP)(0);
  } else {
    double ρ = Q(0);
    double e = internal_energy(Q, MP);
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

double temperature_prim(double ρ, double p, Params &MP) {
  // Returns the temperature for a Mie-Gruneisen material
  return temperature_mg(ρ, p, MP);
}

double temperature(VecVr Q, Par &MP) {
  double ρ = Q(0);
  double p = pressure(Q, MP);
  return temperature_prim(ρ, p, MP);
}

Vec3 heat_flux(double T, Vec3r J, Par &MP) {
  // Returns the heat flux vector
  return MP.cα2 * T * J;
}
