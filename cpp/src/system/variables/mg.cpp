#include "mg.h"
#include "../objects/gpr_objects.h"
#include <cmath>

double Γ_MG(double ρ, Par &MP) {
  // Returns the MG parameter

  double ret;

  switch (MP.EOS) {

  case STIFFENED_GAS: {
    double γ = MP.γ;
    ret = γ - 1;
    break;
  }
  case SHOCK_MG: {
    double Γ0 = MP.Γ0;
    double ρ0 = MP.ρ0;
    ret = Γ0 * ρ0 / ρ;
    break;
  }
  case GODUNOV_ROMENSKI:
    ret = MP.γ;
    break;
  case JWL:
    ret = MP.Γ0;
    break;
  case COCHRAN_CHAN:
    ret = MP.Γ0;
    break;
  }
  return ret;
}

double p_ref(double ρ, Par &MP) {
  // Returns the reference pressure in the MG EOS

  double ret;

  switch (MP.EOS) {

  case STIFFENED_GAS: {
    double γ = MP.γ;
    double pINF = MP.pINF;
    ret = -γ * pINF;
    break;
  }
  case SHOCK_MG: {
    double c02 = MP.c02;
    double ρ0 = MP.ρ0;
    double s = MP.s;
    if (ρ > ρ0) {
      double tmp = 1 / ρ0 - s * (1 / ρ0 - 1 / ρ);
      ret = c02 * (1 / ρ0 - 1 / ρ) / (tmp * tmp);
    } else {
      ret = c02 * (ρ - ρ0);
    }
    break;
  }
  case GODUNOV_ROMENSKI: {
    double c02 = MP.c02;
    double α = MP.α;
    double ρ0 = MP.ρ0;
    double tmp = pow(ρ / ρ0, α);
    ret = c02 * ρ / α * (tmp - 1) * tmp;
    break;
  }
  case JWL: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    ret = A * exp(-R1 * v_) + B * exp(-R2 * v_);
    break;
  }
  case COCHRAN_CHAN: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    ret = A * pow(v_, -R1) - B * pow(v_, -R2);
    break;
  }
  }
  return ret;
}

double e_ref(double ρ, Par &MP) {
  // Returns the reference energy for the MG EOS

  double ret;

  switch (MP.EOS) {

  case STIFFENED_GAS:
    ret = 0.;
    break;
  case SHOCK_MG: {
    double ρ0 = MP.ρ0;
    double pr = p_ref(ρ, MP);
    if (ρ > ρ0)
      ret = 0.5 * pr * (1 / ρ0 - 1 / ρ);
    else
      ret = 0.;
    break;
  }
  case GODUNOV_ROMENSKI: {
    double c02 = MP.c02;
    double α = MP.α;
    double ρ0 = MP.ρ0;
    double tmp = pow(ρ / ρ0, α);
    ret = c02 / (2 * α * α) * (tmp - 1) * (tmp - 1);
    break;
  }
  case JWL: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    ret = A / (ρ0 * R1) * exp(-R1 * v_) + B / (ρ0 * R2) * exp(-R2 * v_);
    break;
  }
  case COCHRAN_CHAN: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    ret = -A / (ρ0 * (1 - R1)) * (pow(v_, 1 - R1) - 1) +
          B / (ρ0 * (1 - R2)) * (pow(v_, 1 - R2) - 1);
    break;
  }
  }
  return ret;
}

double dΓ_MG(double ρ, Par &MP) {
  // Returns the derivative of the MG parameter

  double ret;

  switch (MP.EOS) {

  case SHOCK_MG: {
    double Γ0 = MP.Γ0;
    double ρ0 = MP.ρ0;
    ret = -Γ0 * ρ0 / (ρ * ρ);
    break;
  }
  default:
    ret = 0.;
    break;
  }
  return ret;
}

double dp_ref(double ρ, Par &MP) {
  // Returns the derivative of the reference  pressure in the MG EOS

  double ret;

  switch (MP.EOS) {

  case STIFFENED_GAS:
    ret = 0.;
    break;
  case SHOCK_MG: {
    double c02 = MP.c02;
    double ρ0 = MP.ρ0;
    double s = MP.s;

    if (ρ > ρ0) {
      double tmp = s * (ρ - ρ0) - ρ;
      ret = c02 * ρ0 * ρ0 * (s * (ρ0 - ρ) - ρ) / (tmp * tmp * tmp);
    } else
      ret = c02;
    break;
  }
  case GODUNOV_ROMENSKI: {
    double ρ0 = MP.ρ0;
    double c02 = MP.c02;
    double α = MP.α;
    double tmp = pow(ρ / ρ0, α);
    ret = c02 / α * tmp * ((1 + α) * (tmp - 1) + α * tmp);
    break;
  }
  case JWL: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    ret = v_ / ρ * (A * R1 * exp(-R1 * v_) + B * R2 * exp(-R2 * v_));
    break;
  }
  case COCHRAN_CHAN: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    ret = v_ / ρ * (A * R1 * pow(v_, -R1 - 1) - B * R2 * pow(v_, -R2 - 1));
    break;
  }
  }
  return ret;
}

double de_ref(double ρ, Par &MP) {
  // Returns the derivative of the reference  energy for the MG EOS

  double ret;

  switch (MP.EOS) {

  case STIFFENED_GAS:
    ret = 0.;
    break;
  case SHOCK_MG: {
    double c02 = MP.c02;
    double ρ0 = MP.ρ0;
    double s = MP.s;

    if (ρ > ρ0) {
      double tmp = s * (ρ - ρ0) - ρ;
      ret = -(ρ - ρ0) * ρ0 * c02 / (tmp * tmp * tmp);
    } else {
      ret = 0.;
    }
    break;
  }
  case GODUNOV_ROMENSKI: {
    double c02 = MP.c02;
    double α = MP.α;
    double ρ0 = MP.ρ0;
    double tmp = pow(ρ / ρ0, α);
    ret = c02 / (ρ * α) * (tmp - 1) * tmp;
    break;
  }
  case JWL:
    ret = e_ref(ρ, MP) / (ρ * ρ);
    break;
  case COCHRAN_CHAN:
    ret = e_ref(ρ, MP) / (ρ * ρ);
    break;
  }
  return ret;
}

double dedρ(double ρ, double p, Par &MP) {
  // Returns the derivative of the MG internal energy with respect to ρ
  double Γ = Γ_MG(ρ, MP);
  double dΓ = dΓ_MG(ρ, MP);
  double pr = p_ref(ρ, MP);
  double dpr = dp_ref(ρ, MP);
  double der = de_ref(ρ, MP);
  return der - (dpr * ρ * Γ + (Γ + ρ * dΓ) * (p - pr)) / ((ρ * Γ) * (ρ * Γ));
}

double dedp(double ρ, Par &MP) {
  // Returns the derivative of the MG internal energy with respect to p
  double Γ = Γ_MG(ρ, MP);
  return 1 / (ρ * Γ);
}

double dTdρ(double ρ, double p, Par &MP) {
  // Returns the derivative of the MG temperature with respect to ρ
  double cv = MP.cv;
  double Γ = Γ_MG(ρ, MP);
  double dΓ = dΓ_MG(ρ, MP);
  double pr = p_ref(ρ, MP);
  double dpr = dp_ref(ρ, MP);
  return -(dpr * ρ * Γ + (Γ + ρ * dΓ) * (p - pr)) / ((ρ * Γ) * (ρ * Γ)) / cv;
}

double dTdp(double ρ, Par &MP) {
  // Returns the derivative of the MGtemperature with respect to p
  double cv = MP.cv;
  return dedp(ρ, MP) / cv;
}
