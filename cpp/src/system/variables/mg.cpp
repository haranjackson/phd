#include "mg.h"
#include "../objects/gpr_objects.h"
#include <cmath>

double Γ_MG(double ρ, Par &MP) {
  // Returns the MG parameter

  switch (MP.EOS) {

  case STIFFENED_GAS: {
    return MP.γ - 1;
  }
  case SHOCK_MG: {
    double Γ0 = MP.Γ0;
    double ρ0 = MP.ρ0;
    return Γ0 * ρ0 / ρ;
  }
  case GODUNOV_ROMENSKI:
    return MP.γ;

  case JWL:
  case COCHRAN_CHAN:
    return MP.Γ0;

  default:
    throw "EOS not recognized";
  }
}

double p_ref(double ρ, Par &MP) {
  // Returns the reference pressure in the MG EOS

  switch (MP.EOS) {

  case STIFFENED_GAS:
    return -MP.pINF;

  case SHOCK_MG: {
    double c02 = MP.c02;
    double ρ0 = MP.ρ0;
    double s = MP.s;
    if (ρ > ρ0) {
      double tmp = 1 / ρ0 - s * (1 / ρ0 - 1 / ρ);
      return c02 * (1 / ρ0 - 1 / ρ) / (tmp * tmp);
    } else
      return c02 * (ρ - ρ0);
  }
  case GODUNOV_ROMENSKI: {
    double c02 = MP.c02;
    double α = MP.α;
    double ρ0 = MP.ρ0;
    double tmp = pow(ρ / ρ0, α);
    return c02 * ρ / α * (tmp - 1) * tmp;
  }
  case JWL: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    return A * exp(-R1 * v_) + B * exp(-R2 * v_);
  }
  case COCHRAN_CHAN: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    return A * pow(v_, -R1) - B * pow(v_, -R2);
  }
  default:
    throw "EOS not recognized";
  }
}

double e_ref(double ρ, Par &MP) {
  // Returns the reference energy for the MG EOS

  switch (MP.EOS) {

  case STIFFENED_GAS:
    return MP.pINF / ρ;

  case SHOCK_MG: {
    double ρ0 = MP.ρ0;
    double pr = p_ref(ρ, MP);
    if (ρ > ρ0)
      return 0.5 * pr * (1 / ρ0 - 1 / ρ);
    else
      return 0.;
  }
  case GODUNOV_ROMENSKI: {
    double c02 = MP.c02;
    double α = MP.α;
    double ρ0 = MP.ρ0;
    double tmp = pow(ρ / ρ0, α);
    return c02 / (2 * α * α) * (tmp - 1) * (tmp - 1);
  }
  case JWL: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    return A / (ρ0 * R1) * exp(-R1 * v_) + B / (ρ0 * R2) * exp(-R2 * v_);
  }
  case COCHRAN_CHAN: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    return -A / (ρ0 * (1 - R1)) * (pow(v_, 1 - R1) - 1) +
           B / (ρ0 * (1 - R2)) * (pow(v_, 1 - R2) - 1);
  }
  default:
    throw "EOS not recognized";
  }
}

double dΓ_MG(double ρ, Par &MP) {
  // Returns the derivative of the MG parameter

  switch (MP.EOS) {

  case SHOCK_MG: {
    double Γ0 = MP.Γ0;
    double ρ0 = MP.ρ0;
    return -Γ0 * ρ0 / (ρ * ρ);
  }
  case STIFFENED_GAS:
  case JWL:
  case COCHRAN_CHAN:
  case GODUNOV_ROMENSKI:
    return 0.;

  default:
    throw "EOS not recognized";
  }
}

double dp_ref(double ρ, Par &MP) {
  // Returns the derivative of the reference  pressure in the MG EOS

  switch (MP.EOS) {

  case STIFFENED_GAS:
    return 0.;

  case SHOCK_MG: {
    double c02 = MP.c02;
    double ρ0 = MP.ρ0;
    double s = MP.s;

    if (ρ > ρ0) {
      double tmp = s * (ρ - ρ0) - ρ;
      return c02 * ρ0 * ρ0 * (s * (ρ0 - ρ) - ρ) / (tmp * tmp * tmp);
    } else
      return c02;
  }
  case GODUNOV_ROMENSKI: {
    double ρ0 = MP.ρ0;
    double c02 = MP.c02;
    double α = MP.α;
    double tmp = pow(ρ / ρ0, α);
    return c02 / α * tmp * ((1 + α) * (tmp - 1) + α * tmp);
  }
  case JWL: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    return v_ / ρ * (A * R1 * exp(-R1 * v_) + B * R2 * exp(-R2 * v_));
  }
  case COCHRAN_CHAN: {
    double A = MP.A;
    double B = MP.B;
    double R1 = MP.R1;
    double R2 = MP.R2;
    double ρ0 = MP.ρ0;
    double v_ = ρ0 / ρ;
    return v_ / ρ * (A * R1 * pow(v_, -R1 - 1) - B * R2 * pow(v_, -R2 - 1));
  }
  default:
    throw "EOS not recognized";
  }
}

double de_ref(double ρ, Par &MP) {
  // Returns the derivative of the reference  energy for the MG EOS

  switch (MP.EOS) {

  case STIFFENED_GAS:
    return -MP.pINF / (ρ * ρ);

  case SHOCK_MG: {
    double c02 = MP.c02;
    double ρ0 = MP.ρ0;
    double s = MP.s;

    if (ρ > ρ0) {
      double tmp = s * (ρ - ρ0) - ρ;
      return -(ρ - ρ0) * ρ0 * c02 / (tmp * tmp * tmp);
    } else
      return 0.;
  }
  case GODUNOV_ROMENSKI: {
    double c02 = MP.c02;
    double α = MP.α;
    double ρ0 = MP.ρ0;
    double tmp = pow(ρ / ρ0, α);
    return c02 / (ρ * α) * (tmp - 1) * tmp;
  }
  case JWL:
  case COCHRAN_CHAN:
    return e_ref(ρ, MP) / (ρ * ρ);

  default:
    throw "EOS not recognized";
  }
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
  // Returns the derivative of the MG temperature with respect to p
  double cv = MP.cv;
  return dedp(ρ, MP) / cv;
}
