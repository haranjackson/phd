#ifndef MG_H
#define MG_H

#include "../objects/gpr_objects.h"

const int STIFFENED_GAS = 0;
const int SHOCK_MG = 1;
const int JWL = 2;
const int COCHRAN_CHAN = 3;
const int GODUNOV_ROMENSKI = 4;

double Γ_MG(double ρ, Par &MP);
double p_ref(double ρ, Par &MP);
double e_ref(double ρ, Par &MP);
double dΓ_MG(double ρ, Par &MP);
double dp_ref(double ρ, Par &MP);
double de_ref(double ρ, Par &MP);
double dedρ(double ρ, double p, Par &MP);
double dedp(double ρ, Par &MP);
double dTdρ(double ρ, double p, Par &MP);
double dTdp(double ρ, Par &MP);

#endif // MG_H
