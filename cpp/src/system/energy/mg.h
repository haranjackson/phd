#ifndef MG_H
#define MG_H

#include "../objects.h"

const int VACUUM = -1;
const int STIFFENED_GAS = 0;
const int SHOCK_MG = 1;
const int JWL = 2;
const int COCHRAN_CHAN = 3;
const int GODUNOV_ROMENSKI = 4;

double Γ_MG(double ρ, Params &MP);
double p_ref(double ρ, Params &MP);
double e_ref(double ρ, Params &MP);
double pressure_mg(double ρ, double e, Params &MP);
double temperature_mg(double ρ, double p, Params &MP);
double φ(double ρ, Params &MP);

double dΓ_MG(double ρ, Params &MP);
double dp_ref(double ρ, Params &MP);
double de_ref(double ρ, Params &MP);
double dedρ(double ρ, double p, Params &MP);
double dedp(double ρ, Params &MP);
double dTdρ(double ρ, double p, Params &MP);
double dTdp(double ρ, Params &MP);

#endif // MG_H
