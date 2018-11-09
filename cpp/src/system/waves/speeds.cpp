#include <cmath>

#include "../energy/derivatives.h"
#include "../energy/eos.h"
#include "../objects.h"

double c_0(double ρ, double p, Mat3_3r A, Par &MP) {
  // Returns the adiabatic sound speed for the MG EOS
  double dE_dρ = dEdρ(ρ, p, A, MP);
  double dE_dp = dEdp(ρ, MP);
  return sqrt((p / (ρ * ρ) - dE_dρ) / dE_dp);
}

double c_h(double ρ, double T, Par &MP) {
  // Returns the velocity of the heat characteristic at equilibrium
  double cα2 = MP.cα2;
  double cv = MP.cv;
  return sqrt(cα2 * T / cv) / ρ;
}
