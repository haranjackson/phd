#include <cmath>

#include "../energy/derivatives.h"
#include "../energy/eos.h"
#include "../objects.h"
#include "../variables/state.h"

double c_0(VecVr Q, Par &MP) {
  // Returns the adiabatic sound speed for the MG EOS

  // TODO: make more efficient - pressure solver is used 3 times here

  double ρ = Q(0);
  double p = pressure(Q, MP);
  double dE_dρ = dEdρ(Q, MP);
  double dE_dp = dEdp(Q, MP);
  return sqrt((p / (ρ * ρ) - dE_dρ) / dE_dp);
}

double c_h(double ρ, double T, Par &MP) {
  // Returns the velocity of the heat characteristic at equilibrium
  double cα2 = MP.cα2;
  double cv = MP.cv;
  return sqrt(cα2 * T / cv) / ρ;
}
