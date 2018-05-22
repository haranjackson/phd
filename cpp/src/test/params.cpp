#include "../system/objects/gpr_objects.h"

Par air_params() {

  double κ = 1e-2;
  double μ = 1e-2;

  Par MP;
  MP.EOS = 0;

  MP.ρ0 = 1.;
  MP.p0 = 1.;
  MP.T0 = 1.;

  MP.γ = 1.4;
  MP.cv = 2.5;
  MP.pINF = 0.;

  MP.B0 = 1.;
  MP.τ1 = 6 * μ / (MP.ρ0 * MP.B0);

  MP.cα2 = 4.;
  MP.τ2 = κ * MP.ρ0 / (MP.T0 * MP.cα2);

  return MP;
}

Par aluminium_params() {

  Par MP;
  MP.EOS = 4;

  MP.ρ0 = 1.;
  MP.p0 = 0.;
  MP.Tref = 300.;
  MP.T0 = 300.;
  MP.cv = 900.;

  MP.c02 = 6220 * 6220 - 4 / 3 * 3160 * 3160;
  MP.α = 1.;
  MP.β = 3.577;
  MP.γ = 2.088;

  MP.B0 = 3160. * 3160.;
  MP.τ1 = 1.;
  MP.PLASTIC = true;
  MP.σY = 0.4e9;
  MP.n = 100.;
  return MP;
}

Par vacuum_params() {
  Par MP;
  MP.EOS = -1;
  return MP;
}
