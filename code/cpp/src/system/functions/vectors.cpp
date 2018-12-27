#include "../../etc/types.h"
#include "../energy/eos.h"
#include "../objects.h"
#include "../variables/state.h"

Vec3Map get_ρv(VecVr Q)
{
  // Returns the momentum vector
  return Vec3Map(Q.data() + 2);
}

Mat3_3Map get_A(VecVr Q)
{
  // Returns the distortion matrix.
  return Mat3_3Map(Q.data() + 5);
}

Vec3Map get_ρJ(VecVr Q)
{
  // Returns the density times the thermal impulse vector
  return Vec3Map(Q.data() + 14);
}

int get_material_index(VecVr Q)
{
  int ret = 0;
  for (int i = V - LSET; i < V; i++)
    if (Q(i) >= 0.)
      ret += 1;
  return ret;
}

void Pvec(VecVr Q, Par &MP)
{
  // Turns conserved vector Q into a primitive vector
  double ρ = Q(0);
  double p = pressure(Q, MP);
  Q(1) = p;
  Q.segment<3>(2) /= ρ;
  if (THERMAL)
    Q.segment<3>(14) /= ρ;
  if (MULTI)
    Q(mV) /= ρ;
}

void Cvec(VecVr P, Par &MP)
{
  // Turns primitive vector Q into a conserved vector
  double ρ = P(0);
  double p = P(1);
  Vec3 v = P.segment<3>(2);
  Mat3_3Map A = get_A(P);
  if (THERMAL)
  {
    Vec3 J = P.segment<3>(14);
    if (MULTI)
    {
      double λ = P(mV);
      P(1) = ρ * total_energy(ρ, p, A, J, v, λ, MP);
    }
    else
    {
      P(1) = ρ * total_energy(ρ, p, A, J, v, MP);
    }
  }
  else
  {
    if (MULTI)
    {
      double λ = P(mV);
      P(1) = ρ * total_energy(ρ, p, A, v, λ, MP);
    }
    else
    {
      P(1) = ρ * total_energy(ρ, p, A, v, MP);
    }
  }
  P.segment<3>(2) *= ρ;
  if (THERMAL)
    P.segment<3>(14) *= ρ;
  if (MULTI)
    P(mV) *= ρ;
}

VecV Cvec_to_Pvec(VecV Q, Par &MP)
{
  // Returns vector of primitive variables (atypical ordering), given a vector
  // of conserved variables (typical ordering)
  double ρ = Q(0);
  double p = pressure(Q, MP);
  Vec3Map ρv = get_ρv(Q);
  Mat3_3Map A = get_A(Q);

  Q(1) = p;
  Q.segment<3>(2) = A.col(0);
  Q.segment<3>(5) = A.col(1);
  Q.segment<3>(8) = A.col(2);
  Q.segment<3>(11) = ρv / ρ;

  if (THERMAL)
    Q.segment<3>(14) /= ρ;

  if (MULTI)
    Q(mV) /= ρ;

  return Q;
}

VecV Pvec_to_Cvec(VecV P, Par &MP)
{
  // Returns vector of conserved variables (typical ordering), given a vector of
  // primitive variables (atypical ordering)
  double ρ = P(0);
  double p = P(1);

  Mat3_3 A;
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
      A(i, j) = P(2 + 3 * j + i);

  Vec3 v = P.segment<3>(11);

  P.segment<3>(2) = ρ * v;
  P.segment<9>(5) = Vec9Map(A.data());

  if (MULTI)
    P(mV) *= ρ;

  if (MP.REACTION > -1)
  {
    double λ = P(mV) / ρ;

    if (THERMAL)
    {
      Vec3 J = P.segment<3>(14);
      P(1) = ρ * total_energy(ρ, p, A, J, v, λ, MP);
      P.segment<3>(14) *= ρ;
    }
    else
      P(1) = ρ * total_energy(ρ, p, A, v, λ, MP);
  }
  else
  {
    if (THERMAL)
    {
      Vec3 J = P.segment<3>(14);
      P(1) = ρ * total_energy(ρ, p, A, J, v, MP);
      P.segment<3>(14) *= ρ;
    }
    else
      P(1) = ρ * total_energy(ρ, p, A, v, MP);
  }
  return P;
}
