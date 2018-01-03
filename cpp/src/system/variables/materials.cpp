#include "../../etc/globals.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../variables/state.h"


double theta_1(VecVr Q, Par & MP)
{   // Returns the function used in the source terms for the distortion tensor
    // NB May be more suitable to use a different form for other fluids/solids

    Mat3_3Map A = get_A(Q);
    double den = pow(det(A), 5./3.);
    return (MP.cs2 * MP.τ1) / (3 * den);
}

double theta_2(VecVr Q, Par & MP)
{   // Returns the function used in the source terms for the thermal impulse vector
    // NB May be more suitable to use a different form for other fluids/solids

    double ρ = Q(0);
    double p = pressure(Q, MP);
    double T = temperature(ρ, p, MP);
    return MP.α2 * MP.τ2 * (ρ / MP.ρ0) * (MP.T0 / T);
}
