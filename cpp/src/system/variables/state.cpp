#include "eos.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"


// Functions depending only on conserved variables

double pressure(VecVr Q, Par & MP)
{   // Returns the pressure, given the total energy, velocity, distortion matrix, and density.
    // NB Only valid for EOS used for fluids by Dumbser et al.

    double r = Q(0);
    double E = Q(1) / r;
    double E1 = E - E_2A(Q, MP) - E_2J(Q, MP) - E_3(Q);
    return (MP.γ-1) * r * E1 - MP.γ * MP.pINF;
}

Mat3_3 sigma(VecVr Q, Par & MP)
{   // Returns the symmetric viscous shear stress tensor

    double r = Q(0);
    Mat3_3Map A = get_A(Q);
    return -r * A.transpose() * dEdA(Q, MP);
}

Vec3 sigma(VecVr Q, Par & MP, int d)
{   // Returns the dth column of the symmetric viscous shear stress tensor

    double r = Q(0);
    Mat3_3Map A = get_A(Q);
    Mat3_3 E_A = dEdA(Q, MP);
    return -r * E_A.transpose() * A.col(d);
}


// Functions depending on compound variables

double temperature(double r, double p, Par & MP)
{   // Returns the temperature for an stiffened gas

    return (p + MP.pINF) / ((MP.γ-1) * r * MP.cv);
}

Vec3 heat_flux(double T, Vec3r J, Par & MP)
{   // Returns the heat flux vector

    return MP.α2 * T * J;
}
