#include <cmath>

#include "../../etc/globals.h"
#include "../functions/matrices.h"
#include "../functions/vectors.h"
#include "../objects/gpr_objects.h"


double E_1(double r, double p, Par & MP)
{   // Returns the microscale energy corresponding to a stiffened gas
    // NB The ideal gas equation is obtained if pINF=0

    double γ = MP.γ;
    double pINF = MP.pINF;
    return (p + γ*pINF) / ((γ-1) * r);
}

double E_2A(VecVr Q, Par & MP)
{   // Returns the mesoscale energy dependent on the distortion

    Mat3_3Map A = get_A(Q);
    double G00 = dot(A.row(0), A.row(0));
    double G11 = dot(A.row(1), A.row(1));
    double G22 = dot(A.row(2), A.row(2));
    double G01 = dot(A.row(0), A.row(1));
    double G02 = dot(A.row(0), A.row(2));
    double G12 = dot(A.row(1), A.row(2));
    double t = (G00 + G11 + G22) / 3;
    return MP.cs2 / 4 * ( (G00-t)*(G00-t) + (G11-t)*(G11-t) + (G22-t)*(G22-t) \
            + 2 * (G01*G01 + G02*G02 + G12*G12) );
}

double E_2J(VecVr Q, Par & MP)
{   // Returns the mesoscale energy dependent on the thermal impulse

    double r = Q(0);
    Vec3Map rJ = get_rJ(Q);
    return MP.α2 * L2_1D(rJ) / (2*r*r);
}

double E_3(VecVr Q)
{   // Returns the macroscale kinetic energy

    double r = Q(0);
    Vec3Map rv = get_rv(Q);
    return L2_1D(rv) / (2*r*r);
}

Mat3_3 dEdA(VecVr Q, Par & MP)
{   // Returns the partial derivative of E by A

    Mat3_3Map A = get_A(Q);
    return MP.cs2 * AdevG(A);
}

Vec3 dEdJ(VecVr Q, Par & MP)
{   // Returns the partial derivative of E by J

    double r = Q(0);
    Vec3Map rJ = get_rJ(Q);
    return MP.α2 * rJ / r;
}
