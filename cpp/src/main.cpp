#include <iostream>

#include "etc/globals.h"
#include "solvers/iterator.h"
#include "system/objects/gpr_objects.h"
#include "system/functions/vectors.h"


Vec heat_conduction_1d(Par MP)
{
    int nx = 200;
    double rL = 2;
    double rR = 0.5;
    double rL_3 = pow(rL, 1/3.);
    double rR_3 = pow(rR, 1/3.);
    Mat AL = rL_3 * Mat::Identity(3,3);
    Mat AR = rR_3 * Mat::Identity(3,3);
    Vec v = Vec::Zero(3);
    Vec J = Vec::Zero(3);

    VecV QL = Qvec(rL, 1., v, AL, J, MP);
    VecV QR = Qvec(rR, 1., v, AR, J, MP);

    Vec u (nx*V);
    double dx = 1. / nx;
    for (int i=0; i<nx; i++)
    {
        if (i*dx < 0.5)
            u.segment(i*V,V) = QL;
        else
            u.segment(i*V,V) = QR;
    }
    return u;
}


int main()
{
    Par MP;
    MP.gamma = 1.4;
    MP.cv = 2.5;
    MP.pinf = 0.;
    MP.r0 = 1.;
    MP.p0 = 1.;
    MP.T0 = 1.;
    MP.cs2 = 1.;
    MP.mu = 1e-2;
    MP.tau1 = 6 * MP.mu / (MP.r0 * MP.cs2);
    MP.alpha2 = 4.;
    MP.kappa = 1e-2;
    MP.tau2 = MP.kappa * MP.r0 / (MP.T0 * MP.alpha2);
    MP.Pr = 0.;

    Vec u = heat_conduction_1d(MP);

    iterator(u, 0.1, 20, 10, 1, 0.005, 0.005, 0.005, 0.6,
             false, true, true, true, true, MP);

    std::cout << "Hello World" << std::endl;
    return 0;
}
