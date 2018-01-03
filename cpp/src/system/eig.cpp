#include "functions/matrices.h"
#include "functions/vectors.h"
#include "objects/gpr_objects.h"
#include "variables/state.h"


Mat4_4 thermo_acoustic_tensor(double r, Mat3_3r A, double p, double T, int d,
                              Par & MP)
{   // Returns the tensor T_dij corresponding to the (i,j) component of the
    // thermo-acoustic tensor in the dth direction

    Mat3_3 G = gram(A);
    Mat4_4 ret;
    Vec3 Gd = G.col(d);

    double γ = MP.γ;
    double pINF = MP.pINF;

    Mat3_3 O = GdevG(G);
    O.col(d) *= 2.;
    O.row(d) *= 2.;
    O(d,d) *= 3./4.;
    O += Gd(d) * G + 1./3. * outer(Gd, Gd);
    O *= MP.cs2;
    ret.topLeftCorner<3,3>() = O;

    ret(d,d) += γ * p / r;

    ret(3,0) = ((γ-1) * p - pINF) * T / (r * (p+pINF));
    ret(3,1) = 0.;
    ret(3,2) = 0.;
    double tmp = (γ-1) * MP.α2 * T / r;
    ret(0,3) = tmp;
    ret(1,3) = 0.;
    ret(2,3) = 0.;
    ret(3,3) = tmp * T / (p+pINF);

    return ret;
}

double max_abs_eigs(VecVr Q, int d, bool PERRON_FROBENIUS,
                    Par & MP)
{   // Returns the maximum of the absolute values of the eigenvalues of the GPR
    // system

    double r = Q(0);
    double vd = Q(2+d) / r;
    Mat3_3Map A = get_A(Q);

    double p = pressure(Q, MP);
    double T = temperature(r, p, MP);

    Mat4_4 O = thermo_acoustic_tensor(r, A, p, T, d, MP);

    double lam;
    if (PERRON_FROBENIUS)
    {
        double r01 = std::max(O.row(0).sum(), O.row(1).sum());
        double r23 = std::max(O.row(2).sum(), O.row(3).sum());
        double c01 = std::max(O.col(0).sum(), O.col(1).sum());
        double c23 = std::max(O.col(2).sum(), O.col(3).sum());
        double r = std::max(r01, r23);
        double c = std::max(c01, c23);
        lam = sqrt(std::min(r, c));
    }
    else
    {
        lam = sqrt(O.eigenvalues().array().abs().maxCoeff());
    }
    if (vd > 0)
        return vd + lam;
    else
        return lam - vd;
}
