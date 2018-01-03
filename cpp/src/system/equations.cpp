#include "../etc/globals.h"
#include "objects/gpr_objects.h"
#include "functions/matrices.h"
#include "functions/vectors.h"
#include "variables/eos.h"
#include "variables/materials.h"
#include "variables/state.h"


void flux(VecVr ret, VecVr Q, int d,
          Par & MP)
{   // Adds the flux vector in the dth direction, given conserved variables Q
    // NOTE It may be necessary to initialize ret to 0 first

    double ρ = Q(0);
    double rE = Q(1);

    Vec3Map rv = get_rv(Q);
    Mat3_3Map A = get_A(Q);
    Vec3Map rJ = get_rJ(Q);

    double p = pressure(Q, MP);
    double T = temperature(ρ, p, MP);

    double vd = rv(d) / ρ;

    ret(0) += rv(d);
    ret(1) += vd * (rE + p);
    ret.segment<3>(2) += vd * rv;
    ret(2+d) += p;

    Vec3 sigmad = sigma(Q, MP, d);
    ret(1) -= dot(sigmad, rv) / ρ;
    ret.segment<3>(2) -= sigmad;

    Vec3 Av = A * rv / ρ;
    ret(5+d) += Av(0);
    ret(8+d) += Av(1);
    ret(11+d) += Av(2);

    ret(1) += MP.α2 * T * rJ(d) / ρ;
    ret.tail<3>() += vd * rJ;
    ret(14+d) += T;
}


void source(VecVr ret, VecVr Q,
            Par & MP)
{
    double ρ = Q(0);

    ret.head<5>().setZero();

    Mat3_3 Asource = - dEdA(Q, MP) / theta_1(Q, MP);
    ret.segment<9>(5) = VecMap(Asource.data(), 9);

    ret.tail<3>() = - ρ * dEdJ(Q, MP) / theta_2(Q, MP);
}


void block(MatV_Vr ret, VecVr Q, int d)
{
    double ρ = Q(0);
    Vec3Map rv = get_rv(Q);
    double vd = rv(d) / ρ;
    for (int i=5; i<14; i++)
        ret(i,i) = vd;
    for (int i=0; i<3; i++)
    {
        double vi = rv(i) / ρ;
        ret(5+d, 5+d+i) -= vi;
        ret(8+d, 8+d+i) -= vi;
        ret(11+d, 11+d+i) -= vi;
    }
}


void B0dot(VecVr ret, VecVr x, Vec3 v)
{
    double v0 = v(0);
    double v1 = v(1);
    double v2 = v(2);
    ret.head<5>().setZero();
    ret.tail<3>().setZero();
    ret(5) = - v1 * x(6) - v2 * x(7);
    ret(6) = v0 * x(6);
    ret(7) = v0 * x(7);
    ret(8) = - v1 * x(9) - v2 * x(10);
    ret(9) = v0 * x(9);
    ret(10) = v0 * x(10);
    ret(11) = - v1 * x(12) - v2 * x(13);
    ret(12) = v0 * x(12);
    ret(13) = v0 * x(13);
}

void B1dot(VecVr ret, VecVr x, Vec3 v)
{
    double v0 = v(0);
    double v1 = v(1);
    double v2 = v(2);
    ret.head<5>().setZero();
    ret.tail<3>().setZero();
    ret(5) = v1 * x(5);
    ret(6) = - v0 * x(5) - v2 * x(7);
    ret(7) = v1 * x(7);
    ret(8) = v1 * x(8);
    ret(9) = - v0 * x(8) - v2 * x(10);
    ret(10) = v1 * x(10);
    ret(11) = v1 * x(11);
    ret(12) = - v0 * x(11) - v2 * x(13);
    ret(13) = v1 * x(13);
}

void B2dot(VecVr ret, VecVr x, Vec3 v)
{
    double v0 = v(0);
    double v1 = v(1);
    double v2 = v(2);
    ret.head<5>().setZero();
    ret.tail<3>().setZero();
    ret(5) = v2 * x(5);
    ret(6) = v2 * x(6);
    ret(7) = - v0 * x(5) - v1 * x(6);
    ret(8) = v2 * x(8);
    ret(9) = v2 * x(9);
    ret(10) = - v0 * x(8) - v1 * x(9);
    ret(11) = v2 * x(11);
    ret(12) = v2 * x(12);
    ret(13) = - v0 * x(11) - v1 * x(12);
}

void Bdot(VecVr ret, VecVr Q, VecVr x, int d)
{
    double ρ = Q(0);
    Vec3 v = get_rv(Q) / ρ;
    if (d==0)
        B0dot(ret, x, v);
    else if (d==1)
        B1dot(ret, x, v);
    else
        B2dot(ret, x, v);
}
