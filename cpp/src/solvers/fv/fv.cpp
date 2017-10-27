#include "fluxes.h"
#include "../evaluations.h"
#include "../../etc/globals.h"
#include "../../system/functions/vectors.h"
#include "../../system/equations.h"


int ind(int t, int i, int nx)
{
    return t * nx + i;
}

int ind(int t, int i, int j, int nx, int ny)
{
    return (t * nx + i) * ny + j;
}

void centers1(Vecr u, Vecr rec, int nt, int nx, double dt, double dx,
              Vecnr WGHTS_T, bool SOURCES, Par & MP)
{
    Matn_V dqh_dx;
    VecV dqdxs, qs, S, tmpx;

    for (int t=0; t<nt; t++)
        for (int i=0; i<nx; i++)
        {
            int idx = ind(t, i+1, nx+2) * N1V;
            Matn_VMap qh (rec.data() + idx, OuterStride(V));
            dqh_dx.noalias() = DERVALS * qh;
            double wght_t = dt * WGHTS_T(t);

            for (int s=0; s<N1; s++)
            {
                qs = qh.row(s);
                dqdxs = dqh_dx.row(s);

                if (SOURCES)
                    source(S, qs, MP);
                else
                    S.setZero(V);

                Bdot(tmpx, qs, dqdxs, 0);

                S -= tmpx / dx;

                u.segment<V>(i*V) += wght_t * WGHTS(s) * S;
            }
        }
}

void centers2(Vecr u, Vecr rec, int nt, int nx, int ny,
              double dt, double dx, double dy, Vecnr WGHTS_T, bool SOURCES,
              Par & MP)
{
    Matn2_V dqh_dx, dqh_dy;
    VecV qs, dqdxs, dqdys, S, tmpx, tmpy;

    for (int t=0; t<nt; t++)
        for (int i=0; i<nx; i++)
            for (int j=0; j<ny; j++)
            {
                int idx = ind(t, i+1, j+1, nx+2, ny+2) * N1N1V;

                Matn2_VMap qh (rec.data()+idx, OuterStride(V));
                derivs2d(dqh_dx, qh, 0);
                derivs2d(dqh_dy, qh, 1);
                double wght_t = dt * WGHTS_T(t);

                for (int a=0; a<N1; a++)
                    for (int b=0; b<N1; b++)
                    {
                        int s = a*N1 + b;
                        qs = qh.row(s);
                        dqdxs = dqh_dx.row(s);
                        dqdys = dqh_dy.row(s);

                        if (SOURCES)
                            source(S, qs, MP);
                        else
                            S.setZero(V);

                        Bdot(tmpx, qs, dqdxs, 0);
                        Bdot(tmpy, qs, dqdys, 1);

                        S -= tmpx / dx;
                        S -= tmpy / dy;

                        u.segment<V>((i*ny+j)*V) += wght_t * WGHTS(a) * WGHTS(b) * S;
                    }
            }
}

void interfs1(Vecr u, Vecr rec, int nt, int nx, double dt, double dx,
              Vecnr WGHTS_T, bool PERRON_FROBENIUS, Par & MP)
{
    double k = dt/(2*dx);
    VecV ql, qr, f, b;

    for (int t=0; t<nt; t++)
        for (int i=0; i<nx+1; i++)
        {
            int indl = ind(t, i, nx+2) * N1V;
            int indr = ind(t, i+1, nx+2) * N1V;
            Matn_VMap qhl (rec.data()+indl, OuterStride(V));
            Matn_VMap qhr (rec.data()+indr, OuterStride(V));
            ql.noalias() = ENDVALS.row(1) * qhl;
            qr.noalias() = ENDVALS.row(0) * qhr;

            f = Smax(ql, qr, 0, PERRON_FROBENIUS, MP);
            flux(f, ql, 0, MP);
            flux(f, qr, 0, MP);
            b = Bint(ql, qr, 0);

            if (i>0)
                u.segment<V>((i-1)*V) -= WGHTS_T(t) * k * (b + f);
            if (i<nx)
                u.segment<V>(i*V) -= WGHTS_T(t) * k * (b - f);
        }
}

void interfs2(Vecr u, Vecr rec, int nt, int nx, int ny,
              double dt, double dx, double dy, Vecnr WGHTS_T,
              bool PERRON_FROBENIUS, Par & MP)
{
    double kx = dt/(2*dx);
    double ky = dt/(2*dy);

    Matn_V q0x, q0y, q1x, q1y;
    VecV qlx, qrx, qly, qry, fx, bx, fy, by;

    for (int t=0; t<nt; t++)
    {
        double wghts_t = WGHTS_T(t);

        for (int i=0; i<nx+1; i++)
            for (int j=0; j<ny+1; j++)
            {
                if ((i==0 || i==nx+1) && (j==0 || j==ny+1))
                    continue;

                int uind0 = ind(i-1, j-1, ny) * V;
                int uindx = ind(i,   j-1, ny) * V;
                int uindy = ind(i-1, j,   ny) * V;

                int ind0 = ind(t,  i,  j, nx+2, ny+2) * N1N1V;
                int indx = ind(t, i+1, j, nx+2, ny+2) * N1N1V;
                int indy = ind(t, i, j+1, nx+2, ny+2) * N1N1V;

                Matn2_VMap qh0 (rec.data()+ind0, OuterStride(V));
                Matn2_VMap qhx (rec.data()+indx, OuterStride(V));
                Matn2_VMap qhy (rec.data()+indy, OuterStride(V));

                endpts2d(q0x, qh0, 0, 1);
                endpts2d(q0y, qh0, 1, 1);
                endpts2d(q1x, qhx, 0, 0);
                endpts2d(q1y, qhy, 1, 0);

                for (int s=0; s<N1; s++)
                {
                    qlx = q0x.row(s);
                    qrx = q1x.row(s);
                    qly = q0y.row(s);
                    qry = q1y.row(s);

                    fx = Smax(qlx, qrx, 0, PERRON_FROBENIUS, MP);
                    flux(fx, qlx, 0, MP);
                    flux(fx, qrx, 0, MP);
                    bx = Bint(qlx, qrx, 0);

                    fy = Smax(qly, qry, 1, PERRON_FROBENIUS, MP);
                    flux(fy, qly, 1, MP);
                    flux(fy, qry, 1, MP);
                    by = Bint(qly, qry, 1);

                    double wght = wghts_t * WGHTS(s);

                    if (i>0 && i<nx+1 && j>0 && j<ny+1)
                    {
                        u.segment<V>(uind0) -= wght * kx * (bx+fx);
                        u.segment<V>(uind0) -= wght * ky * (by+fy);
                    }
                    if (i<nx && j>0 and j<ny+1)
                        u.segment<V>(uindx) -=  wght * kx * (bx-fx);
                    if (i>0 && i<nx+1 && j<ny)
                        u.segment<V>(uindy) -=  wght * ky * (by-fy);
                }
            }
        }
}

void fv_launcher(Vecr u, Vecr rec,
                 int ndim, int nx, int ny, int nz,
                 double dt, double dx, double dy, double dz,
                 bool SOURCES, bool TIME, bool PERRON_FROBENIUS,
                 Par & MP)
{
    int nt = TIME ? N1 : 1;
    Vecn WGHTS_T;
    if (TIME)
        WGHTS_T = WGHTS;
    else
        WGHTS_T = Vecn::Constant(1.);

    switch (ndim)
    {
        case 1 : centers1(u, rec, nt, nx, dt, dx, WGHTS_T, SOURCES, MP);
                 interfs1(u, rec, nt, nx, dt, dx, WGHTS_T,
                          PERRON_FROBENIUS, MP);
                 break;
        case 2 : centers2(u, rec, nt, nx, ny, dt, dx, dy, WGHTS_T, SOURCES, MP);
                 interfs2(u, rec, nt, nx, ny, dt, dx, dy, WGHTS_T,
                          PERRON_FROBENIUS, MP);
                 break;
        // case 3 : TODO
    }
}
