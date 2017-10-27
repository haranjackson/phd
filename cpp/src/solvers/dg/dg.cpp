#include "../evaluations.h"
#include "../../etc/globals.h"
#include "../../scipy/newton_krylov.h"
#include "../../system/equations.h"
#include "../../system/objects/gpr_objects.h"


Matn2_V rhs1(Matn2_Vr q, Matn2_Vr Ww, double dt, double dx, Par & MP)
{
    Matn2_V ret, F, dq_dx;
    F.setZero(N1N1,V);
    VecV tmp;

    for (int t=0; t<N1; t++)
        dq_dx.block<N1,V>(t*N1,0) = DERVALS * q.block<N1,V>(t*N1,0) / dx;

    int ind = 0;
    for (int t=0; t<N1; t++)
        for (int i=0; i<N1; i++)
        {
            source(ret.row(ind), q.row(ind), MP);
            Bdot(tmp, q.row(ind), dq_dx.row(ind), 0);
            ret.row(ind) -= tmp;
            ret.row(ind) *= WGHTS(t) * WGHTS(i);
            flux(F.row(ind), q.row(ind), 0, MP);
            ind += 1;
        }

    for (int t=0; t<N1; t++)
        ret.block<N1,V>(t*N1,0) -= WGHTS(t) * DG_DER * F.block<N1,V>(t*N1,0) / dx;

    ret *= dt;
    ret += Ww;
    return ret;
}

Matn3_V rhs2(Matn3_Vr q, Matn3_Vr Ww, double dt, double dx, double dy, Par & MP)
{
    Matn3_V ret, F, G, dq_dx, dq_dy;
    F.setZero(N1N1N1,V);
    G.setZero(N1N1N1,V);
    VecV tmpx, tmpy;

    for (int t=0; t<N1; t++)
    {
        derivs2d(dq_dx.block<N1N1,V>(t*N1N1,0), q.block<N1N1,V>(t*N1N1,0), 0);
        derivs2d(dq_dy.block<N1N1,V>(t*N1N1,0), q.block<N1N1,V>(t*N1N1,0), 1);
    }
    dq_dx /= dx;
    dq_dy /= dy;

    int ind = 0;
    for (int t=0; t<N1; t++)
        for (int i=0; i<N1; i++)
            for (int j=0; j<N1; j++)
            {
                source(ret.row(ind), q.row(ind), MP);
                Bdot(tmpx, q.row(ind), dq_dx.row(ind), 0);
                Bdot(tmpy, q.row(ind), dq_dy.row(ind), 1);
                ret.row(ind) -= tmpx + tmpy;
                ret.row(ind) *= WGHTS(t) * WGHTS(i) * WGHTS(j);
                flux(F.row(ind), q.row(ind), 0, MP);
                flux(G.row(ind), q.row(ind), 1, MP);
                ind += 1;
            }

    Matn3_V F2, G2;
    for (int i=0; i<N1N1N1; i++)
    {
        F2.row(i) = WGHTS(i%N1) * F.row(i);
    }
    for (int i=0; i<N1N1; i++)
    {
        G2.block<N1,V>(i*N1,0) = DG_DER * G.block<N1,V>(i*N1,0);
    }

    ind = 0;
    for (int t=0; t<N1; t++)
        for (int i=0; i<N1; i++)
        {
            for (int j=0; j<N1; j++)
            {
                int indi = t*N1+i;
                int indj = t*N1+j;
                ret.block<N1,V>(indi*N1,0) -= WGHTS(t) * DG_DER(i,j) * F2.block<N1,V>(indj*N1,0) / dx;
            }
            ret.block<N1,V>(ind*N1,0) -= WGHTS(t) * WGHTS(i) * G2.block<N1,V>(ind*N1,0) / dy;
            ind += 1;
        }

    ret *= dt;
    ret += Ww;
    return ret;
}

Vec obj1(Vec q, Matn2_Vr Ww, double dt, double dx, Par & MP)
{
    Matn2_VMap qmat (q.data(), OuterStride(V));
    Matn2_V tmp = rhs1(qmat, Ww, dt, dx, MP);

    for (int t=0; t<N1; t++)
        for (int i=0; i<N1; i++)
            for (int k=0; k<N1; k++)
            {
                tmp.row(t*N1+i) -= DG_MAT(t,k) * WGHTS(i) * qmat.row(k*N1+i);
            }
    VecMap ret (tmp.data(), N1N1V);
    return ret;
}

Vec obj2(Vec q, Matn3_Vr Ww, double dt, double dx, double dy, Par & MP)
{
    Matn3_VMap qmat (q.data(), OuterStride(V));
    Matn3_V tmp = rhs2(qmat, Ww, dt, dx, dy, MP);

    for (int t=0; t<N1; t++)
        for (int k=0; k<N1; k++)
            for (int i=0; i<N1; i++)
                for (int j=0; j<N1; j++)
                {
                    int indt = (t*N1+i)*N1+j;
                    int indk = (k*N1+i)*N1+j;
                    tmp.row(indt) -= (DG_MAT(t,k) * WGHTS(i) * WGHTS(j))
                                     * qmat.row(indk);
                }
    VecMap ret (tmp.data(), N1N1N1V);
    return ret;
}

void standard_initial_guess(Matr q, Matr w, int NT)
{   // Returns a Galerkin intial guess consisting of the value of q at t=0

    for (int i=0; i<N1; i++)
        q.block(i*NT, 0, NT, V) = w;
}

void hidalgo_initial_guess(Matr q, Matr w, int NT, double dt, Par & MP)
{   // Returns the initial guess found in DOI: 10.1007/s10915-010-9426-6

    standard_initial_guess(q, w, NT);
    /*
    Mat qt = w;

    for (int t=0; t<N1; t++)
    {
        double DT;
        if (t==0)
            Dt = dt * NODES(0);
        else
            DT = dt * (NODES(t) - NODES(t-1));

        Mat dqdxt = dot(derivs, qt);

        for (int i=0; i<N1; i++)
        {
            M = dot(system_conserved(qi, 0, PAR, SYS), dqdxt[i]);
            S = source(qt[i], PAR, SYS);

            if (superStiff)
            {
                f = lambda X: X - qt[i] + dt/dx * M - dt/2 * (S+source(X,PAR,SYS));
                q[t,i] = newton_krylov(f, qi, f_tol=TOL);
            }
            else
            {
                q[t,i] = qi - dt/dx * M + dt * Sj;
            }
        }
        qt = q[t];
    }
    */
}

void initial_condition(Matr Ww, Matr w)
{
    for (int t=0; t<N1; t++)
        for (int i=0; i<N1; i++)
            Ww.row(t*N1+i) = ENDVALS(0,t) * WGHTS(i) * w.row(i);
}

void predictor(Vecr qh, Vecr wh, int ndim,
               double dt, double dx, double dy, double dz,
               bool STIFF, bool HIDALGO, Par & MP)

{
    int ncell = qh.size() / (int(pow(N1, ndim+1)) * V);
    int NT = int(pow(N1, ndim));

    Mat Ww (NT*N1, V);
    Mat q0 (NT*N1, V);

    for (int ind=0; ind<ncell; ind++)
    {
        MatMap wi (wh.data()+(ind*NT*V), NT, V, OuterStride(V));
        MatMap qi (qh.data()+(ind*NT*N1V), NT*N1, V, OuterStride(V));

        initial_condition(Ww, wi);

        using std::placeholders::_1;
        VecFunc obj_bound;
        switch(ndim)
        {
            case 1 : obj_bound = std::bind(obj1, _1, Ww, dt, dx, MP);
                     break;
            case 2 : obj_bound = std::bind(obj2, _1, Ww, dt, dx, dy, MP);
                     break;
        }

        if (HIDALGO)
            hidalgo_initial_guess(q0, wi, NT, dt, MP);
        else
            standard_initial_guess(q0, wi, NT);

        if (STIFF)
        {
            VecMap q0v (q0.data(), NT*N1V);
            qh.segment(ind*NT*N1V, NT*N1V) = nonlin_solve(obj_bound, q0v);
        }
        else
        {
            bool FAIL = true;
            for (int count=0; count<DG_ITER; count++)
            {
                Mat q1;
                switch(ndim)
                {
                    case 1 : q1 = DG_U1.solve(rhs1(q0, Ww, dt, dx, MP));
                             break;
                    case 2 : q1 = DG_U2.solve(rhs2(q0, Ww, dt, dx, dy, MP));
                             break;
                }
                Arr absDiff = (q1-q0).array().abs();

                if (( absDiff > DG_TOL * (1 + q0.array().abs() ) ).any())
                {
                    q0 = q1;
                    continue;
                }
                else
                {
                    qi = q1;
                    FAIL = false;
                    break;
                }
            }
            if (FAIL)
            {
                hidalgo_initial_guess(q0, wi, NT, dt, MP);
                VecMap q0v (q0.data(), NT*N1V);
                qh.segment(ind*NT*N1V, NT*N1V) = nonlin_solve(obj_bound, q0v);
            }
        }
    }
}
