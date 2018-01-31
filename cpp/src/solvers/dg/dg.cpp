#include "../../etc/globals.h"
#include "../../scipy/newton_krylov.h"
#include "../../system/equations.h"
#include "../../system/objects/gpr_objects.h"
#include "../evaluations.h"

MatN2_V rhs1(MatN2_Vr q, MatN2_Vr Ww, double dt, double dx, Par &MP) {

  MatN2_V ret, F, dq_dx;
  F.setZero(N * N, V);
  VecV tmp;

  for (int t = 0; t < N; t++)
    dq_dx.block<N, V>(t * N, 0) = DERVALS * q.block<N, V>(t * N, 0) / dx;

  int ind = 0;
  for (int t = 0; t < N; t++)
    for (int i = 0; i < N; i++) {
      source(ret.row(ind), q.row(ind), MP);
      Bdot(tmp, q.row(ind), dq_dx.row(ind), 0, MP);
      ret.row(ind) -= tmp;
      ret.row(ind) *= WGHTS(t) * WGHTS(i);
      flux(F.row(ind), q.row(ind), 0, MP);
      ind += 1;
    }

  for (int t = 0; t < N; t++)
    ret.block<N, V>(t * N, 0) -=
        WGHTS(t) * DG_DER * F.block<N, V>(t * N, 0) / dx;

  ret *= dt;
  ret += Ww;
  return ret;
}

MatN3_V rhs2(MatN3_Vr q, MatN3_Vr Ww, double dt, double dx, double dy,
             Par &MP) {

  MatN3_V ret, F, G, dq_dx, dq_dy;
  F.setZero(N * N * N, V);
  G.setZero(N * N * N, V);
  VecV tmpx, tmpy;

  for (int t = 0; t < N; t++) {
    derivs2d(dq_dx.block<N * N, V>(t * N * N, 0),
             q.block<N * N, V>(t * N * N, 0), 0);
    derivs2d(dq_dy.block<N * N, V>(t * N * N, 0),
             q.block<N * N, V>(t * N * N, 0), 1);
  }
  dq_dx /= dx;
  dq_dy /= dy;

  int ind = 0;
  for (int t = 0; t < N; t++)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        source(ret.row(ind), q.row(ind), MP);
        Bdot(tmpx, q.row(ind), dq_dx.row(ind), 0, MP);
        Bdot(tmpy, q.row(ind), dq_dy.row(ind), 1, MP);
        ret.row(ind) -= tmpx + tmpy;
        ret.row(ind) *= WGHTS(t) * WGHTS(i) * WGHTS(j);
        flux(F.row(ind), q.row(ind), 0, MP);
        flux(G.row(ind), q.row(ind), 1, MP);
        ind += 1;
      }

  MatN3_V F2, G2;
  for (int i = 0; i < N * N * N; i++) {
    F2.row(i) = WGHTS(i % N) * F.row(i);
  }
  for (int i = 0; i < N * N; i++) {
    G2.block<N, V>(i * N, 0) = DG_DER * G.block<N, V>(i * N, 0);
  }

  ind = 0;
  for (int t = 0; t < N; t++)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        int indi = t * N + i;
        int indj = t * N + j;
        ret.block<N, V>(indi * N, 0) -=
            WGHTS(t) * DG_DER(i, j) * F2.block<N, V>(indj * N, 0) / dx;
      }
      ret.block<N, V>(ind * N, 0) -=
          WGHTS(t) * WGHTS(i) * G2.block<N, V>(ind * N, 0) / dy;
      ind += 1;
    }

  ret *= dt;
  ret += Ww;
  return ret;
}

Vec obj1(Vecr q, MatN2_Vr Ww, double dt, double dx, Par &MP) {

  MatN2_VMap qmat(q.data(), OuterStride(V));
  MatN2_V tmp = rhs1(qmat, Ww, dt, dx, MP);

  for (int t = 0; t < N; t++)
    for (int i = 0; i < N; i++)
      for (int k = 0; k < N; k++)
        tmp.row(t * N + i) -= DG_MAT(t, k) * WGHTS(i) * qmat.row(k * N + i);

  VecMap ret(tmp.data(), N * N * V);
  return ret;
}

Vec obj2(Vecr q, MatN3_Vr Ww, double dt, double dx, double dy, Par &MP) {
  MatN3_VMap qmat(q.data(), OuterStride(V));
  MatN3_V tmp = rhs2(qmat, Ww, dt, dx, dy, MP);

  for (int t = 0; t < N; t++)
    for (int k = 0; k < N; k++)
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
          int indt = (t * N + i) * N + j;
          int indk = (k * N + i) * N + j;
          tmp.row(indt) -=
              (DG_MAT(t, k) * WGHTS(i) * WGHTS(j)) * qmat.row(indk);
        }
  VecMap ret(tmp.data(), N * N * N * V);
  return ret;
}

void standard_initial_guess(Matr q, Matr w, int NT) {
  // Returns a Galerkin intial guess consisting of the value of q at t=0
  for (int i = 0; i < N; i++)
    q.block(i * NT, 0, NT, V) = w;
}

void hidalgo_initial_guess(Matr q, Matr w, int NT, double dt, Par &MP) {
  // Returns the initial guess found in DOI: 10.1007/s10915-010-9426-6

  standard_initial_guess(q, w, NT);
  /*
  Mat qt = w;

  for (int t=0; t<N; t++)
  {
      double DT;
      if (t==0)
          Dt = dt * NODES(0);
      else
          DT = dt * (NODES(t) - NODES(t-1));

      Mat dqdxt = derivs.dot(qt);

      for (int i=0; i<N; i++)
      {
          M = system_conserved(qi, 0, PAR, SYS).dot(dqdxt[i]);
          S = source(qt[i], PAR, SYS);

          if (superStiff)
          {
              f = lambda X: X - qt[i] + dt/dx * M - dt/2 *
  (S+source(X,PAR,SYS));
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

void initial_condition(Matr Ww, Matr w) {
  for (int t = 0; t < N; t++)
    for (int i = 0; i < N; i++)
      Ww.row(t * N + i) = ENDVALS(0, t) * WGHTS(i) * w.row(i);
}

void predictor(Vecr qh, Vecr wh, int ndim, double dt, double dx, double dy,
               double dz, bool STIFF, bool HIDALGO, Par &MP) {

  int ncell = qh.size() / (int(pow(N, ndim + 1)) * V);
  int NT = int(pow(N, ndim));

  Mat Ww(NT * N, V);
  Mat q0(NT * N, V);

  for (int ind = 0; ind < ncell; ind++) {

    MatMap wi(wh.data() + (ind * NT * V), NT, V, OuterStride(V));
    MatMap qi(qh.data() + (ind * NT * N * V), NT * N, V, OuterStride(V));

    initial_condition(Ww, wi);

    using std::placeholders::_1;
    VecFunc obj_bound;
    if (ndim == 1)
      obj_bound = std::bind(obj1, _1, Ww, dt, dx, MP);
    else if (ndim == 2)
      obj_bound = std::bind(obj2, _1, Ww, dt, dx, dy, MP);

    if (HIDALGO)
      hidalgo_initial_guess(q0, wi, NT, dt, MP);
    else
      standard_initial_guess(q0, wi, NT);

    if (STIFF) {
      VecMap q0v(q0.data(), NT * N * V);
      qh.segment(ind * NT * N * V, NT * N * V) = nonlin_solve(obj_bound, q0v);
    } else {

      bool FAIL = true;
      for (int count = 0; count < DG_ITER; count++) {

        Mat q1;
        if (ndim == 1)
          q1 = DG_U1.solve(rhs1(q0, Ww, dt, dx, MP));
        else if (ndim == 2)
          q1 = DG_U2.solve(rhs2(q0, Ww, dt, dx, dy, MP));

        Arr absDiff = (q1 - q0).array().abs();

        if ((absDiff > DG_TOL * (1 + q0.array().abs())).any()) {
          q0 = q1;
          continue;
        } else if (q1.array().isNaN().any()) {
          FAIL = true;
          break;
        } else {
          qi = q1;
          FAIL = false;
          break;
        }
      }
      if (FAIL) {
        hidalgo_initial_guess(q0, wi, NT, dt, MP);
        VecMap q0v(q0.data(), NT * N * V);
        qh.segment(ind * NT * N * V, NT * N * V) = nonlin_solve(obj_bound, q0v);
      }
    }
  }
}
