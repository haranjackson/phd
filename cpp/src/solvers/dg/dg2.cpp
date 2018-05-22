#include "../../etc/globals.h"
#include "../../scipy/newton_krylov.h"
#include "../../system/equations.h"
#include "../../system/objects/gpr_objects.h"
#include "../evaluations.h"
#include "initial_guess.h"

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

void initial_condition2(Matr Ww, Matr w) {

  for (int t = 0; t < N; t++)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        Ww.row(t * N * N + i * N + j) =
            ENDVALS(0, t) * WGHTS(i) * WGHTS(j) * w.row(i * N + j);
}

void predictor2(Vecr qh, Vecr wh, double dt, double dx, double dy, bool STIFF,
                bool STIFF_IG, Par &MP, bVecr mask) {

  int ncell = mask.size();
  Mat Ww(N * N * N, V);
  Mat q0(N * N * N, V);

  for (int ind = 0; ind < ncell; ind++) {

    if (mask(ind)) {

      MatMap wi(wh.data() + (ind * N * N * V), N * N, V, OuterStride(V));
      MatMap qi(qh.data() + (ind * N * N * N * V), N * N * N, V,
                OuterStride(V));

      initial_condition2(Ww, wi);

      using std::placeholders::_1;
      VecFunc obj_bound = std::bind(obj2, _1, Ww, dt, dx, dy, MP);

      if (STIFF_IG)
        stiff_initial_guess(q0, wi, N * N, dt, MP);
      else
        standard_initial_guess2(q0, wi);

      if (STIFF) {
        VecMap q0v(q0.data(), N * N * N * V);
        qh.segment(ind * N * N * N * V, N * N * N * V) =
            nonlin_solve(obj_bound, q0v, DG_TOL);

      } else {

        Mat q1(N * N * N, V);
        aMat absDiff(N * N * N, V);

        for (int count = 0; count < DG_IT; count++) {

          q1 = DG_U2.solve(rhs2(q0, Ww, dt, dx, dy, MP));

          absDiff = (q1 - q0).array().abs();

          if ((absDiff > DG_TOL * (1 + q0.array().abs())).any()) {
            q0 = q1;
            continue;

          } else if (q1.array().isNaN().any()) {
            stiff_initial_guess(q0, wi, N * N, dt, MP);
            VecMap q0v(q0.data(), N * N * N * V);
            qh.segment(ind * N * N * N * V, N * N * N * V) =
                nonlin_solve(obj_bound, q0v, DG_TOL);
            break;

          } else {
            qi = q1;
            break;
          }
        }
      }
    }
  }
}
