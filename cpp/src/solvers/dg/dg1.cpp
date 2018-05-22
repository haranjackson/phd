#include "../../etc/globals.h"
#include "../../scipy/newton_krylov.h"
#include "../../system/equations.h"
#include "../../system/objects/gpr_objects.h"
#include "../evaluations.h"
#include "initial_guess.h"

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

void initial_condition1(Matr Ww, Matr w) {

  for (int t = 0; t < N; t++)
    for (int i = 0; i < N; i++)
      Ww.row(t * N + i) = ENDVALS(0, t) * WGHTS(i) * w.row(i);
}

void predictor1(Vecr qh, Vecr wh, double dt, double dx, bool STIFF,
                bool STIFF_IG, Par &MP, bVecr mask) {

  int ncell = mask.size();
  Mat Ww(N * N, V);
  Mat q0(N * N, V);

  for (int ind = 0; ind < ncell; ind++) {

    if (mask(ind)) {

      MatMap wi(wh.data() + (ind * N * V), N, V, OuterStride(V));
      MatMap qi(qh.data() + (ind * N * N * V), N * N, V, OuterStride(V));

      initial_condition1(Ww, wi);

      using std::placeholders::_1;
      VecFunc obj_bound = std::bind(obj1, _1, Ww, dt, dx, MP);

      if (STIFF_IG)
        stiff_initial_guess(q0, wi, N, dt, MP);
      else
        standard_initial_guess1(q0, wi);

      if (STIFF) {
        VecMap q0v(q0.data(), N * N * V);
        qh.segment<N * N * V>(ind * N * N * V) =
            nonlin_solve(obj_bound, q0v, DG_TOL);
      } else {

        Mat q1(N * N, V);
        aMat absDiff(N * N, V);

        for (int count = 0; count < DG_IT; count++) {

          q1 = DG_U1.solve(rhs1(q0, Ww, dt, dx, MP));

          absDiff = (q1 - q0).array().abs();

          if ((absDiff > DG_TOL * (1 + q0.array().abs())).any()) {
            q0 = q1;
            continue;

          } else if (q1.array().isNaN().any()) {
            stiff_initial_guess(q0, wi, N, dt, MP);
            VecMap q0v(q0.data(), N * N * V);
            qh.segment<N * N * V>(ind * N * N * V) =
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
