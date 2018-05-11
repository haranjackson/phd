#include "../../etc/globals.h"
#include "../../system/equations.h"
#include "../evaluations.h"

void midstepper(Vecr wh, int ndim, double dt, Vecr dX, Par &MP, bVecr mask) {
  // Steps the WENO reconstruction forwards by dt/2
  // NOTE: Only for the homogeneous system

  int ncell = mask.size();
  double dx = dX(0);

  if (ndim == 1) {
    MatN_V F, dwdx, dFdx;
    VecV Bdwdx;

    for (int ind = 0; ind < ncell; ind += 1) {
      if (mask(ind)) {

        MatN_VMap w(wh.data() + ind * N * V, OuterStride(V));

        F.setZero(N, V);
        for (int a = 0; a < N; a++)
          flux(F.row(a), w.row(a), 0, MP);

        dwdx.noalias() = DERVALS * w;
        dFdx.noalias() = DERVALS * F;

        for (int a = 0; a < N; a++) {
          Bdot(Bdwdx, w.row(a), dwdx.row(a), 0, MP);
          w.row(a) -= dt / 2 * (dFdx.row(a) + Bdwdx.transpose()) / dx;
        }
      }
    }
  }
  if (ndim == 2) {
    double dy = dX(1);
    MatN2_V F, G, dwdx, dwdy, dFdx, dGdy;
    VecV Bdwdx, Bdwdy;

    for (int ind = 0; ind < ncell; ind += 1) {
      if (mask(ind)) {

        MatN2_VMap w(wh.data() + ind * N * N * V, OuterStride(V));

        F.setZero(N * N, V);
        G.setZero(N * N, V);
        for (int s = 0; s < N * N; s++) {
          flux(F.row(s), w.row(s), 0, MP);
          flux(G.row(s), w.row(s), 1, MP);
        }

        derivs2d(dwdx, w, 0);
        derivs2d(dwdy, w, 1);
        derivs2d(dFdx, F, 0);
        derivs2d(dGdy, G, 1);

        for (int s = 0; s < N * N; s++) {
          Bdot(Bdwdx, w.row(s), dwdx.row(s), 0, MP);
          Bdot(Bdwdy, w.row(s), dwdy.row(s), 1, MP);
          w.row(s) -= dt / 2 *
                      ((dFdx.row(s) + Bdwdx.transpose()) / dx +
                       (dGdy.row(s) + Bdwdy.transpose()) / dy);
        }
      }
    }
  }
}
