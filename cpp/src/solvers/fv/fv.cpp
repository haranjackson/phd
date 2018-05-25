#include "../../etc/globals.h"
#include "../../system/equations.h"
#include "../../system/functions/vectors.h"
#include "../evaluations.h"
#include "fluxes.h"

int ind(int i, int t, int nt) { return i * nt + t; }

int ind(int i, int j, int t, int ny, int nt) { return (i * ny + j) * nt + t; }

void centers1_inner(Vecr u, Vecr rec, int nx, double dx, int nt, int t,
                    double wght_t, bool SOURCES, Par &MP, bVecr mask) {
  MatN_V dqh_dx;
  VecV dqdxs, qs, S, tmpx;

  for (int i = 0; i < nx; i++) {
    if (mask(i + 1)) {

      int idx = ind(i + 1, t, nt) * N * V;
      MatN_VMap qh(rec.data() + idx, OuterStride(V));
      dqh_dx.noalias() = DERVALS * qh;

      for (int s = 0; s < N; s++) {
        qs = qh.row(s);
        dqdxs = dqh_dx.row(s);

        if (SOURCES)
          source(S, qs, MP);
        else
          S.setZero(V);

        Bdot(tmpx, qs, dqdxs, 0, MP);

        S -= tmpx / dx;

        u.segment<V>(i * V) += wght_t * WGHTS(s) * S;
      }
    }
  }
}

void centers1(Vecr u, Vecr rec, int nx, double dt, double dx, bool SOURCES,
              bool TIME, Par &MP, bVecr mask) {
  if (TIME)
    for (int t = 0; t < N; t++)
      centers1_inner(u, rec, nx, dx, N, t, dt * WGHTS(t), SOURCES, MP, mask);
  else
    centers1_inner(u, rec, nx, dx, 1, 0, dt, SOURCES, MP, mask);
}

void centers2_inner(Vecr u, Vecr rec, int nx, int ny, double dx, double dy,
                    int nt, int t, double wght_t, bool SOURCES, Par &MP,
                    bVecr mask) {
  MatN2_V dqh_dx, dqh_dy;
  VecV qs, dqdxs, dqdys, S, tmpx, tmpy;

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      if (mask(ind(i + 1, j + 1, ny + 2))) {
        int idx = ind(i + 1, j + 1, t, ny + 2, nt) * N * N * V;

        MatN2_VMap qh(rec.data() + idx, OuterStride(V));
        derivs2d(dqh_dx, qh, 0);
        derivs2d(dqh_dy, qh, 1);

        for (int a = 0; a < N; a++)
          for (int b = 0; b < N; b++) {
            int s = a * N + b;
            qs = qh.row(s);
            dqdxs = dqh_dx.row(s);
            dqdys = dqh_dy.row(s);

            if (SOURCES)
              source(S, qs, MP);
            else
              S.setZero(V);

            Bdot(tmpx, qs, dqdxs, 0, MP);
            Bdot(tmpy, qs, dqdys, 1, MP);

            S -= tmpx / dx;
            S -= tmpy / dy;

            u.segment<V>((i * ny + j) * V) += wght_t * WGHTS(a) * WGHTS(b) * S;
          }
      }
    }
}

void centers2(Vecr u, Vecr rec, int nx, int ny, double dt, double dx, double dy,
              bool SOURCES, bool TIME, Par &MP, bVecr mask) {

  if (TIME)
    for (int t = 0; t < N; t++)
      centers2_inner(u, rec, nx, ny, dx, dy, N, t, dt * WGHTS(t), SOURCES, MP,
                     mask);
  else
    centers2_inner(u, rec, nx, ny, dx, dy, 1, 0, dt, SOURCES, MP, mask);
}

void interfs1_inner(Vecr u, Vecr rec, int nx, double dx, int nt, int t,
                    double wght_t, int FLUX, Par &MP, bVecr mask) {

  double k = wght_t / (2. * dx);
  VecV ql, qr, f, b;

  for (int i = 0; i < nx + 1; i++) {
    if (mask(i) && mask(i + 1)) {
      int indl = ind(i, t, nt) * N * V;
      int indr = ind(i + 1, t, nt) * N * V;
      MatN_VMap qhl(rec.data() + indl, OuterStride(V));
      MatN_VMap qhr(rec.data() + indr, OuterStride(V));
      ql.noalias() = ENDVALS.row(1) * qhl;
      qr.noalias() = ENDVALS.row(0) * qhr;

      switch (FLUX) {
      case OSHER:
        f = D_OSH(ql, qr, 0, MP);
        break;
      case ROE:
        f = D_ROE(ql, qr, 0, MP);
        break;
      case RUSANOV:
        f = D_RUS(ql, qr, 0, MP);
        break;
      }
      b = Bint(ql, qr, 0, MP);

      if (i > 0)
        u.segment<V>((i - 1) * V) -= k * (b + f);
      if (i < nx)
        u.segment<V>(i * V) -= k * (b - f);
    }
  }
}

void interfs1(Vecr u, Vecr rec, int nx, double dt, double dx, bool TIME,
              int FLUX, Par &MP, bVecr mask) {

  if (TIME)
    for (int t = 0; t < N; t++)
      interfs1_inner(u, rec, nx, dx, N, t, dt * WGHTS(t), FLUX, MP, mask);
  else
    interfs1_inner(u, rec, nx, dx, 1, 0, dt, FLUX, MP, mask);
}

void interfs2_inner(Vecr u, Vecr rec, int nx, int ny, double dx, double dy,
                    int nt, int t, double wghts_t, int FLUX, Par &MP,
                    bVecr mask) {

  MatN_V q0, q1;
  VecV f, b, u0, u1;

  VecN xWGHTS = wghts_t / (2. * dx) * WGHTS;
  VecN yWGHTS = wghts_t / (2. * dy) * WGHTS;

  int NNV = N * N * V;

#pragma omp parallel for collapse(2) private(q0, q1, f, b, u0, u1)             \
    schedule(static, 8) num_threads(4)
  for (int i = 0; i < nx + 1; i++)
    for (int j = 0; j < ny + 1; j++) {

      if (mask(ind(i, j, ny + 2))) {

        int uind0 = ind(i - 1, j - 1, ny) * V;
        int ind0 = ind(i, j, t, ny + 2, nt) * NNV;
        MatN2_VMap qh0(rec.data() + ind0, OuterStride(V));

        if (mask(ind(i + 1, j, ny + 2)) && j > 0) {

          int uindx = ind(i, j - 1, ny) * V;
          int indx = ind(i + 1, j, t, ny + 2, nt) * NNV;

          MatN2_VMap qhx(rec.data() + indx, OuterStride(V));
          endpts2d(q0, qh0, 0, 1);
          endpts2d(q1, qhx, 0, 0);

          u0.setZero(V);
          u1.setZero(V);

          for (int s = 0; s < N; s++) {

            switch (FLUX) {
            case OSHER:
              f = D_OSH(q0.row(s), q1.row(s), 0, MP);
              break;
            case ROE:
              f = D_ROE(q0.row(s), q1.row(s), 0, MP);
              break;
            case RUSANOV:
              f = D_RUS(q0.row(s), q1.row(s), 0, MP);
              break;
            }
            b = Bint(q0.row(s), q1.row(s), 0, MP);

            if (i > 0)
              u0 += xWGHTS(s) * (b + f);
            if (i < nx)
              u1 += xWGHTS(s) * (b - f);
          }
          if (i > 0)
            u.segment<V>(uind0) -= u0;
          if (i < nx)
            u.segment<V>(uindx) -= u1;
        }
        if (mask(ind(i, j + 1, ny + 2)) && i > 0) {

          int uindy = ind(i - 1, j, ny) * V;
          int indy = ind(i, j + 1, t, ny + 2, nt) * NNV;

          MatN2_VMap qhy(rec.data() + indy, OuterStride(V));
          endpts2d(q0, qh0, 1, 1);
          endpts2d(q1, qhy, 1, 0);

          u0.setZero(V);
          u1.setZero(V);

          for (int s = 0; s < N; s++) {

            switch (FLUX) {
            case OSHER:
              f = D_OSH(q0.row(s), q1.row(s), 1, MP);
              break;
            case ROE:
              f = D_ROE(q0.row(s), q1.row(s), 1, MP);
              break;
            case RUSANOV:
              f = D_RUS(q0.row(s), q1.row(s), 1, MP);
              break;
            }
            b = Bint(q0.row(s), q1.row(s), 1, MP);

            if (j > 0)
              u0 += yWGHTS(s) * (b + f);
            if (j < ny)
              u1 += yWGHTS(s) * (b - f);
          }
          if (j > 0)
            u.segment<V>(uind0) -= u0;
          if (j < ny)
            u.segment<V>(uindy) -= u1;
        }
      }
    }
}

void interfs2(Vecr u, Vecr rec, int nx, int ny, double dt, double dx, double dy,
              bool TIME, int FLUX, Par &MP, bVecr mask) {
  if (TIME)
    for (int t = 0; t < N; t++)
      interfs2_inner(u, rec, nx, ny, dx, dy, N, t, dt * WGHTS(t), FLUX, MP,
                     mask);
  else
    interfs2_inner(u, rec, nx, ny, dx, dy, 1, 0, dt, FLUX, MP, mask);
}

void fv_launcher(Vecr u, Vecr rec, iVecr nX, double dt, Vecr dX, bool SOURCES,
                 bool TIME, int FLUX, Par &MP, bVecr mask) {

  int ndim = nX.size();
  switch (ndim) {
  case 1:
    centers1(u, rec, nX(0), dt, dX(0), SOURCES, TIME, MP, mask);
    interfs1(u, rec, nX(0), dt, dX(0), TIME, FLUX, MP, mask);
    break;
  case 2:
    centers2(u, rec, nX(0), nX(1), dt, dX(0), dX(1), SOURCES, TIME, MP, mask);
    interfs2(u, rec, nX(0), nX(1), dt, dX(0), dX(1), TIME, FLUX, MP, mask);
    break;
  }
}
