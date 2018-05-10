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

  MatN_V q0x, q0y, q1x, q1y;
  VecV qlx, qrx, qly, qry, fx, bx, fy, by;

  double kx = wghts_t / (2 * dx);
  double ky = wghts_t / (2 * dy);

  for (int i = 0; i < nx + 1; i++)
    for (int j = 0; j < ny + 1; j++) {

      if ((i == 0 || i == nx + 1) && (j == 0 || j == ny + 1))
        continue;

      if (mask(ind(i, j, ny + 2))) {

        int uind0 = ind(i - 1, j - 1, ny) * V;
        int ind0 = ind(i, j, t, ny + 2, nt) * N * N * V;

        MatN2_VMap qh0(rec.data() + ind0, OuterStride(V));

        if (mask(ind(i + 1, j, ny + 2))) {

          int uindx = ind(i, j - 1, ny) * V;
          int indx = ind(i + 1, j, t, ny + 2, nt) * N * N * V;

          MatN2_VMap qhx(rec.data() + indx, OuterStride(V));
          endpts2d(q0x, qh0, 0, 1);
          endpts2d(q1x, qhx, 0, 0);

          for (int s = 0; s < N; s++) {

            qlx = q0x.row(s);
            qrx = q1x.row(s);

            switch (FLUX) {
            case OSHER:
              fx = D_OSH(qlx, qrx, 0, MP);
              break;
            case ROE:
              fx = D_ROE(qlx, qrx, 0, MP);
              break;
            case RUSANOV:
              fx = D_RUS(qlx, qrx, 0, MP);
              break;
            }
            bx = Bint(qlx, qrx, 0, MP);

            if (i > 0 && i < nx + 1 && j > 0 && j < ny + 1) {
              u.segment<V>(uind0) -= WGHTS(s) * kx * (bx + fx);
            }
            if (i < nx && j > 0 and j < ny + 1)
              u.segment<V>(uindx) -= WGHTS(s) * kx * (bx - fx);
          }
        }
        if (mask(ind(i, j + 1, ny + 2))) {

          int uindy = ind(i - 1, j, ny) * V;
          int indy = ind(i, j + 1, t, ny + 2, nt) * N * N * V;

          MatN2_VMap qhy(rec.data() + indy, OuterStride(V));
          endpts2d(q0y, qh0, 1, 1);
          endpts2d(q1y, qhy, 1, 0);

          for (int s = 0; s < N; s++) {
            qly = q0y.row(s);
            qry = q1y.row(s);

            switch (FLUX) {
            case OSHER:
              fy = D_OSH(qly, qry, 1, MP);
              break;
            case ROE:
              fy = D_ROE(qly, qry, 1, MP);
              break;
            case RUSANOV:
              fy = D_RUS(qly, qry, 1, MP);
              break;
            }
            by = Bint(qly, qry, 1, MP);

            if (i > 0 && i < nx + 1 && j > 0 && j < ny + 1) {
              u.segment<V>(uind0) -= WGHTS(s) * ky * (by + fy);
            }
            if (i > 0 && i < nx + 1 && j < ny)
              u.segment<V>(uindy) -= WGHTS(s) * ky * (by - fy);
          }
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

void fv_launcher(Vecr u, Vecr rec, int ndim, iVec3r nX, double dt, Vec3r dX,
                 bool SOURCES, bool TIME, int FLUX, Par &MP, bVecr mask) {
  int nx = nX(0);
  int ny = nX(1);
  double dx = dX(0);
  double dy = dX(1);

  switch (ndim) {
  case 1:
    centers1(u, rec, nx, dt, dx, SOURCES, TIME, MP, mask);
    interfs1(u, rec, nx, dt, dx, TIME, FLUX, MP, mask);
    break;
  case 2:
    centers2(u, rec, nx, ny, dt, dx, dy, SOURCES, TIME, MP, mask);
    interfs2(u, rec, nx, ny, dt, dx, dy, TIME, FLUX, MP, mask);
    break;
  }
}
