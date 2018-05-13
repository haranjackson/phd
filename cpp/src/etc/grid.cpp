#include "types.h"

int uind(int i, int j, int ny) {
  // Returns the starting index of cell (i,j)
  return (i * ny + j) * V;
}

int uind(int i, int j, int k, int ny, int nz) {
  // Returns the starting index of cell (i,j,k)
  return ((i * ny + j) * nz + k) * V;
}

void boundaries1(Vecr u, Vecr ub, int nx, bool PERIODIC) {
  ub.segment(N * V, nx * V) = u;

  if (PERIODIC) {
    ub.head<N * V>() = u.tail<N * V>();
    ub.tail<N * V>() = u.head<N * V>();
  } else {
    for (int i = 0; i < N; i++) {
      ub.segment<V>(i * V) = u.head<V>();
      ub.segment<V>((i + nx + N) * V) = u.tail<V>();
    }
  }
}

void boundaries2(Vecr u, Vecr ub, int nx, int ny, bool PERIODIC) {

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      ub.segment<V>(uind(i + N, j + N, ny + 2 * N)) =
          u.segment<V>(uind(i, j, ny));

  if (PERIODIC) {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        ub.segment<V>(uind(i, j, ny + 2 * N)) =
            u.segment<V>(uind(nx - N + i, ny - N + j, ny));
        ub.segment<V>(uind(nx + N + i, j, ny + 2 * N)) =
            u.segment<V>(uind(i, ny - N + j, ny));
        ub.segment<V>(uind(i, ny + N + j, ny + 2 * N)) =
            u.segment<V>(uind(nx - N + i, j, ny));
        ub.segment<V>(uind(nx + N + i, ny + N + j, ny + 2 * N)) =
            u.segment<V>(uind(i, j, ny));
      }
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < N; j++) {
        ub.segment<V>(uind(i + N, j, ny + 2 * N)) =
            u.segment<V>(uind(i, ny - N + j, ny));
        ub.segment<V>(uind(i + N, ny + N + j, ny + 2 * N)) =
            u.segment<V>(uind(i, j, ny));
      }
    for (int i = 0; i < N; i++)
      for (int j = 0; j < ny; j++) {
        ub.segment<V>(uind(i, N + j, ny + 2 * N)) =
            u.segment<V>(uind(nx - N + i, j, ny));
        ub.segment<V>(uind(nx + N + i, j + N, ny + 2 * N)) =
            u.segment<V>(uind(i, j, ny));
      }
  } else {
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < N; j++) {
        ub.segment<V>(uind(i + N, j, ny + 2 * N)) =
            u.segment<V>(uind(i, 0, ny));
        ub.segment<V>(uind(i + N, ny + N + j, ny + 2 * N)) =
            u.segment<V>(uind(i, ny - 1, ny));
      }
    for (int i = 0; i < N; i++)
      for (int j = 0; j < ny; j++) {
        ub.segment<V>(uind(i, j + N, ny + 2 * N)) =
            u.segment<V>(uind(0, j, ny));
        ub.segment<V>(uind(nx + N + i, j + N, ny + 2 * N)) =
            u.segment<V>(uind(nx - 1, j, ny));
      }

    // Make corners averages of cells either side
    VecV BL = (ub.segment<V>(uind(N - 1, N, ny + 2 * N)) +
               ub.segment<V>(uind(N, N - 1, ny + 2 * N))) /
              2;
    VecV BR = (ub.segment<V>(uind(nx + N - 1, N - 1, ny + 2 * N)) +
               ub.segment<V>(uind(nx + N, N, ny + 2 * N))) /
              2;
    VecV TL = (ub.segment<V>(uind(N - 1, ny + N - 1, ny + 2 * N)) +
               ub.segment<V>(uind(N, ny + N, ny + 2 * N))) /
              2;
    VecV TR = (ub.segment<V>(uind(nx + N - 1, ny + N, ny + 2 * N)) +
               ub.segment<V>(uind(nx + N, ny + N - 1, ny + 2 * N))) /
              2;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        ub.segment<V>(uind(i, N - 1, ny + 2 * N)) = BL;
        ub.segment<V>(uind(N - 1, j, ny + 2 * N)) = BL;
        ub.segment<V>(uind(nx + N + i, N - 1, ny + 2 * N)) = BR;
        ub.segment<V>(uind(nx + N, j, ny + 2 * N)) = BR;
        ub.segment<V>(uind(i, ny + N, ny + 2 * N)) = TL;
        ub.segment<V>(uind(N - 1, ny + N + j, ny + 2 * N)) = TL;
        ub.segment<V>(uind(nx + N + i, ny + N, ny + 2 * N)) = TR;
        ub.segment<V>(uind(nx + N, ny + N + j, ny + 2 * N)) = TR;
      }
  }
}

void boundaries(Vecr u, Vecr ub, iVecr nX, bool PERIODIC) {
  // If periodic is true, applies periodic boundary conditions,
  // else applies transmissive boundary conditions
  int ndim = nX.size();
  switch (ndim) {
  case 1:
    boundaries1(u, ub, nX(0), PERIODIC);
    break;
  case 2:
    boundaries2(u, ub, nX(0), nX(1), PERIODIC);
    break;
  }
}

int extended_dimensions(iVecr nX, int ext) {
  return (nX.array() + 2 * ext).prod();
}

void extend_mask(bVecr mask, bVecr maskb, iVecr nX) {
  // Given a mask corresponding to the cells that the user wishes to update,
  // this function returns a mask of the cells for which the DG predictor
  // must be calculated (i.e. the masked cells and their neighbors)

  int ndim = nX.size();
  int nx = nX(0);
  switch (ndim) {

  case 1:
    for (int i = 0; i < nx; i++) {
      if (mask(i) || (i > 0 && mask(i - 1)) || (i < nx - 1 && mask(i + 1)))
        maskb(i + 1) = true;
      else
        maskb(i + 1) = false;
    }
    maskb(0) = maskb(1);
    maskb(nx + 1) = maskb(nx);
    break;

  case 2:
    int ny = nX(1);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {

        int ind = i * ny + j;
        int ind_ = (i + 1) * (ny + 2) + (j + 1);
        int indU = (i + 1) * ny + j;
        int indD = (i - 1) * ny + j;
        int indL = i * ny + (j - 1);
        int indR = i * ny + (j + 1);

        if (mask(ind) || (i > 0 && mask(indD)) || (i < nx - 1 && mask(indU)) ||
            (j > 0 && mask(indL)) || (j < ny - 1 && mask(indR)))
          maskb(ind_) = true;
        else
          maskb(ind_) = false;

        for (int i = 0; i < nx; i++) {
          maskb((i + 1) * (ny + 2)) = mask(i * ny);
          maskb((i + 1) * (ny + 2) + (ny + 1)) = mask(i * ny + (ny - 1));
        }
        for (int j = 0; j < ny; j++) {
          maskb(j + 1) = mask(j);
          maskb((nx + 1) * (ny + 2) + (j + 1)) = mask((nx - 1) * ny + j);
        }

        maskb(0) = false;
        maskb((ny + 1)) = false;
        maskb((nx + 1) * (ny + 2)) = false;
        maskb((nx + 1) * (ny + 2) + (ny + 1)) = false;
      }
    break;
  }
}
