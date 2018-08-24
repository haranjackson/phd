#include "types.h"

const int TRANSMISSIVE = 0;
const int PERIODIC = 1;
const int SLIP = 2;
const int STICK = 3;
const int MOVING = 4;

void boundaries1(Matr u, Matr ub, int nx, iVecr boundaryTypes, int d) {

  switch (boundaryTypes(0)) {

  case TRANSMISSIVE:
    for (int i = 0; i < N; i++)
      ub.row(i) = u.row(0);
    break;

  case PERIODIC:
    for (int i = 0; i < N; i++)
      ub.row(i) = u.row(nx - N + i);
    break;

  case SLIP:
    for (int i = 0; i < N; i++) {
      ub.row(i) = u.row(N - 1 - i);
      ub(i, 2 + d) *= -1.;
    }
    break;

  case STICK:
    for (int i = 0; i < N; i++) {
      ub.row(i) = u.row(N - 1 - i);
      ub(i, 2) *= -1.;
      ub(i, 3) *= -1.;
      ub(i, 4) *= -1.;
    }
    break;

  case MOVING:
    for (int i = 0; i < N; i++) {
      ub.row(i) = u.row(N - 1 - i);
      ub(i, 2) = ub(i, 0) * 2. - ub(i, 2);
      ub(i, 3) *= -1.;
      ub(i, 4) *= -1.;
      ub(i, 1) += 1 / (2 * ub(i, 0)) *
                  (ub(i, 2) * ub(i, 2) - u(N - 1 - i, 2) * u(N - 1 - i, 2));
    }
    break;
  }

  switch (boundaryTypes(1)) {

  case TRANSMISSIVE:
    for (int i = 0; i < N; i++)
      ub.row(i + nx + N) = u.row(nx - 1);
    break;

  case PERIODIC:
    for (int i = 0; i < N; i++)
      ub.row(i + nx + N) = u.row(i);
    break;

  case SLIP:
    for (int i = 0; i < N; i++) {
      ub.row(i + nx + N) = u.row(nx - 1 - i);
      ub(i + nx + N, 2 + d) *= -1.;
    }
    break;

  case STICK:
    for (int i = 0; i < N; i++) {
      ub.row(i + nx + N) = u.row(nx - 1 - i);
      ub(i + nx + N, 2) *= -1.;
      ub(i + nx + N, 3) *= -1.;
      ub(i + nx + N, 4) *= -1.;
    }
    break;

  case MOVING:
    for (int i = 0; i < N; i++) {
      int ind = i + nx + N;
      ub.row(ind) = u.row(nx - 1 - i);
      ub(ind, 2) = ub(ind, 0) * 2. - ub(ind, 2);
      ub(ind, 3) *= -1.;
      ub(ind, 4) *= -1.;
      ub(ind, 1) +=
          1 / (2 * ub(ind, 0)) *
          (ub(ind, 2) * ub(ind, 2) - u(nx - 1 - i, 2) * u(nx - 1 - i, 2));
    }
    break;
  }
}

void boundaries2(Vecr u, Vecr ub, int nx, int ny, iVecr boundaryTypes) {

  // (i * ny + j) * V;
  // (i * (ny + 2 * N) + j) * V;

  for (int j = 0; j < ny; j++) {
    MatMap u0(u.data() + (j * V), nx, V, OuterStride(ny * V));
    MatMap ub0(ub.data() + ((j + N) * V), nx + 2 * N, V,
               OuterStride((ny + 2 * N) * V));
    boundaries1(u0, ub0, nx, boundaryTypes.head<2>(), 0);
  }

  for (int i = 0; i < nx + 2 * N; i++) {
    MatMap u0(ub.data() + ((i * (ny + 2 * N) + N) * V), ny, V, OuterStride(V));
    MatMap ub0(ub.data() + ((i * (ny + 2 * N)) * V), ny + 2 * N, V,
               OuterStride(V));
    boundaries1(u0, ub0, ny, boundaryTypes.tail<2>(), 1);
  }
}

void boundaries(Vecr u, Vecr ub, iVecr nX, iVecr boundaryTypes) {
  // If periodic is true, applies periodic boundary conditions,
  // else applies transmissive boundary conditions
  long ndim = nX.size();
  int nx = nX(0);

  switch (ndim) {

  case 1: {
    ub.segment(N * V, nx * V) = u;
    MatMap u0(u.data(), nx, V, OuterStride(V));
    MatMap ub0(ub.data(), nx + 2 * N, V, OuterStride(V));
    boundaries1(u0, ub0, nx, boundaryTypes, 0);
  } break;

  case 2:
    int ny = nX(1);
    for (int i = 0; i < nx; i++)
      ub.segment(((i + N) * (ny + 2 * N) + N) * V, ny * V) =
          u.segment(i * ny * V, ny * V);
    boundaries2(u, ub, nx, ny, boundaryTypes);
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

  long ndim = nX.size();
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
