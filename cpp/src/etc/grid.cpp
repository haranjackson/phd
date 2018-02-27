#include "types.h"

int uind(int i, int j, int ny) {
  // Returns the starting index of cell (i,j,k)
  return (i * ny + j) * V;
}

int uind(int i, int j, int k, int ny, int nz) {
  // Returns the starting index of cell (i,j,k)
  return ((i * ny + j) * nz + k) * V;
}

void boundaries(Vecr u, Vecr ub, int ndim, Veci3r nX, bool PERIODIC) {
  // If periodic is true, applies periodic boundary conditions,
  // else applies transmissive boundary conditions

  int nx = nX(0);
  int ny = nX(1);
  if (ndim == 1) {
    ub.segment(V, nx * V) = u;

    if (PERIODIC) {
      ub.head<V>() = u.tail<V>();
      ub.tail<V>() = u.head<V>();
    } else {
      ub.head<V>() = u.head<V>();
      ub.tail<V>() = u.tail<V>();
    }
    return;
  }
  if (ndim == 2) {
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
        ub.segment<V>(uind(i + 1, j + 1, ny + 2)) =
            u.segment<V>(uind(i, j, ny));

    if (PERIODIC) {
      for (int i = 0; i < nx; i++) {
        ub.segment<V>(uind(i + 1, 0, ny + 2)) =
            u.segment<V>(uind(i, ny - 1, ny));
        ub.segment<V>(uind(i + 1, ny + 1, ny + 2)) =
            u.segment<V>(uind(i, 0, ny));
      }
      for (int j = 0; j < ny; j++) {
        ub.segment<V>(uind(0, j + 1, ny + 2)) =
            u.segment<V>(uind(nx - 1, j, ny));
        ub.segment<V>(uind(nx + 1, j + 1, ny + 2)) =
            u.segment<V>(uind(0, j, ny));
      }
    } else {
      for (int i = 0; i < nx; i++) {
        ub.segment<V>(uind(i + 1, 0, ny + 2)) = u.segment<V>(uind(i, 0, ny));
        ub.segment<V>(uind(i + 1, ny + 1, ny + 2)) =
            u.segment<V>(uind(i, ny - 1, ny));
      }
      for (int j = 0; j < ny; j++) {
        ub.segment<V>(uind(0, j + 1, ny + 2)) = u.segment<V>(uind(0, j, ny));
        ub.segment<V>(uind(nx + 1, j + 1, ny + 2)) =
            u.segment<V>(uind(nx - 1, j, ny));
      }
    }

    ub.segment<V>(uind(0, 0, ny + 2)) = (ub.segment<V>(uind(0, 1, ny + 2)) +
                                         ub.segment<V>(uind(1, 0, ny + 2))) /
                                        2;
    ub.segment<V>(uind(0, ny + 1, ny + 2)) =
        (ub.segment<V>(uind(0, ny, ny + 2)) +
         ub.segment<V>(uind(1, ny + 1, ny + 2))) /
        2;
    ub.segment<V>(uind(nx + 1, 0, ny + 2)) =
        (ub.segment<V>(uind(nx, 0, ny + 2)) +
         ub.segment<V>(uind(nx + 1, 1, ny + 2))) /
        2;
    ub.segment<V>(uind(nx + 1, ny + 1, ny + 2)) =
        (ub.segment<V>(uind(nx, ny + 1, ny + 2)) +
         ub.segment<V>(uind(nx + 1, ny, ny + 2))) /
        2;

    return;
  }
  if (ndim == 3) {
    // TODO
  }
}

int extended_dimensions(Veci3r nX) {
  int nx = nX(0);
  int ny = nX(1);
  int nz = nX(2);
  if (nz > 1)
    return (nx + 2) * (ny + 2) * (nz + 2);
  else if (ny > 1)
    return (nx + 2) * (ny + 2);
  else
    return nx + 2;
}

/*
Vec expandx(Vecr arr, int nx, int ny, int nz) {
  // Expands arr by (N-1) cells either side of the x dimension
  // Size of ret: (nx+2*(N-1))*ny*nz*V
  // Size of arr: nx*ny*nz*V
  Vec ret((nx + 2 * (N - 1)) * ny * nz * V);

  for (int i = 0; i < nx + 2 * (N - 1); i++) {
    int ii = std::min(nx - 1, std::max(0, i - (N - 1)));
    int indi = i * ny * nz * V;
    int indii = ii * ny * nz * V;

    for (int s = 0; s < ny * nz * V; s++)
      ret(indi + s) = arr(indii + s);
  }
  return ret;
}

Vec expandy(Vecr arr, int nx, int ny, int nz) {
  // Expands arr by (N-1) cells either side of the y dimension
  // Size of ret: nx*(ny+2*(N-1))*nz*N*V
  // Size of arr: nx*ny*nz*N*V
  Vec ret(nx * (ny + 2 * (N - 1)) * nz * N * V);

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny + 2 * (N - 1); j++) {
      int jj = std::min(ny - 1, std::max(0, j - (N - 1)));
      int indj = (i * (ny + 2 * (N - 1)) + j) * nz * N * V;
      int indjj = (i * ny + jj) * nz * N * V;

      for (int s = 0; s < nz * N * V; s++)
        ret(indj + s) = arr(indjj + s);
    }
  return ret;
}

Vec expandz(Vecr arr, int nx, int ny, int nz) {
  // Expands arr by (N-1) cells either side of the z dimension
  // Size of ret: nx*ny*(nz+2*(N-1))*N*N*V
  // Size of arr: nx*ny*nz*N*N*V
  Vec ret(nx * ny * (nz + 2 * (N - 1)) * N * N * V);

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz + 2 * (N - 1); k++) {
        int kk = std::min(nz - 1, std::max(0, k - (N - 1)));
        int indk = ((i * ny + j) * (nz + 2 * (N - 1)) + k) * N * N * V;
        int indkk = ((i * ny + j) * nz + kk) * N * N * V;

        for (int s = 0; s < N * N * V; s++)
          ret(indk + s) = arr(indkk + s);
      }
  return ret;
}
*/
