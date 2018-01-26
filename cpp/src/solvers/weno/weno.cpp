#include "../../etc/globals.h"

void weight(VecVr ret, MatN_Vr w, double LAM) {
  // Produces the WENO weight for this stencil
  // NOTE: The denominator is raised to the 8th power.
  //       The square method is used because pow is slow.
  for (int i = 0; i < V; i++) {
    double tmp = w.col(i).transpose() * SIG * w.col(i) + EPS;
    double tmp2 = tmp * tmp;
    double tmp4 = tmp2 * tmp2;
    double den = tmp4 * tmp4;
    ret(i) = LAM / den;
  }
}

void coeffs(MatN_Vr ret, Mat2N_Vr data) {
  // Calculate coefficients of basis polynomials and weights
  VecV oL, oR, oCL, oCR, oSum;
  MatN_V wL, wR, wCL, wCR;

  if (N < 4) {
    wL.noalias() = mLinv * data.block<N, V>(0, 0);
    wR.noalias() = mRinv * data.block<N, V>((N - 1), 0);
  } else {
    wL = ML.solve(data.block<N, V>(0, 0));
    wR = MR.solve(data.block<N, V>((N - 1), 0));
  }

  weight(oL, wL, LAMS);
  weight(oR, wR, LAMS);
  oSum = oL + oR;

  if (N > 2) {
    if (N < 4)
      wCL.noalias() = mCLinv * data.block<N, V>(FN2, 0);
    else
      wCL = MCL.solve(data.block<N, V>(FN2, 0));
    weight(oCL, wCL, LAMC);
    oSum += oCL;

    if ((N - 1) % 2) // Two central stencils (N>3)
    {
      wCR = MCR.solve(data.block<N, V>(CN2, 0));
      weight(oCR, wCR, LAMC);
      oSum += oCR;
    }
  }

  for (int i = 0; i < N; i++)
    for (int j = 0; j < V; j++) {
      if (N == 2)
        ret(i, j) = (oL(j) * wL(i, j) + oR(j) * wR(i, j)) / oSum(j);
      else if ((N - 1) % 2 == 0)
        ret(i, j) = (oL(j) * wL(i, j) + oR(j) * wR(i, j) + oCL(j) * wCL(i, j)) /
                    oSum(j);
      else
        ret(i, j) = (oL(j) * wL(i, j) + oR(j) * wR(i, j) + oCL(j) * wCL(i, j) +
                     oCR(j) * wCR(i, j)) /
                    oSum(j);
    }
}

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

void weno1(Vecr ret, Vecr u, int nx, int ny, int nz) {
  // Returns the WENO reconstruction of u using polynomials in x
  // Size of ret: nx*ny*nz*N*V
  // Size of u:   nx*ny*nz*V
  Vec tmp = expandx(u, nx, ny, nz);
  for (int ind = 0; ind < nx * ny * nz; ind++) {
    MatN_VMap ret_ref(ret.data() + (ind * N * V), OuterStride(V));
    Mat2N_VMap tmp_ref(tmp.data() + (ind * V), OuterStride(ny * nz * V));
    coeffs(ret_ref, tmp_ref);
  }
}

void weno2(Vecr ret, Vecr u, int nx, int ny, int nz) {
  // Returns the WENO reconstruction of u using polynomials in y
  // Size of ret: nx*ny*nz*N*N*V
  // Size of u:   nx*ny*nz*V
  Vec ux(nx * ny * nz * N * V);
  weno1(ux, u, nx, ny, nz);
  Vec tmp = expandy(ux, nx, ny, nz);

  for (int i = 0; i < nx; i++) {
    int indi = i * ny * nz * N;
    int indii = i * (ny + 2 * (N - 1)) * nz * N;

    for (int s = 0; s < ny * nz * N; s++) {
      int indr = (indi + s) * N * V;
      int indt = (indii + s) * V;
      MatN_VMap ret_ref(ret.data() + indr, OuterStride(V));
      Mat2N_VMap tmp_ref(tmp.data() + indt, OuterStride(nz * N * V));
      coeffs(ret_ref, tmp_ref);
    }
  }
}

void weno3(Vecr ret, Vecr u, int nx, int ny, int nz) {
  // Returns the WENO reconstruction of u using polynomials in z
  // Size of ret: nx*ny*nz*N*N*N*V
  // Size of u:   nx*ny*nz*V
  Vec uy(nx * ny * nz * N * N * V);
  weno2(uy, u, nx, ny, nz);
  Vec tmp = expandz(uy, nx, ny, nz);

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      int indi = (i * ny + j) * nz * N * N;
      int indii = (i * ny + j) * (nz + 2 * (N - 1)) * N * N;

      for (int s = 0; s < nz * N * N; s++) {
        int indr = (indi + s) * N * V;
        int indt = (indii + s) * V;
        MatN_VMap ret_ref(ret.data() + indr, OuterStride(V));
        Mat2N_VMap tmp_ref(tmp.data() + indt, OuterStride(N * N * V));
        coeffs(ret_ref, tmp_ref);
      }
    }
}

void weno_launcher(Vecr ret, Vecr u, int ndim, int nx, int ny, int nz) {
  // NOTE: boundary conditions extend u by two cells in each dimension
  switch (ndim) {
  case 1:
    weno1(ret, u, nx + 2, ny, nz);
    break;
  case 2:
    weno2(ret, u, nx + 2, ny + 2, nz);
    break;
  case 3:
    weno3(ret, u, nx + 2, ny + 2, nz + 2);
    break;
  }
}
