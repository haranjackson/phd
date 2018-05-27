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

  if (N <= 4) {
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
    if (N <= 4)
      wCL.noalias() = mCLinv * data.block<N, V>(FN2, 0);
    else
      wCL = MCL.solve(data.block<N, V>(FN2, 0));
    weight(oCL, wCL, LAMC);
    oSum += oCL;

    if ((N - 1) % 2) // Two central stencils (N>3)
    {
      if (N <= 4)
        wCR.noalias() = mCRinv * data.block<N, V>(CN2, 0);
      else
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

void weno1(Vecr wh, Vecr ub, int nx, int ny, int nz) {
  // Returns the WENO reconstruction of u using polynomials in x
  // Size of wh: nx*ny*nz*N*V
  // Size of ub: (nx+2(N-1))*ny*nz*V

  for (int ind = 0; ind < nx * ny * nz; ind++) {
    MatN_VMap wh_ref(wh.data() + (ind * N * V), OuterStride(V));
    Mat2N_VMap ub_ref(ub.data() + (ind * V), OuterStride(ny * nz * V));
    coeffs(wh_ref, ub_ref);
  }
}

void weno2(Vecr wh, Vecr ub, int nx, int ny, int nz) {
  // Returns the WENO reconstruction of u using polynomials in y
  // Size of wh: nx*ny*nz*N*N*V
  // Size of ub: (nx+2(N-1))*(ny+2(N-1))*nz*V
  Vec ux(nx * (ny + 2 * (N - 1)) * nz * N * V);
  weno1(ux, ub, nx, ny + 2 * (N - 1), nz);

  for (int i = 0; i < nx; i++) {
    int indi = i * ny * nz * N;
    int indii = i * (ny + 2 * (N - 1)) * nz * N;

    for (int s = 0; s < ny * nz * N; s++) {
      int indr = (indi + s) * N * V;
      int indt = (indii + s) * V;
      MatN_VMap wh_ref(wh.data() + indr, OuterStride(V));
      Mat2N_VMap ux_ref(ux.data() + indt, OuterStride(nz * N * V));
      coeffs(wh_ref, ux_ref);
    }
  }
}

void weno3(Vecr wh, Vecr ub, int nx, int ny, int nz) {
  // Returns the WENO reconstruction of u using polynomials in z
  // Size of wh: nx*ny*nz*N*N*N*V
  // Size of ub: (nx+2(N-1))*(ny+2(N-1))*(nz+2(N-1))*V
  Vec uy(nx * ny * (nz + 2 * (N - 1)) * N * N * V);
  weno2(uy, ub, nx, ny, nz + 2 * (N - 1));

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      int indi = (i * ny + j) * nz * N * N;
      int indii = (i * ny + j) * (nz + 2 * (N - 1)) * N * N;

      for (int s = 0; s < nz * N * N; s++) {
        int indr = (indi + s) * N * V;
        int indt = (indii + s) * V;
        MatN_VMap wh_ref(wh.data() + indr, OuterStride(V));
        Mat2N_VMap uy_ref(uy.data() + indt, OuterStride(N * N * V));
        coeffs(wh_ref, uy_ref);
      }
    }
}

void weno_launcher(Vecr wh, Vecr ub, iVecr nX) {
  // NOTE: boundary conditions extend u by two cells in each dimension

  int ndim = nX.size();

  switch (ndim) {

  case 1:
    weno1(wh, ub, nX(0) + 2, 1, 1);
    break;

  case 2:
    weno2(wh, ub, nX(0) + 2, nX(1) + 2, 1);
    break;

  case 3:
    weno3(wh, ub, nX(0) + 2, nX(1) + 2, nX(2) + 2);
    break;
  }
}
