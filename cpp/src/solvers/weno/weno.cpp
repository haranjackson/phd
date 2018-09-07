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

void coeffs(MatN_Vr ret, Mat2N_Vr data, int nx, int gap, int ind) {
  // Calculate coefficients of basis polynomials and weights

  VecV oL, oR, oCL, oCR;
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

  if (N > 2) {
    if (N <= 4)
      wCL.noalias() = mCLinv * data.block<N, V>(FN2, 0);
    else
      wCL = MCL.solve(data.block<N, V>(FN2, 0));
    weight(oCL, wCL, LAMC);

    if (N % 2 == 0) // Two central stencils (N>3)
    {
      if (N <= 4)
        wCR.noalias() = mCRinv * data.block<N, V>(CN2, 0);
      else
        wCR = MCR.solve(data.block<N, V>(CN2, 0));
      weight(oCR, wCR, LAMC);
    }
  }

  MatN_V num = MatN_V::Zero();
  VecV den = VecV::Zero();

  if (ind >= gap) {
    num.array() += wL.array().rowwise() * oL.transpose().array();
    den += oL;
  }
  if (ind <= nx - gap - 1) {
    num.array() += wR.array().rowwise() * oR.transpose().array();
    den += oR;
  }
  if (N > 2) {
    if (ind + FN2 >= gap) {
      num.array() += wCL.array().rowwise() * oCL.transpose().array();
      den += oCL;
    }
    if ((N % 2 == 0) && (ind - FN2 <= nx - gap - 1)) {
      num.array() += wCR.array().rowwise() * oCR.transpose().array();
      den += oCR;
    }
  }
  ret.array() = num.array().rowwise() / den.transpose().array();
}

void weno1(Matr wh, Matr ub, int nx, int gap) {
  // Returns the WENO reconstruction of u using polynomials in x
  // Shape of ub: (nx + 2(N-1), V)
  // Shapw of wh: (nx * N, V)
  for (int i = 0; i < nx; i++)
    coeffs(wh.block<N, V>(i * N, 0), ub.block<2 * N, V>(i, 0), nx, gap, i);
}

void weno2(Matr wh, Matr ub, int nx, int ny) {
  // Returns the WENO reconstruction of u using polynomials in y
  // Size of ub: (nx + 2(N-1)) * (ny + 2(N-1)) * V
  // Size of wh: nx * ny * N * N * V

  Vec ux(nx * (ny + 2 * (N - 1)) * N * V);

  Mat tmp0(nx * N, V);
  MatMap tmp0Map(tmp0.data(), nx, N * V, OuterStride(N * V));

  for (int j = 0; j < ny + 2 * (N - 1); j++) {
    MatMap ubMap(ub.data() + j * V, nx + 2 * (N - 1), V,
                 OuterStride((ny + 2 * (N - 1)) * V));
    MatMap uxMap(ux.data() + j * N * V, nx, N * V,
                 OuterStride((ny + 2 * (N - 1)) * N * V));

    if (NO_CORNERS && (j < N || j >= ny + 2 * (N - 1) - N))
      weno1(tmp0, ubMap, nx, N);
    else
      weno1(tmp0, ubMap, nx, 0);
    uxMap = tmp0Map;
  }

  Mat tmp1(ny * N, V);
  MatMap tmp1Map(tmp1.data(), ny, N * V, OuterStride(N * V));

  for (int i = 0; i < nx; i++)
    for (int ii = 0; ii < N; ii++) {
      MatMap uxMap(ux.data() + (i * (ny + 2 * (N - 1)) * N + ii) * V,
                   (ny + 2 * (N - 1)), V, OuterStride(N * V));
      MatMap whMap(wh.data() + (i * ny * N + ii) * N * V, ny, N * V,
                   OuterStride(N * N * V));

      if (NO_CORNERS && (i == 0 || i == nx - 1))
        weno1(tmp1, uxMap, ny, 1);
      else
        weno1(tmp1, uxMap, ny, 0);
      whMap = tmp1Map;
    }
  // TODO: make this n-dimensional by turning these two sets of for loops into
  // n sets of for loops, contained within a large loop over the dimensions
}

void weno_launcher(Vecr wh, Vecr ub, iVecr nX) {
  // NOTE: boundary conditions extend u by two cells in each dimension

  int ndim = nX.size();
  MatMap whMap(wh.data(), wh.size() / V, V, OuterStride(V));
  MatMap ubMap(ub.data(), ub.size() / V, V, OuterStride(V));

  switch (ndim) {

  case 1:
    weno1(whMap, ubMap, nX(0) + 2, 0);
    break;

  case 2:
    weno2(whMap, ubMap, nX(0) + 2, nX(1) + 2);
    break;
  }
}
