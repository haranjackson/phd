#include "../../etc/globals.h"
#include "../../etc/types.h"
#include "../../scipy/poly.h"
#include <vector>

int ceil(int x, int y) { return x / y + (x % y != 0); }

std::vector<MatN_N> coefficient_matrices(const std::vector<poly> &basis) {
  // Generate linear systems governing  coefficients of the basis polynomials

  MatN_N mL, mR, mCL, mCR; // Left, right, center-left, center-right stencils

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      poly Pj = basis[j].intt();
      mL(i, j) = Pj.eval(i - N + 2) - Pj.eval(i - N + 1);
      mR(i, j) = Pj.eval(i + 1) - Pj.eval(i);
      mCL(i, j) = Pj.eval(i - CN2 + 1) - Pj.eval(i - CN2);
      mCR(i, j) = Pj.eval(i - FN2 + 1) - Pj.eval(i - FN2);
    }

  std::vector<MatN_N> ret(4);
  ret[0] = mL;
  ret[1] = mR;
  ret[2] = mCL;
  ret[3] = mCR;
  return ret;
}

MatN_N oscillation_indicator(const std::vector<poly> &basis) {
  // Generate the oscillation indicator matrix from a set of basis polynomials
  MatN_N ret;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      for (int a = 1; a < N; a++) {
        poly p = basis[i].diff(a) * basis[j].diff(a);
        poly P = p.intt();
        ret(i, j) += P.eval(1) - P.eval(0);
      }
  return ret;
}
