#include <vector>

#include "../../etc/types.h"
#include "../../scipy/poly.h"

int ceil(int x, int y) { return x / y + (x % y != 0); }

std::vector<Matn_n> coefficient_matrices(const std::vector<poly> &basis) {
  // Generate linear systems governing  coefficients of the basis polynomials
  int N1 = N + 1;
  int FN2 = (int)floor(N / 2.);
  int CN2 = (int)ceil(N / 2.);

  Matn_n mL, mR, mCL, mCR; // Left, right, center-left, center-right stencils

  for (int i = 0; i < N1; i++)
    for (int j = 0; j < N1; j++) {
      poly Pj = basis[j].intt();
      mL(i, j) = Pj.eval(i - N1 + 2) - Pj.eval(i - N1 + 1);
      mR(i, j) = Pj.eval(i + 1) - Pj.eval(i);
      mCL(i, j) = Pj.eval(i - CN2 + 1) - Pj.eval(i - CN2);
      mCR(i, j) = Pj.eval(i - FN2 + 1) - Pj.eval(i - FN2);
    }

  std::vector<Matn_n> ret(4);
  ret[0] = mL;
  ret[1] = mR;
  ret[2] = mCL;
  ret[3] = mCR;
  return ret;
}

Matn_n oscillation_indicator(const std::vector<poly> &basis) {
  // Generate the oscillation indicator matrix from a set of basis polynomials
  Matn_n ret;
  for (int i = 0; i < N + 1; i++)
    for (int j = 0; j < N + 1; j++)
      for (int a = 1; a < N + 1; a++) {
        poly p = basis[i].diff(a) * basis[j].diff(a);
        poly P = p.intt();
        ret(i, j) += P.eval(1) - P.eval(0);
      }
  return ret;
}
