#include "../etc/types.h"
#include "../scipy/legendre.h"
#include "../scipy/poly.h"
#include <vector>

Vecn scaled_nodes() {
  std::vector<Vec> tmp = leggauss(N + 1);
  Vecn nodes = tmp[0];
  nodes.array() += 1;
  nodes /= 2;
  return nodes;
}

Vecn scaled_weights() {
  std::vector<Vec> tmp = leggauss(N + 1);
  Vecn wghts = tmp[1];
  wghts /= 2;
  return wghts;
}

poly lagrange(Vecr x, int i) {
  /*  Return a Lagrange interpolating polynomial at the ith Legendre node.
      Warning: This implementation is numerically unstable. Do not use more than
      about 20 points even if they are chosen optimally.

      Input
      ----------
      x : array
          Legendre nodes

      Output
      -------
      lagrange : numpy.poly1d instance
          The Lagrange interpolating polynomial.
  */
  poly p = poly((Vec(1) << 1).finished());
  for (int j = 0; j < x.size(); j++) {
    if (j == i)
      continue;
    p = p * poly((Vec(2) << -x(j), 1).finished()) / (x(i) - x(j));
  }
  return p;
}

std::vector<poly> basis_polys() {
  // Returns basis polynomials
  Vecn nodes = scaled_nodes();
  std::vector<poly> psi(N + 1);
  for (int i = 0; i < N + 1; i++)
    psi[i] = lagrange(nodes, i);
  return psi;
}

Mat2_n end_values(const std::vector<poly> &basis) {
  // ret[i,0], ret[i,1] are the the values of ith basis polynomial at 0,1
  Mat2_n ret;
  for (int j = 0; j < N + 1; j++) {
    ret(0, j) = basis[j].eval(0.);
    ret(1, j) = basis[j].eval(1.);
  }
  return ret;
}

Matn_n derivative_values(const std::vector<poly> &basis, const Vecn nodes) {
  // ret[i,j] is the derivative of the jth basis function at the ith node
  Matn_n ret;
  for (int i = 0; i < N + 1; i++)
    for (int j = 0; j < N + 1; j++)
      ret(i, j) = basis[j].diff(1).eval(nodes(i));
  return ret;
}

/*
def derivative_end_values():
    """ Returns the value of the derivative of the ith basis function at 0 and 1
    """
    _, psiDer, _ = basis_polys()
    ret = zeros([N1, 2])
    for i in range(N1):
        ret[i,0] = psiDer[1][i](0)
        ret[i,1] = psiDer[1][i](1)
    return ret

def mid_values():
    psi, _, _ = basis_polys()
    return array([psii(0.5) for psii in psi])
*/
