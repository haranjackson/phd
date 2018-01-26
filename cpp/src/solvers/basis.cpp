#include "../etc/types.h"
#include "../scipy/legendre.h"
#include "../scipy/poly.h"
#include <vector>

VecN scaled_nodes() {
  std::vector<Vec> tmp = leggauss(N);
  VecN nodes = tmp[0];
  nodes.array() += 1;
  nodes /= 2;
  return nodes;
}

VecN scaled_weights() {
  std::vector<Vec> tmp = leggauss(N);
  VecN wghts = tmp[1];
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
  VecN nodes = scaled_nodes();
  std::vector<poly> psi(N);
  for (int i = 0; i < N; i++)
    psi[i] = lagrange(nodes, i);
  return psi;
}

Mat2_N end_values(const std::vector<poly> &basis) {
  // ret[i,0], ret[i,1] are the the values of ith basis polynomial at 0,1
  Mat2_N ret;
  for (int j = 0; j < N; j++) {
    ret(0, j) = basis[j].eval(0.);
    ret(1, j) = basis[j].eval(1.);
  }
  return ret;
}

MatN_N derivative_values(const std::vector<poly> &basis, const VecN nodes) {
  // ret[i,j] is the derivative of the jth basis function at the ith node
  MatN_N ret;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      ret(i, j) = basis[j].diff(1).eval(nodes(i));
  return ret;
}

/*
def derivative_end_values():
    """ Returns the value of the derivative of the ith basis function at 0 and 1
    """
    _, psiDer, _ = basis_polys()
    ret = zeros([N, 2])
    for i in range(N):
        ret[i,0] = psiDer[1][i](0)
        ret[i,1] = psiDer[1][i](1)
    return ret

def mid_values():
    psi, _, _ = basis_polys()
    return array([psii(0.5) for psii in psi])
*/
