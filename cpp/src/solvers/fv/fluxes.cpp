#include "../../etc/globals.h"
#include "../../include/eigen3/Eigenvalues"
#include "../../system/eig.h"
#include "../../system/equations.h"
#include <cmath>

VecV Aint(VecVr qL, VecVr qR, int d, Par &MP) {
  // Returns the Osher-Solomon jump matrix for A, in the dth direction
  VecV ret = VecV::Zero();
  VecV Δq = qR - qL;
  for (int i = 0; i < N; i++) {
    VecV q = qL + NODES[i] * Δq;
    MatV_V J = system(q, d, MP);
    Eigen::EigenSolver<MatV_V> es(J);
    MatV_V R = es.eigenvectors().real();
    VecV b = R.colPivHouseholderQr().solve(Δq);
    for (int j = 0; j < V; j++)
      b(j) = std::abs(es.eigenvalues().array().abs()(j)) * b(j);
    ret += WGHTS[i] * R * b;
  }
  return ret;
}

VecV Bint(VecVr qL, VecVr qR, int d, Par &MP) {
  // Returns the jump matrix for B, in the dth direction.
  VecV ret = VecV::Zero();
  VecV Δq = qR - qL;
  VecV q, tmp;
  for (int i = 0; i < N; i++) {
    q = qL + NODES(i) * Δq;
    Bdot(tmp, q, Δq, d, MP);
    ret += WGHTS(i) * tmp;
  }
  return ret;
}

VecV Smax(VecVr qL, VecVr qR, int d, bool PERRON_FROBENIUS, Par &MP) {
  double max1 = max_abs_eigs(qL, d, PERRON_FROBENIUS, MP);
  double max2 = max_abs_eigs(qR, d, PERRON_FROBENIUS, MP);
  return std::max(max1, max2) * (qL - qR);
}
