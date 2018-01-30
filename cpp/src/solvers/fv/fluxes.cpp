#include "../../etc/globals.h"
#include "../../include/eigen3/Eigenvalues"
#include "../../system/eig.h"
#include "../../system/equations.h"
#include <cmath>

VecV Aint(VecVr qL, VecVr qR, int d, Par &MP) {
  // Returns the Osher-Solomon jump matrix for A, in the dth direction
  VeccV ret = VeccV::Zero();
  VecV Δq = qL - qR;
  VeccV Δqc = VeccV(Δq);
  Eigen::EigenSolver<MatV_V> es;

  VecV q;
  VeccV b;
  MatV_V J;
  for (int i = 0; i < N; i++) {
    q = qR + NODES(i) * Δq;
    J = system_matrix(q, d, MP);
    es.compute(J);
    b = es.eigenvectors().colPivHouseholderQr().solve(Δqc).array() *
        es.eigenvalues().array().abs();
    ret += WGHTS(i) * (es.eigenvectors() * b);
  }
  return ret.real();
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

VecV Smax(VecVr qL, VecVr qR, int d, bool PERR_FROB, Par &MP) {
  double max1 = max_abs_eigs(qL, d, PERR_FROB, MP);
  double max2 = max_abs_eigs(qR, d, PERR_FROB, MP);
  return std::max(max1, max2) * (qL - qR);
}
