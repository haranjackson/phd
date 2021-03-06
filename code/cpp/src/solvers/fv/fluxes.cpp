#include <cmath>

#include "eigen3/Eigenvalues"

#include "../../etc/globals.h"
#include "../../system/eig.h"
#include "../../system/equations.h"

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

VecV D_OSH(VecVr qL, VecVr qR, int d, Par &MP) {
  // Returns the Osher flux component, in the dth direction
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
  VecV f = ret.real();
  flux(f, qL, d, MP);
  flux(f, qR, d, MP);
  return f;
}

VecV D_ROE(VecVr qL, VecVr qR, int d, Par &MP) {
  // Returns the Osher flux component, in the dth direction
  VecV Δq = qL - qR;
  VeccV Δqc = VeccV(Δq);
  VecV q;
  MatV_V J = MatV_V::Zero();
  for (int i = 0; i < N; i++) {
    q = qR + NODES(i) * Δq;
    J += WGHTS(i) * system_matrix(q, d, MP);
  }
  Eigen::EigenSolver<MatV_V> es(J);
  VeccV b = es.eigenvectors().colPivHouseholderQr().solve(Δqc).array() *
            es.eigenvalues().array().abs();

  VecV f = (es.eigenvectors() * b).real();
  flux(f, qL, d, MP);
  flux(f, qR, d, MP);
  return f;
}

VecV D_RUS(VecVr qL, VecVr qR, int d, Par &MP) {

  double max1 = max_abs_eigs(qL, d, MP);
  double max2 = max_abs_eigs(qR, d, MP);

  VecV f = std::max(max1, max2) * (qL - qR);
  flux(f, qL, d, MP);
  flux(f, qR, d, MP);
  return f;
}

VecV D_RUS(VecVr qL, VecVr qR, int d, Par &MP) {

  double max1 = max_abs_eigs(qL, d, MP);
  double max2 = max_abs_eigs(qR, d, MP);

  VecV f = std::max(max1, max2) * (qL - qR);
  flux(f, qL, d, MP);
  flux(f, qR, d, MP);
  return f;
}
