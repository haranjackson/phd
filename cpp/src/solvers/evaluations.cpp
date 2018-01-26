#include "../etc/globals.h"

void derivs2d(MatN2_Vr ret, MatN2_Vr qh, int d) {
  // ret[s] is the value of the derivative of qh in  direction d at node s
  if (d == 0) {
    for (int ind = 0; ind < N * V; ind += V) {
      MatN_VMap qhj(qh.data() + ind, OuterStride(N * V));
      MatN_VMap retj(ret.data() + ind, OuterStride(N * V));
      retj.noalias() = DERVALS * qhj;
    }
  } else if (d == 1) {
    for (int ind = 0; ind < N * N * V; ind += N * V) {
      MatN_VMap qhi(qh.data() + ind, OuterStride(V));
      MatN_VMap reti(ret.data() + ind, OuterStride(V));
      reti.noalias() = DERVALS * qhi;
    }
  }
}

void endpts2d(MatN_Vr ret, MatN2_Vr qh, int d, int e) {
  // ret[i] is value of qh at end e (0 or 1) of the dth axis
  if (d == 0) {
    for (int j = 0; j < N; j++) {
      int ind = j * V;
      MatN_VMap qhj(qh.data() + ind, OuterStride(N * V));
      ret.row(j) = ENDVALS.row(e) * qhj;
    }
  } else if (d == 1) {
    for (int i = 0; i < N; i++) {
      int ind = i * N * V;
      MatN_VMap qhi(qh.data() + ind, OuterStride(V));
      ret.row(i) = ENDVALS.row(e) * qhi;
    }
  }
}
