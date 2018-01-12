#include "../etc/globals.h"

void derivs2d(Matn2_Vr ret, Matn2_Vr qh, int d) {
  // ret[s] is the value of the derivative of qh in  direction d at node s
  if (d == 0) {
    for (int ind = 0; ind < N1V; ind += V) {
      Matn_VMap qhj(qh.data() + ind, OuterStride(N1V));
      Matn_VMap retj(ret.data() + ind, OuterStride(N1V));
      retj.noalias() = DERVALS * qhj;
    }
  } else if (d == 1) {
    for (int ind = 0; ind < N1N1V; ind += N1V) {
      Matn_VMap qhi(qh.data() + ind, OuterStride(V));
      Matn_VMap reti(ret.data() + ind, OuterStride(V));
      reti.noalias() = DERVALS * qhi;
    }
  }
}

void endpts2d(Matn_Vr ret, Matn2_Vr qh, int d, int e) {
  // ret[i] is value of qh at end e (0 or 1) of the dth axis
  if (d == 0) {
    for (int j = 0; j < N1; j++) {
      int ind = j * V;
      Matn_VMap qhj(qh.data() + ind, OuterStride(N1V));
      ret.row(j) = ENDVALS.row(e) * qhj;
    }
  } else if (d == 1) {
    for (int i = 0; i < N1; i++) {
      int ind = i * N1V;
      Matn_VMap qhi(qh.data() + ind, OuterStride(V));
      ret.row(i) = ENDVALS.row(e) * qhi;
    }
  }
}
