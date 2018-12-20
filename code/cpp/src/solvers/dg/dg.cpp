#include "dg1.h"
#include "dg2.h"

void predictor(Vecr qh, Vecr wh, double dt, Vecr dX, bool STIFF, bool STIFF_IG,
               Par &MP, bVecr mask) {

  int ndim = dX.size();

  switch (ndim) {
  case 1:
    predictor1(qh, wh, dt, dX(0), STIFF, STIFF_IG, MP, mask);
    break;
  case 2:
    predictor2(qh, wh, dt, dX(0), dX(1), STIFF, STIFF_IG, MP, mask);
    break;
  }
}
