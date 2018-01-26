#ifndef GLOBALS_H
#define GLOBALS_H

#include <limits>

#include "../options.h"
#include "../solvers/basis.h"
#include "../solvers/weno/weno_matrices.h"
#include "types.h"

const int FN2 = (int)floor((N - 1) / 2.);
const int CN2 = (int)ceil((N - 1) / 2.);

extern const VecN NODES;
extern const VecN WGHTS;
extern const Mat2_N ENDVALS;
extern const MatN_N DERVALS;

extern const Mat mLinv;
extern const Mat mRinv;
extern const Mat mCLinv;
extern const Mat mCRinv;

extern const Dec ML;
extern const Dec MR;
extern const Dec MCL;
extern const Dec MCR;
extern const MatN_N SIG;

extern const MatN_N DG_END;
extern const MatN_N DG_DER;
extern const MatN_N DG_MAT;
extern const Dec DG_U1;
extern const Dec DG_U2;
extern const Dec DG_U3;

const double mEPS = 2.2204460492503131e-16;
const double INF = std::numeric_limits<double>::max();

#endif // GLOBALS_H
