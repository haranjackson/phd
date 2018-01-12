#ifndef GLOBALS_H
#define GLOBALS_H

#include <limits>

#include "../options.h"
#include "../solvers/basis.h"
#include "../solvers/weno/weno_matrices.h"
#include "types.h"

const int N1 = N + 1;
const int FN2 = (int)floor(N / 2.);
const int CN2 = (int)ceil(N / 2.);

const int N1N1 = N1 * N1;
const int N1N1N1 = N1 * N1 * N1;
const int N1V = N1 * V;
const int N1N1V = N1 * N1 * V;
const int N1N1N1V = N1 * N1 * N1 * V;

extern const Vecn NODES;
extern const Vecn WGHTS;
extern const Mat2_n ENDVALS;
extern const Matn_n DERVALS;

extern const Mat mLinv;
extern const Mat mRinv;
extern const Mat mCLinv;
extern const Mat mCRinv;

extern const Dec ML;
extern const Dec MR;
extern const Dec MCL;
extern const Dec MCR;
extern const Matn_n SIG;

extern const Matn_n DG_END;
extern const Matn_n DG_DER;
extern const Matn_n DG_MAT;
extern const Dec DG_U1;
extern const Dec DG_U2;
extern const Dec DG_U3;

const double mEPS = 2.2204460492503131e-16;
const double INF = std::numeric_limits<double>::max();

#endif // GLOBALS_H
