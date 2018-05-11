#include "globals.h"
#include "../options.h"
#include "../solvers/basis.h"
#include "../solvers/dg/dg_matrices.h"
#include "../solvers/weno/weno_matrices.h"
#include "../system/functions/matrices.h"

MatN_N minv(Mat m) {
  if (N <= 4)
    return m.inverse();
  else
    return MatN_N::Zero();
}

std::vector<poly> basis = basis_polys();

std::vector<MatN_N> coeffMats = coefficient_matrices(basis);
MatN_N mL = coeffMats[0];
MatN_N mR = coeffMats[1];
MatN_N mCL = coeffMats[2];
MatN_N mCR = coeffMats[3];

const VecN NODES = scaled_nodes();
const VecN WGHTS = scaled_weights();
const Mat2_N ENDVALS = end_values(basis);
const MatN_N DERVALS = derivative_values(basis, NODES);

const Mat mLinv = minv(mL);
const Mat mRinv = minv(mR);
const Mat mCLinv = minv(mCL);
const Mat mCRinv = minv(mCR);

const Dec ML(mL);
const Dec MR(mR);
const Dec MCL(mCL);
const Dec MCR(mCR);
const MatN_N SIG = oscillation_indicator(basis);

const MatN_N DG_END = end_value_products(basis);
const MatN_N DG_DER = derivative_products(basis, NODES, WGHTS);
const MatN_N DG_MAT = DG_END - DG_DER.transpose();

Mat wghts = WGHTS.asDiagonal();
std::vector<Mat> tmp1 = {DG_MAT, wghts};
std::vector<Mat> tmp2 = {DG_MAT, wghts, wghts};
std::vector<Mat> tmp3 = {DG_MAT, wghts, wghts, wghts};
const Dec DG_U1(kron(tmp1));
const Dec DG_U2(kron(tmp2));
const Dec DG_U3(kron(tmp3));
