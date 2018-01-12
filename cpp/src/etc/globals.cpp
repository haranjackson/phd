#include "globals.h"
#include "../options.h"
#include "../solvers/basis.h"
#include "../solvers/dg/dg_matrices.h"
#include "../solvers/weno/weno_matrices.h"
#include "../system/functions/matrices.h"

Mat minv(Mat m) {
  if (N == 0)
    return m.inverse();
  else if (N == 1)
    return inv2(m);
  else if (N == 2)
    return inv3(m);
  else
    return Mat::Zero(N1, N1);
}

std::vector<poly> basis = basis_polys();

std::vector<Matn_n> coeffMats = coefficient_matrices(basis);
Matn_n mL = coeffMats[0];
Matn_n mR = coeffMats[1];
Matn_n mCL = coeffMats[2];
Matn_n mCR = coeffMats[3];

const Vecn NODES = scaled_nodes();
const Vecn WGHTS = scaled_weights();
const Mat2_n ENDVALS = end_values(basis);
const Matn_n DERVALS = derivative_values(basis, NODES);

const Mat mLinv = minv(mL);
const Mat mRinv = minv(mR);
const Mat mCLinv = minv(mCL);
const Mat mCRinv = minv(mCR);

const Dec ML(mL);
const Dec MR(mR);
const Dec MCL(mCL);
const Dec MCR(mCR);
const Matn_n SIG = oscillation_indicator(basis);

const Matn_n DG_END = end_value_products(basis);
const Matn_n DG_DER = derivative_products(basis, NODES, WGHTS);
const Matn_n DG_MAT = DG_END - DG_DER.transpose();

Mat wghts = WGHTS.asDiagonal();
std::vector<Mat> tmp1 = {DG_MAT, wghts};
std::vector<Mat> tmp2 = {DG_MAT, wghts, wghts};
std::vector<Mat> tmp3 = {DG_MAT, wghts, wghts, wghts};
const Dec DG_U1(kron(tmp1));
const Dec DG_U2(kron(tmp2));
const Dec DG_U3(kron(tmp3));
