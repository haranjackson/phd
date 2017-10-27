#ifndef TYPES_H
#define TYPES_H

//#define EIGEN_USE_MKL_ALL
//#define EIGEN_MKL_DIRECT_CALL

#include "../../include/eigen3/Eigen"

#include "../options.h"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vec;
typedef Eigen::Ref<Mat> Matr;
typedef Eigen::Ref<Vec> Vecr;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Arr;


typedef Eigen::Matrix<double, 2, N+1, Eigen::RowMajor> Mat2_n;
typedef Eigen::Matrix<double, V, V, Eigen::RowMajor> MatV_V;
typedef Eigen::Matrix<double, N, V, Eigen::RowMajor> MatN_V;
typedef Eigen::Matrix<double, N+1, V, Eigen::RowMajor> Matn_V;
typedef Eigen::Matrix<double, (N+1)*(N+1), V, Eigen::RowMajor> Matn2_V;
typedef Eigen::Matrix<double, (N+1)*(N+1)*(N+1), V, Eigen::RowMajor> Matn3_V;
typedef Eigen::Matrix<double, N+1, N+1, Eigen::RowMajor> Matn_n;
typedef Eigen::Matrix<double, 2*N+1, V, Eigen::RowMajor> Mat2N1_V;

typedef Eigen::Ref<Mat2_n> Mat2_nr;
typedef Eigen::Ref<MatV_V> MatV_Vr;
typedef Eigen::Ref<MatN_V> MatN_Vr;
typedef Eigen::Ref<Matn_V> Matn_Vr;
typedef Eigen::Ref<Matn2_V> Matn2_Vr;
typedef Eigen::Ref<Matn3_V> Matn3_Vr;
typedef Eigen::Ref<Matn_n> Matn_nr;
typedef Eigen::Ref<Mat2N1_V> Mat2N1_Vr;

typedef Eigen::Matrix<double, V, 1> VecV;
typedef Eigen::Matrix<double, N, 1> VecN;
typedef Eigen::Matrix<double, N+1, 1> Vecn;

typedef Eigen::Ref<VecV> VecVr;
typedef Eigen::Ref<VecN> VecNr;
typedef Eigen::Ref<Vecn> Vecnr;

typedef Eigen::OuterStride<Eigen::Dynamic> OuterStride;
typedef Eigen::Map<Mat, 0, Eigen::OuterStride<Eigen::Dynamic> > MatMap;
typedef Eigen::Map<Matn_V, 0, Eigen::OuterStride<Eigen::Dynamic> > Matn_VMap;
typedef Eigen::Map<Matn2_V, 0, Eigen::OuterStride<Eigen::Dynamic> > Matn2_VMap;
typedef Eigen::Map<Matn3_V, 0, Eigen::OuterStride<Eigen::Dynamic> > Matn3_VMap;
typedef Eigen::Map<Mat2N1_V, 0, Eigen::OuterStride<Eigen::Dynamic> > Mat2N1_VMap;
typedef Eigen::Map<Vec, 0, Eigen::InnerStride<1> > VecMap;

typedef Eigen::HouseholderQR<Mat> DecQR;
typedef Eigen::ColPivHouseholderQR<Mat> Dec;

typedef std::function<Vec(Vec)> VecFunc; // change to Vec(Vecr)


#endif
