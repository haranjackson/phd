#ifndef TYPES_H
#define TYPES_H

//#define EIGEN_USE_MKL_ALL
//#define EIGEN_MKL_DIRECT_CALL

//#define EIGEN_DONT_VECTORIZE
//#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT

#include "../../include/eigen3/Eigen"
#include "../options.h"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vec;
typedef Eigen::Ref<Mat> Matr;
typedef Eigen::Ref<Vec> Vecr;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Arr;

typedef Eigen::Matrix<double, 2, N, Eigen::RowMajor> Mat2_N;
typedef Eigen::Matrix<double, V, V, Eigen::RowMajor> MatV_V;
typedef Eigen::Matrix<double, N, V, Eigen::RowMajor> MatN_V;
typedef Eigen::Matrix<double, N * N, V, Eigen::RowMajor> MatN2_V;
typedef Eigen::Matrix<double, N * N * N, V, Eigen::RowMajor> MatN3_V;
typedef Eigen::Matrix<double, N, N, Eigen::RowMajor> MatN_N;
typedef Eigen::Matrix<double, 2 * N, V, Eigen::RowMajor> Mat2N_V;

typedef Eigen::Ref<Mat2_N> Mat2_Nr;
typedef Eigen::Ref<MatV_V> MatV_Vr;
typedef Eigen::Ref<MatN_V> MatN_Vr;
typedef Eigen::Ref<MatN2_V> MatN2_Vr;
typedef Eigen::Ref<MatN3_V> MatN3_Vr;
typedef Eigen::Ref<MatN_N> MatN_Nr;
typedef Eigen::Ref<Mat2N_V> Mat2N_Vr;

typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Ref<Vec3> Vec3r;
typedef Eigen::Matrix<int, 3, 1> Veci3;
typedef Eigen::Ref<Veci3> Veci3r;

typedef Eigen::Matrix<double, N, 1> VecN;
typedef Eigen::Matrix<double, V, 1> VecV;

typedef Eigen::Matrix<std::complex<double>, V, 1> VeccV;

typedef Eigen::Ref<VecV> VecVr;
typedef Eigen::Ref<VecN> VecNr;

typedef Eigen::OuterStride<Eigen::Dynamic> OuterStride;
typedef Eigen::Map<Mat, 0, Eigen::OuterStride<Eigen::Dynamic>> MatMap;
typedef Eigen::Map<MatN_V, 0, Eigen::OuterStride<Eigen::Dynamic>> MatN_VMap;
typedef Eigen::Map<MatN2_V, 0, Eigen::OuterStride<Eigen::Dynamic>> MatN2_VMap;
typedef Eigen::Map<MatN3_V, 0, Eigen::OuterStride<Eigen::Dynamic>> MatN3_VMap;
typedef Eigen::Map<Mat2N_V, 0, Eigen::OuterStride<Eigen::Dynamic>> Mat2N_VMap;
typedef Eigen::Map<Vec, 0, Eigen::InnerStride<1>> VecMap;

typedef Eigen::HouseholderQR<Mat> DecQR;
typedef Eigen::ColPivHouseholderQR<Mat> Dec;

typedef std::function<Vec(Vecr)> VecFunc;

#endif
