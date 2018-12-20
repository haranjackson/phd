#include "../etc/globals.h"
#include "../options.h"
#include "../system/energy/eos.h"
#include "../system/functions/vectors.h"

void renorm_distortion(Vecr u, std::vector<Par> &MPs) {

  int ncell = u.size() / V;
  for (int i = 0; i < ncell; i++) {

    int mi = get_material_index(u.segment<V>(i * V));
    if (MPs[mi].EOS > -1) {

      double ρ = u(i * V);
      Mat3_3 A = get_A(u.segment<V>(i * V));
      double c = cbrt(ρ / (MPs[mi].ρ0 * A.determinant()));
      u.segment<9>(i * V + 5) *= c;
    }
  }
}

void rotate_distortion(Vecr u, std::vector<Par> &MPs) {
  int ncell = u.size() / V;
  for (int i = 0; i < ncell; i++) {

    int mi = get_material_index(u.segment<V>(i * V));
    if (MPs[mi].EOS > -1) {

      Mat3_3Map A(u.data() + i * V + 5);
      Eigen::JacobiSVD<Mat3_3> svd(A, Eigen::ComputeFullV);

      Vec3 s = svd.singularValues();
      A = svd.matrixV().transpose();

      for (int i = 0; i < 3; i++)
        A.row(i) *= s[i];
    }
  }
}

void reset_distortion(Vecr u, std::vector<Par> &MPs) {
  int ncell = u.size() / V;
  for (int i = 0; i < ncell; i++) {

    int mi = get_material_index(u.segment<V>(i * V));
    if (MPs[mi].EOS > -1) {

      double ρ = u(i * V);
      double c = cbrt(ρ / MPs[mi].ρ0);

      Mat3_3Map A(u.data() + i * V + 5);
      u(i * V + 1) -= ρ * E_2A(ρ, A, MPs[mi]);

      u.segment<9>(i * V + 5).setZero();
      u(i * V + 5) = c;
      u(i * V + 9) = c;
      u(i * V + 13) = c;
    }
  }
}

bool contorted(Vecr u, double contorted_tol) {
  int ncell = u.size() / V;
  for (int i = 0; i < ncell; i++) {

    Mat3_3Map A(u.data() + i * V + 5);
    Eigen::JacobiSVD<Mat3_3> svd(A);

    Vec3 s = svd.singularValues();
    double detA = s(0) * s(1) * s(2);
    double detA1_3 = cbrt(detA);
    double detA2_3 = detA1_3 * detA1_3;
    double s0 = s(0) * s(0) / detA2_3;
    double s1 = s(1) * s(1) / detA2_3;
    double s2 = s(2) * s(2) / detA2_3;

    double m0 = (s0 + s1 + s2) / 3.;
    if (m0 - 1 > contorted_tol)
      return true;
  }
  return false;
}
