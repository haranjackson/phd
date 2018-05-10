#include "../../include/eigen3/SVD"
#include "../etc/debug.h"
#include "../etc/globals.h"
#include "functions/matrices.h"
#include "functions/vectors.h"
#include "variables/eos.h"
#include "variables/shear.h"
#include <cmath>

double pos(double x) { return std::max(0., x); }

double nondimensionalized_time(double ρ, double detA3, double m0, double u0,
                               double dt, Par &MP) {

  double τ1 = MP.τ1;
  if (MP.n != 0) {
    double n = MP.n;
    double σY = MP.σY;
    double cs2 = c_s2(ρ, MP);
    double a = 9. * m0 - u0 - 9.;
    double b = 6. * m0 - u0 - 6.;
    double c = (108. * a - 324. * b + 108. * a * a - 396. * a * b +
                297. * b * b - 24. * (a * a * b - 2. * a * b * b + b * b * b) -
                4. * std::pow(a - b, 4.));
    if (c <= 0.)
      return 0.;

    double λ = c / (18. * a - 36. * b + 9. * a * a - 132. / 5. * a * b +
                    33. / 2. * b * b - 8. / 7. * a * a * b + 2. * a * b * b -
                    8. / 9. * b * b * b - std::pow(a, 4.) / 6. +
                    16. / 27. * a * a * a * b - 4. / 5. * a * a * b * b +
                    16. / 33. * a * b * b * b - std::pow(b, 4.) / 9.);
    double tmp = std::pow(sqrt(c) * ρ * cs2 / (6. * σY), n);
    return 2. / (n * λ) *
           log(n * λ / τ1 * std::pow(detA3, 4. * n + 7.) * tmp * dt + 1.);
  } else
    return 2. * std::pow(detA3, 7.) / τ1 * dt;
}

void analyticSolver_distortion(VecVr Q, double dt, Par &MP) {
  Mat3_3Map A(Q.data() + 5);
  Eigen::JacobiSVD<Mat3_3> svd(A, Eigen::ComputeFullV | Eigen::ComputeFullU);

  Vec3 s = svd.singularValues();
  double detA = s(0) * s(1) * s(2);
  double ρ = detA * MP.ρ0;
  double detA1_3 = cbrt(detA);
  double detA2_3 = detA1_3 * detA1_3;
  double s0 = s(0) * s(0) / detA2_3;
  double s1 = s(1) * s(1) / detA2_3;
  double s2 = s(2) * s(2) / detA2_3;

  double m0 = (s0 + s1 + s2) / 3.;
  double u0 =
      ((s0 - s1) * (s0 - s1) + (s1 - s2) * (s1 - s2) + (s2 - s0) * (s2 - s0)) /
      3.;

  if (u0 >= 1e-12) {

    double τ = nondimensionalized_time(ρ, detA1_3, m0, u0, dt, MP);
    double e_6τ = exp(-6. * τ);
    double e_9τ = exp(-9. * τ);

    double a = 3. * m0 - u0 / 3. - 3.;
    double b = 2. * m0 - u0 / 3. - 2.;
    double m = 1. + a * e_6τ - b * e_9τ;
    double u = pos(6. * a * e_6τ - 9. * b * e_9τ);

    double Δ = -2. * m * m * m + m * u + 2.;
    double arg1 = pos(6. * u * u * u - 81. * Δ * Δ);

    double x1;
    if (abs(Δ) < 1e-12) {
      // θ = pi / 2
      x1 = sqrt(6. * u) / 3. * cos(M_PI / 6.) + m;
    } else if (arg1 == 0.) {
      double tmp = cbrt(54. * Δ);
      x1 = tmp / 6. + u / tmp + m;
    } else {
      double θ = atan(sqrt(arg1) / (9. * Δ));
      x1 = sqrt(6. * u) / 3. * cos(θ / 3.) + m;
    }
    double tmp2 = 3. * m - x1;
    double arg2 = pos(x1 * tmp2 * tmp2 - 4.);
    double x2 = 0.5 * (sqrt(arg2 / x1) + tmp2);
    double x3 = 1. / (x1 * x2);

    double x[3]{x1, x2, x3};
    std::sort(x, x + 3); // sorts in ascending order
    Mat3_3 Vmat = svd.matrixV().transpose();

    for (int i = 0; i < 3; i++)
      Vmat.row(i) *= detA1_3 * sqrt(x[2 - i]); // s sorted in descending order

    A.noalias() = svd.matrixU() * Vmat;
  }
}

void analyticSolver_thermal(VecVr Q, double dt, Par &MP) {
  // Solves the thermal impulse ODE analytically in 3D for the ideal gas EOS
  double ρ = Q(0);
  double E = Q(1) / ρ;
  Vec3 v = get_ρv(Q) / ρ;
  Mat3_3Map A = get_A(Q);
  Vec3Map ρJ = get_ρJ(Q);

  double c1 = E - E_2A(ρ, A, MP) - E_3(v);
  double c2 = MP.cα2 / 2.;
  double k = 2 * MP.ρ0 / (MP.τ2 * MP.T0 * ρ * MP.cv);
  c1 *= k;
  c2 *= k;

  double ea = exp(-c1 * dt / 2);
  double den = 1 - c2 / c1 * (1 - ea * ea) * ρJ.squaredNorm() / (ρ * ρ);
  Q.segment<3>(14) *= ea / sqrt(den);
}

void ode_stepper_analytic(VecVr Q, double dt, Par &MP) {
  // Solves the ODE analytically by linearising the distortion equations and
  // providing an analytic approximation to the thermal impulse evolution

  if (VISCOUS)
    analyticSolver_distortion(Q, dt, MP);
  if (THERMAL)
    analyticSolver_thermal(Q, dt, MP);
}
