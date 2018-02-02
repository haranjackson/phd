#include <iostream>

#include "etc/globals.h"
#include "solvers/iterator.h"
#include "system/functions/vectors.h"
#include "system/objects/gpr_objects.h"
#include "system/variables/eos.h"

VecV Qvec(double ρ, double p, Vec3r v, Mat3_3r A, Vec3r J, Par &MP) {
  // Returns the vector of conserved variables
  VecV Q;
  Q(0) = ρ;
  Q.segment<3>(2) = ρ * v;
  Q.segment<9>(5) = VecMap(A.data(), 9);
  Q.segment<3>(14) = ρ * J;
  double E = E_1(ρ, p, MP);
  E += E_2A(Q, MP);
  E += E_2J(Q, MP);
  E += E_3(Q);
  Q(1) = ρ * E;
  return Q;
}

Vec heat_conduction_1d(Par MP, int nx) {
  double ρL = 2;
  double ρR = 0.5;
  Mat AL = pow(ρL, 1 / 3.) * Mat::Identity(3, 3);
  Mat AR = pow(ρR, 1 / 3.) * Mat::Identity(3, 3);
  Vec v = Vec::Zero(3);
  Vec J = Vec::Zero(3);

  VecV QL = Qvec(ρL, 1., v, AL, J, MP);
  VecV QR = Qvec(ρR, 1., v, AR, J, MP);

  Vec u(nx * V);
  double dx = 1. / nx;
  for (int i = 0; i < nx; i++) {
    if (i * dx < 0.5)
      u.segment(i * V, V) = QL;
    else
      u.segment(i * V, V) = QR;
  }
  return u;
}

int main() {

  double γ = 1.4;
  double cv = 2.5;
  double κ = 1e-2;
  double μ = 1e-2;

  Par MP;
  MP.EOS = 0;
  MP.γ = γ;
  MP.cv = cv;
  MP.pINF = 0.;
  MP.ρ0 = 1.;
  MP.p0 = 1.;
  MP.T0 = 1.;
  MP.B0 = 1.;
  MP.τ1 = 6 * μ / (MP.ρ0 * MP.B0);
  MP.cα2 = 4.;
  MP.τ2 = κ * MP.ρ0 / (MP.T0 * MP.cα2);

  double tf = 0.1;
  int nx = 200;
  int ny = 1;
  int nz = 1;
  double dx = 1. / nx;
  double dy = 1. / ny;
  double dz = 1. / nz;
  double CFL = 0.6;
  bool PERIODIC = false;
  bool SPLIT = false;
  bool STRANG = true;
  bool HALF_STEP = true;
  bool STIFF = false;
  bool OSHER = false;
  bool PERR_FROB = false;

  Vec u = heat_conduction_1d(MP, nx);

  iterator(u, tf, nx, ny, nz, dx, dy, dz, CFL, PERIODIC, SPLIT, STRANG,
           HALF_STEP, STIFF, OSHER, PERR_FROB, MP);

  std::cout << "Hello World" << std::endl;
  return 0;
}
