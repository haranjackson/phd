#ifndef EOS_H
#define EOS_H

#include "../../etc/globals.h"
#include "../objects.h"

double E_1(double ρ, double p, Params &MP);
double E_2A(double ρ, Mat3_3r A, Par &MP);
double E_2J(Vec3r J, Par &MP);
double E_3(Vec3r v);
double total_energy(double ρ, double p, Mat3_3r A, Vec3r v, Par &MP);
double total_energy(double ρ, double p, Mat3_3r A, Vec3r J, Vec3r v, Par &MP);
double total_energy(double ρ, double p, Mat3_3r A, Vec3r v, double λ, Par &MP);
double total_energy(double ρ, double p, Mat3_3r A, Vec3r J, Vec3r v, double λ,
                    Par &MP);
double internal_energy(VecVr Q, Par &MP);

#endif // EOS_H
