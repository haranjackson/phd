#ifndef STATE_H
#define STATE_H

#include "../../etc/types.h"
#include "../objects.h"

double pressure(VecVr Q, Par &MP);

Mat3_3 sigma(VecVr Q, Par &MP);

Vec3 sigma(VecVr Q, Par &MP, int d);

Vec3 Sigma(VecVr Q, Par &MP, int d);

Vec3 dsigmadρ(VecVr Q, Par &MP, int d);

Mat3_3 dsigmadA(VecVr Q, Par &MP, int d);

double dsigmadA(double ρ, double cs2, Mat3_3r A, Mat3_3r G, Mat3_3r AdevG,
                int i, int j, int m, int n);

double temperature_prim(double ρ, double p, Params &MP);

double temperature(VecVr Q, Par &MP);

Vec3 heat_flux(double T, Vec3r J, Par &MP);

#endif // STATE_H
