#ifndef FV_H
#define FV_H

#include "../../etc/globals.h"
#include "../../system/objects/gpr_objects.h"


void fv_launcher(Vecr u, Vecr rec,
                 int ndim, int nx, int ny, int nz,
                 double dt, double dx, double dy, double dz,
                 bool SOURCES, bool TIME, bool PERRON_FROBENIUS,
                 Par & MP);


#endif // FV_H
