#ifndef FV_H
#define FV_H

#include "../../etc/globals.h"
#include "../../system/objects/gpr_objects.h"

void centers1(Vecr u, Vecr rec, int nx, double dt, double dx, bool SOURCES,
              bool TIME, Par &MP);
void centers2(Vecr u, Vecr rec, int nx, int ny, double dt, double dx, double dy,
              bool SOURCES, bool TIME, Par &MP);

void interfs1(Vecr u, Vecr rec, int nx, double dt, double dx, bool TIME,
              bool OSHER, bool PERR_FROB, Par &MP);
void interfs2(Vecr u, Vecr rec, int nx, int ny, double dt, double dx, double dy,
              bool TIME, bool OSHER, bool PERR_FROB, Par &MP);

void fv_launcher(Vecr u, Vecr rec, int ndim, int nx, int ny, int nz, double dt,
                 double dx, double dy, double dz, bool SOURCES, bool TIME,
                 bool OSHER, bool PERR_FROB, Par &MP);

#endif // FV_H
