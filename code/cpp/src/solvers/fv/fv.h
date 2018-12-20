#ifndef FV_H
#define FV_H

#include "../../etc/globals.h"
#include "../../system/objects.h"

void centers1(Vecr u, Vecr rec, int nx, double dt, double dx, bool SOURCES,
              bool TIME, Par &MP, bVecr mask);
void centers2(Vecr u, Vecr rec, int nx, int ny, double dt, double dx, double dy,
              bool SOURCES, bool TIME, Par &MP, bVecr mask);

void interfs1(Vecr u, Vecr rec, int nx, double dt, double dx, bool TIME,
              int FLUX, Par &MP, bVecr mask);
void interfs2(Vecr u, Vecr rec, int nx, int ny, double dt, double dx, double dy,
              bool TIME, int FLUX, Par &MP, bVecr mask);

void fv_launcher(Vecr u, Vecr rec, iVecr nX, double dt, Vecr dX, bool SOURCES,
                 bool TIME, int FLUX, Par &MP, bVecr mask);

#endif // FV_H
