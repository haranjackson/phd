#ifndef FILL_H
#define FILL_H

#include "../etc/types.h"
#include "../system/objects/gpr_objects.h"

void find_interface_cells(iVecr intMask, Vecr u, int interf, int nmat,
                          iVecr nX);

void fill_boundary_cells(Vecr u, Vecr grid0, Vecr grid1, iVecr intMask, Vecr φ,
                         Matr Δφ, double dx, Par &MPL, Par &MPR, double dt,
                         iVecr nX);

void fill_from_neighbor(Vecr grid, Vecr Δφi, iVecr inds, double dx, double sgn,
                        iVecr nX);

void fill_neighbor_cells(Vecr grid0, Vecr grid1, iVecr intMask, Matr Δφ,
                         double dx, iVecr nX);

void fill_ghost_cells(std::vector<Vec> &grids, std::vector<bVec> &masks, Vecr u,
                      iVecr nX, Vecr dX, double dt, std::vector<Par> &MPs);

#endif // FILL_H
