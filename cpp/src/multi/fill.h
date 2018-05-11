#ifndef FILL_H
#define FILL_H

#include "../etc/types.h"
#include "../system/objects/gpr_objects.h"

iVec find_interface_cells(Vecr u, int ind, int m, iVecr nX);

void fill_boundary_cells(Vecr u, std::vector<Vec> &grids, iVecr intMask,
                         int mat, Vecr φ, Matr Δφ, double dx, Par &MPL,
                         Par &MPR, double dt, iVecr nX);

void fill_from_neighbor(Vecr grid, Matr Δφ, iVecr inds, double dx, double sgn);

void fill_neighbor_cells(std::vector<Vec> &grids, iVecr intMask, int mat,
                         Matr Δφ, double dx, iVecr nX);

void fill_ghost_cells(std::vector<Vec> &grids, std::vector<bVec> &masks, Vecr u,
                      iVecr nX, Vecr dX, double dt, std::vector<Par> &MPs);

#endif // FILL_H
