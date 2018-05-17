#ifndef FILL_H
#define FILL_H

#include "../etc/types.h"
#include "../system/objects/gpr_objects.h"

void find_interface_cells(iVecr intMask, Vecr φ, iVecr nX);

void fill_boundary_cells(Vecr u, Vecr grid, iVecr intMask, int mat, Vecr φ,
                         Matr Δφ, aVecr dX, std::vector<Par> &MPs, double dt,
                         iVecr nX);

void fill_neighbor_cells(Vecr grid, iVecr intMask, Matr Δφ, aVecr dX, iVecr nX);

void fill_ghost_cells(std::vector<Vec> &grids, std::vector<bVec> &masks, Vecr u,
                      iVecr nX, aVecr dX, double dt, std::vector<Par> &MPs);

#endif // FILL_H
