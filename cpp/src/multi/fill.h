#ifndef FILL_H
#define FILL_H

fill_ghost_cells(std::vector<Vec> &grids, std::vector<bVec> &masks, Vecr u,
                      int NDIM, iVec3r nX, Vecr dX, double dt,
                      std::vector<Par> &MPs);

#endif // FILL_H
