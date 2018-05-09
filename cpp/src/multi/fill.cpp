#include "../etc/types.h"
#include "../system/multi/riemann.h"
#include "../system/objects/gpr_objects.h"
#include "functions.h"
#include "pfmm.h"

void update_int_mask(iVecr intMask, Vecr u, int indL, int indR) {
  double φL = u(indL);
  double φR = u(indR);
  if (φL * φR <= 0) {
    intMask(indL) = sign(φL);
    intMask(indR) = sign(φR);
  }
}

iVec find_interface_cells(Vecr u, int ind, int m, int nx, int ny, int NDIM) {
  // Finds the cells lying on interface ind,
  // given that there are m materials
  int ii = ind - (m - 1);
  iVec intMask(u.size() / V);

  if (NDIM == 1) {
    for (int i = 0; i < nx - 1; i++) {
      int indL = i * V + ii;
      int indR = (i + 1) * V + ii;
      update_int_mask(intMask, u, indL, indR);
    }
  } else if (NDIM == 2) {
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        int indL = (i * ny + j) * V + ii;
        int indR;
        if (i < nx - 1) {
          indR = ((i + 1) * ny + j) * V + ii;
          update_int_mask(intMask, u, indL, indR);
        }
        if (j < ny - 1) {
          indR = (i * ny + (j + 1)) * V + ii;
          update_int_mask(intMask, u, indL, indR);
        }
      }
  }
  return intMask;
}

struct BoundaryInds {
  iVec ii, iL, iR, i_;
};

BoundaryInds boundary_inds(iVec inds, double φ, Vecr n, double dx) {
  // Calculates indexes of the boundary states at position given by inds
  Vec xp = (inds.cast<double>().array() + 0.5) * dx;
  double d = 1.5;

  Vec xi = xp - φ * n;            // interface position
  Vec xL = xi - d * dx * n;       // probe on left side
  Vec xR = xi + d * dx * n;       // probe on right side
  Vec x_ = xi - dx * sign(φ) * n; // point on opposite side of interface

  BoundaryInds ret;

  ret.ii = (xi / dx).cast<int>();
  ret.iL = (xL / dx).cast<int>();
  ret.iR = (xR / dx).cast<int>();
  ret.i_ = (x_ / dx).cast<int>();

  return ret;
}

void fill_boundary_cells(Vecr u, std::vector<Vec> &grids, iVecr intMask,
                         int lvl, Vecr φ, Vecr Δφ, double dx, Par &MPL,
                         Par &MPR, double dt, int nx, int ny) {

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      int ind = i * ny + j;
      if (intMask(ind) != 0) {
        Vec n = normal(Δφ);
        iVec inds(2);
        inds << i, j;
        BoundaryInds bInds = boundary_inds(inds, φ(ind), n, dx);

        VecV QL = u.segment<V>(iVec_to_ind(bInds.iL));
        VecV QR = u.segment<V>(iVec_to_ind(bInds.iR));

        StarStates S = star_states(QL, QR, MPL, MPR, dt, n);

        if (intMask[ind] == -1) {
          grids[lvl].segment<V>(iVec_to_ind(bInds.ii)) = S.QL_;
          grids[lvl].segment<V>(iVec_to_ind(bInds.i_)) = S.QL_;
        } else if (intMask(ind) == 1) {
          grids[lvl + 1].segment<V>(iVec_to_ind(bInds.ii)) = S.QR_;
          grids[lvl + 1].segment<V>(iVec_to_ind(bInds.i_)) = S.QR_;
        }
      }
    }
}

void fill_from_neighbor(Vecr grid, Vecr Δφ, iVecr inds, double dx, double sgn) {
  // makes the value of cell ind equal to the value of its neighbor in the
  // direction of the interface
  int ind = iVec_to_ind(inds);
  Vec n = normal(Δφ);
  Vec x = (inds.cast<double>().array() + 0.5) * dx;
  Vec xn = x + sgn * dx * n;
  iVec newInds = (xn / dx).cast<int>();
  int newInd = iVec_to_ind(newInds);
  grid.segment<V>(ind) = grid.segment<V>(newInd);
}

void fill_neighbor_cells(std::vector<Vec> &grids, iVecr intMask, int lvl,
                         Vecr Δφ, double dx, int nx, int ny, int NDIM) {

  for (int N0 = 1; N0 <= N; N0++)
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {

        int ind = i * ny + j;
        if (intMask(ind) == 0) {

          bool pos = false;
          bool neg = false;

          if (i > 0) {
            int ind_ = (i - 1) * ny + j;
            pos = pos || intMask(ind_) == N0;
            neg = neg || intMask(ind_) == -N0;
          }
          if (i < nx - 1) {
            int ind_ = (i + 1) * ny + j;
            pos = pos || intMask(ind_) == N0;
            neg = neg || intMask(ind_) == -N0;
          }
          if (j > 0) {
            int ind_ = i * ny + (j - 1);
            pos = pos || intMask(ind_) == N0;
            neg = neg || intMask(ind_) == -N0;
          }
          if (j < ny - 1) {
            int ind_ = i * ny + (j + 1);
            pos = pos || intMask(ind_) == N0;
            neg = neg || intMask(ind_) == -N0;
          }

          if (pos) {
            intMask(ind) = N0 + 1;
            iVec inds(2);
            inds << i, j;
            fill_from_neighbor(grids[lvl], Δφ.segment(NDIM, ind), inds, dx,
                               -1.);
          }
          if (neg) {
            intMask(ind) = -(N0 + 1);
            iVec inds(2);
            inds << i, j;
            fill_from_neighbor(grids[lvl + 1], Δφ.segment(NDIM, ind), inds, dx,
                               1.);
          }
        }
      }
}

void fill_ghost_cells(std::vector<Vec> &grids, std::vector<bVec> &masks, Vecr u,
                      int NDIM, iVec3r nX, Vecr dX, double dt,
                      std::vector<Par> &MPs) {
  // Fills in ghost cells for each fluid
  // NOTE: doesn't work for different cell spacings in each direction

  int m = grids.size();
  int ncell = u.size() / V;
  int nx = nX(0);
  int ny = nX(1);
  double dx = dX(0);

  for (int i = 0; i < m; i++) {
    grids[i] = u;
    masks[i].setConstant(true);
  }

  for (int i = 0; i < m - 1; i++) {

    Par MPL = MPs[i];
    Par MPR = MPs[i + 1];

    iVec intMask = find_interface_cells(u, i, m, nx, ny, NDIM);
    MatMap uMap(u.data(), ncell, V, OuterStride(V));
    Vec phi = uMap.col(i - (m - 1));
    Vec φ = distance(phi, dX, nX, NDIM);
    Vec Δφ = finite_difference(φ, dX, NDIM, nX);

    fill_boundary_cells(u, grids, intMask, i, φ, Δφ, dx, MPL, MPR, dt, nx, ny);
    fill_neighbor_cells(grids, intMask, i, Δφ, dX(0), nX(0), nX(1), NDIM);

    for (int j = 0; j < ncell; j++) {
      masks[i](j) = masks[i](j) && (φ(j) <= 0. || intMask(j) == 1);
      masks[i + 1](j) = masks[i](j) && (φ(j) >= 0. || intMask(j) == -1);
    }

    for (int j = 0; j < m; j++) {
      MatMap gridMap(grids[j].data(), ncell, V, OuterStride(V));
      gridMap.col(i - (m - 1)) = φ;
    }
  }
}
