#include "../etc/debug.h"
#include "../etc/types.h"
#include "../system/multi/riemann.h"
#include "../system/objects/gpr_objects.h"
#include "functions.h"
#include "pfmm.h"

int iVec_to_ind(iVecr v) { return v(0) * V; }

int iVec_to_ind(iVecr v, int ny) { return (v(0) * ny + v(1)) * V; }

void update_int_mask(iVecr intMask, Vecr u, int indL, int indR, int ii) {
  double φL = u(indL * V + ii);
  double φR = u(indR * V + ii);
  if (φL * φR <= 0.) {
    intMask(indL) = sign(φL);
    intMask(indR) = sign(φR);
  }
}

void find_interface_cells(iVecr intMask, Vecr u, int interf, int nmat,
                          iVecr nX) {
  // Finds the cells lying on interface ind, given that there are nmat materials
  int ndim = nX.size();
  int ii = V - (nmat - 1) + interf;
  intMask.setZero();

  int nx = nX(0);
  if (ndim == 1) {
    for (int i = 0; i < nx - 1; i++) {
      int indL = i;
      int indR = i + 1;
      update_int_mask(intMask, u, indL, indR, ii);
    }
  } else if (ndim == 2) {
    int ny = nX(1);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        int indL = i * ny + j;
        int indR;
        if (i < nx - 1) {
          indR = (i + 1) * ny + j;
          update_int_mask(intMask, u, indL, indR, ii);
        }
        if (j < ny - 1) {
          indR = i * ny + (j + 1);
          update_int_mask(intMask, u, indL, indR, ii);
        }
      }
  }
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

void fill_boundary_cells_inner(Vecr u, Vecr grid0, Vecr grid1,
                               BoundaryInds &bInds, Par &MPL, Par &MPR,
                               double dt, Vecr n, int maskVal) {

  VecV QL = u.segment<V>(iVec_to_ind(bInds.iL));
  VecV QR = u.segment<V>(iVec_to_ind(bInds.iR));

  std::vector<VecV> S = star_states(QL, QR, MPL, MPR, dt, n);

  if (maskVal == -1) {
    grid0.segment<V>(iVec_to_ind(bInds.ii)) = S[0];
    grid0.segment<V>(iVec_to_ind(bInds.i_)) = S[0];
  } else if (maskVal == 1) {
    grid1.segment<V>(iVec_to_ind(bInds.ii)) = S[1];
    grid1.segment<V>(iVec_to_ind(bInds.i_)) = S[1];
  }
}

void fill_boundary_cells(Vecr u, Vecr grid0, Vecr grid1, iVecr intMask, Vecr φ,
                         Matr Δφ, double dx, Par &MPL, Par &MPR, double dt,
                         iVecr nX) {

  int ndim = nX.size();
  int nx = nX(0);

  if (ndim == 1) {
    for (int i = 0; i < nx; i++) {
      if (intMask(i) != 0) {
        Vec n = normal(Δφ.row(i));
        iVec inds(1);
        inds << i;

        BoundaryInds bInds = boundary_inds(inds, φ(i), n, dx);
        fill_boundary_cells_inner(u, grid0, grid1, bInds, MPL, MPR, dt, n,
                                  intMask(i));
      }
    }
  } else {
    int ny = nX(1);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        int ind = i * ny + j;
        if (intMask(ind) != 0) {
          Vec n = normal(Δφ.row(ind));
          iVec inds(2);
          inds << i, j;

          BoundaryInds bInds = boundary_inds(inds, φ(ind), n, dx);
          fill_boundary_cells_inner(u, grid0, grid1, bInds, MPL, MPR, dt, n,
                                    intMask(ind));
        }
      }
  }
}

void fill_from_neighbor(Vecr grid, Vecr Δφi, iVecr inds, double dx,
                        double sgn) {
  // makes the value of cell ind equal to the value of its neighbor in the
  // direction of the interface
  int ind = iVec_to_ind(inds);
  Vec n = normal(Δφi);
  Vec x = (inds.cast<double>().array() + 0.5) * dx;
  Vec xn = x + sgn * dx * n;
  iVec newInds = (xn / dx).cast<int>();
  int newInd = iVec_to_ind(newInds);
  grid.segment<V>(ind) = grid.segment<V>(newInd);
}

void fill_neighbor_cells(Vecr grid0, Vecr grid1, iVecr intMask, Matr Δφ,
                         double dx, iVecr nX) {

  int ndim = nX.size();
  int nx = nX(0);

  for (int N0 = 1; N0 <= N; N0++) {

    if (ndim == 1) {
      for (int i = 0; i < nx; i++) {

        if (intMask(i) == 0) {

          bool pos = false;
          bool neg = false;

          if (i > 0) {
            int ind_ = i - 1;
            pos = pos || intMask(ind_) == N0;
            neg = neg || intMask(ind_) == -N0;
          }
          if (i < nx - 1) {
            int ind_ = i + 1;
            pos = pos || intMask(ind_) == N0;
            neg = neg || intMask(ind_) == -N0;
          }
          if (pos) {
            intMask(i) = N0 + 1;
            iVec inds(1);
            inds << i;
            fill_from_neighbor(grid0, Δφ.row(i), inds, dx, -1.);
          }
          if (neg) {
            intMask(i) = -(N0 + 1);
            iVec inds(1);
            inds << i;
            fill_from_neighbor(grid1, Δφ.row(i), inds, dx, 1.);
          }
        }
      }
    } else if (ndim == 2) {

      int ny = nX(1);
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
              fill_from_neighbor(grid0, Δφ.row(ind), inds, dx, -1.);
            }
            if (neg) {
              intMask(ind) = -(N0 + 1);
              iVec inds(2);
              inds << i, j;
              fill_from_neighbor(grid1, Δφ.row(ind), inds, dx, 1.);
            }
          }
        }
    }
  }
}

void fill_ghost_cells(std::vector<Vec> &grids, std::vector<bVec> &masks, Vecr u,
                      iVecr nX, Vecr dX, double dt, std::vector<Par> &MPs) {
  // Fills in ghost cells for each fluid
  // NOTE: doesn't work for different cell spacings in each direction

  int nmat = grids.size();
  int ncell = u.size() / V;
  double dx = dX(0);

  for (int i = 0; i < nmat; i++) {
    grids[i] = u;
    masks[i].setConstant(true);
  }
  iVec intMask(ncell);
  MatMap uMap(u.data(), ncell, V, OuterStride(V));

  for (int i = 0; i < nmat - 1; i++) {

    Par MPL = MPs[i];
    Par MPR = MPs[i + 1];

    find_interface_cells(intMask, u, i, nmat, nX);
    Vec phi = uMap.col(V - (nmat - 1) + i);
    Vec φ = distance(phi, dX, nX);
    Mat Δφ = finite_difference(φ, dX, nX);

    fill_boundary_cells(u, grids[i], grids[i + 1], intMask, φ, Δφ, dx, MPL, MPR,
                        dt, nX);
    fill_neighbor_cells(grids[i], grids[i + 1], intMask, Δφ, dX(0), nX);

    for (int j = 0; j < ncell; j++) {
      masks[i](j) = masks[i](j) && (φ(j) <= 0. || intMask(j) == 1);
      masks[i + 1](j) = masks[i + 1](j) && (φ(j) >= 0. || intMask(j) == -1);
    }

    for (int j = 0; j < nmat; j++) {
      MatMap gridMap(grids[j].data(), ncell, V, OuterStride(V));
      gridMap.col(V - (nmat - 1) + i) = φ;
    }
  }
}
