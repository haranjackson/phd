#include "../etc/debug.h"
#include "../etc/types.h"
#include "../system/functions/vectors.h"
#include "../system/multi/riemann.h"
#include "../system/objects/gpr_objects.h"
#include "functions.h"

void update_interface_mask(iVecr intMask, Vecr φ, int indL, int indR) {
  double φL = φ(indL);
  double φR = φ(indR);
  if (φL * φR <= 0.) {
    intMask(indL) = sign(φL);
    intMask(indR) = sign(φR);
  }
}

void find_interface_cells(iVecr intMask, Vecr φ, iVecr nX) {
  // Finds the cells lying on interface ind, given that there are nmat materials
  int ndim = nX.size();
  intMask.setZero();

  int nx = nX(0);
  switch (ndim) {

  case 1:
    for (int i = 0; i < nx - 1; i++) {
      int indL = i;
      int indR = i + 1;
      update_interface_mask(intMask, φ, indL, indR);
    }
    break;

  case 2:
    int ny = nX(1);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        int indL = i * ny + j;
        int indR;
        if (i < nx - 1) {
          indR = (i + 1) * ny + j;
          update_interface_mask(intMask, φ, indL, indR);
        }
        if (j < ny - 1) {
          indR = i * ny + (j + 1);
          update_interface_mask(intMask, φ, indL, indR);
        }
      }
    break;
  }
}

BoundaryInds boundary_inds(iVec inds, double φi, aVecr n, aVecr dX, iVecr nX) {
  // Calculates indexes of the boundary states at position given by inds

  aVec xp = (inds.cast<double>().array() + 0.5) * dX;
  double d = 1.5;

  aVec xi = xp - φi * n;            // interface position
  aVec xL = xi - d * dX * n;        // probe on left side
  aVec xR = xi + d * dX * n;        // probe on right side
  aVec x_ = xi - sign(φi) * dX * n; // point on opposite side of interface

  BoundaryInds ret;

  iVec xiVec = (xi / dX).cast<int>();
  iVec xLVec = (xL / dX).cast<int>();
  iVec xRVec = (xR / dX).cast<int>();
  iVec x_Vec = (x_ / dX).cast<int>();

  int ndim = nX.size();
  switch (ndim) {

  case 1:
    ret.ii = iVec_to_ind(xiVec);
    ret.iL = iVec_to_ind(xLVec);
    ret.iR = iVec_to_ind(xRVec);
    ret.i_ = iVec_to_ind(x_Vec);
    break;

  case 2:
    int ny = nX(1);
    ret.ii = iVec_to_ind(xiVec, ny);
    ret.iL = iVec_to_ind(xLVec, ny);
    ret.iR = iVec_to_ind(xRVec, ny);
    ret.i_ = iVec_to_ind(x_Vec, ny);
    break;
  }

  return ret;
}

void fill_boundary_inner(Vecr u, Vecr grid, iVecr inds, aVecr dX, iVecr nX,
                         double φi, int mat, std::vector<Par> &MPs, double dt,
                         Vecr n) {

  BoundaryInds bInds = boundary_inds(inds, φi, n, dX, nX);

  VecV QL = u.segment<V>(bInds.iL);
  VecV QR = u.segment<V>(bInds.iR);

  int rInd = get_material_index(QR);

  std::vector<VecV> S = star_states(QL, QR, MPs[mat], MPs[rInd], dt, n);
  grid.segment<V>(bInds.ii) = S[0];
  grid.segment<V>(bInds.i_) = S[0];
}

void fill_boundary_cells(Vecr u, Vecr grid, iVecr intMask, int mat, Vecr φ,
                         Matr Δφ, aVecr dX, std::vector<Par> &MPs, double dt,
                         iVecr nX) {
  int ndim = nX.size();
  int nx = nX(0);

  switch (ndim) {

  case 1:
    for (int ind = 0; ind < nx; ind++) {
      if (intMask(ind) == -1) {
        Vec n = normal(Δφ.row(ind));
        iVec inds(1);
        inds << ind;
        fill_boundary_inner(u, grid, inds, dX, nX, φ(ind), mat, MPs, dt, n);
      }
    }
    break;
  case 2:
    int ny = nX(1);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        int ind = i * ny + j;
        if (intMask(ind) == -1) {
          Vec n = normal(Δφ.row(ind));
          iVec inds(2);
          inds << i, j;
          fill_boundary_inner(u, grid, inds, dX, nX, φ(ind), mat, MPs, dt, n);
        }
      }
    break;
  }
}

void fill_neighbor_inner(Vecr grid, Vecr Δφi, iVecr inds, aVecr dX, iVecr nX) {
  // makes the value of cell ind equal to the value of its neighbor in the
  // direction of the interface
  int ndim = nX.size();
  int ind, newInd;

  aVec n = normal(Δφi);
  aVec x = (inds.cast<double>().array() + 0.5) * dX;
  aVec xn = x - dX * n;
  iVec newInds = (xn / dX).cast<int>();

  switch (ndim) {
  case 1:
    ind = iVec_to_ind(inds);
    newInd = iVec_to_ind(newInds);
    break;
  case 2:
    int ny = nX(1);
    ind = iVec_to_ind(inds, ny);
    newInd = iVec_to_ind(newInds, ny);
    break;
  }
  grid.segment<V>(ind) = grid.segment<V>(newInd);
}

void fill_neighbor_cells(Vecr grid, iVecr intMask, Matr Δφ, aVecr dX,
                         iVecr nX) {

  int ndim = nX.size();
  int nx = nX(0);

  for (int N0 = 1; N0 < N + 1; N0++) {

    if (ndim == 1) {
      for (int i = 0; i < nx; i++) {

        if (intMask(i) == 0) {

          bool pos = false;

          if (i > 0) {
            int ind_ = i - 1;
            pos = pos || intMask(ind_) == N0;
          }
          if (i < nx - 1) {
            int ind_ = i + 1;
            pos = pos || intMask(ind_) == N0;
          }
          if (pos) {
            intMask(i) = N0 + 1;
            iVec inds(1);
            inds << i;
            fill_neighbor_inner(grid, Δφ.row(i), inds, dX, nX);
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

            if (i > 0) {
              int ind_ = (i - 1) * ny + j;
              pos = pos || intMask(ind_) == N0;
            }
            if (i < nx - 1) {
              int ind_ = (i + 1) * ny + j;
              pos = pos || intMask(ind_) == N0;
            }
            if (j > 0) {
              int ind_ = i * ny + (j - 1);
              pos = pos || intMask(ind_) == N0;
            }
            if (j < ny - 1) {
              int ind_ = i * ny + (j + 1);
              pos = pos || intMask(ind_) == N0;
            }
            if (pos) {
              intMask(ind) = N0 + 1;
              iVec inds(2);
              inds << i, j;
              fill_neighbor_inner(grid, Δφ.row(ind), inds, dX, nX);
            }
          }
        }
    }
  }
}

void fill_ghost_cells(std::vector<Vec> &grids, std::vector<bVec> &masks, Vecr u,
                      iVecr nX, aVecr dX, double dt, std::vector<Par> &MPs) {
  // Fills in ghost cells for each fluid
  // NOTE: doesn't work for different cell spacings in each direction

  int nmat = grids.size();
  int ncell = u.size() / V;

  MatMap uMap(u.data(), ncell, V, OuterStride(V));
  renormalize_levelsets(uMap, nmat, dX, nX);
  iVec intMask(ncell);

  for (int mat = 0; mat < nmat; mat++) {

    if (MPs[mat].EOS > -1) {

      grids[mat] = u;

      Vec φ = material_indicator(uMap, mat, nmat, dX, nX);
      Mat Δφ = finite_difference(φ, dX, nX);

      find_interface_cells(intMask, φ, nX);

      fill_boundary_cells(u, grids[mat], intMask, mat, φ, Δφ, dX, MPs, dt, nX);

      fill_neighbor_cells(grids[mat], intMask, Δφ, dX, nX);

      for (int j = 0; j < ncell; j++)
        masks[mat](j) = φ(j) <= 0. || intMask(j) == 1;

      MatMap gridMap(grids[mat].data(), ncell, V, OuterStride(V));
      gridMap.block(0, V - (nmat - 1), ncell, nmat - 1) =
          uMap.block(0, V - (nmat - 1), ncell, nmat - 1);
    }
  }
}
