#include "../etc/types.h"
#include "../system/functions/vectors.h"
#include "../system/multi/riemann.h"
#include "../system/objects.h"
#include "functions.h"

int iVec_to_ind(iVecr v) { return v(0) * V; }

int iVec_to_ind(iVecr v, int ny) { return (v(0) * ny + v(1)) * V; }

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
    MatMap φMap(φ.data(), nx, ny, OuterStride(ny));
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        int ind0 = i * ny + j;
        if (φ(ind0) <= 0.) {
          bool condL = (i > 0) && (φMap(i - 1, j) <= 0.);
          bool condR = (i < nx - 1) && (φMap(i + 1, j) <= 0.);
          bool condD = (j > 0) && (φMap(i, j - 1) <= 0.);
          bool condU = (j < ny - 1) && (φMap(i, j + 1) <= 0.);

          // not an isolated point
          if (condL || condR || condD || condU) {
            int ind1;
            if (i > 0) {
              ind1 = (i - 1) * ny + j;
              update_interface_mask(intMask, φ, ind0, ind1);
            }
            if (i < nx - 1) {
              ind1 = (i + 1) * ny + j;
              update_interface_mask(intMask, φ, ind0, ind1);
            }
            if (j > 0) {
              ind1 = i * ny + (j - 1);
              update_interface_mask(intMask, φ, ind0, ind1);
            }
            if (j < ny - 1) {
              ind1 = i * ny + (j + 1);
              update_interface_mask(intMask, φ, ind0, ind1);
            }
          } else { // isolated point
            intMask(ind0) = -2;
          }
        }
      }
    break;
  }
}

BoundaryInds boundary_inds(iVec inds, double φi, aVecr n, aVecr dX, iVecr nX) {
  // Calculates indexes of the boundary states at position given by inds

  aVec xp = (inds.cast<double>().array() + 0.5) * dX;
  double d = 1.;

  aVec xi = xp - φi * n;     // interface position
  aVec xL = xi - d * dX * n; // probe on left side
  aVec xR = xi + d * dX * n; // probe on right side

  BoundaryInds ret;

  iVec xiVec = (xi / dX).cast<int>();
  iVec xLVec = (xL / dX).cast<int>();
  iVec xRVec = (xR / dX).cast<int>();

  int ndim = nX.size();
  switch (ndim) {

  case 1:
    ret.ind = iVec_to_ind(inds);
    ret.interf = iVec_to_ind(xiVec);
    ret.L = iVec_to_ind(xLVec);
    ret.R = iVec_to_ind(xRVec);
    break;

  case 2:
    int ny = nX(1);
    ret.ind = iVec_to_ind(inds, ny);
    ret.interf = iVec_to_ind(xiVec, ny);
    ret.L = iVec_to_ind(xLVec, ny);
    ret.R = iVec_to_ind(xRVec, ny);
    break;
  }

  return ret;
}

void fill_boundary_inner(Vecr u, Vecr grid, iVecr inds, aVecr dX, iVecr nX,
                         double φi, int mat, std::vector<Par> &MPs, double dt,
                         Vecr n) {
  // Attempts to fill the boundary cell at location given by inds.
  // Returns 1 if successful, and -2 if fails to find a suitable
  // left state for the interface.

  // TODO: handle case when QL is chosen to be an isolated point

  BoundaryInds bInds = boundary_inds(inds, φi, n, dX, nX);

  VecV QL;
  VecV QR = u.segment<V>(bInds.R);
  int miR = get_material_index(QR);

  if (get_material_index(u.segment<V>(bInds.interf)) == mat)
    QL = u.segment<V>(bInds.interf);

  else if (get_material_index(u.segment<V>(bInds.L)) == mat)
    QL = u.segment<V>(bInds.L);

  else { // should only happen if ndim>1
    iVec Linds = inds;
    int Lind;

    // note: n points from left side to right
    int n0 = sgn(n(0));
    int n1 = sgn(n(1));

    if (std::abs(n(0)) > std::abs(n(1))) {
      Linds(0) -= n0;
      Lind = iVec_to_ind(Linds, nX(1));
      if (get_material_index(u.segment<V>(Lind)) == mat) {
        n(0) = n0;
        n(1) = 0.;
      } else {
        Linds(0) += n0;
        Linds(1) -= n1;
        Lind = iVec_to_ind(Linds, nX(1));
        n(0) = 0.;
        n(1) = n1;
      }
    } else {
      Linds(1) -= n1;
      Lind = iVec_to_ind(Linds, nX(1));
      if (get_material_index(u.segment<V>(Lind)) == mat) {
        n(0) = 0.;
        n(1) = n1;
      } else {
        Linds(0) -= n0;
        Linds(1) += n1;
        Lind = iVec_to_ind(Linds, nX(1));
        n(0) = n0;
        n(1) = 0.;
      }
    }
    QL = u.segment<V>(Lind);
  }

  VecV QL_ = left_star_state(QL, QR, MPs[mat], MPs[miR], dt, n);
  grid.segment<V - LSET>(bInds.ind) = QL_.head<V - LSET>();

  if (get_material_index(grid.segment<V>(bInds.interf)) != mat)
    grid.segment<V - LSET>(bInds.interf) = QL_.head<V - LSET>();
}

void fill_boundary_cells(Vecr u, Vecr grid, iVecr intMask, int mat, Vecr φ,
                         Matr Δφ, aVecr dX, std::vector<Par> &MPs, double dt,
                         iVecr nX) {
  int ndim = nX.size();
  int nx = nX(0);
  Vec n(ndim);

  switch (ndim) {

  case 1:
    for (int ind = 0; ind < nx; ind++) {
      if (intMask(ind) == 1) {
        n = normal(Δφ.row(ind));
        iVec inds(1);
        inds << ind;
        fill_boundary_inner(u, grid, inds, dX, nX, φ(ind), mat, MPs, dt, n);
      }
    }
    break;
  case 2:
    int ny = nX(1);
#pragma omp parallel for collapse(2) private(n)
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        int ind = i * ny + j;
        if (intMask(ind) == 1) {
          n = normal(Δφ.row(ind));
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
  grid.segment<V - LSET>(ind) = grid.segment<V - LSET>(newInd);
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
          if (intMask(ind) == 0 || intMask(ind) == -2) {

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
  int ndim = nX.size();

  MatMap uMap(u.data(), ncell, V, OuterStride(V));
  renormalize_levelsets(uMap, dX, nX);
  iVec intMask(ncell);
  Vec φ(ncell);
  Mat Δφ(ncell, ndim);

  for (int mat = 0; mat < nmat; mat++) {

    if (MPs[mat].EOS > -1) {

      grids[mat] = u;

      material_indicator(φ, uMap, mat, nmat, dX, nX);

      finite_difference(Δφ, φ, dX, nX);

      find_interface_cells(intMask, φ, nX);

      fill_boundary_cells(u, grids[mat], intMask, mat, φ, Δφ, dX, MPs, dt, nX);

      fill_neighbor_cells(grids[mat], intMask, Δφ, dX, nX);

      for (int j = 0; j < ncell; j++)
        masks[mat](j) = (φ(j) <= 0. && intMask(j) != -2) || intMask(j) == 1;

      MatMap gridMap(grids[mat].data(), ncell, V, OuterStride(V));
      gridMap.block(0, V - LSET, ncell, LSET) =
          uMap.block(0, V - LSET, ncell, LSET);
    } else
      masks[mat].setZero(ncell);
  }
}
