#include "../etc/globals.h"
#include "../etc/grid.h"
#include "dg/dg.h"
#include "fv/fv.h"
#include "omp.h"
#include "split/homogeneous.h"
#include "split/ode.h"
#include "weno/weno.h"

void ader_stepper(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX, bool STIFF,
                  int FLUX, Par &MP, bVecr mask) {

  int ndim = nX.size();
  Vec wh(extended_dimensions(nX, 1) * int(pow(N, ndim)) * V);
  Vec qh(extended_dimensions(nX, 1) * int(pow(N, ndim + 1)) * V);

  weno_launcher(wh, ub, nX, MP);

  predictor(qh, wh, dt, dX, STIFF, false, MP, mask);

  fv_launcher(u, qh, nX, dt, dX, true, true, FLUX, MP, mask);
}

void ader_stepper_para(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX,
                       bool STIFF, int FLUX, Par &MP, bVecr mask) {
  int nx = nX(0);
  int uRowSize = u.size() / nx;
  int ubRowSize = ub.size() / (nx + 2 * N);
  int maskRowSize = mask.size() / (nx + 2);

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int ithread = omp_get_thread_num();

    int start = ithread * nx / nthreads;
    int finish = (ithread + 1) * nx / nthreads;

    iVec nX0 = nX;
    nX0(0) = finish - start;

    ader_stepper(
        u.segment(start * uRowSize, (finish - start) * uRowSize),
        ub.segment(start * ubRowSize, (finish + 2 * N - start) * ubRowSize),
        nX0, dt, dX, STIFF, FLUX, MP,
        mask.segment(start * maskRowSize, (finish + 2 - start) * maskRowSize));
  }
}

void split_stepper(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX,
                   bool HALF_STEP, int FLUX, Par &MP, bVecr mask) {

  int ndim = nX.size();
  ode_launcher(u, dt / 2, MP);
  ode_launcher(ub, dt / 2, MP);

  Vec wh(extended_dimensions(nX, 1) * int(pow(N, ndim)) * V);
  weno_launcher(wh, ub, nX, MP);

  if (HALF_STEP)
    midstepper(wh, dt, dX, MP, mask);

  fv_launcher(u, wh, nX, dt, dX, false, false, FLUX, MP, mask);

  ode_launcher(u, dt / 2, MP);
}

void split_stepper_para(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX,
                        bool STIFF, int FLUX, Par &MP, bVecr mask) {
  int nx = nX(0);
  int uRowSize = u.size() / nx;
  int ubRowSize = ub.size() / (nx + 2 * N);
  int maskRowSize = mask.size() / (nx + 2);

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int ithread = omp_get_thread_num();

    int start = ithread * nx / nthreads;
    int finish = (ithread + 1) * nx / nthreads;

    iVec nX0 = nX;
    nX0(0) = finish - start;

    Vec ub_ =
        ub.segment(start * ubRowSize, (finish + 2 * N - start) * ubRowSize);

    split_stepper(
        u.segment(start * uRowSize, (finish - start) * uRowSize), ub_, nX0, dt,
        dX, STIFF, FLUX, MP,
        mask.segment(start * maskRowSize, (finish + 2 - start) * maskRowSize));
  }
}
