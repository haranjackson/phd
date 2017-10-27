#include "../etc/globals.h"
#include "../etc/grid.h"
#include "weno/weno.h"
#include "dg/dg.h"
#include "split/homogeneous.h"
#include "split/ode.h"
#include "fv/fv.h"


void ader_stepper(Vecr u, Vecr ub, Vecr wh, Vecr qh,
                  int ndim, int nx, int ny, int nz,
                  double dt, double dx, double dy, double dz,
                  bool PERIODIC, bool PERRON_FROBENIUS, Par & MP)
{
    boundaries(u, ub, ndim, nx, ny, nz, PERIODIC);

    weno_launcher(wh, ub, ndim, nx, ny, nz);

    predictor(qh, wh, ndim, dt, dx, dy, dz, false, false, MP);

    fv_launcher(u, qh, ndim, nx, ny, nz, dt, dx, dy, dz, true, true,
                PERRON_FROBENIUS, MP);
}

void split_stepper(Vecr u, Vecr ub, Vecr wh, int ndim, int nx, int ny, int nz,
                   double dt, double dx, double dy, double dz,
                   bool PERIODIC, bool STRANG, bool HALF_STEP,
                   bool PERRON_FROBENIUS, Par & MP)
{
    double Dt = STRANG ? dt/2 : dt;

    ode_launcher(u, Dt, MP);

    boundaries(u, ub, ndim, nx, ny, nz, PERIODIC);

    weno_launcher(wh, ub, ndim, nx, ny, nz);

    if (HALF_STEP)
        midstepper(wh, ndim, dt, dx, dy, dz, MP);

    fv_launcher(u, wh, ndim, nx, ny, nz, dt, dx, dy, dz, false, false,
                PERRON_FROBENIUS, MP);

    if (STRANG)
        ode_launcher(u, Dt, MP);
}
