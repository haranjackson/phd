#ifndef OPTIONS_H
#define OPTIONS_H

const int N = 3;  // Order of accuracy
const int V = 17; // Number of variables

// WENO Constants //
const double LAMS = 1.;   // WENO side stencil weighting
const double LAMC = 1e5;  // WENO central stencil weighting
const double EPS = 1e-14; // WENO epsilon parameter

// DG Constants //
const int DG_ITER = 50;     // No. of iterations of non-Newton solver attempted
const double DG_TOL = 6e-6; // Convergence tolerance

#endif // OPTIONS_H
