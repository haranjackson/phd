#ifndef OPTIONS_H
#define OPTIONS_H

// System Options //
const bool VISCOUS = true;
const bool THERMAL = false;
const bool MULTI = true;
const int LSET = 0;

// Boundary Options //
const bool NO_CORNERS = true;
const bool DESTRESS = false;

// Solver Options //
const int N = 3;

// Riemann Solver Options //
const bool RIEMANN_STICK = false;
const bool RIEMANN_RELAXATION = true;
const double STAR_TOL = 1e-8;

// WENO Constants //
const double LAMS = 1.;   // WENO side stencil weighting
const double LAMC = 1e5;  // WENO central stencil weighting
const double EPS = 1e-14; // WENO epsilon parameter

// DG Constants //
const int DG_IT = 50;       // No. of iterations of non-Newton solver attempted
const double DG_TOL = 1e-8; // Convergence tolerance

const int V = 5 + int(VISCOUS) * 9 + int(THERMAL) * 3 + int(MULTI) + LSET;
const int mV = 5 + int(VISCOUS) * 9 + int(THERMAL) * 3;

#endif // OPTIONS_H
