#ifndef OPTIONS_H
#define OPTIONS_H

// System //
const bool VISCOUS = true;
const bool THERMAL = false;
const bool MULTI = false;
const int LSET = 2;

// Boundaries //
const bool NO_CORNERS = true;
const bool DESTRESS = false;

// Solver //
const int N = 2;
const bool ROTATE_DISTORTION = false;

// RGFM //
const bool RIEMANN_STICK = false;
const bool RIEMANN_RELAXATION = true;
const double STAR_TOL = 1e-8;

// WENO //
const double LAMS = 1.;              // WENO side stencil weighting
const double LAMC = 1e5;             // WENO central stencil weighting
const double EPS = 1e-14;            // WENO epsilon parameter
const bool PRIM_RECONSTRUCT = false; // Whether to reconstruct with prim vars

// DG //
const int DG_IT = 50;       // No. of iterations of non-Newton solver attempted
const double DG_TOL = 6e-6; // Convergence tolerance

const int V = 5 + int(VISCOUS) * 9 + int(THERMAL) * 3 + int(MULTI) + LSET;
const int mV = 5 + int(VISCOUS) * 9 + int(THERMAL) * 3;

#endif // OPTIONS_H
