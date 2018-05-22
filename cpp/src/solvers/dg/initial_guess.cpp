#include "../../etc/types.h"
#include "../../system/objects/gpr_objects.h"

void standard_initial_guess1(Matr q, Matr w) {
  // Returns a Galerkin intial guess consisting of the value of q at t=0
  for (int i = 0; i < N; i++)
    q.block<N, V>(i * N, 0) = w;
}
void standard_initial_guess2(Matr q, Matr w) {
  // Returns a Galerkin intial guess consisting of the value of q at t=0
  for (int i = 0; i < N; i++)
    q.block<N * N, V>(i * N * N, 0) = w;
}

void stiff_initial_guess(Matr q, Matr w, int NT, double dt, Par &MP) {
  // Returns the initial guess found in DOI: 10.1007/s10915-010-9426-6

  standard_initial_guess1(q, w);
  /*
  Mat qt = w;

  for (int t=0; t<N; t++)
  {
      double DT;
      if (t==0)
          Dt = dt * NODES(0);
      else
          DT = dt * (NODES(t) - NODES(t-1));

      Mat dqdxt = derivs.dot(qt);

      for (int i=0; i<N; i++)
      {
          M = system_conserved(qi, 0, PAR, SYS).dot(dqdxt[i]);
          S = source(qt[i], PAR, SYS);

          if (superStiff)
          {
              f = lambda X: X - qt[i] + dt/dx * M - dt/2 *
  (S+source(X,PAR,SYS));
              q[t,i] = newton_krylov(f, qi, f_tol=TOL);
          }
          else
          {
              q[t,i] = qi - dt/dx * M + dt * Sj;
          }
      }
      qt = q[t];
  }
  */
}
