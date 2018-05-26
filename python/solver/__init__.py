from itertools import product
from time import time

from numpy import expand_dims, ones

from ader.solver import Solver
from ader.etc.boundaries import extend_mask
from ader.fv.fv import FVSolver

from solver.cpp import solve_full_cpp, cpp_split_stepper, cpp_ader_stepper
from solver.split import SplitSolver


flux_types = {'rusanov': 0,
              'roe': 1,
              'osher': 2}


class SolverPlus(Solver):

    def __init__(self, nvar, ndim, F, B=None, S=None, model_params=None,
                 M=None, max_eig=None, order=2, ncore=4,
                 riemann_solver='rusanov', stiff_dg=False,
                 stiff_dg_guess=False, newton_dg_guess=False, split=False,
                 half_step=True, ode_solver=None):

        Solver.__init__(self, nvar, ndim, F, B=B, S=S,
                        model_params=model_params, M=M, max_eig=max_eig,
                        order=order, ncore=ncore,
                        riemann_solver=riemann_solver, stiff_dg=stiff_dg,
                        stiff_dg_guess=stiff_dg_guess,
                        newton_dg_guess=newton_dg_guess)

        self.split = split
        self.half_step = half_step
        self.ode_solver = ode_solver
        self.stiff_dg = stiff_dg
        self.flux_type = flux_types[riemann_solver]

        if split:
            self.fvSolver = FVSolver(self.N, self.NV, self.NDIM, F=self.F,
                                     B=self.B, M=self.M, max_eig=self.max_eig,
                                     pars=self.pars,
                                     riemann_solver=riemann_solver,
                                     time_rec=False)

            self.splitSolver = SplitSolver(order, nvar, ndim, F, B=B, S=S,
                                           ode_solver=ode_solver,
                                           model_params=model_params)
        else:
            self.fvSolver = FVSolver(self.N, self.NV, self.NDIM, F=self.F,
                                     S=self.S, B=self.B, M=self.M,
                                     max_eig=self.max_eig, pars=self.pars,
                                     riemann_solver=riemann_solver)

    def timestep(self, mask=None):
        """ Calculates dt, based on the maximum wavespeed across the domain
        """
        MAX = 0
        for coords in product(*[range(s) for s in self.u.shape[:self.NDIM]]):

            if mask is None or mask[coords]:
                Q = self.u[coords]
                for d in range(self.NDIM):
                    MAX = max(MAX, self.max_eig(Q, d, self.pars) / self.dX[d])

        dt = self.cfl / MAX

        # Reduce early time steps to avoid initialization errors
        if self.count <= 5:
            dt /= 5

        return min(self.tf - self.t, dt)


    def split_stepper(self, dt, maskBC):

        t0 = time()

        self.splitSolver.ode_launcher(self.u, dt / 2)
        t1 = time()

        uBC = self.BC(self.u, self.N, self.NDIM)
        wh = self.wenoSolver.solve(uBC)
        if self.half_step:
            self.splitSolver.weno_midstepper(wh, dt, self.dX, maskBC)
        t2 = time()

        self.u += self.fvSolver.solve(wh, dt, self.dX, maskBC)
        t3 = time()

        self.splitSolver.ode_launcher(self.u, dt / 2)
        t4 = time()

        if self.verbose:
            print('ODE1:', t1 - t0)
            print('WENO:', t2 - t1)
            print('FV:  ', t3 - t2)
            print('ODE2:', t4 - t3, '\n')

    def cpp_stepper(self, uBC, dt, maskBC):
        if self.split:
            cpp_split_stepper(self, self.u, uBC, dt, self.dX, maskBC)
        else:
            cpp_ader_stepper(self, self.u, uBC, dt, self.dX, maskBC)

    def stepper(self, executor, dt, mask=None):

        uBC = self.BC(self.u, self.N, self.NDIM)

        if mask is None:
            maskBC = ones(uBC.shape[:-1], dtype=bool)
        else:
            maskBC = extend_mask(mask)

        if self.cpp_level > 0:
            self.cpp_stepper(uBC, dt, maskBC)

        elif self.split:
            self.split_stepper(dt, maskBC)

        else:
            if self.ncore == 1:
                du = self.ader_stepper(uBC, dt, self.verbose, maskBC)
            else:
                du = self.parallel_ader_stepper(executor, uBC, dt, maskBC)

            if mask is None:
                self.u += du
            else:
                self.u += du * expand_dims(mask, -1)

    def solve(self, u0, tf, dX, cfl=0.9,
              bcs='transitive', verbose=False, callback=None,
              cpp_level=0):

        self.cpp_level = cpp_level

        if cpp_level == 2:
            self.u = solve_full_cpp(self, u0, tf, dX, cfl)
            return self.u

        else:
            return Solver.solve(self, u0, tf, dX, cfl=cfl, bcs=bcs,
                                verbose=verbose, callback=callback)
