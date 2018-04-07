from time import time

from ader.solver import Solver
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
                 strang=True, half_step=True, ode_solver=None):

        Solver.__init__(self, nvar, ndim, F, B=B, S=S,
                        model_params=model_params, M=M, max_eig=max_eig,
                        order=order, ncore=ncore,
                        riemann_solver=riemann_solver, stiff_dg=stiff_dg,
                        stiff_dg_guess=stiff_dg_guess,
                        newton_dg_guess=newton_dg_guess)

        self.split = split
        self.strang = strang
        self.half_step = half_step
        self.ode_solver = ode_solver
        self.stiff_dg = stiff_dg
        self.flux_type = flux_types[riemann_solver]

        self.fvSolver = FVSolver(self.N, self.NV, self.NDIM, F=self.F,
                                 S=self.S, B=self.B, M=self.M,
                                 max_eig=self.max_eig, pars=self.pars,
                                 riemann_solver=riemann_solver,
                                 time_rec=not split)

        if split:
            self.splitSolver = SplitSolver(order, nvar, ndim, F, B=B, S=S,
                                           ode_solver=ode_solver,
                                           model_params=model_params)

    def split_stepper(self, uBC, dt, dX, verbose=False):

        t0 = time()

        self.splitSolver.ode_launcher(uBC, dt / 2)
        t1 = time()

        wh = self.wenoSolver.solve(uBC)
        if self.half_step:
            self.splitSolver.weno_midstepper(wh, dt, dX)
        t2 = time()

        self.u += self.fvSolver.solve(wh, dt, dX)
        t3 = time()

        self.splitSolver.ode_launcher(self.u, dt / 2)
        t4 = time()

        if verbose:
            print('ODE1:', t1 - t0)
            print('WENO:', t2 - t1)
            print('FV:  ', t3 - t2)
            print('ODE2:', t4 - t3, '\n')

    def cpp_stepper(self, uBC, dt, dX):
        if self.split:
            cpp_split_stepper(self, self.u, uBC, dt, dX)
        else:
            cpp_ader_stepper(self, self.u, uBC, dt, dX)

    def resume(self, verbose=False):

        if self.split or self.cpp_level > 0:

            while self.t < self.final_time:

                dt = self.timestep(self.u, self.dX, count=self.count, t=self.t,
                                   final_time=self.final_time)

                if verbose:
                    print('Iteration:', self.count)
                    print('t  = {:.3e}'.format(self.t))
                    print('dt = {:.3e}'.format(dt))

                uBC = self.BC(self.u, self.N, self.NDIM)

                if self.cpp_level == 1:
                    self.cpp_stepper(uBC, dt, self.dX)
                else:
                    self.split_stepper(uBC, dt, self.dX, verbose)

                self.t += dt
                self.count += 1

                if self.callback is not None:
                    self.callback(self.u, self.t, self.count)

            return self.u

        else:
            return Solver.resume(self, verbose)

    def solve(self, initial_grid, final_time, dX, cfl=0.9,
              boundary_conditions='transitive', verbose=False, callback=None,
              cpp_level=0):

        self.cpp_level = cpp_level

        if cpp_level == 2:
            return solve_full_cpp(self, initial_grid, final_time, dX, cfl)

        else:
            return Solver.solve(self, initial_grid, final_time, dX, cfl=cfl,
                                boundary_conditions=boundary_conditions,
                                verbose=verbose, callback=callback)
