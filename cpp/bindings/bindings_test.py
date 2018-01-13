### Some scripts to test the Python bindings for the C++ GPR implementation
### Make sure Git/GPR-cpp and Git/GPR/python are in PYTHONPATH

from itertools import product

from numpy import amax, arange, sign, zeros
from numpy.random import rand

import GPRpy

from gpr.misc.objects import material_parameters
from gpr.misc.structures import Cvec, Cvec_to_Pvec
from gpr.variables.eos import total_energy
from system import flux_ref, source_ref, block_ref, Bdot

from solvers.weno.weno import coeffs, weno_launcher, extend
from solvers.dg.dg import predictor, rhs
from solvers.dg.matrices import DG_U
from solvers.split.homogeneous import weno_midstepper
from solvers.split.ode import ode_stepper_analytical
from solvers.fv.fluxes import Smax, Bint
from solvers.fv.fv import fv_terms, centers, interfaces, extend_dimensions, endpoints

from tests_1d.fluids import viscous_shock_IC

from options import dx, N1, SPLIT, nx, ny, nz, ndim, nV

### ENSURE N,V are equal ###


γ = 1.4
cv = 2.5
pINF = 0
ρ0 = 1
p0 = 1
cs = 2
μ = 1e-3
cα = 1.5
κ = 1e-4

d = 0
dt = 0.0001

MP = material_parameters(EOS='sg', ρ0=ρ0, cv=cv, p0=p0,
                          γ=γ, pINF=pINF, b0=cs, cα=cα, μ=μ, κ=κ)

def generate_vector():
    A = rand(3,3)
    A *= sign(det(A))
    ρ = det(A)
    p = rand()
    v = rand(3)
    J = rand(3)
    E = total_energy(ρ, p, v, A, J, 0, MP)
    return Cvec(ρ, p, v, A, J, MP)

def diff(x1, x2):
    return amax(abs(x1-x2))

Q = generate_vector()


### EQUATIONS ###


F_cp = zeros(nV)
F_py = zeros(nV)
GPRpy.system.flux(F_cp, Q, d, MP)
flux_ref(F_py, Q, d, MP)

print("F    diff =", diff(F_cp, F_py))

S_cp = zeros(nV)
S_py = zeros(nV)
GPRpy.system.source(S_cp, Q, MP)
source_ref(S_py, Q, MP)

print("S    diff =", diff(S_cp, S_py))

B_cp = zeros([nV,nV])
B_py = zeros([nV,nV])
GPRpy.system.block(B_cp, Q, d)
block_ref(B_py, Q, d, MP)

print("B    diff =", diff(B_cp, B_py))

Bx_cp = zeros(nV)
Bx_py = zeros(nV)
x = rand(nV)
GPRpy.system.Bdot(Bx_cp, Q, x, d)
Bdot(Bx_py, x, Q, d, MP)

print("Bdot diff =", diff(Bx_cp, Bx_py))


### WENO ###


upy = rand(nx+2, ny+2*(ndim>1), nz+2*(ndim>2), nV)
ucp = upy.ravel()

if ndim==1:
    wh_py = weno_launcher(upy).ravel()
    wh_cp = zeros((nx+2)*(ny)*(nz)*N1*1*1*nV)
elif ndim==2:
    wh_py = weno_launcher(upy).ravel()
    wh_cp = zeros((nx+2)*(ny+2)*(nz)*N1*N1*1*nV)
else:
    wh_py = weno_launcher(upy).ravel()
    wh_cp = zeros((nx+2)*(ny+2)*(nz+2)*N1*N1*N1*nV)

GPRpy.solvers.weno.weno_launcher(wh_cp, ucp, ndim, nx, ny, nz)
print("WENO diff =", diff(wh_cp, wh_py))


### DISCONTINUOUS GALERKIN ###


upy, _, _ = viscous_shock_IC()
rec_py = weno_launcher(upy)
rec_cp = rec_py.ravel()

Q = rec_py[100,0,0]
Q_py = array([Q]*N1).reshape([N1*N1,nV])
Q_cp = Q_py[:,:nV]
Ww_py = rand(N1*N1, nV)
Ww_py[:,-1] = 0
Ww_cp = Ww_py[:,:nV]
rhs_py = rhs(Q_py, Ww_py, dt, MP, 0)
rhs_cp = GPRpy.solvers.dg.rhs1(Q_cp, Ww_cp, dt, dx, MP)

print("RHS  diff =", diff(rhs_cp, rhs_py))

obj_cp = GPRpy.solvers.dg.obj1(Q_cp.ravel(), Ww_cp, dt, dx, MP).reshape([N1*N1, nV])
obj_py = rhs_py - dot(DG_U, Q_py)

print("obj  diff =", diff(obj_cp, obj_py))

qh_cp = zeros(len(rec_cp)*N1)
STIFF = False
HIDALGO = False
GPRpy.solvers.dg.predictor(qh_cp, rec_cp, ndim, dt, dx, dx, dx, STIFF, HIDALGO, MP)

qh_py0 = predictor(rec_py, dt, MP)
qh_py = qh_py0.ravel()

print("DG   diff =", diff(qh_cp, qh_py))


### FLUXES ###


Q1 = generate_vector()
Q2 = generate_vector()

Smax_cp = GPRpy.solvers.fv.Smax(Q1, Q2, d, False, MP)
Smax_py = -Smax(Q1, Q2, d, MP)

print("Smax diff =", diff(Smax_cp, Smax_py))

Bint_cp = GPRpy.solvers.fv.Bint(Q1, Q2, d)
Bint_py = Bint(Q1, Q2, d, MP)

print("Bint diff =", diff(Bint_cp, Bint_py))


### FINITE VOLUME (ndim=1) ###


HOMOGENEOUS = SPLIT
TIME = not SPLIT
SOURCES = not SPLIT

qh0 = extend_dimensions(qh_py0)[0]
FVc_py = dt/dx * centers(qh0, nx-2, ny, nz, MP, HOMOGENEOUS)
FVc_cp = zeros([(nx-2)*nV])
GPRpy.solvers.fv.centers1(FVc_cp, qh_py, nx-2, dt, dx, SOURCES, TIME, MP)

FVc_cp = FVc_cp.reshape([(nx-2),nV])
FVc_py = FVc_py.reshape([(nx-2),nV])

print("FVc  diff =", diff(FVc_cp, FVc_py))

qEnd = endpoints(qh0)
FVi_py = -0.5 * dt/dx * interfaces(qEnd, MP)
FVi_cp = zeros([(nx-2)*nV])
GPRpy.solvers.fv.interfs1(FVi_cp, qh_py, nx-2, dt, dx, TIME, 0, MP)

FVi_cp = FVi_cp.reshape([(nx-2),nV])
FVi_py = FVi_py.reshape([(nx-2),nV])

print("FVi  diff =", diff(FVi_cp, FVi_py))

FV_py = fv_terms(qh_py0, dt, MP, HOMOGENEOUS)
FV_cp = zeros([(nx-2)*nV])
GPRpy.solvers.fv.fv_launcher(FV_cp, qh_py, 1, nx-2, 1, 1, dt, dx, 1, 1,
                             SOURCES, TIME, False, MP)

FV_cp = FV_cp.reshape([(nx-2),nV])
FV_py = FV_py.reshape([(nx-2),nV])

# NOTE: Differences occur in cells where the reconstuction given is non-physical
# This arises because the max_abs_eigs functions are slightly different
print("FV   diff =", diff(FV_cp, FV_py))


### SPLIT (ndim=1) ###


mid_py = rec_py.reshape([nx,1,1,N1,nV])
mid_cp = mid_py.ravel()

weno_midstepper(mid_py, dt, MP)
GPRpy.solvers.split.midstepper(mid_cp, 1, dt, dx, dx, dx, MP)
mid_cp = mid_cp.reshape([nx,N1,nV])
mid_py = mid_py.reshape([nx,N1,nV])

print("Step diff =", diff(mid_cp, mid_py))

ode_py = Q1.copy()
ode_cp = Q1.copy()
u = zeros([1,1,1,nV])
u[0] = ode_py
GPRpy.solvers.split.ode_launcher(ode_cp, dt, MP)
ode_stepper_analytical(u, dt, MP)
ode_py = u[0,0,0]

print("ODEs diff =", diff(ode_cp, ode_py))
