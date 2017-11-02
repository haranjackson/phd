### Some scripts to test the Python bindings for the C++ GPR implementation
### Make sure Git/GPR-cpp and Git/GPR/python are in PYTHONPATH

from itertools import product

from numpy import amax, arange, sign, zeros
from numpy.random import rand

import GPRpy

from system.gpr.misc.objects import CParameters, material_parameters
from system.gpr.misc.structures import Cvec, Cvec_to_Pvec
from system.gpr.variables.eos import total_energy
from system.gpr.variables.state import temperature
from system.system import flux_ref, source_ref, block_ref, Bdot

from solvers.weno.weno import coeffs, weno_launcher, extend
from solvers.dg.dg import predictor, rhs
from solvers.dg.matrices import system_matrices
from solvers.split.homogeneous import weno_midstepper
from solvers.split.ode import ode_stepper_analytical
from solvers.fv.fluxes import Smax, Bint
from solvers.fv.fv import fv_terms

from tests_1d.validation import viscous_shock_IC

from options import dx, N1, SPLIT, nx, ny, nz, ndim, nV


_, U, _, _, _ = system_matrices()


### ENSURE N,V are equal ###


γ = 1.4
cv = 2.5
pinf = 0
ρ0 = 1
p0 = 1
T0 = temperature(ρ0, p0, γ, pinf, cv)
cs = 2
μ = 1e-3
τ1 = 6 * μ / (ρ0 * cs**2)
α = 1.5
κ = 1e-4
τ2 = κ * ρ0 / (T0 * α**2)


def generate_vector():
    A = rand(3,3)
    A *= sign(det(A))
    ρ = det(A)
    p = rand()
    v = rand(3)
    J = rand(3)
    E = total_energy(ρ, p, v, A, J, 0, PAR)
    return Cvec(ρ, p, v, A, J, 0, PAR)

def generate_reconstruction(recDims, FACTOR=100):

    NX = nx+2
    NY = ny+2 if ny>1 else 1
    NZ = nz+2 if nz>1 else 1

    if recDims==1:
        rec = zeros([NX, NY, NZ, N1, nV])
        for i, j, k in product(range(NX), range(NY), range(NZ)):
            Q = generate_vector()
            for r1 in range(N1):
                Q += rand(nV) / FACTOR
                rec[i,j,k,r1] = Q

    elif recDims==2:
        rec = zeros([NX, NY, NZ, N1, N1, nV])
        for i, j, k in product(range(NX), range(NY), range(NZ)):
            Q = generate_vector()
            for r1, r2 in product(range(N1), range(N1)):
                Q += rand(nV) / FACTOR
                rec[i,j,k,r1,r2] = Q

    elif recDims==3:
        rec = zeros([NX, NY, NZ, N1, N1, N1, nV])
        for i, j, k in product(range(NX), range(NY), range(NZ)):
            Q = generate_vector()
            for r1 in product(range(N1), range(N1), range(N1)):
                Q[:nV] += rand(nV) / FACTOR
                rec[i,j,k,r1,r2,r3] = Q

    return rec, rec.ravel()

d = 0
dt = 0.0001

PAR = material_parameters(γ=γ, pINF=pinf, cv=cv, ρ0=ρ0, p0=p0, cs=cs, α=α, μ=μ, κ=κ)
MP = CParameters(GPRpy, PAR)
Q = generate_vector()


def diff(x1, x2):
    return amax(abs(x1-x2))

### EQUATIONS ###


F_cp = zeros(nV)
F_py = zeros(nV)
GPRpy.system.flux(F_cp, Q, d, MP)
flux_ref(F_py, Q, d, PAR)

print("F    diff =", diff(F_cp, F_py))

S_cp = zeros(nV)
S_py = zeros(nV)
GPRpy.system.source(S_cp, Q, MP)
source_ref(S_py, Q, PAR)

print("S    diff =", diff(S_cp, S_py))

B_cp = zeros([nV,nV])
B_py = zeros([nV,nV])
GPRpy.system.block(B_cp, Q, d)
block_ref(B_py, Q, d)

print("B    diff =", diff(B_cp, B_py))

Bx_cp = zeros(nV)
Bx_py = zeros(nV)
x = rand(nV)
GPRpy.system.Bdot(Bx_cp, Q, x, d)
Bdot(Bx_py, x, Q, d)

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
rhs_py = rhs(Q_py, Ww_py, dt, PAR, 0)
rhs_cp = GPRpy.solvers.dg.rhs1(Q_cp, Ww_cp, dt, dx, MP)

print("RHS  diff =", diff(rhs_cp, rhs_py))

obj_cp = GPRpy.solvers.dg.obj1(Q_cp.ravel(), Ww_cp, dt, dx, MP).reshape([N1*N1, nV])
obj_py = rhs_py - dot(U, Q_py)

print("obj  diff =", diff(obj_cp, obj_py))

qh_cp = zeros(len(rec_cp)*N1)
STIFF = False
HIDALGO = False
GPRpy.solvers.dg.predictor(qh_cp, rec_cp, ndim, dt, dx, dx, dx, STIFF, HIDALGO, MP)

qh_py = predictor(rec_py, dt, PAR).ravel()

print("DG   diff =", diff(qh_cp, qh_py))


### FLUXES ###


Q1 = generate_vector()
Q2 = generate_vector()

Smax_cp = GPRpy.solvers.fv.Smax(Q1, Q2, d, False, MP)
Smax_py = -Smax(Q1, Q2, d, PAR)

print("Smax diff =", diff(Smax_cp, Smax_py))

Bint_cp = GPRpy.solvers.fv.Bint(Q1, Q2, d)
Bint_py = Bint(Q1, Q2, d)

print("Bint diff =", diff(Bint_cp, Bint_py))


### FINITE VOLUME (ndim=1) ###

HOMOGENEOUS = SPLIT
TIME = not SPLIT
SOURCES = not SPLIT

rec_py, rec_cp = generate_reconstruction(ndim+TIME)

FV_py = fv_terms(rec_py, dt, PAR, HOMOGENEOUS)
FV_cp = zeros([nx*nV])
GPRpy.solvers.fv.fv_launcher(FV_cp, rec_cp, 1, nx, 1, 1, dt, dx, 1, 1,
                             SOURCES, TIME, False, MP)

FV_cp = FV_cp.reshape([nx,nV])
FV_py = FV_py.reshape([nx,nV])

# NOTE: Differences occur in cells where the reconstuction given is non-physical
# This arises because the max_abs_eigs functions are slightly different
print("FV   diff =", diff(FV_cp, FV_py))


### SPLIT (ndim=1) ###


if SPLIT:
    mid_py = rec_py.reshape([nx+2,1,1,N1,nV])
    mid_cp = mid_py.ravel()
else:
    mid_py = rec_py.reshape([nx+2,1,1,N1,N1,nV])[:,:,:,:,0]
    mid_cp = mid_py.ravel()

weno_midstepper(mid_py, dt, PAR)
GPRpy.solvers.split.midstepper(mid_cp, 1, dt, dx, dx, dx, MP)
mid_cp = mid_cp.reshape([nx+2,N1,nV])
mid_py = mid_py.reshape([nx+2,N1,nV])

print("Step diff =", diff(mid_cp, mid_py))

ode_py = Q1.copy()
ode_cp = Q1.copy()
u = zeros([1,1,1,nV])
u[0] = ode_py
GPRpy.solvers.split.ode_launcher(ode_cp, dt, MP)
ode_stepper_analytical(u, dt, PAR)
ode_py = u[0,0,0]

print("ODEs diff =", diff(ode_cp, ode_py))
