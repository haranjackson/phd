import numpy as np


""" Domain Parameters """

tf = 0.25                      # Final time of simulation
Lx = 1                      # Length of domain in x direction
Ly = 1                      # Length of domain in x direction
Lz = 1                      # Length of domain in x direction
nx = 200                    # Number of cells in x direction
ny = 1                      # Number of cells in y direction
nz = 1                      # Number of cells in z direction

""" Model Options """

mechanical = 1              # Include evolution of density and velocity
viscous    = 1              # Include viscosity
thermal    = 1              # Include thermal conduction
reactive   = 0              # Include reactivity

""" Problem Parameters """

W  = 1                      # Power input at wall in cook off
doubleTime = 1e-6           # Time taken for temperature at wall to double
reactionType = 'a'          # 'a' (Arrhenius) or 'd' (Discrete)
fullBurn = 0                # Run simulation until 50% of reactant has burned
burnProp = 0.5              # If fullBurn=1, simulation stops when proportion of cells with reactant
                            # remaining falls below burnProp

""" GFM Options """

GFM  = 0                    # Use original GFM
RGFM = 1                    # Use RGFM (requires GFM=1)
isoFix = 0                  # Use isobaric fix
SFix   = 0                  # Fix entropy in ghost cells
TFix   = 1                  # Fix temperature in ghost cells (requries entropyFix=0)
UPDATE_STEP = 5             # Number of timesteps used to update interface locations

""" Solver Options """

solver = 'SPLIT-WENO'       # 'ADER-WENO', SPLIT-WENO', 'SPLIT-DG'
convertTemp   = 1           # Use constant-pressure approximation in cookoff
altThermSolve = 1           # Use operator splitting for thermal subsystem

fullODE = 1                 # Use numerical ODE solver (Split-WENO)
wenoHalfStep = 1            # Step forwards WENO solver by dt/2 (Split-WENO)
StrangSplit  = 1            # Use Strang splitting (Split-WENO)

approxInterface = 0         # Calculate fluxes with average value of interface states
reconstructPrim = 0         # Perform WENO and DG in primitive variables
wenoAverage = 1             # Average x-then-y and y-then-x WENO reconstruction

N      = 2                  # Method is order N+1
CFL    = 0.4                # CFL number
method = 'rusanov'          # Intercell fluxes ('osher' or 'rusanov')
perronFrob = 0              # Use Perron-Frobenius approximation to max λ

""" DG Options """

hidalgo    = 0              # Use Hidalgo initial guess
stiff      = 0              # Use Newton-Krylov solver
superStiff = 0              # Use Newton-Krylov solver for Hidalgo initial guess
failLim    = 180            # Max number of non-stiff solves allowed to fail
TOL        = 6e-6           # Tolerance to which the predictor must converge
MAX_ITER   = 50             # Max number of non-stiff iterations attempted

""" WENO Parameters """

rc = 8                      # Exponent used in oscillation indicator
λc = 1e5                    # Coefficient of oscillation indicator of central stencils
λs = 1                      # Coefficient of oscillation indicator of side stencils
eps  = 1e-14                # Constant ensuring oscillation indicators don't blow up

""" Speed-Up Parameters """

paraDG = 0                  # Parallelise DG step
paraFV = 0                  # Parallelise FV step
ncore  = 4                  # Number of cores used if running in parallel

""" Debug Options """

NO_WARNING = 1      # Turn off all SciPy/NumPy warnings. Potentially dangerous.


""" Derived Values """

ndim = sum(np.array([nx, ny, nz]) > 1)
N1 = N+1
NT = N1**(ndim+1)
dx = Lx / nx
dy = Ly / ny
dz = Lz / nz
if solver in ['SPLIT-WENO', 'WENO']:
    timeDim = 0
else:
    timeDim = 1
if not timeDim:
    approxInterface = 0

class systems():
    def __init__(self, mechanical, viscous, thermal, reactive):
        self.mechanical = mechanical
        self.viscous = viscous
        self.thermal = thermal
        self.reactive = reactive
SYS = systems(mechanical, viscous, thermal, reactive)

""" Compatibility Settings """

if fullBurn:
    tf = 1e16
if GFM:
    reducedDomain = 0

from numpy import seterr
if NO_WARNING:
    seterr(all='ignore')
