""" Domain Parameters """

tf = 0.2                      # Final time of simulation
Lx = 1                      # Length of domain in x direction
Ly = 1                      # Length of domain in x direction
Lz = 1                      # Length of domain in x direction
nx = 200                    # Number of cells in x direction
ny = 1                      # Number of cells in y direction
nz = 1                      # Number of cells in z direction

""" System Options """

VISCOUS    = 1              # Include viscosity
THERMAL    = 1              # Include thermal conduction
REACTIVE   = 0              # Include reactivity
REACTION_TYPE = 'a'         # 'a' (Arrhenius) or 'd' (Discrete)

""" GFM Options """

RGFM = 1                    # Use original GFM
isoFix = 0                  # Use isobaric fix

""" Solver Options """

SOLVER = 'ADER-WENO'       # 'ADER-WENO', SPLIT-WENO'

fullODE = 0                 # Use numerical ODE solver (Split-WENO)
wenoHalfStep = 1            # Step forwards WENO solver by dt/2 (Split-WENO)
StrangSplit  = 1            # Use Strang splitting (Split-WENO)

approxInterface = 0         # Calculate fluxes with average value of interface states
reconstructPrim = 0         # Perform WENO and DG in primitive variables
wenoAverage = 0             # Average x-then-y and y-then-x WENO reconstruction

N      = 1                  # Method is order N+1
CFL    = 0.9                # CFL number
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


""" Derived Values """

ndim = (nx>1) + (ny>1) + (nz>1)
N1 = N+1
NT = N1**(ndim+1)
dx = Lx / nx
dy = Ly / ny
dz = Lz / nz

if SOLVER == 'SPLIT-WENO':
    timeDim = 0
else:
    timeDim = 1

if not timeDim:
    approxInterface = 0
