import numpy as np
import time  as timer
from math import cos, pi
import BoxModule
import MeshModule
import LagrangeModule as LM
import MacroModule
import VlasovModule
import LenardBernsteinModule
import PoissonModule
import MomentsModule
import InputOutputModule
import ImexModule

polynomialDegree = 2

nX  = 16
bcX = 1 # --- ( 0 = No BC, 1 = Periodic, 2 = Homogeneous )
ngX = 1
loX = - 2.0 * pi
hiX = + 2.0 * pi
nDOFx = polynomialDegree + 1
nQx = nDOFx

nV  = 16
bcV = 0 # --- ( 0 = No BC, 1 = Periodic, 2 = Homogeneous )
ngV = 1
loV = - 6.0
hiV = + 6.0
nDOFv = polynomialDegree + 1
nQv = nDOFv

nDOFz = nDOFv * nDOFx

D_0   = 1.0
U_0   = 0.0
T_0   = 1.0
alpha = 1.0e-4
k     = 0.5

CollFreq = 1.0e2
CFL      = 0.75 / ( 2.0 * polynomialDegree + 1.0 )
t_end    = 5.0e+1

ImexScheme   = 'PDARS'
doEuler      = False
KineticFlux  = False # --- If True, macro model uses kinetic flux
SlopeLimiter = False
CleanMicro   = True
EvolveField  = True
InfMaxwell   = False

iCycleD = 1  # --- Display Info Every iCycleD Cycles
iCycleW = 10 # --- Write Output Every iCycleW Cycles

BX_X = BoxModule.Box()
BX_X.CreateBox( [ nX ], [ ngX ], [ bcX ] )

BX_V = BoxModule.Box()
BX_V.CreateBox( [ nV ], [ ngV ], [ bcV ] )

BX_Z = BoxModule.Box()
BX_Z.CreateBox( [ nV, nX ], [ ngV, ngX, ], [ bcV, bcX ] )

Mesh_X = MeshModule.Mesh()
Mesh_X.CreateMesh( nX, ngX, loX, hiX )

Mesh_V = MeshModule.Mesh()
Mesh_V.CreateMesh( nV, ngV, loV, hiV )

# --- Macro Fields ---

rhoC = np.zeros( ( nDOFx, 3, BX_X.w[0] ), np.float64, 'F' ) # Conserved Moments
rhoP = np.zeros( ( nDOFx, 3, BX_X.w[0] ), np.float64, 'F' ) # Primitive Moments
Phi  = np.zeros( ( BX_X.nx[0] + 1 )     , np.float64, 'F' ) # Electrostatic Potential
E    = np.zeros( ( nDOFx   , BX_X.w[0] ), np.float64, 'F' ) # Electric Field
D_g  = np.zeros( ( nDOFx   , BX_X.w[0] ), np.float64, 'F' ) # 0th Moment of g
S_g  = np.zeros( ( nDOFx   , BX_X.w[0] ), np.float64, 'F' ) # 1st Moment of g
G_g  = np.zeros( ( nDOFx   , BX_X.w[0] ), np.float64, 'F' ) # 2nd Moment of g

# --- Micro Fields ---

f = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' ) # Distribution Function
g = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' ) # Deviation from Maxwellian
M = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' ) # Maxwellian

# --- Set Initial Condition ---

pQx, wQx = np.polynomial.legendre.leggauss( nDOFx )
pQx = pQx / 2.0
wQx = wQx / 2.0

# --- Quadrature to Project Initial Condition ---

nQQ = nDOFx
pQQ, wQQ = np.polynomial.legendre.leggauss( nQQ )
pQQ = pQQ / 2.0
wQQ = wQQ / 2.0

for i_x in range( BX_X.lo[0], BX_X.hi[0] + 1 ):
    
    for q_x in range( nDOFx ):
        
        for qq_x in range( nQQ ):
                        
            x_q = Mesh_X.x_c[i_x] + Mesh_X.dx[i_x] * pQQ[qq_x]
            
            rhoC[q_x,0,i_x] \
            += ( wQQ[qq_x] / wQx[q_x] ) * LM.LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) * D_0 * ( 1.0 + alpha * cos( k * x_q ) )
            
            rhoC[q_x,1,i_x] \
            += ( wQQ[qq_x] / wQx[q_x] ) * LM.LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) * D_0 * U_0 * ( 1.0 + alpha * cos( k * x_q ) )
            
            rhoC[q_x,2,i_x] \
            += ( wQQ[qq_x] / wQx[q_x] ) * LM.LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) * 0.5 * D_0 * ( T_0 + U_0**2 ) * ( 1.0 + alpha * cos( k * x_q ) )

# --- Initialize Macro Solver ---

Macro = MacroModule.MacroSolver()
Macro.Initialize \
    ( 3.0, BX_Z, BX_X, Mesh_V.dx, Mesh_V.x_c, nDOFv, nQv, nDOFx, nQx, \
      UseKineticFlux = KineticFlux, UseSlopeLimiter = SlopeLimiter, InfiniteMaxwellianDomain = InfMaxwell )

# --- Initialize Vlasov Solver ---

Vlasov = VlasovModule.VlasovSolver()
Vlasov.Initialize( BX_Z, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx, UseMicroCleaning = CleanMicro )

# --- Initialize Lenard-Bernstein Solver ---

LB = LenardBernsteinModule.LenardBernsteinSolver()
LB.Initialize( CollFreq, BX_Z, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx )

# --- Initialize Poisson Solver ---

Poisson = PoissonModule.PoissonSolver()
Poisson.Initialize( nDOFx, nQx )

# --- Compute Auxiliary Variables ---

Macro.ApplySlopeLimiter( BX_X, rhoC )

Macro.ComputePrimitiveFromConserved( BX_X, rhoC, rhoP )

Macro.ComputeMaxwellianProjectionFromPrimitive( BX_Z, rhoP, M )

f = M + g

if( EvolveField ):
    Poisson.Solve( BX_X, Mesh_X.dx, 1.0-rhoC[:,0,:], Phi, [ 0.0, 0.0 ] )
    Poisson.ComputeElectricField( BX_X, Mesh_X.dx, Phi, E )

# --- Initialize Velocity Moments ---

VM = MomentsModule.VelocityMoments()
VM.Initialize( BX_V, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx )
VM.ComputeVelocityMoments( BX_Z, Mesh_V.dx, g, D_g, S_g, G_g )

# --- Initialize IO ---

IO = InputOutputModule.InputOutput()
IO.Initialize( 'CollisionalLandauDampingMM', nDOFv, nDOFx )

# --- Write Initial Condition ---

IO.WriteFields \
( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, 0.0, \
  D = rhoP[:,Macro.iD,:], U = rhoP[:,Macro.iU,:], T = rhoP[:,Macro.iT,:], \
  S = rhoC[:,Macro.iS,:], G = rhoC[:,Macro.iG,:], E = E, f = f, M = M, g = g, \
  D_g = D_g, S_g = S_g, G_g = G_g )

# --- Initialize Time Stepper ---

ImEx = ImexModule.IMEX()
ImEx.Initialize( ImexScheme, Verbose = True )

# --- Evolve ---

B_Macro    = np.zeros( ( nDOFx, 3        , BX_X.w[0], ImEx.nStages ), np.float64, 'F' ) # --- Macro Bilinear Form
B_Micro_EX = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1], ImEx.nStages ), np.float64, 'F' ) # --- Micro Bilinear Form (Explicit)
B_Micro_IM = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1], ImEx.nStages ), np.float64, 'F' ) # --- Micro Bilinear Form (Implicit)
B_Micro_Mx = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1], ImEx.nStages ), np.float64, 'F' ) # --- Micro Bilinear Form (Explicit Maxwellian)

# --- Timers ---

timerTotal    = timer.time()
timerBLF_M    = 0.0
timerBLF_m    = 0.0
timerBLF_VP   = 0.0
timerBLF_LB   = 0.0
timerPoisson  = 0.0
timerMaxwell  = 0.0
timerCleaning = 0.0
timerIO       = 0.0

t = 0.0
iCycle = 0
while( t < t_end ):
    
    iCycle += 1
    
    # --- Compute Time Step ---
    
    dt = CFL * np.min( Mesh_X.dx[BX_Z.lo[1]:BX_Z.hi[1]+1] / np.max( [ np.abs( loV ), np.abs( hiV ) ] ) )
    
    if( t + dt > t_end ):
        dt = t_end - t
    
    if( np.mod( iCycle, iCycleD ) == 0 ):
        
        print( "  iCycle =%9.8d, t =%10.3E, dt =%10.3E" %(iCycle,t,dt) )
    
    # --- IMEX-RK Algorithm ---
    
    rhoC_0 = rhoC.copy()
    g_0    = g.copy()
    M_0    = M.copy()
    
    for iS in range( ImEx.nStages ):
        
        rhoC = rhoC_0.copy()
        g    = g_0.copy()
        
        for jS in range( iS ):
            
            # --- Macro ---
            
            if( ImEx.a_EX[iS,jS] != 0.0 ):
                
                rhoC -= dt * ImEx.a_EX[iS,jS] * B_Macro[:,:,:,jS]
            
            if( jS == iS - 1 ):
                
                Macro.ApplySlopeLimiter( BX_X, rhoC )
                
                Macro.ComputePrimitiveFromConserved( BX_X, rhoC, rhoP )
                
                tic = timer.time()
                Macro.ComputeMaxwellianProjectionFromPrimitive( BX_Z, rhoP, M )
                toc = timer.time()
                timerMaxwell += (toc-tic)
                
                if( EvolveField ):
                    
                    tic = timer.time()
                    Poisson.Solve( BX_X, Mesh_X.dx, 1.0-rhoC[:,0,:], Phi, [ 0.0, 0.0 ] )
                    Poisson.ComputeElectricField( BX_X, Mesh_X.dx, Phi, E )
                    toc = timer.time()
                    timerPoisson += (toc-tic)
            
            if( not doEuler ):
                
                # --- Micro ---
                
                if( ImEx.a_EX[iS,jS] != 0.0 ):
                    
                    g -= dt * ImEx.a_EX[iS,jS] * ( B_Micro_EX[:,:,:,jS] + B_Micro_Mx[:,:,:,jS] )
                
                if( ImEx.a_IM[iS,jS] != 0.0 ):
                    
                    g -= dt * ImEx.a_IM[iS,jS] * LB.CollisionFrequency * B_Micro_IM[:,:,:,jS]
                
                if( jS == iS - 1 ):
                    
                    g += ( M_0 - M )
                    
                    tic = timer.time()
                    Vlasov.CleanMicroDistribution( BX_Z, g )
                    toc = timer.time()
                    timerCleaning += (toc-tic)
                    
                    f = M + g
            
        if( any( ImEx.a_IM[:,iS] != 0.0 ) or ( ImEx.w_IM[iS] != 0.0 ) ):
            
            if( not doEuler and LB.CollisionFrequency > 0.0 ):
                
                # --- Implicit Increment (Micro) ---
                
                tic = timer.time()
                LB.ComputeBLF \
                ( ImEx.a_IM[iS,iS] * dt, BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, \
                  rhoP[:,Macro.iU,:], rhoP[:,Macro.iT,:], g, B_Micro_IM[:,:,:,iS], True )
                toc = timer.time()
                timerBLF_LB += (toc-tic)
                
                g -= dt * ImEx.a_IM[iS,iS] * LB.CollisionFrequency * B_Micro_IM[:,:,:,iS]
                
            f = M + g
        
        if( any( ImEx.a_EX[:,iS] != 0.0 ) or ( ImEx.w_EX[iS] != 0.0 ) ):
            
            # --- Explicit Increment (Macro) ---
            
            tic = timer.time()
            Macro.ComputeBLF \
            ( BX_Z, BX_X, Mesh_X.dx, rhoC, g, - E, B_Macro[:,:,:,iS], True )
            toc = timer.time()
            timerBLF_M += (toc-tic)
            
            if( not doEuler ):
                
                # --- Explicit Increment (Micro) ---
                
                tic = timer.time()
                Vlasov.ComputeBLF \
                ( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, \
                  - E, g, B_Micro_EX[:,:,:,iS], True )
                toc = timer.time()
                timerBLF_VP += (toc-tic)
                
                tic = timer.time()
                Macro.ComputeBLF_Micro \
                ( BX_Z, BX_X, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, \
                  rhoC, - E, B_Micro_Mx[:,:,:,iS], True )
                toc = timer.time()
                timerBLF_m += (toc-tic)
    
    iS = ImEx.nStages - 1
    
    if( any( ImEx.a_IM[iS,:] != ImEx.w_IM ) or any( ImEx.a_EX[iS,:] != ImEx.w_EX ) ):
        
        # --- Macro ---
        
        rhoC = rhoC_0.copy()
        
        for iS in range( ImEx.nStages ):
            
            if( ImEx.w_EX[iS] != 0.0 ):
                
                rhoC -= dt * ImEx.w_EX[iS] * B_Macro[:,:,:,iS]
        
        Macro.ApplySlopeLimiter( BX_X, rhoC )
        
        Macro.ComputePrimitiveFromConserved( BX_X, rhoC, rhoP )
        
        tic = timer.time()
        Macro.ComputeMaxwellianProjectionFromPrimitive( BX_Z, rhoP, M )
        toc = timer.time()
        timerMaxwell += (toc-tic)
        
        if( EvolveField ):
            
            tic = timer.time()
            Poisson.Solve( BX_X, Mesh_X.dx, 1.0-rhoC[:,0,:], Phi, [ 0.0, 0.0 ] )
            Poisson.ComputeElectricField( BX_X, Mesh_X.dx, Phi, E )
            toc = timer.time()
            timerPoisson += (toc-tic)
        
        if( not doEuler ):
            
            # --- Micro ---
            
            g = g_0.copy()
            
            g += ( M_0 - M )
            
            for iS in range( ImEx.nStages ):
                
                if( ImEx.w_IM[iS] != 0.0 ):
                    
                    g -= dt * ImEx.w_IM[iS] * LB.CollisionFrequency * B_Micro_IM[:,:,:,iS]
                
                if( ImEx.w_EX[iS] != 0.0 ):
                    
                    g -= dt * ImEx.w_EX[iS] * ( B_Micro_EX[:,:,:,iS] + B_Micro_Mx[:,:,:,iS] )
                
            tic = timer.time()
            Vlasov.CleanMicroDistribution( BX_Z, g )
            toc = timer.time()
            timerCleaning += (toc-tic)
        
        f = M + g
    
    # -------------------------
    
    t += dt
    
    if( np.mod( iCycle, iCycleW ) == 0 ):
        
        VM.ComputeVelocityMoments( BX_Z, Mesh_V.dx, g, D_g, S_g, G_g )
        
        tic = timer.time()
        IO.WriteFields \
        ( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, t, \
          D = rhoP[:,Macro.iD,:], U = rhoP[:,Macro.iU,:], T = rhoP[:,Macro.iT,:], \
          S = rhoC[:,Macro.iS,:], G = rhoC[:,Macro.iG,:], E = E, f = f, M = M, g = g, \
          D_g = D_g, S_g = S_g, G_g = G_g )
        toc = timer.time()
        timerIO += (toc-tic)

VM.ComputeVelocityMoments( BX_Z, Mesh_V.dx, g, D_g, S_g, G_g )

tic = timer.time()
IO.WriteFields \
( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, t, \
  D = rhoP[:,Macro.iD,:], U = rhoP[:,Macro.iU,:], T = rhoP[:,Macro.iT,:], \
  S = rhoC[:,Macro.iS,:], G = rhoC[:,Macro.iG,:], E = E, f = f, M = M, g = g, \
  D_g = D_g, S_g = S_g, G_g = G_g )
toc = timer.time()
timerIO += (toc-tic)

timerTotal = timer.time() - timerTotal
timerSum   = (timerBLF_M+timerBLF_m+timerBLF_VP+timerBLF_LB+timerPoisson+timerMaxwell+timerCleaning+timerIO)

print( "--- Program Timers -----------------------------------------------" )
print( "" )
print( "  timerTotal    = ", timerTotal )
print( "  timerBLF_M    = ", timerBLF_M   , timerBLF_M    / timerTotal )
print( "  timerBLF_m    = ", timerBLF_m   , timerBLF_m    / timerTotal )
print( "  timerBLF_VP   = ", timerBLF_VP  , timerBLF_VP   / timerTotal )
print( "  timerBLF_LB   = ", timerBLF_LB  , timerBLF_LB   / timerTotal )
print( "  timerPoisson  = ", timerPoisson , timerPoisson  / timerTotal )
print( "  timerMaxwell  = ", timerMaxwell , timerMaxwell  / timerTotal )
print( "  timerCleaning = ", timerCleaning, timerCleaning / timerTotal )
print( "  timerIO       = ", timerIO      , timerIO       / timerTotal )
print( "  timerSum      = ", timerSum     , timerSum      / timerTotal )
print( "" )
print( "------------------------------------------------------------------" )
