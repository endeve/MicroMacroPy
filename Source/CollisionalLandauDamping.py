import numpy as np
import time  as timer
from math import exp, sqrt, cos, pi
import BoxModule
import MeshModule
from LagrangeModule import LagrangeP
import MomentsModule
import PoissonModule
import InputOutputModule
import VlasovModule
import LenardBernsteinModule
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

EvolveField = True
UseGLF_Flux = False

ImexScheme = 'PDARS'
iCycleD    = 1  # --- Display Info Every iCycleD Cycles
iCycleW    = 10 # --- Write Output Every iCycleW Cycles

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

D   = np.zeros( ( nDOFx, BX_X.w[0] )           , np.float64, 'F' ) # 0th Moment
U   = np.zeros( ( nDOFx, BX_X.w[0] )           , np.float64, 'F' ) # 1st Moment
T   = np.zeros( ( nDOFx, BX_X.w[0] )           , np.float64, 'F' ) # 2nd Moment
Phi = np.zeros( ( BX_X.nx[0] + 1   )           , np.float64, 'F' ) # Electrostatic Potential
E   = np.zeros( ( nDOFx, BX_X.w[0] )           , np.float64, 'F' ) # Electric Field
f   = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' ) # Distribution Function

# --- Set Initial Condition ---

pQx, wQx = np.polynomial.legendre.leggauss( nDOFx )
pQx = pQx / 2.0
wQx = wQx / 2.0
pQv, wQv = np.polynomial.legendre.leggauss( nDOFv )
pQv = pQv / 2.0
wQv = wQv / 2.0

# --- Quadrature to Project Initial Condition ---

nQQ = nDOFx
pQQ, wQQ = np.polynomial.legendre.leggauss( nQQ )
pQQ = pQQ / 2.0
wQQ = wQQ / 2.0

for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
    for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
        
        q_z = 0
        for q_x in range( nDOFx ):
            for q_v in range( nDOFv ):
                
                f[q_z,i_v,i_x] = 0.0
                for qq_x in range( nQQ ):
                    for qq_v in range( nQQ ):
                        
                        v_q = Mesh_V.x_c[i_v] + Mesh_V.dx[i_v] * pQQ[qq_v]
                        x_q = Mesh_X.x_c[i_x] + Mesh_X.dx[i_x] * pQQ[qq_x]
                        
                        f[q_z,i_v,i_x] \
                        += ( wQQ[qq_v] / wQv[q_v] ) * ( wQQ[qq_x] / wQx[q_x] ) \
                        * LagrangeP( [pQQ[qq_v]], q_v, pQv, nDOFv ) \
                        * LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) \
                        * ( D_0 / sqrt( 2.0 * pi * T_0 ) ) \
                        * exp( - ( v_q - U_0 )**2 / ( 2.0 * T_0 ) ) \
                        * ( 1.0 + alpha * cos( k * x_q ) )
                
                q_z += 1

# --- Initialize Velocity Moments ---

VM = MomentsModule.VelocityMoments()
VM.Initialize( BX_V, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx )
VM.ComputeVelocityMoments_DUT( BX_Z, Mesh_V.dx, f, D, U, T )

# --- Initialize Poisson Solver ---

Poisson = PoissonModule.PoissonSolver()
Poisson.Initialize( nDOFx, nQx )
if( EvolveField ):
    Poisson.Solve( BX_X, Mesh_X.dx, 1.0-D, Phi, [ 0.0, 0.0 ] )
    Poisson.ComputeElectricField( BX_X, Mesh_X.dx, Phi, E )

# --- Initialize IO ---

IO = InputOutputModule.InputOutput()
IO.Initialize( 'CollisionalLandauDamping', nDOFv, nDOFx )

# --- Write Initial Condition ---

IO.WriteFields \
( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, 0.0, D = D, U = U, T = T, E = E, f = f )

# --- Initialize Vlasov Solver ---

Vlasov = VlasovModule.VlasovSolver()
Vlasov.Initialize( BX_Z, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx, UseGLF_Flux=UseGLF_Flux )

# --- Initialize Lenard-Bernstein Solver ---

LB = LenardBernsteinModule.LenardBernsteinSolver()
LB.Initialize( CollFreq, BX_Z, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx )

# --- Initialize Time Stepper ---

ImEx = ImexModule.IMEX()
ImEx.Initialize( ImexScheme, Verbose = True )

# --- Evolve ---

B_EX = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1], ImEx.nStages ), np.float64, 'F' ) # Explicit BLF
B_IM = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1], ImEx.nStages ), np.float64, 'F' ) # Implicit BLF

# --- Timers ---

timerTotal   = timer.time()
timerBLF_VP  = 0.0
timerBLF_LB  = 0.0
timerPoisson = 0.0
timerIO      = 0.0

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
    
    f0 = f.copy()
    
    for iS in range( ImEx.nStages ):
        
        fi = f0.copy()
        
        for jS in range( iS ):
            
            if( ImEx.a_IM[iS,jS] != 0.0 ):
                
                fi -= dt * ImEx.a_IM[iS,jS] * LB.CollisionFrequency * B_IM[:,:,:,jS]
            
            if( ImEx.a_EX[iS,jS] != 0.0 ):
                
                fi -= dt * ImEx.a_EX[iS,jS] * B_EX[:,:,:,jS]
            
            if( jS == iS - 1 ):
                
                Vlasov.ApplyPositivityLimiter( BX_Z, fi )
                
                VM.ComputeVelocityMoments_DUT( BX_Z, Mesh_V.dx, fi, D, U, T )
                
                if( EvolveField ):
                    
                    tic = timer.time()
                    Poisson.Solve( BX_X, Mesh_X.dx, 1.0-D, Phi, [ 0.0, 0.0 ] )                    
                    Poisson.ComputeElectricField( BX_X, Mesh_X.dx, Phi, E )
                    toc = timer.time()
                    timerPoisson += (toc-tic)
        
        if( any( ImEx.a_IM[:,iS] != 0.0 ) or ( ImEx.w_IM[iS] != 0.0 ) ):
            
            # --- Implicit Increment ---
            
            if( LB.CollisionFrequency > 0.0 ):
                
                tic = timer.time()
                LB.ComputeBLF \
                ( ImEx.a_IM[iS,iS] * dt, BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, \
                  U, T, fi, B_IM[:,:,:,iS], True )
                toc = timer.time()
                timerBLF_LB += (toc-tic)
                
                fi -= dt * ImEx.a_IM[iS,iS] * LB.CollisionFrequency * B_IM[:,:,:,iS]
                
                Vlasov.ApplyPositivityLimiter( BX_Z, fi )
        
        if( any( ImEx.a_EX[:,iS] != 0.0 ) or ( ImEx.w_EX[iS] != 0.0 ) ):
            
            # --- Explicit Increment ---
            
            tic = timer.time()
            Vlasov.ComputeBLF \
            ( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, - E, fi, B_EX[:,:,:,iS], True )
            toc = timer.time()
            timerBLF_VP += (toc-tic)
    
    iS = ImEx.nStages - 1
    
    if( any( ImEx.a_IM[iS,:] != ImEx.w_IM ) or any( ImEx.a_EX[iS,:] != ImEx.w_EX ) ):
        
        fi = f0.copy()
        
        for iS in range( ImEx.nStages ):
            
            if( ImEx.w_IM[iS] != 0.0 ):
                
                fi -= dt * ImEx.w_IM[iS] * LB.CollisionFrequency * B_IM[:,:,:,iS]
            
            if( ImEx.w_EX[iS] != 0.0 ):
                
                fi -= dt * ImEx.w_EX[iS] * B_EX[:,:,:,iS]
        
        Vlasov.ApplyPositivityLimiter( BX_Z, fi )
    
    f = fi.copy()
    
    VM.ComputeVelocityMoments_DUT( BX_Z, Mesh_V.dx, f, D, U, T )
    
    if( EvolveField ):
        
        tic = timer.time()
        Poisson.Solve( BX_X, Mesh_X.dx, 1.0-D, Phi, [ 0.0, 0.0 ] )
        Poisson.ComputeElectricField( BX_X, Mesh_X.dx, Phi, E )
        toc = timer.time()
        timerPoisson += (toc-tic)
    
    t += dt
    
    if( np.mod( iCycle, iCycleW ) == 0 ):
        
        tic = timer.time()
        IO.WriteFields \
        ( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, t, \
          D = D, U = U, T = T, E = E, f = f )
        toc = timer.time()
        timerIO += (toc-tic)

tic = timer.time()
IO.WriteFields \
( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, t, \
  D = D, U = U, T = T, E = E, f = f )
toc = timer.time()
timerIO += (toc-tic)

timerTotal = timer.time() - timerTotal
timerSum   = (timerBLF_VP+timerBLF_LB+timerPoisson+timerIO)

print( "--- Program Timers -----------------------------------------------" )
print( "" )
print( "  timerTotal    = ", timerTotal )
print( "  timerBLF_VP   = ", timerBLF_VP  , timerBLF_VP   / timerTotal )
print( "  timerBLF_LB   = ", timerBLF_LB  , timerBLF_LB   / timerTotal )
print( "  timerPoisson  = ", timerPoisson , timerPoisson  / timerTotal )
print( "  timerIO       = ", timerIO      , timerIO       / timerTotal )
print( "  timerSum      = ", timerSum     , timerSum      / timerTotal )
print( "" )
print( "------------------------------------------------------------------" )
