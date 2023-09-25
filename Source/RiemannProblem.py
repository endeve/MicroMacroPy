import numpy as np
from math import exp, sqrt, pi
import BoxModule
import MeshModule
from LagrangeModule import LagrangeP
import InputOutputModule
import MomentsModule
import TallyModule
import VlasovModule
import LenardBernsteinModule
import ImexModule

polynomialDegree = 2

nX  = 128
bcX = 2 # --- ( 0 = No BC, 1 = Periodic, 2 = Homogeneous )
ngX = 1
loX = - 1.0
hiX = + 1.0
nDOFx = polynomialDegree + 1
nQx = nDOFx

nV  = 8
bcV = 0 # --- ( 0 = No BC, 1 = Periodic, 2 = Homogeneous )
ngV = 1
loV = - 6.0
hiV = + 6.0
nDOFv = polynomialDegree + 1
nQv = nDOFv

nDOFz = nDOFv * nDOFx

D_L = 1.0
U_L = 0.0
P_L = 1.0
T_L = P_L / D_L

D_R = 0.125
U_R = 0.0
P_R = 0.1
T_R = P_R / D_R

CollFreq = 1.0e3
CFL      = 0.75 / ( 2.0 * polynomialDegree + 1.0 )
t_end    = 1.0e-1

UsePositivityLimiter = False

ImexScheme = 'PDARS'

ImEx = ImexModule.IMEX()
ImEx.Initialize( ImexScheme, Verbose = True )

iCycleD = 1   # --- Display Info Every iCycleD Cycles
iCycleT = 1   # --- Tally Fields Every iCycleT Cycles
iCycleW = 10  # --- Write Output Every iCycleW Cycles

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

D = np.zeros( ( nDOFx, BX_X.w[0] )           , np.float64, 'F' ) # 0th Moment
U = np.zeros( ( nDOFx, BX_X.w[0] )           , np.float64, 'F' ) # 1st Moment
T = np.zeros( ( nDOFx, BX_X.w[0] )           , np.float64, 'F' ) # 2nd Moment
E = np.zeros( ( nDOFx, BX_X.w[0] )           , np.float64, 'F' ) # Electric Field
f = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' ) # Distribution Function

# --- Initialize IO ---

IO = InputOutputModule.InputOutput()
IO.Initialize( 'RiemannProblem', nDOFv, nDOFx )

# --- Set Initial Condition ---

pQx, wQx = np.polynomial.legendre.leggauss( nDOFx )
pQx = pQx / 2.0
wQx = wQx / 2.0
pQv, wQv = np.polynomial.legendre.leggauss( nDOFv )
pQv = pQv / 2.0
wQv = wQv / 2.0

# --- Quadrature to Project Initial Condition ---

nQQ = 4
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
                        
                        if( x_q <= 0.0 ):
                        
                            f[q_z,i_v,i_x] \
                            += ( wQQ[qq_v] / wQv[q_v] ) * ( wQQ[qq_x] / wQx[q_x] ) \
                            * LagrangeP( [pQQ[qq_v]], q_v, pQv, nDOFv ) \
                            * LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) \
                            * ( D_L / sqrt( 2.0 * pi * T_L ) ) \
                            * exp( - ( v_q - U_L )**2 / ( 2.0 * T_L ) )
                            
                        else:
                            
                            f[q_z,i_v,i_x] \
                            += ( wQQ[qq_v] / wQv[q_v] ) * ( wQQ[qq_x] / wQx[q_x] ) \
                            * LagrangeP( [pQQ[qq_v]], q_v, pQv, nDOFv ) \
                            * LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) \
                            * ( D_R / sqrt( 2.0 * pi * T_R ) ) \
                            * exp( - ( v_q - U_R )**2 / ( 2.0 * T_R ) )
                
                q_z += 1

# --- Initialize Tally ---

Tally = TallyModule.Tally()
Tally.Initialize( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, nDOFv, nDOFx, nQv, nQx )
Tally.Compute( 0.0, BX_Z, f )

# --- Initialize Velocity Moments ---

VM = MomentsModule.VelocityMoments()
VM.Initialize( BX_V, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx )
VM.ComputeVelocityMoments_DUT( BX_Z, Mesh_V.dx, f, D, U, T )

# --- Initialize Vlasov Solver ---

Vlasov = VlasovModule.VlasovSolver()
Vlasov.Initialize( BX_Z, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx, UsePositivityLimiter )

# --- Initialize Lenard-Bernstein Solver ---

LB = LenardBernsteinModule.LenardBernsteinSolver()
LB.Initialize( CollFreq, BX_Z, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx )

# --- Write Initial Condition ---

IO.WriteFields \
( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, 0.0, D = D, U = U, T = T, f = f )

# --- Evolve ---

B_EX = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1], ImEx.nStages ), np.float64, 'F' ) # Explicit BLF
B_IM = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1], ImEx.nStages ), np.float64, 'F' ) # Implicit BLF

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
        
        if( any( ImEx.a_IM[:,iS] != 0.0 ) or ( ImEx.w_IM[iS] != 0.0 ) ):
            
            # --- Implicit Increment ---
            
            LB.ComputeBLF \
            ( ImEx.a_IM[iS,iS] * dt, BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, U, T, fi, B_IM[:,:,:,iS], True )
            
            fi -= dt * ImEx.a_IM[iS,iS] * LB.CollisionFrequency * B_IM[:,:,:,iS]
            
            Vlasov.ApplyPositivityLimiter( BX_Z, fi )
        
        if( any( ImEx.a_EX[:,iS] != 0.0 ) or ( ImEx.w_EX[iS] != 0.0 ) ):
            
            # --- Explicit Increment ---
            
            Vlasov.ComputeBLF \
            ( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, E, fi, B_EX[:,:,:,iS], True )
            
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
    
    t += dt
    
    if( np.mod( iCycle, iCycleT ) == 0 ):
        
        Tally.Compute( t, BX_Z, f )
    
    if( np.mod( iCycle, iCycleW ) == 0 ):
        
        IO.WriteFields \
        ( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, t, \
          D = D, U = U, T = T, f = f )

Tally.Compute( t, BX_Z, f )
Tally.Write( 'RiemannProblem' )

IO.WriteFields \
( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, t, \
  D = D, U = U, T = T, f = f )
