import numpy as np
from math import pi
import BoxModule
import MeshModule
from LagrangeModule import LagrangeP
import InputOutputModule
import MomentsModule
import TallyModule
import LenardBernsteinModule

polynomialDegree = 2

nX  = 1
bcX = 1 # --- ( 0 = No BC, 1 = Periodic )
ngX = 0
loX = 0.0
hiX = 1.0
nDOFx = 1
nQx = nDOFx

nV  = 48
bcV = 0 # --- ( 0 = No BC, 1 = Periodic )
ngV = 1
loV = - 12.0
hiV = + 12.0
nDOFv = polynomialDegree + 1
nQv = nDOFv

nDOFz = nDOFv * nDOFx

Profile = 'TopHat' #( 'TopHat' 'Maxwell' or 'DoubleMaxwell' )

if  ( Profile == 'TopHat' ):
    D_1 = 1.0
    U_1 = 1.0
    T_1 = 4.0/3.0
elif( Profile == 'Maxwell' ):
    D_1 = 1.0
    U_1 = 1.0
    T_1 = 4.0/3.0
elif( Profile == 'DoubleMaxwell' ):
    D_1 = 1.0
    U_1 = - 1.5
    T_1 = 0.5
    D_2 = 1.0
    U_2 = 2.5
    T_2 = 0.5
else:
    print( 'Invalid Profile: ', Profile )
    raise SystemExit

CollFreq = 1.0e0

dt    = 1.0e-1
t_end = 10.0

iCycleD = 1  # --- Display Info Every iCycleD Cycles
iCycleT = 1  # --- Tally Fields Every iCycleT Cycles
iCycleW = 1  # --- Write Output Every iCycleW Cycles

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
f   = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' ) # Distribution Function
B   = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' ) # Bilinear Form

# --- Initialize IO ---

IO = InputOutputModule.InputOutput()
IO.Initialize( 'Relaxation', nDOFv, nDOFx )

# --- Set Initial Condition ---

W_0 = np.sqrt( 12.0 * T_1 )
C_0 = U_1
H_0 = D_1 / W_0

pQx, wQx = np.polynomial.legendre.leggauss( nDOFx )
pQx = pQx / 2.0
wQx = wQx / 2.0
pQv, wQv = np.polynomial.legendre.leggauss( nDOFv )
pQv = pQv / 2.0
wQv = wQv / 2.0
nQ8 = nDOFv
pQ8, wQ8 = np.polynomial.legendre.leggauss( nQ8 )
pQ8 = pQ8 / 2.0
wQ8 = wQ8 / 2.0

for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
    for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
        
        q_z = 0
        for q_x in range( nDOFx ):
            for q_v in range( nDOFv ):
                
                f[q_z,i_v,i_x] = 0.0
                for qq_x in range( nQ8 ):
                    for qq_v in range( nQ8 ):
                        
                        v_q = Mesh_V.x_c[i_v] + Mesh_V.dx[i_v] * pQ8[qq_v]
                        x_q = Mesh_X.x_c[i_x] + Mesh_X.dx[i_x] * pQ8[qq_x]
                        
                        if  ( Profile == 'TopHat' ):
                            
                            f[q_z,i_v,i_x] \
                                += ( wQ8[qq_v] / wQv[q_v] ) \
                                 * ( wQ8[qq_x] / wQx[q_x] ) \
                                 * LagrangeP( [pQ8[qq_v]], q_v, pQv, nDOFv ) \
                                 * LagrangeP( [pQ8[qq_x]], q_x, pQx, nDOFx ) \
                                 * H_0 * ( np.heaviside( (v_q-C_0)+W_0/2.0, 0.5 ) - np.heaviside( (v_q-C_0)-W_0/2.0, 0.5 ) )
                        
                        elif( Profile == 'Maxwell' ):
                            
                            f[q_z,i_v,i_x] \
                                += ( wQ8[qq_v] / wQv[q_v] ) \
                                 * ( wQ8[qq_x] / wQx[q_x] ) \
                                 * LagrangeP( [pQ8[qq_v]], q_v, pQv, nDOFv ) \
                                 * LagrangeP( [pQ8[qq_x]], q_x, pQx, nDOFx ) \
                                 * (  D_1 / np.sqrt( 2.0 * pi * T_1 ) ) * np.exp( - ( v_q - U_1 )**2 / ( 2.0 * T_1 ) )
                        
                        elif( Profile == 'DoubleMaxwell' ):
                            
                            f[q_z,i_v,i_x] \
                                += ( wQ8[qq_v] / wQv[q_v] ) \
                                 * ( wQ8[qq_x] / wQx[q_x] ) \
                                 * LagrangeP( [pQ8[qq_v]], q_v, pQv, nDOFv ) \
                                 * LagrangeP( [pQ8[qq_x]], q_x, pQx, nDOFx ) \
                                 * (  D_1 / np.sqrt( 2.0 * pi * T_1 ) * np.exp( - ( v_q - U_1 )**2 / ( 2.0 * T_1 ) ) \
                                    + D_2 / np.sqrt( 2.0 * pi * T_2 ) * np.exp( - ( v_q - U_2 )**2 / ( 2.0 * T_2 ) ) )
                
                q_z += 1

# --- Initialize Tally ---

Tally = TallyModule.Tally()
Tally.Initialize( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, nDOFv, nDOFx, nQv, nQx )
Tally.Compute( 0.0, BX_Z, f )

# --- Initialize Velocity Moments ---

VM = MomentsModule.VelocityMoments()
VM.Initialize( BX_V, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx )
VM.ComputeVelocityMoments_DUT( BX_Z, Mesh_V.dx, f, D, U, T )

# --- Initialize Lenard-Bernstein Solver ---

LB = LenardBernsteinModule.LenardBernsteinSolver()
LB.Initialize( CollFreq, BX_Z, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx )

# --- Write Initial Condition ---

IO.WriteFields \
( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, 0.0, D = D, U = U, T = T, f = f )

t = 0.0
iCycle = 0
while( t < t_end ):
    
    iCycle += 1
    
    if( np.mod( iCycle, iCycleD ) == 0 ):
        
        print( "  iCycle =%9.8d, t =%10.3E, dt =%10.3E" %(iCycle,t,dt) )

    # --- Compute Bilinear Form ---

    LB.ComputeBLF( dt, BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, U, T, f, B, True )

    # --- Update Distribution ---

    f -= dt * LB.CollisionFrequency * B
    
    VM.ComputeVelocityMoments_DUT( BX_Z, Mesh_V.dx, f, D, U, T )
    
    t += dt
    
    if( np.mod( iCycle, iCycleT ) == 0 ):
        
        Tally.Compute( t, BX_Z, f )
    
    if( np.mod( iCycle, iCycleW ) == 0 ):
        
        IO.WriteFields \
        ( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, t, D = D, U = U, T = T, f = f )

Tally.Write( 'Relaxation' )
