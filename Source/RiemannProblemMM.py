import numpy as np
import BoxModule
import MeshModule
import LagrangeModule as LM
import TallyModule
import MacroModule
import VlasovModule
import LenardBernsteinModule
import InputOutputModule
import MomentsModule
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
S_L = D_L * U_L
G_L = 0.5 * D_L * ( U_L**2 + T_L )

D_R = 0.125
U_R = 0.0
P_R = 0.1
T_R = P_R / D_R
S_R = D_R * U_R
G_R = 0.5 * D_R * ( U_R**2 + T_R )

CollFreq = 1.0e3
CFL      = 0.75 / ( 2.0 * polynomialDegree + 1.0 )
t_end    = 1.0e-1

ImexScheme   = 'PDARS'
doEuler      = False # --- If True, micro solve is omitted
KineticFlux  = False # --- If True, macro model uses kinetic flux
SlopeLimiter = False
CleanMicro   = True

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

# --- Macro Fields ---

rhoC = np.zeros( ( nDOFx, 3, BX_X.w[0] ), np.float64, 'F' ) # Conserved Moments
rhoP = np.zeros( ( nDOFx, 3, BX_X.w[0] ), np.float64, 'F' ) # Primitive Moments
E    = np.zeros( ( nDOFx   , BX_X.w[0] ), np.float64, 'F' ) # Electric Field
D_g  = np.zeros( ( nDOFx   , BX_X.w[0] ), np.float64, 'F' ) # 0th Moment of g
S_g  = np.zeros( ( nDOFx   , BX_X.w[0] ), np.float64, 'F' ) # 1st Moment of g
G_g  = np.zeros( ( nDOFx   , BX_X.w[0] ), np.float64, 'F' ) # 2nd Moment of g

# --- Micro Fields ---

f   = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' ) # Distribution Function
g   = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' ) # Deviation from Maxwellian
M   = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' ) # Maxwellian

# --- Set Initial Condition ---

pQx, wQx = np.polynomial.legendre.leggauss( nDOFx )
pQx = pQx / 2.0
wQx = wQx / 2.0

# --- Quadrature to Project Initial Condition ---

nQQ = 4
pQQ, wQQ = np.polynomial.legendre.leggauss( nQQ )
pQQ = pQQ / 2.0
wQQ = wQQ / 2.0

for i_x in range( BX_X.lo[0], BX_X.hi[0] + 1 ):
    
    for q_x in range( nDOFx ):
        
        for qq_x in range( nQQ ):
                        
            x_q = Mesh_X.x_c[i_x] + Mesh_X.dx[i_x] * pQQ[qq_x]
            
            if( x_q <= 0.0 ):
                
                rhoC[q_x,0,i_x] \
                += ( wQQ[qq_x] / wQx[q_x] ) * LM.LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) * D_L
                
                rhoC[q_x,1,i_x] \
                += ( wQQ[qq_x] / wQx[q_x] ) * LM.LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) * S_L
                
                rhoC[q_x,2,i_x] \
                += ( wQQ[qq_x] / wQx[q_x] ) * LM.LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) * G_L
            
            else:
                
                rhoC[q_x,0,i_x] \
                += ( wQQ[qq_x] / wQx[q_x] ) * LM.LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) * D_R
                
                rhoC[q_x,1,i_x] \
                += ( wQQ[qq_x] / wQx[q_x] ) * LM.LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) * S_R
                
                rhoC[q_x,2,i_x] \
                += ( wQQ[qq_x] / wQx[q_x] ) * LM.LagrangeP( [pQQ[qq_x]], q_x, pQx, nDOFx ) * G_R

# --- Initialize Macro Solver ---

Macro = MacroModule.MacroSolver()
Macro.Initialize( 3.0, BX_Z, BX_X, Mesh_V.dx, Mesh_V.x_c, nDOFv, nQv, nDOFx, nQx, UseKineticFlux = KineticFlux, UseSlopeLimiter = SlopeLimiter )

# --- Initialize Vlasov Solver ---

Vlasov = VlasovModule.VlasovSolver()
Vlasov.Initialize( BX_Z, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx, UseMicroCleaning = CleanMicro )

# --- Initialize Lenard-Bernstein Solver ---

LB = LenardBernsteinModule.LenardBernsteinSolver()
LB.Initialize( CollFreq, BX_Z, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx )

# --- Compute Auxiliary Variables ---

Macro.ApplySlopeLimiter( BX_X, rhoC )

Macro.ComputePrimitiveFromConserved( BX_X, rhoC, rhoP )

Macro.ComputeMaxwellianProjectionFromPrimitive( BX_Z, rhoP, M )

f = M + g

# --- Initialize Tally ---

Tally = TallyModule.Tally()
Tally.Initialize( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, nDOFv, nDOFx, nQv, nQx )
Tally.Compute( 0.0, BX_Z, f )

# --- Initialize Velocity Moments ---

VM = MomentsModule.VelocityMoments()
VM.Initialize( BX_V, Mesh_V.dx, Mesh_V.x_c, nDOFv, nDOFx, nQv, nQx )
VM.ComputeVelocityMoments( BX_Z, Mesh_V.dx, g, D_g, S_g, G_g )

# --- Initialize IO ---

IO = InputOutputModule.InputOutput()
IO.Initialize( 'RiemannProblemMM', nDOFv, nDOFx )

# --- Write Initial Condition ---

IO.WriteFields \
( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, 0.0, \
  D = rhoP[:,Macro.iD,:], U = rhoP[:,Macro.iU,:], T = rhoP[:,Macro.iT,:], \
  S = rhoC[:,Macro.iS,:], G = rhoC[:,Macro.iG,:], f = f, M = M, g = g, \
  D_g = D_g, S_g = S_g, G_g = G_g )

ImEx = ImexModule.IMEX()
ImEx.Initialize( ImexScheme, Verbose = True )

B_Macro    = np.zeros( ( nDOFx, 3        , BX_X.w[0], ImEx.nStages ), np.float64, 'F' ) # --- Macro Bilinear Form
B_Micro_EX = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1], ImEx.nStages ), np.float64, 'F' ) # --- Micro Bilinear Form (Explicit)
B_Micro_IM = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1], ImEx.nStages ), np.float64, 'F' ) # --- Micro Bilinear Form (Implicit)
B_Micro_Mx = np.zeros( ( nDOFz, BX_Z.w[0], BX_Z.w[1], ImEx.nStages ), np.float64, 'F' ) # --- Micro Bilinear Form (Explicit Maxwellian)

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
                
                Macro.ComputeMaxwellianProjectionFromPrimitive( BX_Z, rhoP, M )
            
            if( not doEuler ):

                # --- Micro ---
                
                if( ImEx.a_EX[iS,jS] != 0.0 ):
                    
                    g -= dt * ImEx.a_EX[iS,jS] * ( B_Micro_EX[:,:,:,jS] + B_Micro_Mx[:,:,:,jS] )
                
                if( ImEx.a_IM[iS,jS] != 0.0 ):
                    
                    g -= dt * ImEx.a_IM[iS,jS] * LB.CollisionFrequency * B_Micro_IM[:,:,:,jS]
                
                if( jS == iS - 1 ):
                    
                    g += ( M_0 - M )
                    
                    Vlasov.CleanMicroDistribution( BX_Z, g )
                    
                    f = M + g
            
        if( any( ImEx.a_IM[:,iS] != 0.0 ) or ( ImEx.w_IM[iS] != 0.0 ) ):
            
            if( not doEuler and LB.CollisionFrequency > 0.0 ):
                
                # --- Implicit Increment (Micro) ---
                
                LB.ComputeBLF \
                ( ImEx.a_IM[iS,iS] * dt, BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, \
                  rhoP[:,Macro.iU,:], rhoP[:,Macro.iT,:], g, B_Micro_IM[:,:,:,iS], True )
            
                g -= dt * ImEx.a_IM[iS,iS] * LB.CollisionFrequency * B_Micro_IM[:,:,:,iS]
            
            f = M + g
        
        if( any( ImEx.a_EX[:,iS] != 0.0 ) or ( ImEx.w_EX[iS] != 0.0 ) ):
            
            # --- Explicit Increment (Macro) ---
            
            Macro.ComputeBLF \
            ( BX_Z, BX_X, Mesh_X.dx, rhoC, g, E, B_Macro[:,:,:,iS], True )
            
            if( not doEuler ):
                
                # --- Explicit Increment (Micro) ---
            
                Vlasov.ComputeBLF \
                ( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, \
                  E, g, B_Micro_EX[:,:,:,iS], True )
                
                Macro.ComputeBLF_Micro \
                ( BX_Z, BX_X, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, \
                  rhoC, E, B_Micro_Mx[:,:,:,iS], True )
    
    iS = ImEx.nStages - 1
    
    if( any( ImEx.a_IM[iS,:] != ImEx.w_IM ) or any( ImEx.a_EX[iS,:] != ImEx.w_EX ) ):
        
        rhoC = rhoC_0.copy()
        
        for iS in range( ImEx.nStages ):
            
            if( ImEx.w_EX[iS] != 0.0 ):
                
                rhoC -= dt * ImEx.w_EX[iS] * B_Macro[:,:,:,iS]
        
        Macro.ApplySlopeLimiter( BX_X, rhoC )
        
        Macro.ComputePrimitiveFromConserved( BX_X, rhoC, rhoP )
        
        Macro.ComputeMaxwellianProjectionFromPrimitive( BX_Z, rhoP, M )
        
        if( not doEuler ):
            
            g = g_0.copy()
            
            g += ( M_0 - M )
            
            for iS in range( ImEx.nStages ):
                
                if( ImEx.w_IM[iS] != 0.0 ):
                    
                    g -= dt * ImEx.w_IM[iS] * LB.CollisionFrequency * B_Micro_IM[:,:,:,iS]
                
                if( ImEx.w_EX[iS] != 0.0 ):
                    
                    g -= dt * ImEx.w_EX[iS] * ( B_Micro_EX[:,:,:,iS] + B_Micro_Mx[:,:,:,iS] )
            
            Vlasov.CleanMicroDistribution( BX_Z, g )
        
        f = M + g
    
    # -------------------------
    
    t += dt
    
    if( np.mod( iCycle, iCycleT ) == 0 ):
        
        Tally.Compute( t, BX_Z, f )
    
    if( np.mod( iCycle, iCycleW ) == 0 ):
        
        VM.ComputeVelocityMoments( BX_Z, Mesh_V.dx, g, D_g, S_g, G_g )
        
        IO.WriteFields \
        ( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, t, \
          D = rhoP[:,Macro.iD,:], U = rhoP[:,Macro.iU,:], T = rhoP[:,Macro.iT,:], \
          S = rhoC[:,Macro.iS,:], G = rhoC[:,Macro.iG,:], f = f, M = M, g = g, \
          D_g = D_g, S_g = S_g, G_g = G_g )

Tally.Compute( t, BX_Z, f )
Tally.Write( 'RiemannProblemMM' )

VM.ComputeVelocityMoments( BX_Z, Mesh_V.dx, g, D_g, S_g, G_g )

IO.WriteFields \
( BX_Z, Mesh_V.dx, Mesh_V.x_c, Mesh_X.dx, Mesh_X.x_c, t, \
  D = rhoP[:,Macro.iD,:], U = rhoP[:,Macro.iU,:], T = rhoP[:,Macro.iT,:], \
  S = rhoC[:,Macro.iS,:], G = rhoC[:,Macro.iG,:], f = f, M = M, g = g, \
  D_g = D_g, S_g = S_g, G_g = G_g )
