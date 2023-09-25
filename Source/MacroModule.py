
# --- Macro Module ---

import numpy as np
import time  as timer
from math          import pi
from scipy.special import erf

import BoundaryConditionsModule as BC
from LagrangeModule import LagrangeP, dLagrangeP

class MacroSolver:
    
    def __init__( self ):
        
        self.Gamma = [] # "Ratio of Specific Heats"
        
        self.iD = 0
        self.iU = 1
        self.iT = 2
        
        self.iS = 1
        self.iG = 2
        
        self.minT = 1.0e-8
        
    def Initialize( self, Gamma, BX_Z, BX_X, dv, v_c, nDOFv, n_v, nDOFx, n_x, UseKineticFlux = False, UseSlopeLimiter = False, InfiniteMaxwellianDomain = False ):
        
        self.Gamma = Gamma
        
        self.UseKineticFlux = UseKineticFlux
        
        self.UseSlopeLimiter = UseSlopeLimiter
        
        self.InfiniteMaxwellianDomain = InfiniteMaxwellianDomain
        
        self.dv  = dv.copy()
        self.v_c = v_c.copy()
        
        self.nDOFv = nDOFv
        self.nDOFx = nDOFx
        self.nDOFz = nDOFv * nDOFx
        
        self.pLG_v, self.wLG_v = np.polynomial.legendre.leggauss( self.nDOFv )
        self.pLG_v = self.pLG_v / 2.0 # --- Unit Interval
        self.wLG_v = self.wLG_v / 2.0 # --- Unit Interval
        
        self.pLG_x, self.wLG_x = np.polynomial.legendre.leggauss( self.nDOFx )
        self.pLG_x = self.pLG_x / 2.0 # --- Unit Interval
        self.wLG_x = self.wLG_x / 2.0 # --- Unit Interval
        
        self.wLG_z = np.empty( self.nDOFz, np.float64, 'F' )

        l_z = 0
        for l_x in range( self.nDOFx ):
            for l_v in range( self.nDOFv ):
                
                self.wLG_z[l_z] = self.wLG_v[l_v] * self.wLG_x[l_x]
                
                l_z += 1
        
        self.n_v = n_v
        
        self.pv_q, self.wv_q = np.polynomial.legendre.leggauss( self.n_v )
        self.pv_q = self.pv_q / 2.0 # --- Unit Interval
        self.wv_q = self.wv_q / 2.0 # --- Unit Interval
        
        self.n_x = n_x
        
        self.px_q, self.wx_q = np.polynomial.legendre.leggauss( self.n_x )
        self.px_q = self.px_q / 2.0 # --- Unit Interval
        self.wx_q = self.wx_q / 2.0 # --- Unit Interval
        
        self.n_z = self.n_v * self.n_x
        
        self.pz_q = np.empty( [ self.n_z, 2 ], np.float64, 'F' )
        self.wz_q = np.empty( [ self.n_z    ], np.float64, 'F' )
        
        i_z = 0
        for i_x in range( n_x ):
            for i_v in range( n_v ):
                
                self.pz_q[i_z,0] = self.pv_q[i_v]
                self.pz_q[i_z,1] = self.px_q[i_x]
                
                self.wz_q[i_z] = self.wv_q[i_v] * self.wx_q[i_x]
                
                i_z += 1

        # --- Mass Matrix ---
        
        self.MassM = self.wLG_x
        
        # --- phi_x_dn, phi_x_up:
        
        self.phi_x_dn = np.empty( [ 1, self.nDOFx ], np.float64, 'F' )
        self.phi_x_up = np.empty( [ 1, self.nDOFx ], np.float64, 'F' )
        
        for l_x in range( self.nDOFx ):
            
            self.phi_x_dn[0,l_x] \
            = LagrangeP( [ - 0.5 ], l_x, self.pLG_x, self.nDOFx )
            
            self.phi_x_up[0,l_x] \
            = LagrangeP( [ + 0.5 ], l_x, self.pLG_x, self.nDOFx )
        
        # --- phi_v_dn, phi_v_up:
        
        self.phi_v_dn = np.empty( [ 1, self.nDOFv ], np.float64, 'F' )
        self.phi_v_up = np.empty( [ 1, self.nDOFv ], np.float64, 'F' )
        
        for l_v in range( self.nDOFv ):
            
            self.phi_v_dn[0,l_v] \
            = LagrangeP( [ - 0.5 ], l_v, self.pLG_v, self.nDOFv )
            
            self.phi_v_up[0,l_v] \
            = LagrangeP( [ + 0.5 ], l_v, self.pLG_v, self.nDOFv )
        
        # --- phi_x, dphidx_x ---
        
        self.phi_x    = np.empty( [ self.n_x, self.nDOFx ], np.float64, 'F' )
        self.dphidx_x = np.empty( [ self.n_x, self.nDOFx ], np.float64, 'F' )
        
        for l_x in range( self.nDOFx ):
            for q_x in range( self.n_x ):
                
                self.phi_x[q_x,l_x] \
                = LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                        
                self.dphidx_x[q_x,l_x] \
                = dLagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
        
        # --- phi_z_dn, phi_z_up ---
        
        self.phi_z_dn = np.empty( [ self.n_v, self.nDOFz ], np.float64, 'F' )
        self.phi_z_up = np.empty( [ self.n_v, self.nDOFz ], np.float64, 'F' )
        
        l_z = 0
        for l_x in range( self.nDOFx ):
            for l_v in range( self.nDOFv ):

                for q_v in range( self.n_v ):
                    
                    self.phi_z_dn[q_v,l_z] \
                    = LagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                    * LagrangeP( [ - 0.5          ], l_x, self.pLG_x, self.nDOFx )
                    
                    self.phi_z_up[q_v,l_z] \
                    = LagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                    * LagrangeP( [ + 0.5          ], l_x, self.pLG_x, self.nDOFx )
                
                l_z += 1
        
        # --- phi_z ---
        
        self.phi_z = np.empty( [ self.n_z, self.nDOFz ], np.float64, 'F' )
        
        l_z = 0
        for l_x in range( self.nDOFx ):
            for l_v in range( self.nDOFv ):
                
                q_z = 0
                for q_x in range( self.n_x ):
                    for q_v in range( self.n_v ):
                        
                        self.phi_z[q_z,l_z] \
                        = LagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                        * LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                        
                        q_z += 1
                
                l_z += 1
        
        # --- vLG ---
        
        self.vLG = np.empty( [ self.nDOFv * BX_Z.nx[0] ], np.float64, 'F' )
        
        i = 0
        for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
            for q_v in range( self.nDOFv ):
                
                self.vLG[i] = v_c[i_v] + dv[i_v] * self.pLG_v[q_v]
                
                i += 1
        
        # --- v_q ---
        
        self.v_q = np.empty( [ self.n_v, BX_Z.w[0] ], np.float64, 'F' )
        
        for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
            
            self.v_q[:,i_v] = v_c[i_v] + dv[i_v] * self.pv_q[:]
        
        # --- vVec, vHatP, vHatM ---
        
        self.vVec  = np.empty( [ self.n_v * BX_Z.nx[0] ], np.float64, 'F' )
        self.vHatP = np.empty( [ self.n_v * BX_Z.nx[0] ], np.float64, 'F' )
        self.vHatM = np.empty( [ self.n_v * BX_Z.nx[0] ], np.float64, 'F' )
        
        i = 0
        for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
            for q_v in range( self.n_v ):
                
                v_q = v_c[i_v] + dv[i_v] * self.pv_q[q_v]
                
                self.vVec[i]  = v_q
                self.vHatP[i] = 0.5 * ( v_q + np.abs( v_q ) )
                self.vHatM[i] = 0.5 * ( v_q - np.abs( v_q ) )
                
                i += 1
        
        # --- eVec = w ( 1, v, 0.5*v^2 ) ---
        
        self.eVec = np.empty( [ self.n_v * BX_Z.nx[0], 3 ], np.float64, 'F' )
        
        i = 0
        for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
            for q_v in range( self.n_v ):
                
                v_q = v_c[i_v] + dv[i_v] * self.pv_q[q_v]
                
                self.eVec[i,0] = self.wv_q[q_v] * dv[i_v]
                self.eVec[i,1] = self.wv_q[q_v] * dv[i_v] * v_q
                self.eVec[i,2] = self.wv_q[q_v] * dv[i_v] * 0.5 * v_q**2
                
                i += 1
        
        # --- Maxwellian Weights ---
        
        # --- For ComputeMaxwellianProjectionFromPrimitive ---
        
        self.C_0 = np.empty( [ self.nDOFv, BX_Z.w[0] ], np.float64, 'F' )
        self.C_1 = np.empty( [ self.nDOFv, BX_Z.w[0] ], np.float64, 'F' )
        self.C_2 = np.empty( [ self.nDOFv, BX_Z.w[0] ], np.float64, 'F' )
        
        # --- For ComputeBLF_Micro ---
        
        self.W_0 = np.empty( [ self.nDOFv, BX_Z.w[0] ], np.float64, 'F' )
        self.W_1 = np.empty( [ self.nDOFv, BX_Z.w[0] ], np.float64, 'F' )
        self.W_2 = np.empty( [ self.nDOFv, BX_Z.w[0] ], np.float64, 'F' )
        
        for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
            for q_v in range( self.nDOFv ):
                
                r_v, s_v = np.delete( np.arange( self.nDOFv ), q_v )
                
                v_q = self.v_c[i_v] + self.pLG_v[q_v] * self.dv[i_v]
                v_r = self.v_c[i_v] + self.pLG_v[r_v] * self.dv[i_v]
                v_s = self.v_c[i_v] + self.pLG_v[s_v] * self.dv[i_v]
                
                self.C_2[q_v,i_v] = 1.0 / ( ( v_q - v_r ) * ( v_q - v_s ) * self.dv[i_v] * self.wLG_v[q_v] )
                self.C_1[q_v,i_v] = - ( v_r + v_s ) * self.C_2[q_v,i_v]
                self.C_0[q_v,i_v] =   ( v_r * v_s ) * self.C_2[q_v,i_v]
                
                self.W_2[q_v,i_v] = 1.0 / ( ( v_q - v_r ) * ( v_q - v_s ) )
                self.W_1[q_v,i_v] = - ( v_r + v_s ) * self.W_2[q_v,i_v]
                self.W_0[q_v,i_v] =   ( v_r * v_s ) * self.W_2[q_v,i_v]
        
        # --- Slope Limiter ---
        
        self.N2M = np.empty( [ 2, self.nDOFx ], np.float64, 'F' )
        
        for l_x in range( self.nDOFx ):
            
            self.N2M[0,l_x] =        np.sum( self.wx_q             * LagrangeP( self.px_q, l_x, self.pLG_x, self.nDOFx ) )
            self.N2M[1,l_x] = 12.0 * np.sum( self.wx_q * self.px_q * LagrangeP( self.px_q, l_x, self.pLG_x, self.nDOFx ) )
        
        self.M2N = np.empty( [ self.nDOFx, 2 ], np.float64, 'F' )
        
        for l_x in range( self.nDOFx ):
            
            self.M2N[l_x,0] = 1.0
            self.M2N[l_x,1] = self.pLG_x[l_x]
        
    def ComputeBLF( self, BX_Z, BX_X, dx, rhoC_h, g_h, E_h, B_h, MultiplyInverseMass = False ):
        
        print( "---------------------------" )
        print( "--- ComputeBLF (Macro) ----" )
        print( "---------------------------" )
        
        # --- Boundary Conditions ---
        
        BC.ApplyBC_X( BX_X, rhoC_h[:,self.iD,:] )
        BC.ApplyBC_X( BX_X, rhoC_h[:,self.iS,:] )
        BC.ApplyBC_X( BX_X, rhoC_h[:,self.iG,:] )
        
        BC.ApplyBC_Z( BX_Z, g_h )
        
        if( self.UseKineticFlux ):
            
            rhoP_h = np.zeros( ( self.nDOFx, 3        , BX_X.w[0] ), np.float64, 'F' )
            M_h    = np.zeros( ( self.nDOFz, BX_Z.w[0], BX_Z.w[1] ), np.float64, 'F' )

            self.ComputePrimitiveFromConserved( BX_X, rhoC_h, rhoP_h )
            
            self.ComputeMaxwellianFromPrimitive( BX_Z, rhoP_h, M_h )
            
            BC.ApplyBC_Z( BX_Z, M_h )
            
        
        # --- Initialize to Zero ---
        
        B_h[:,:,:] = 0.0
        
        #
        # --- Macroscopic Flux ---
        #
        
        if( not self.UseKineticFlux ):
            
            # --- Surface Term ---
            
            # --- Interpolate Left and Right State ---
            
            rhoC_L = np.reshape( ( self.phi_x_up @ np.reshape( rhoC_h[:,:,BX_X.lo[0]-1:BX_X.hi[0]+1], ( self.nDOFx, 3*(BX_X.nx[0]+1) ), order='F' ) ), ( 3, BX_X.nx[0]+1 ), order='F' )
            rhoC_R = np.reshape( ( self.phi_x_dn @ np.reshape( rhoC_h[:,:,BX_X.lo[0]-0:BX_X.hi[0]+2], ( self.nDOFx, 3*(BX_X.nx[0]+1) ), order='F' ) ), ( 3, BX_X.nx[0]+1 ), order='F' )
            
            # --- Compute Numerical Fluxes ---
            
            rhoP_L = np.empty( [ 3 ], np.float64, 'F' )
            rhoP_R = np.empty( [ 3 ], np.float64, 'F' )
            
            Flux_L = np.empty( [ 3 ], np.float64, 'F' )
            Flux_R = np.empty( [ 3 ], np.float64, 'F' )
            
            NumericalFlux = np.empty( [ 3, BX_X.nx[0] + 1 ], np.float64, 'F' )
            
            for i_x in range( BX_X.nx[0] + 1 ):
                
                # --- Left States ---
                
                rhoP_L[self.iD], rhoP_L[self.iU], rhoP_L[self.iT] \
                = self.ComputePrimitive \
                    ( rhoC_L[self.iD,i_x], rhoC_L[self.iS,i_x], rhoC_L[self.iG,i_x] )
                
                Flux_L[self.iD], Flux_L[self.iS], Flux_L[self.iG] \
                = self.ComputeMacroFlux \
                    ( rhoP_L[self.iD], rhoP_L[self.iU], rhoP_L[self.iT] )
                
                # --- Right States ---
                
                rhoP_R[self.iD], rhoP_R[self.iU], rhoP_R[self.iT] \
                = self.ComputePrimitive \
                    ( rhoC_R[self.iD,i_x], rhoC_R[self.iS,i_x], rhoC_R[self.iG,i_x] )
                
                Flux_R[self.iD], Flux_R[self.iS], Flux_R[self.iG] \
                = self.ComputeMacroFlux \
                    ( rhoP_R[self.iD], rhoP_R[self.iU], rhoP_R[self.iT] )
                
                # --- Numerical Flux (Lax-Friedrichs) ---
                
                # lambda_L = max( abs( rhoP_L[self.iU] + np.sqrt( 3.0 * max( rhoP_L[self.iT], self.minT ) ) ), \
                #                 abs( rhoP_L[self.iU] - np.sqrt( 3.0 * max( rhoP_L[self.iT], self.minT ) ) ) )
                
                # lambda_R = max( abs( rhoP_R[self.iU] + np.sqrt( 3.0 * max( rhoP_R[self.iT], self.minT ) ) ), \
                #                 abs( rhoP_R[self.iU] - np.sqrt( 3.0 * max( rhoP_R[self.iT], self.minT ) ) ) )
                
                # alpha = 6.0#max( lambda_L, lambda_R )
                
                # NumericalFlux[:,i_x] \
                # = 0.5 * ( Flux_L + Flux_R - alpha * ( rhoC_R[:,i_x] - rhoC_L[:,i_x] ) )
                
                # --- Numerical Flux (Quadrature-Based Kinetic-Type) ---
                
                # M_L = self.ComputeMaxwellian( rhoP_L[self.iD], rhoP_L[self.iU], rhoP_L[self.iT], self.vVec )
                
                # M_R = self.ComputeMaxwellian( rhoP_R[self.iD], rhoP_R[self.iU], rhoP_R[self.iT], self.vVec )
                
                # NumericalFlux[0,i_x] \
                # = 0.5 * ( Flux_L[0] + Flux_R[0] ) \
                # - 0.5 * np.sum( np.abs( self.vVec[:] ) * self.eVec[:,0] * ( M_R[:] - M_L[:] ) )
                
                # NumericalFlux[1,i_x] \
                # = 0.5 * ( Flux_L[1] + Flux_R[1] ) \
                # - 0.5 * np.sum( np.abs( self.vVec[:] ) * self.eVec[:,1] * ( M_R[:] - M_L[:] ) )
                
                # NumericalFlux[2,i_x] \
                # = 0.5 * ( Flux_L[2] + Flux_R[2] ) \
                # - 0.5 * np.sum( np.abs( self.vVec[:] ) * self.eVec[:,2] * ( M_R[:] - M_L[:] ) )
                
                # --- Numerical Flux (Analytic Kinetic-Type) ---
                
                Int_absV_e0_M_L, Int_absV_e1_M_L, Int_absV_e2_M_L \
                = self.ComputeIntegral_absV_e_M( rhoP_L[self.iD], rhoP_L[self.iU], rhoP_L[self.iT] )
                
                Int_absV_e0_M_R, Int_absV_e1_M_R, Int_absV_e2_M_R \
                = self.ComputeIntegral_absV_e_M( rhoP_R[self.iD], rhoP_R[self.iU], rhoP_R[self.iT] )
                
                C_DISS = 1.0 # --- Variable Dissipation Strength (C_DISS = 1.0 is Fiducial)
                
                NumericalFlux[0,i_x] \
                = 0.5 * ( Flux_L[0] + Flux_R[0] ) \
                - C_DISS * 0.5 * ( Int_absV_e0_M_R - Int_absV_e0_M_L )
                
                NumericalFlux[1,i_x] \
                = 0.5 * ( Flux_L[1] + Flux_R[1] ) \
                - C_DISS * 0.5 * ( Int_absV_e1_M_R - Int_absV_e1_M_L )
                
                NumericalFlux[2,i_x] \
                = 0.5 * ( Flux_L[2] + Flux_R[2] ) \
                - C_DISS * 0.5 * ( Int_absV_e2_M_R - Int_absV_e2_M_L )
            
            # --- Assign to Bilinear Form ---
                
            B_h[:,:,BX_X.lo[0]:BX_X.hi[0]+1] \
            += np.reshape( ( self.phi_x_up.T ) @ ( np.reshape( NumericalFlux[:,1:BX_X.nx[0]+2], ( 1, 3*BX_X.nx[0] ), order='F' ) ), ( self.nDOFx, 3, BX_X.nx[0] ), order='F' )
            
            B_h[:,:,BX_X.lo[0]:BX_X.hi[0]+1] \
            -= np.reshape( ( self.phi_x_dn.T ) @ ( np.reshape( NumericalFlux[:,0:BX_X.nx[0]  ], ( 1, 3*BX_X.nx[0] ), order='F' ) ), ( self.nDOFx, 3, BX_X.nx[0] ), order='F' )
            
            # --- Volume Term ---
            
            # --- Interpolate to Quadrature Points ---
            
            rhoC_K = np.reshape( ( self.phi_x @ np.reshape( rhoC_h[:,:,BX_X.lo[0]:BX_X.hi[0]+1], ( self.nDOFx, 3*BX_X.nx[0] ), order='F' ) ), ( self.n_x, 3, BX_X.nx[0] ), order='F' )
        
            rhoP_K = np.empty( [ self.n_x, 3 ], np.float64, 'F' )
            
            wFlux_K = np.empty( [ self.n_x, 3, BX_X.nx[0] ], np.float64, 'F' )
            
            # --- Compute Macro Fluxes in Quadrature Points ---
            
            for i_x in range( BX_X.nx[0] ):
                
                rhoP_K[:,self.iD], rhoP_K[:,self.iU], rhoP_K[:,self.iT] \
                = self.ComputePrimitive \
                    ( rhoC_K[:,self.iD,i_x], rhoC_K[:,self.iS,i_x], rhoC_K[:,self.iG,i_x] )
                
                wFlux_K[:,self.iD,i_x], wFlux_K[:,self.iS,i_x], wFlux_K[:,self.iG,i_x] \
                = self.ComputeMacroFlux \
                    ( rhoP_K[:,self.iD], rhoP_K[:,self.iU], rhoP_K[:,self.iT], self.wx_q[:] )
                
            # --- Assign to Bilinear Form ---
            
            B_h[:,:,BX_X.lo[0]:BX_X.hi[0]+1] \
            -= np.reshape( ( self.dphidx_x.T ) @ ( np.reshape( wFlux_K, ( self.n_x, 3*BX_X.nx[0] ), order='F' ) ), ( self.nDOFx, 3, BX_X.nx[0] ), order='F' )
        
        #
        # --- Microscopic Flux ---
        #
        
        if( self.UseKineticFlux ):
            
            f_h = M_h + g_h
            
        else:
            
            f_h = g_h
        
        # --- Surface Term ---
        
        g_L = np.reshape( self.phi_z_up @ ( np.reshape( f_h[:,BX_Z.lo[0]:BX_Z.hi[0]+1,BX_Z.lo[1]-1:BX_Z.hi[1]+1], ( self.nDOFz, BX_Z.nx[0]*(BX_Z.nx[1]+1) ), order='F' ) ), ( self.n_v*BX_Z.nx[0], BX_Z.nx[1]+1 ), order='F' )
        g_R = np.reshape( self.phi_z_dn @ ( np.reshape( f_h[:,BX_Z.lo[0]:BX_Z.hi[0]+1,BX_Z.lo[1]-0:BX_Z.hi[1]+2], ( self.nDOFz, BX_Z.nx[0]*(BX_Z.nx[1]+1) ), order='F' ) ), ( self.n_v*BX_Z.nx[0], BX_Z.nx[1]+1 ), order='F' )
        
        NumericalFlux = np.empty( [ 3, BX_X.nx[0] + 1 ], np.float64, 'F' )
        
        for i_x in range( BX_X.nx[0] + 1 ):
            
            NumericalFlux[0,i_x] \
            = np.sum( self.eVec[:,0] * ( self.vHatP * g_L[:,i_x] + self.vHatM * g_R[:,i_x] ) )
            NumericalFlux[1,i_x] \
            = np.sum( self.eVec[:,1] * ( self.vHatP * g_L[:,i_x] + self.vHatM * g_R[:,i_x] ) )
            NumericalFlux[2,i_x] \
            = np.sum( self.eVec[:,2] * ( self.vHatP * g_L[:,i_x] + self.vHatM * g_R[:,i_x] ) )
        
        # --- Assign to Bilinear Form ---
        
        B_h[:,:,BX_X.lo[0]:BX_X.hi[0]+1] \
        += np.reshape( ( self.phi_x_up.T ) @ ( np.reshape( NumericalFlux[:,1:BX_X.nx[0]+2], ( 1, 3*BX_X.nx[0] ), order='F' ) ), ( self.nDOFx, 3, BX_X.nx[0] ), order='F' )
        
        B_h[:,:,BX_X.lo[0]:BX_X.hi[0]+1] \
        -= np.reshape( ( self.phi_x_dn.T ) @ ( np.reshape( NumericalFlux[:,0:BX_X.nx[0]  ], ( 1, 3*BX_X.nx[0] ), order='F' ) ), ( self.nDOFx, 3, BX_X.nx[0] ), order='F' )
        
        # --- Volume Term ---
        
        g_K = np.reshape( self.phi_z @ ( np.reshape( f_h[:,BX_Z.lo[0]:BX_Z.hi[0]+1,BX_Z.lo[1]:BX_Z.hi[1]+1], ( self.nDOFz, BX_Z.nx[0]*BX_Z.nx[1] ), order='F' ) ), ( self.n_v, self.n_x, BX_Z.nx[0], BX_Z.nx[1] ), order='F' )
        g_K = np.reshape( np.swapaxes( g_K, 1, 2 ), ( self.n_v*BX_Z.nx[0], self.n_x, BX_Z.nx[1] ), order='F' )
        
        wFlux_K = np.empty( [ self.n_x, 3, BX_X.nx[0] ], np.float64, 'F' )
        
        for i_x in range( BX_X.nx[0] ):
            
            wFlux_K[:,0,i_x] = self.wx_q[:] * np.ndarray.flatten( g_K[:,:,i_x].T @ np.reshape( self.eVec[:,0] * self.vVec[:], ( self.n_v*BX_Z.nx[0], 1 ), order='F' ), 'F' )
            wFlux_K[:,1,i_x] = self.wx_q[:] * np.ndarray.flatten( g_K[:,:,i_x].T @ np.reshape( self.eVec[:,1] * self.vVec[:], ( self.n_v*BX_Z.nx[0], 1 ), order='F' ), 'F' )
            wFlux_K[:,2,i_x] = self.wx_q[:] * np.ndarray.flatten( g_K[:,:,i_x].T @ np.reshape( self.eVec[:,2] * self.vVec[:], ( self.n_v*BX_Z.nx[0], 1 ), order='F' ), 'F' )
        
        # --- Assign to Bilinear Form ---
        
        B_h[:,:,BX_X.lo[0]:BX_X.hi[0]+1] \
        -= np.reshape( ( self.dphidx_x.T ) @ ( np.reshape( wFlux_K, ( self.n_x, 3*BX_X.nx[0] ), order='F' ) ), ( self.nDOFx, 3, BX_X.nx[0] ), order='F' )

        #
        # --- Electric Field ---
        #
        
        if( self.UseKineticFlux ):
            
            f_K = np.reshape( self.phi_z @ ( np.reshape( f_h[:,BX_Z.lo[0]:BX_Z.hi[0]+1,BX_Z.lo[1]:BX_Z.hi[1]+1], ( self.nDOFz, BX_Z.nx[0]*BX_Z.nx[1] ), order='F' ) ), ( self.n_v, self.n_x, BX_Z.nx[0], BX_Z.nx[1] ), order='F' )
            f_K = np.reshape( np.swapaxes( f_K, 1, 2 ), ( self.n_v*BX_Z.nx[0], self.n_x, BX_Z.nx[1] ), order='F' )
            
            for i_x in range( BX_X.nx[0] ):
                
                B_h[:,1,BX_X.lo[0]+i_x] \
                -= dx[BX_X.lo[0]+i_x] * self.wLG_x * E_h[:,BX_X.lo[0]+i_x] * rhoC_h[:,0,BX_X.lo[0]+i_x]
                
                S = np.ndarray.flatten( f_K[:,:,i_x].T @ np.reshape( self.eVec[:,1], ( self.n_v*BX_Z.nx[0], 1 ), order='F' ), 'F' )
                B_h[:,2,BX_X.lo[0]+i_x] \
                -= dx[BX_X.lo[0]+i_x] * self.wLG_x * E_h[:,BX_X.lo[0]+i_x] * S
            
        else:
            
            for i_x in range( BX_X.lo[0], BX_X.hi[0] + 1 ):
                
                B_h[:,1,i_x] \
                -= dx[i_x] * self.wLG_x * E_h[:,i_x] * rhoC_h[:,0,i_x]
                
                B_h[:,2,i_x] \
                -= dx[i_x] * self.wLG_x * E_h[:,i_x] * rhoC_h[:,1,i_x]
        
        if( MultiplyInverseMass ):
            
            for i_x in range( BX_X.lo[0], BX_X.hi[0] + 1 ):
                for i_comp in range( 3 ):
                    
                    B_h[:,i_comp,i_x] = B_h[:,i_comp,i_x] / ( dx[i_x] * self.MassM[:] )
    
    def ComputeBLF_Micro( self, BX_Z, BX_X, dv, v_c, dx, x_c, rhoC_h, E_h, B_h, MultiplyInverseMass = False ):
        
        UseQuadrature = False
        
        print( "---------------------------" )
        print( "--- ComputeBLF (Micro) ----" )
        print( "--- UseQuadrature: ", UseQuadrature, "-" )
        print( "---------------------------" )

        timerT = timer.time()
        
        # --- Boundary Conditions ---
        
        BC.ApplyBC_X( BX_X, rhoC_h[:,self.iD,:] )
        BC.ApplyBC_X( BX_X, rhoC_h[:,self.iS,:] )
        BC.ApplyBC_X( BX_X, rhoC_h[:,self.iG,:] )
        
        # --- Set to Zero ---
        
        B_h[:,:,:] = 0.0
        
        if( not UseQuadrature ):
            
            # --- Element Edges ---
            
            vLo = v_c - 0.5 * dv
            vHi = v_c + 0.5 * dv
            
            if( self.InfiniteMaxwellianDomain ):
                
                # --- Simulate Infinite Domain ---
                
                vLo[BX_Z.lo[0]] = - 1.0e2
                vHi[BX_Z.hi[0]] = + 1.0e2
            
            # --- Surface Term ---
            
            timerXS = timer.time()
            
            # --- Interpolate Left and Right States ---
            
            rhoC_L = np.transpose( np.reshape( ( self.phi_x_up @ np.reshape( rhoC_h[:,:,BX_X.lo[0]-1:BX_X.hi[0]+1], ( self.nDOFx, 3*(BX_X.nx[0]+1) ), order='F' ) ), ( 3, BX_X.nx[0]+1 ), order='F' ) )
            rhoC_R = np.transpose( np.reshape( ( self.phi_x_dn @ np.reshape( rhoC_h[:,:,BX_X.lo[0]-0:BX_X.hi[0]+2], ( self.nDOFx, 3*(BX_X.nx[0]+1) ), order='F' ) ), ( 3, BX_X.nx[0]+1 ), order='F' ) )
            
            # --- Compute Primitive ---
            
            D_L, U_L, T_L \
                = self.ComputePrimitive \
                    ( rhoC_L[:,self.iD], rhoC_L[:,self.iS], rhoC_L[:,self.iG] )
            D_R, U_R, T_R \
                = self.ComputePrimitive \
                    ( rhoC_R[:,self.iD], rhoC_R[:,self.iS], rhoC_R[:,self.iG] )
            
            # --- Compute Maxwellian Moments ---
            
            I_1 = np.empty( [BX_Z.nx[1]+1,BX_Z.w[0]], np.float64, 'F' )
            I_2 = np.empty( [BX_Z.nx[1]+1,BX_Z.w[0]], np.float64, 'F' )
            I_3 = np.empty( [BX_Z.nx[1]+1,BX_Z.w[0]], np.float64, 'F' )
            
            for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                
                if( vHi[i_v] <= 0.0 ):
                    
                    I_1[:,i_v], I_2[:,i_v], I_3[:,i_v] \
                        = self.ComputeMaxwellianMoments_123 \
                            ( vLo[i_v], vHi[i_v], D_R, U_R, T_R )
                    
                else:
                    
                    I_1[:,i_v], I_2[:,i_v], I_3[:,i_v] \
                        = self.ComputeMaxwellianMoments_123 \
                            ( vLo[i_v], vHi[i_v], D_L, U_L, T_L )
            
            # --- Compute Numerical Flux ---
            
            NumericalFlux = np.empty( [ self.nDOFv, BX_Z.w[0], BX_Z.nx[1] + 1 ], np.float64, 'F' )
            
            for i_x in range( BX_Z.nx[1] + 1 ):
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    NumericalFlux[:,i_v,i_x] \
                        = self.W_2[:,i_v] * I_3[i_x,i_v] \
                        + self.W_1[:,i_v] * I_2[i_x,i_v] \
                        + self.W_0[:,i_v] * I_1[i_x,i_v]
            
            # --- Assign to Bilinear Form ---
            
            for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    B_h[:,i_v,i_x] \
                        = np.reshape \
                            (   np.reshape( NumericalFlux[:,i_v,i_x  ], (self.nDOFv,1), 'F' ) @ self.phi_x_up \
                              - np.reshape( NumericalFlux[:,i_v,i_x-1], (self.nDOFv,1), 'F' ) @ self.phi_x_dn \
                              , (self.nDOFz), 'F' )
            
            timerXS = timer.time() - timerXS
            
            # --- Volume Term ---
            
            timerXV = timer.time()
            
            # --- Moments ---
            
            rho_0 = np.reshape( rhoC_h[:,self.iD,BX_Z.lo[1]:BX_Z.hi[1]+1], (self.nDOFx*BX_Z.nx[1]), 'F' )
            rho_1 = np.reshape( rhoC_h[:,self.iS,BX_Z.lo[1]:BX_Z.hi[1]+1], (self.nDOFx*BX_Z.nx[1]), 'F' )
            rho_2 = np.reshape( rhoC_h[:,self.iG,BX_Z.lo[1]:BX_Z.hi[1]+1], (self.nDOFx*BX_Z.nx[1]), 'F' )
            
            D, U, T = self.ComputePrimitive( rho_0, rho_1, rho_2 )
            
            # --- Compute Maxwellian Moments ---
            
            I_1 = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            I_2 = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            I_3 = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            
            for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                
                I_1[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v], \
                I_2[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v], \
                I_3[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v]  \
                    = self.ComputeMaxwellianMoments_123( vLo[i_v], vHi[i_v], D, U, T )
            
            I_1 = np.reshape( I_1, ( self.nDOFx,BX_Z.w[1],BX_Z.w[0] ), 'F' )
            I_2 = np.reshape( I_2, ( self.nDOFx,BX_Z.w[1],BX_Z.w[0] ), 'F' )
            I_3 = np.reshape( I_3, ( self.nDOFx,BX_Z.w[1],BX_Z.w[0] ), 'F' )
            
            # --- Compute Volume Flux ---
            
            for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    Flux \
                    = np.reshape( I_1[:,i_x,i_v], (self.nDOFx,1), 'F' ) @ np.transpose( np.reshape( self.W_0[:,i_v], (self.nDOFv,1), 'F' ) ) \
                    + np.reshape( I_2[:,i_x,i_v], (self.nDOFx,1), 'F' ) @ np.transpose( np.reshape( self.W_1[:,i_v], (self.nDOFv,1), 'F' ) ) \
                    + np.reshape( I_3[:,i_x,i_v], (self.nDOFx,1), 'F' ) @ np.transpose( np.reshape( self.W_2[:,i_v], (self.nDOFv,1), 'F' ) )
            
                    # --- Assign to Bilinear Form ---
                    
                    q_z = 0
                    for q_x in range( self.nDOFx ):
                        for q_v in range( self.nDOFv ):
                            
                            B_h[q_z,i_v,i_x] \
                            -= np.transpose( self.wLG_x * self.dphidx_x[:,q_x] ).dot( Flux[:,q_v] )
                            
                            q_z += 1
            
            timerXV = timer.time() - timerXV
            
            # --- Electric Field ---
            
            # --- Surface Term ---
            
            timerES = timer.time()
            
            weightLo = np.empty( [ BX_Z.w[0] ], np.float64, 'F' )
            weightHi = np.empty( [ BX_Z.w[0] ], np.float64, 'F' )
            
            for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                
                if( i_v == BX_Z.lo[0] and BX_Z.bc[0] == 0 ):
                    weightLo[i_v] = 0.0 # To Enforce Zero Flux at Inner Boundary in v
                else:
                    weightLo[i_v] = 1.0
                
                if( i_v == BX_Z.hi[0] and BX_Z.bc[0] == 0 ):
                    weightHi[i_v] = 0.0 # To Enforce Zero Flux at Outer Boundary in v
                else:
                    weightHi[i_v] = 1.0
            
            MLo = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            MHi = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            
            for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                
                MLo[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v] \
                    = Maxwellian( D, U, T, vLo[i_v] )
                MHi[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v] \
                    = Maxwellian( D, U, T, vHi[i_v] )
            
            MLo = np.reshape( MLo, (self.nDOFx,BX_Z.w[1],BX_Z.w[0]), 'F' )
            MHi = np.reshape( MHi, (self.nDOFx,BX_Z.w[1],BX_Z.w[0]), 'F' )
            
            for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    q_z = 0
                    for q_x in range( self.nDOFx ):
                        for q_v in range( self.nDOFv ):
                            
                            B_h[q_z,i_v,i_x] \
                            += dx[i_x] * self.wLG_x[q_x] * E_h[q_x,i_x] \
                                * ( weightHi[i_v] * MHi[q_x,i_x,i_v] * self.phi_v_up[0,q_v] \
                                  - weightLo[i_v] * MLo[q_x,i_x,i_v] * self.phi_v_dn[0,q_v] )
                            
                            q_z += 1
            
            timerES = timer.time() - timerES
            
            # --- Volume Term ---
            
            timerEV = timer.time()
            
            # --- Compute Maxwellian Moments ---
            
            I_0 = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            I_1 = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            I_2 = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            
            for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                
                I_0[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v], \
                I_1[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v], \
                I_2[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v]  \
                    = self.ComputeMaxwellianMoments_012( vLo[i_v], vHi[i_v], D, U, T )
            
            I_0 = np.reshape( I_0, ( self.nDOFx,BX_Z.w[1],BX_Z.w[0] ), 'F' )
            I_1 = np.reshape( I_1, ( self.nDOFx,BX_Z.w[1],BX_Z.w[0] ), 'F' )
            I_2 = np.reshape( I_2, ( self.nDOFx,BX_Z.w[1],BX_Z.w[0] ), 'F' )
            
            for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    q_z = 0
                    for q_x in range( self.nDOFx ):
                        for q_v in range( self.nDOFv ):
                            
                            B_h[q_z,i_v,i_x] \
                            -= dx[i_x] * self.wLG_x[q_x] * E_h[q_x,i_x] \
                               * ( 2.0 * self.W_2[q_v,i_v] * I_1[q_x,i_x,i_v] + self.W_1[q_v,i_v] * I_0[q_x,i_x,i_v] )
                            
                            q_z += 1
            
            timerEV = timer.time() - timerEV
            
        else:
            
            # --- Surface Term ---
            
            # --- Interpolate Left and Right States ---
            
            rhoC_L = np.reshape( ( self.phi_x_up @ np.reshape( rhoC_h[:,:,BX_X.lo[0]-1:BX_X.hi[0]+1], ( self.nDOFx, 3*(BX_X.nx[0]+1) ), order='F' ) ), ( 3, BX_X.nx[0]+1 ), order='F' )
            rhoC_R = np.reshape( ( self.phi_x_dn @ np.reshape( rhoC_h[:,:,BX_X.lo[0]-0:BX_X.hi[0]+2], ( self.nDOFx, 3*(BX_X.nx[0]+1) ), order='F' ) ), ( 3, BX_X.nx[0]+1 ), order='F' )
            
            # --- Compute Numerical Flux ---
            
            NumericalFlux = np.empty( [ self.nDOFv, BX_Z.w[0], BX_Z.nx[1] + 1 ], np.float64, 'F' )
            
            for i_x in range( BX_Z.nx[1] + 1 ):
                
                D_L, U_L, T_L \
                    = self.ComputePrimitive \
                        ( rhoC_L[self.iD,i_x], rhoC_L[self.iS,i_x], rhoC_L[self.iG,i_x] )
                
                D_R, U_R, T_R \
                    = self.ComputePrimitive \
                        ( rhoC_R[self.iD,i_x], rhoC_R[self.iS,i_x], rhoC_R[self.iG,i_x] )
                
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    vHi = v_c[i_v] + 0.5 * dv[i_v]
                    
                    if( vHi <= 0 ):
                        D = D_R
                        U = U_R
                        T = T_R
                    else:
                        D = D_L
                        U = U_L
                        T = T_L
                    
                    M_q = self.ComputeMaxwellian( D, U, T, self.v_q[:,i_v] )
                    
                    NumericalFlux[:,i_v,i_x] = dv[i_v] * self.wLG_v[:] * self.v_q[:,i_v]* M_q[:]
            
            # --- Assign to Bilinear Form ---
            
            for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    q_z = 0
                    for q_x in range( self.nDOFx ):
                        for q_v in range( self.nDOFv ):
                            
                            B_h[q_z,i_v,i_x] \
                            += ( NumericalFlux[q_v,i_v,i_x  ] * self.phi_x_up[0,q_x] \
                               - NumericalFlux[q_v,i_v,i_x-1] * self.phi_x_dn[0,q_x] )
                            
                            q_z += 1
            
            # --- Volume Term ---
            
            # --- Compute Volume Flux ---
            
            Flux = np.empty( [ self.nDOFx, self.nDOFv, BX_Z.w[0], BX_Z.w[1] ], np.float64, 'F' )
            
            for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
                
                D, U, T \
                = self.ComputePrimitive \
                    ( rhoC_h[:,self.iD,i_x], rhoC_h[:,self.iS,i_x], rhoC_h[:,self.iG,i_x] )
                
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    for q_x in range( self.nDOFx ):
                        for q_v in range( self.nDOFv ):
                            
                            M_q = self.ComputeMaxwellian( D[q_x], U[q_x], T[q_x], self.v_q[q_v,i_v] )
                            
                            Flux[q_x,q_v,i_v,i_x] \
                            = dv[i_v] * self.wLG_v[q_v] * self.v_q[q_v,i_v] * M_q
            
            # --- Assign to Bilinear Form ---
            
            for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    q_z = 0
                    for q_x in range( self.nDOFx ):
                        for q_v in range( self.nDOFv ):
                            
                            for k in range( self.nDOFx ):
                                
                                B_h[q_z,i_v,i_x] \
                                -= self.wLG_x[k] * self.dphidx_x[k,q_x] * Flux[k,q_v,i_v,i_x]
                            
                            q_z += 1
        
        # --- end else UseQuadrature ---
        
        if( MultiplyInverseMass ):
            
            for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    q_z = 0
                    for q_x in range( self.nDOFx ):
                        for q_v in range( self.nDOFv ):
                            
                            B_h[q_z,i_v,i_x] \
                                = B_h[q_z,i_v,i_x] \
                                    / ( dv[i_v] * dx[i_x] * self.wLG_v[q_v] * self.wLG_x[q_x] )
                            
                            q_z += 1
        
        timerT = timer.time() - timerT
        
        print( "          " )
        print( "timerT  = ", timerT )
        if( not UseQuadrature ):
            print( "timerXS = ", timerXS )
            print( "timerXV = ", timerXV )
            print( "timerES = ", timerES )
            print( "timerEV = ", timerEV )
        print( "          " )
    
    def ApplySlopeLimiter( self, BX_X, rhoC_h ):
        
        if( self.nDOFx == 1 ):
            
            return
        
        if( not self.UseSlopeLimiter ):
            
            return
        
        # --- Boundary Conditions ---
        
        BC.ApplyBC_X( BX_X, rhoC_h[:,self.iD,:] )
        BC.ApplyBC_X( BX_X, rhoC_h[:,self.iS,:] )
        BC.ApplyBC_X( BX_X, rhoC_h[:,self.iG,:] )
        
        # Convert from Nodal to Modal ---
        
        rhoC_M = np.reshape( self.N2M @ np.reshape( rhoC_h, ( self.nDOFx, 3*BX_X.w[0] ), order = 'F' ), ( 2, 3, BX_X.w[0] ), order = 'F' )
        
        for i_x in range( BX_X.lo[0], BX_X.hi[0] + 1 ):
            
            D, U, T = self.ComputePrimitive \
            ( rhoC_M[0,self.iD,i_x], rhoC_M[0,self.iS,i_x], rhoC_M[0,self.iG,i_x] )
            
            R, invR = EigenVectors( D, U, T )
            
            drho = MinMod( invR @ rhoC_M[1,:,i_x], \
                           1.75 * invR @ (rhoC_M[0,:,i_x  ]-rhoC_M[0,:,i_x-1]), \
                           1.75 * invR @ (rhoC_M[0,:,i_x+1]-rhoC_M[0,:,i_x  ]) )
            
            drho = R @ drho
            
            diff = np.abs( drho - rhoC_M[1,:,i_x] )
            
            for iComp in range( 3 ):
                if( diff[iComp] > 1.0e-6 * np.abs( rhoC_M[0,iComp,i_x] ) ):
                
                    rhoC_h[:,iComp,i_x] = np.reshape( self.M2N @ np.reshape( [ rhoC_M[0,iComp,i_x], drho[iComp] ], ( 2, 1 ), 'F' ), self.nDOFx, 'F' )
    
    def ComputeMaxwellianFromPrimitive( self, BX_Z, rhoP_h, M_h ):
        
        for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
            for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                
                V = np.reshape( np.repeat( np.reshape( self.v_q[:,i_v], ( self.n_v, 1 ), order='F' ), self.n_x, 1 ), self.n_z, order='F' )
                
                D = np.repeat( self.phi_x @ rhoP_h[:,0,i_x], self.n_v, 0 )
                U = np.repeat( self.phi_x @ rhoP_h[:,1,i_x], self.n_v, 0 )
                T = np.repeat( self.phi_x @ rhoP_h[:,2,i_x], self.n_v, 0 )
                
                if( np.min(T) < self.minT ):
                    
                    print( "Warning: ", np.min(T), self.minT )
                
                M = ( D / np.sqrt( 2.0 * pi * T ) ) * np.exp( - 0.5 * ( V - U )**2 / T )
                
                for l_z in range( self.nDOFz ):
                    
                    M_h[l_z,i_v,i_x] = np.sum( self.wz_q * M * self.phi_z[:,l_z] ) / self.wLG_z[l_z]
    
    def ComputeMaxwellianProjectionFromPrimitive( self, BX_Z, rhoP_h, M ):
        
        print( "------------------------------------------------" )
        print( "--- ComputeMaxwellianProjectionFromPrimitive ---" )
        print( "------------------------------------------------" )
        
        timerMP = timer.time()
        
        if( self.nDOFv != 3 ):
            
            self.ComputeMaxwellianFromPrimitive( BX_Z, rhoP_h, M )
            
        else:
            
            timerPR = timer.time()
            
            # --- Element Edges ---
            
            vLo = self.v_c - 0.5 * self.dv
            vHi = self.v_c + 0.5 * self.dv
            
            if( self.InfiniteMaxwellianDomain ):
                
                # --- Simulate Infinite Domain ---
                
                vLo[BX_Z.lo[0]] = - 1.0e2
                vHi[BX_Z.hi[0]] = + 1.0e2
            
            # --- Moments ---
            
            D = np.reshape( rhoP_h[:,self.iD,BX_Z.lo[1]:BX_Z.hi[1]+1], (self.nDOFx*BX_Z.nx[1]), 'F' )
            U = np.reshape( rhoP_h[:,self.iU,BX_Z.lo[1]:BX_Z.hi[1]+1], (self.nDOFx*BX_Z.nx[1]), 'F' )
            T = np.reshape( rhoP_h[:,self.iT,BX_Z.lo[1]:BX_Z.hi[1]+1], (self.nDOFx*BX_Z.nx[1]), 'F' )
            
            timerPR = timer.time() - timerPR
            
            timerCO = timer.time()
            
            # --- Compute Maxwellian Moments ---
            
            I_0 = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            I_1 = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            I_2 = np.empty( [self.nDOFx*BX_Z.w[1],BX_Z.w[0]], np.float64, 'F' )
            
            for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                
                I_0[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v], \
                I_1[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v], \
                I_2[self.nDOFx*BX_Z.lo[1]:self.nDOFx*BX_Z.lo[1]+self.nDOFx*BX_Z.nx[1],i_v] \
                    = self.ComputeMaxwellianMoments_012( vLo[i_v], vHi[i_v], D, U, T )
            
            I_0 = np.reshape( I_0, ( self.nDOFx,BX_Z.w[1],BX_Z.w[0] ), 'F' )
            I_1 = np.reshape( I_1, ( self.nDOFx,BX_Z.w[1],BX_Z.w[0] ), 'F' )
            I_2 = np.reshape( I_2, ( self.nDOFx,BX_Z.w[1],BX_Z.w[0] ), 'F' )
            
            for i_x in range( BX_Z.lo[1], BX_Z.hi[1] + 1 ):
                for i_v in range( BX_Z.lo[0], BX_Z.hi[0] + 1 ):
                    
                    M[:,i_v,i_x] \
                    = np.reshape( np.reshape( self.C_0[:,i_v], (self.nDOFv,1), 'F' ) @ np.transpose( np.reshape( I_0[:,i_x,i_v], (self.nDOFx,1), 'F' ) ) \
                                + np.reshape( self.C_1[:,i_v], (self.nDOFv,1), 'F' ) @ np.transpose( np.reshape( I_1[:,i_x,i_v], (self.nDOFx,1), 'F' ) ) \
                                + np.reshape( self.C_2[:,i_v], (self.nDOFv,1), 'F' ) @ np.transpose( np.reshape( I_2[:,i_x,i_v], (self.nDOFx,1), 'F' ) ) \
                                , ( self.nDOFz ), 'F' )
            
            timerCO = timer.time() - timerCO
        
        timerMP = timer.time() - timerMP
        
        print( "          " )
        print( "timerMP = ", timerMP )
        print( "timerPR = ", timerPR )
        print( "timerCO = ", timerCO )
        print( "          " )
        
    def ComputeConservedFromPrimitive( self, BX_X, rhoP_h, rhoC_h ):
        
        for i_x in range( BX_X.lo[0], BX_X.hi[0] + 1 ):
            
            rhoC_h[:,self.iD,i_x], rhoC_h[:,self.iS,i_x], rhoC_h[:,self.iG,i_x] \
            = self.ComputeConserved \
                ( rhoP_h[:,self.iD,i_x], rhoP_h[:,self.iU,i_x], rhoP_h[:,self.iT,i_x] )
        
    def ComputePrimitiveFromConserved( self, BX_X, rhoC_h, rhoP_h ):
        
        for i_x in range( BX_X.lo[0], BX_X.hi[0] + 1 ):
            
            rhoP_h[:,self.iD,i_x], rhoP_h[:,self.iU,i_x], rhoP_h[:,self.iT,i_x] \
            = self.ComputePrimitive \
                ( rhoC_h[:,self.iD,i_x], rhoC_h[:,self.iS,i_x], rhoC_h[:,self.iG,i_x] )
    
    def ComputePrimitiveFromConserved_Newton( self, BX_X, rhoC_h, rhoP_h ):
        
        for i_x in range( BX_X.lo[0], BX_X.hi[0] + 1 ):
            
            for i_q in range( self.nDOFx):
                
                rhoP_h[i_q,self.iD,i_x], rhoP_h[i_q,self.iU,i_x], rhoP_h[i_q,self.iT,i_x] \
                = self.ComputePrimitive_Newton \
                    ( rhoC_h[i_q,self.iD,i_x], rhoC_h[i_q,self.iS,i_x], rhoC_h[i_q,self.iG,i_x] )
    
    def ComputeMaxwellian( self, D, U, T, V ):
        
        if( T < self.minT ):
            
            print( "Warning: ", T, self.minT )
        
        M = ( D / np.sqrt( 2.0 * pi * T ) ) * np.exp( - 0.5 * ( V - U )**2 / T )
        
        return M
    
    def ComputeMaxwellianMoments_012( self, vLo, vHi, D, U, T ):
        
        T_lim = np.maximum( T, self.minT )
        
        zLo = ( vLo - U ) / np.sqrt( 2.0 * T_lim )
        zHi = ( vHi - U ) / np.sqrt( 2.0 * T_lim )
        
        i_0 = IntExp_0( zLo, zHi )
        i_1 = IntExp_1( zLo, zHi )
        i_2 = IntExp_2( zLo, zHi )
        
        C_0 = D / np.sqrt( pi )
        C_1 = np.sqrt( 2.0 * T_lim )
        
        I_0 = C_0 * i_0
        I_1 = C_0 * ( C_1 * i_1 + U * i_0 )
        I_2 = C_0 * ( C_1**2 * i_2 + 2.0 * C_1 * U * i_1 + U**2 * i_0 )
        
        return I_0, I_1, I_2
    
    def ComputeMaxwellianMoments_123( self, vLo, vHi, D, U, T ):
        
        T_lim = np.maximum( T, self.minT )
        
        zLo = ( vLo - U ) / np.sqrt( 2.0 * T_lim )
        zHi = ( vHi - U ) / np.sqrt( 2.0 * T_lim )
        
        i_0 = IntExp_0( zLo, zHi )
        i_1 = IntExp_1( zLo, zHi )
        i_2 = IntExp_2( zLo, zHi )
        i_3 = IntExp_3( zLo, zHi )
        
        C_0 = D / np.sqrt( pi )
        C_1 = np.sqrt( 2.0 * T_lim )
        
        I_1 = C_0 * ( C_1 * i_1 + U * i_0 )
        I_2 = C_0 * ( C_1**2 * i_2 + 2.0 * C_1 * U * i_1 + U**2 * i_0 )
        I_3 = C_0 * ( C_1**3 * i_3 + 3.0 * C_1**2 * U * i_2 + 3.0 * C_1 * U**2 * i_1 + U**3 * i_0 )
        
        return I_1, I_2, I_3
    
    def ComputeMaxwellianMoments_0123( self, vLo, vHi, D, U, T ):
        
        T_lim = np.maximum( T, self.minT )
        
        zLo = ( vLo - U ) / np.sqrt( 2.0 * T_lim )
        zHi = ( vHi - U ) / np.sqrt( 2.0 * T_lim )
        
        i_0 = IntExp_0( zLo, zHi )
        i_1 = IntExp_1( zLo, zHi )
        i_2 = IntExp_2( zLo, zHi )
        i_3 = IntExp_3( zLo, zHi )
        
        C_0 = D / np.sqrt( pi )
        C_1 = np.sqrt( 2.0 * T_lim )
        
        I_0 = C_0 * i_0
        I_1 = C_0 * ( C_1 * i_1 + U * i_0 )
        I_2 = C_0 * ( C_1**2 * i_2 + 2.0 * C_1 * U * i_1 + U**2 * i_0 )
        I_3 = C_0 * ( C_1**3 * i_3 + 3.0 * C_1**2 * U * i_2 + 3.0 * C_1 * U**2 * i_1 + U**3 * i_0 )
        
        return I_0, I_1, I_2, I_3
        
    def ComputeConserved( self, D_P, U, T ):
        
        D_C = D_P
        S   = D_P * U
        G   = 0.5 * D_P * ( U * U + T )
        
        return D_C, S, G
        
    def ComputePrimitive( self, D_C, S, G ):
        
        D_P = D_C
        U   = S / D_C
        T   = np.maximum( ( 2.0 * G * D_C - S**2 ) / ( D_C * D_C ), self.minT )
        
        return D_P, U, T
    
    def ComputePrimitive_Newton( self, D_C, S, G ):
        
        Cvec = np.empty( [ 3, 1 ], np.float64, 'F' )
        Uvec = np.empty( [ 3, 1 ], np.float64, 'F' )
        Fvec = np.empty( [ 3, 1 ], np.float64, 'F' )
        Fjac = np.empty( [ 3, 3 ], np.float64, 'F' )
        
        # --- Vector of Constants ---

        Cvec[0,0] = D_C
        Cvec[1,0] = S
        Cvec[2,0] = G
        
        # --- Initial guess ---
        
        Uvec[0,0], Uvec[1,0], Uvec[2,0] = self.ComputePrimitive( D_C, S, G )
        
        k = 0
        converged = False
        while not converged:
            
            k += 1
            
            M = Maxwellian( Uvec[0,0], Uvec[1,0], Uvec[2,0], self.vVec )
            
            Fvec = np.transpose( self.eVec ).dot( np.reshape( M, ( M.size, 1 ), order = 'F' ) ) - Cvec
            
            dMdD = dMaxwelliandD( Uvec[0,0], Uvec[1,0], Uvec[2,0], M, self.vVec )
            dMdU = dMaxwelliandU( Uvec[0,0], Uvec[1,0], Uvec[2,0], M, self.vVec )
            dMdT = dMaxwelliandT( Uvec[0,0], Uvec[1,0], Uvec[2,0], M, self.vVec )
            
            Fjac[:,0] = np.reshape( np.transpose( self.eVec ).dot( np.reshape( dMdD, ( M.size, 1 ), order = 'F' ) ), 3, order = 'F' )
            Fjac[:,1] = np.reshape( np.transpose( self.eVec ).dot( np.reshape( dMdU, ( M.size, 1 ), order = 'F' ) ), 3, order = 'F' )
            Fjac[:,2] = np.reshape( np.transpose( self.eVec ).dot( np.reshape( dMdT, ( M.size, 1 ), order = 'F' ) ), 3, order = 'F' )
            
            dU = - np.linalg.solve( Fjac, Fvec )
            
            Uvec += dU
            
            if( np.all( np.abs( dU ) < 1.0e-12 ) ):
                
                converged = True
        
        return Uvec[0], Uvec[1], Uvec[2]
    
    def ComputeMacroFlux( self, D, U, T, a = 1.0 ):
        
        Flux_D = a * ( D * U )
        Flux_S = a * ( D * ( U**2 + T ) )
        Flux_G = a * ( 0.5 * D * ( U**2 + 3.0 * T ) * U )
        
        return Flux_D, Flux_S, Flux_G
    
    def ComputeIntegral_absV_e_M( self, D, U, T ):
        
        T_lim = np.maximum( T, self.minT )
        
        C_0 = np.sqrt( 2.0 * pi * T_lim )
        C_1 = D / C_0
        C_2 = np.exp( - 0.5 * U**2 / T_lim )
        C_3 = erf( U / np.sqrt( 2.0 * T_lim ) )
        
        Int_absV_e0_M = C_1 * ( 2.0 * T_lim                          * C_2 + U                                * C_0 * C_3 )
        Int_absV_e1_M = C_1 * ( 2.0 * T_lim * U                      * C_2 + ( U**2 + T_lim )                 * C_0 * C_3 )
        Int_absV_e2_M = C_1 * ( 2.0 * T_lim * ( 0.5 * U**2 + T_lim ) * C_2 + 0.5 * ( U**2 + 3.0 * T_lim ) * U * C_0 * C_3 )
        
        return Int_absV_e0_M, Int_absV_e1_M, Int_absV_e2_M

def EigenVectors( D, U, T ):
    
    R = np.empty( ( 3, 3 ), np.float64, 'F' )
    L = np.empty( ( 3, 3 ), np.float64, 'F' )
    
    C = np.sqrt( 3.0 * T )
    K = 0.5 * U**2
    H = K + 1.5 * T
    M = U / C
    
    R[0,:] = [ 1.0  , 1.0, 1.0  ]
    R[1,:] = [ U-C  , U  , U+C  ]
    R[2,:] = [ H-U*C, K  , H+U*C]
    
    L[0,:] = [ 0.5*M*(M+1.0), -0.5*(2.0*M+1.0)/C,  1.0/C**2 ]
    L[1,:] = [ 1.0-M**2     ,  2*M/C            , -2.0/C**2 ]
    L[2,:] = [ 0.5*M*(M-1.0), -0.5*(2.0*M-1.0)/C,  1.0/C**2 ]
    
    return R, L

def MinMod( a, b, c ):
    
    MM = np.empty( a.size, np.float64, 'F' )
    
    for i in range( a.size ):
        MM[i] = MinMod2( a[i], MinMod2( b[i], c[i] ) )
    
    return MM

def MinMod2( a, b ):
    if  ( abs(a) < abs(b) and a * b > 0.0 ):
        return a
    elif( abs(b) < abs(a) and a * b > 0.0 ):
        return b
    else:
        return 0.0

def Maxwellian( D, U, T, V ):
    
    M = ( D / np.sqrt( 2.0 * pi * T ) ) * np.exp( - 0.5 * ( V - U )**2 / T )
    
    return M

def dMaxwelliandD( D, U, T, M, V ):
    
    dMdD = M / D
    
    return dMdD

def dMaxwelliandU( D, U, T, M, V ):
    
    dMdU = M * ( V - U ) / T
    
    return dMdU

def dMaxwelliandT( D, U, T, M, V ):
    
    dMdT = 0.5 * M * ( ( V - U )**2 / T - 1.0 ) / T
    
    return dMdT

def IntExp_0( a, b ):
    
    i_0 = 0.5 * np.sqrt( pi ) * ( erf( b ) - erf( a ) )
    
    return i_0

def IntExp_1( a, b ):
    
    i_1 = 0.5 * ( np.exp( - a**2 ) - np.exp( - b**2 ) )
    
    return i_1

def IntExp_2( a, b ):
    
    i_2 = 0.5 * ( a * np.exp( - a**2 ) - b * np.exp( - b**2 ) ) \
        + 0.25 * np.sqrt( pi ) * ( erf( b ) - erf( a ) )
    
    return i_2

def IntExp_3( a, b ):
    
    i_3 = 0.5 * ( ( a**2 + 1.0 ) * np.exp( - a**2 ) - ( b**2 + 1.0 ) * np.exp( - b**2 ) )
    
    return i_3
