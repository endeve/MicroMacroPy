
# --- Vlasov Module ---

import numpy as np
import scipy as sp
import time as timer

import BoundaryConditionsModule as BC
from LagrangeModule import LagrangeP, dLagrangeP

class VlasovSolver:
    
    def __init__( self ):

        self.iv = 0
        self.ix = 1
        
        self.pLG_x = []
        self.wLG_x = []
        
        self.pLG_v = []
        self.wLG_v = []
        
        self.pz_q = []
        self.wz_q = []
        
        self.pv_q = []
        self.wv_q = []
        
        self.px_q = []
        self.wx_q = []
        
        self.MassM = []
        
        self.phi_z = []
        self.dphidv_z = []
        self.dphidx_z = []

        self.phi_v_dn = []
        self.phi_v_up = []

        self.phi_x_dn = []
        self.phi_x_up = []
        
        self.phi_x = []
        self.phi_x2z = []
        
    def Initialize( self, BX, dv, v_c, nDOFv, nDOFx, n_v, n_x, \
                    UsePositivityLimiter=False, UseMicroCleaning=False, UseGLF_Flux=False ):
        
        self.UsePositivityLimiter = UsePositivityLimiter
        
        self.UseMicroCleaning = UseMicroCleaning
        
        self.UseGLF_Flux = UseGLF_Flux
        
        self.nDOFv = nDOFv
        self.nDOFx = nDOFx
        self.nDOFz = nDOFv * nDOFx
        
        self.pLG_v, self.wLG_v = np.polynomial.legendre.leggauss( self.nDOFv )
        self.pLG_v = self.pLG_v / 2.0 # --- Unit Interval
        self.wLG_v = self.wLG_v / 2.0 # --- Unit Interval
        
        self.pLG_x, self.wLG_x = np.polynomial.legendre.leggauss( self.nDOFx )
        self.pLG_x = self.pLG_x / 2.0 # --- Unit Interval
        self.wLG_x = self.wLG_x / 2.0 # --- Unit Interval
        
        self.wLG_z = np.empty( [ self.nDOFz ], np.float64, 'F' )
        
        i_z = 0
        for i_x in range( self.nDOFx ):
            for i_v in range( self.nDOFv ):
                
                self.wLG_z[i_z] = self.wLG_v[i_v] * self.wLG_x[i_x]
                
                i_z += 1
        
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
                
                self.pz_q[i_z,self.iv] = self.pv_q[i_v]
                self.pz_q[i_z,self.ix] = self.px_q[i_x]
                
                self.wz_q[i_z] = self.wv_q[i_v] * self.wx_q[i_x]
                
                i_z += 1
        
        self.MassM = np.empty( [ self.nDOFz ], np.float64, 'F' )
        
        self.phi_z    = np.empty( [ self.n_z, self.nDOFz ], np.float64, 'F' )
        self.dphidv_z = np.empty( [ self.n_z, self.nDOFz ], np.float64, 'F' )
        self.dphidx_z = np.empty( [ self.n_z, self.nDOFz ], np.float64, 'F' )
        
        self.phi_v_dn = np.empty( [ self.n_x, self.nDOFz ], np.float64, 'F' )
        self.phi_v_up = np.empty( [ self.n_x, self.nDOFz ], np.float64, 'F' )
        
        self.phi_x_dn = np.empty( [ self.n_v, self.nDOFz ], np.float64, 'F' )
        self.phi_x_up = np.empty( [ self.n_v, self.nDOFz ], np.float64, 'F' )
        
        l_z = 0
        for l_x in range( self.nDOFx ):
            for l_v in range( self.nDOFv ):

                # --- MassM:
                
                self.MassM[l_z] = self.wLG_v[l_v] * self.wLG_x[l_x]
                
                # --- phi_z:
                
                q_z = 0
                for q_x in range( self.n_x ):
                    for q_v in range( self.n_v ):
                        
                        self.phi_z[q_z,l_z] \
                        = LagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                        * LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                        
                        self.dphidv_z[q_z,l_z] \
                        = dLagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                        *  LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                        
                        self.dphidx_z[q_z,l_z] \
                        =  LagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                        * dLagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                        
                        q_z += 1
                        
                # --- phi_v_dn, phi_v_up:
                
                for q_x in range( self.n_x ):
                    
                    self.phi_v_dn[q_x,l_z] \
                    = LagrangeP( [ - 0.5 ]         , l_v, self.pLG_v, self.nDOFv ) \
                    * LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                    
                    self.phi_v_up[q_x,l_z] \
                    = LagrangeP( [ + 0.5 ]         , l_v, self.pLG_v, self.nDOFv ) \
                    * LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                
                # --- phi_x_dn, phi_x_up:
                
                for q_v in range( self.n_v ):
                    
                    self.phi_x_dn[q_v,l_z] \
                    = LagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                    * LagrangeP( [ - 0.5 ]         , l_x, self.pLG_x, self.nDOFx )
                    
                    self.phi_x_up[q_v,l_z] \
                    = LagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                    * LagrangeP( [ + 0.5 ]         , l_x, self.pLG_x, self.nDOFx )
                
                l_z += 1
        
        self.phi_x = np.empty( [ self.n_x, self.nDOFx ], np.float64, 'F' )
        
        for l_x in range( self.nDOFx ):
            for q_x in range( self.n_x ):
                
                self.phi_x[q_x,l_x] = LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
        
        self.phi_x2z = np.empty( [ self.n_z, self.nDOFx ], np.float64, 'F' )
        
        for l_x in range( self.nDOFx ):
            for q_z in range( self.n_z ):
                
                self.phi_x2z[q_z,l_x] = LagrangeP( [ self.pz_q[q_z,self.ix] ], l_x, self.pLG_x, self.nDOFx )
        
        # --- Pre-Assemble ---
        
        self.vM_q = np.zeros( [ self.n_v, BX.w[0] ], np.float64, 'F' )
        self.vP_q = np.zeros( [ self.n_v, BX.w[0] ], np.float64, 'F' )
        
        for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
            
            v_q = v_c[i_v] + dv[i_v] * self.pv_q
            
            self.vM_q[:,i_v] = 0.5 * ( v_q - np.abs( v_q ) )
            self.vP_q[:,i_v] = 0.5 * ( v_q + np.abs( v_q ) )
        
        self.v_q  = np.zeros( [ self.n_z, BX.w[0] ], np.float64, 'F' )
        
        for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
            
            self.v_q[:,i_v] = v_c[i_v] + dv[i_v] * self.pz_q[:,self.iv]
        
        self.SA_X = np.zeros( [ self.nDOFz, self.nDOFz, BX.w[0] ], np.float, 'F' ) # --- Multiplies f_i-1j
        self.SB_X = np.zeros( [ self.nDOFz, self.nDOFz, BX.w[0] ], np.float, 'F' ) # --- Multiplies f_ij
        self.SC_X = np.zeros( [ self.nDOFz, self.nDOFz, BX.w[0] ], np.float, 'F' ) # --- Multiplies f_1+1j
        
        for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
            
            for l_z in range( self.nDOFz ):
                for m_z in range( self.nDOFz ):
                    
                    self.SA_X[l_z,m_z,i_v] \
                    = - dv[i_v] * np.sum( self.wv_q * self.vP_q[:,i_v] * self.phi_x_dn[:,l_z] * self.phi_x_up[:,m_z] )
                    
                    self.SB_X[l_z,m_z,i_v] \
                    = dv[i_v] * ( np.sum( self.wv_q * self.vP_q[:,i_v] * self.phi_x_up[:,l_z] * self.phi_x_up[:,m_z] ) \
                                - np.sum( self.wv_q * self.vM_q[:,i_v] * self.phi_x_dn[:,l_z] * self.phi_x_dn[:,m_z] ) )
                    
                    self.SC_X[l_z,m_z,i_v] \
                    = dv[i_v] * np.sum( self.wv_q * self.vM_q[:,i_v] * self.phi_x_up[:,l_z] * self.phi_x_dn[:,m_z] )
        
        self.MV_X = np.zeros( [ self.nDOFz, self.nDOFz, BX.w[0] ], np.float, 'F' )
        
        for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
            
            for l_z in range( self.nDOFz ):
                for m_z in range( self.nDOFz ):
                    
                    self.MV_X[l_z,m_z,i_v] \
                    = - dv[i_v] * np.sum( self.wz_q * self.v_q[:,i_v] * self.dphidx_z[:,l_z] * self.phi_z[:,m_z] )
        
        # --- For Positivity Limiter ---
        
        self.nPP = self.n_z + 2 * self.n_x + 2 * self.n_v
        
        self.zPP_q = np.empty( [ self.nPP, 2 ], np.float64, 'F' )
        
        # --- Positive Points ---
        
        self.zPP_q[0:self.n_z,self.iv] = self.pz_q[:,self.iv]
        self.zPP_q[0:self.n_z,self.ix] = self.pz_q[:,self.ix]
        
        os_1 = self.n_z
        os_2 = os_1 + self.n_v
        for q_v in range( self.n_v ):
            
            self.zPP_q[os_1+q_v,self.iv] = self.pz_q[q_v,self.iv]
            self.zPP_q[os_1+q_v,self.ix] = - 0.5
            self.zPP_q[os_2+q_v,self.iv] = self.pz_q[q_v,self.iv]
            self.zPP_q[os_2+q_v,self.ix] = + 0.5
        
        os_1 = self.n_z + 2 * self.n_v
        os_2 = os_1 + self.n_x
        for q_x in range( self.n_x ):
            
            self.zPP_q[os_1+q_x,self.iv] = - 0.5
            self.zPP_q[os_1+q_x,self.ix] = self.pz_q[q_x,self.ix]
            self.zPP_q[os_2+q_x,self.iv] = + 0.5
            self.zPP_q[os_2+q_x,self.ix] = self.pz_q[q_x,self.ix]
        
        # --- Interpolation Matrix ---
        
        self.phi_PP = np.empty( [ self.nPP, self.nDOFz ], np.float64, 'F' )
        
        l_z = 0
        for l_x in range( self.nDOFx ):
            for l_v in range( self.nDOFv ):
                
                for qPP in range( self.nPP ):
                    
                    self.phi_PP[qPP,l_z] \
                    = LagrangeP( [ self.zPP_q[qPP,self.iv] ], l_v, self.pLG_v, self.nDOFv ) \
                    * LagrangeP( [ self.zPP_q[qPP,self.ix] ], l_x, self.pLG_x, self.nDOFx )
                
                l_z += 1
        
        # --- For Cleaning Micro Distribution ---
        
        self.wVec = np.zeros( [    self.nDOFv * BX.nx[0] ], np.float64, 'F' )
        self.eVec = np.zeros( [ 3, self.nDOFv * BX.nx[0] ], np.float64, 'F' )
        
        i = 0
        for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
            for q_v in range( self.nDOFv ):
                
                v_q = v_c[i_v] + dv[i_v] * self.pLG_v[q_v]
                
                self.wVec[i] = np.sqrt( self.wLG_v[q_v] )
                
                self.eVec[0,i] = self.wLG_v[q_v] * dv[i_v]
                self.eVec[1,i] = self.wLG_v[q_v] * dv[i_v] * v_q
                self.eVec[2,i] = self.wLG_v[q_v] * dv[i_v] * 0.5 * v_q**2
                
                i += 1
    
    def ComputeBLF( self, BX, dv, v_c, dx, x_c, E_h, f_h, B_h, MultiplyInverseMass = False ):
        
        print( "---------------------------" )
        print( "--- ComputeBLF (Vlasov) ---" )
        print( "---------------------------" )

        timerBC   = 0.0
        timerSX   = 0.0
        timerVX   = 0.0
        timerSV   = 0.0
        timerVV   = 0.0
        timerMM   = 0.0
        
        timerT = timer.time()
        
        # --- Boundary Conditions ---
        
        timeStart = timer.time()
        
        BC.ApplyBC_Z( BX, f_h )
        
        timerBC += timer.time() - timeStart
        
        # --- Initialize to Zero ---
        
        B_h[:,:,:] = 0.0
        
        # --- Position Space Fluxes ---
        
        # --- Surface Term ---
        
        timeStart = timer.time()
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                B_h[:,i_v,i_x] += self.SA_X[:,:,i_v].dot( f_h[:,i_v,i_x-1] )                
                B_h[:,i_v,i_x] += self.SB_X[:,:,i_v].dot( f_h[:,i_v,i_x  ] )                
                B_h[:,i_v,i_x] += self.SC_X[:,:,i_v].dot( f_h[:,i_v,i_x+1] )
        
        timerSX += timer.time() - timeStart
        
        # --- Volume Term ---
        
        timeStart = timer.time()
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                B_h[:,i_v,i_x] += self.MV_X[:,:,i_v].dot( f_h[:,i_v,i_x] )
        
        timerVX += timer.time() - timeStart
        
        # --- Velocity Space Fluxes ---
        
        # --- Surface Term ---
        
        timeStart = timer.time()
        
        if( self.UseGLF_Flux ):
            
            # --- Global Lax-Friedrichs ---
            
            MaxAbsE_q  = np.ones( ( self.nDOFx ), np.float64, 'F' ) * np.amax( np.abs( E_h ) )
            
            wMaxAbsE_q = self.wx_q * self.phi_x.dot( MaxAbsE_q )
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            wE_q = self.wx_q * self.phi_x.dot( E_h[:,i_x] )
            
            if( self.UseGLF_Flux ):
                
                # --- Global Lax-Friedrichs ---
                
                wEM_q = 0.5 * ( wE_q - wMaxAbsE_q )
                wEP_q = 0.5 * ( wE_q + wMaxAbsE_q )
            
            else:
                
                wEM_q = 0.5 * ( wE_q - np.abs( wE_q ) )
                wEP_q = 0.5 * ( wE_q + np.abs( wE_q ) )
            
            Mat_A  = dx[i_x] * ( ( wEP_q.reshape(self.n_x,1) * self.phi_v_dn ).T @ self.phi_v_up )
            Mat_B1 = dx[i_x] * ( ( wEP_q.reshape(self.n_x,1) * self.phi_v_up ).T @ self.phi_v_up )
            Mat_B2 = dx[i_x] * ( ( wEM_q.reshape(self.n_x,1) * self.phi_v_dn ).T @ self.phi_v_dn )
            Mat_C  = dx[i_x] * ( ( wEM_q.reshape(self.n_x,1) * self.phi_v_up ).T @ self.phi_v_dn )
            
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                if i_v == BX.lo[0] and BX.bc[0] == 0:
                    weight_lo = 0.0 # To Enforce Zero Flux at Inner Boundary in v (Set to 1.0 to bypass)
                else:
                    weight_lo = 1.0
                
                if i_v == BX.hi[0] and BX.bc[0] == 0:
                    weight_hi = 0.0 # To Enforce Zero Flux at Outer Boundary in v (Set to 1.0 to bypass)
                else:
                    weight_hi = 1.0
                
                B_h[:,i_v,i_x] -= weight_lo * Mat_A @ f_h[:,i_v-1,i_x]
                B_h[:,i_v,i_x] += ( weight_hi * Mat_B1 - weight_lo * Mat_B2 ) @ f_h[:,i_v  ,i_x]
                B_h[:,i_v,i_x] += weight_hi * Mat_C @ f_h[:,i_v+1,i_x]
        
        timerSV += timer.time() - timeStart
        
        # --- Volume Term ---
        
        timeStart = timer.time()
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            wE_q = self.wz_q * self.phi_x2z.dot( E_h[:,i_x] )
            
            Mat_A = dx[i_x] * ( ( wE_q.reshape(self.n_z,1) * self.dphidv_z ).T @ self.phi_z )
            
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                B_h[:,i_v,i_x] -= Mat_A @ f_h[:,i_v,i_x]
        
        timerVV += timer.time() - timeStart
        
        timeStart = timer.time()
        
        if( MultiplyInverseMass ):
            
            for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
                for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                    
                    B_h[:,i_v,i_x] = B_h[:,i_v,i_x] / ( dv[i_v] * dx[i_x] * self.MassM[:] )
        
        timerMM += timer.time() - timeStart
        
        timerT = timer.time() - timerT
        
        print( "          ")
        print( "timerT  = ", timerT  )
        print( "timerBC = ", timerBC )
        print( "timerSX = ", timerSX )
        print( "timerVX = ", timerVX )
        print( "timerSV = ", timerSV )
        print( "timerVV = ", timerVV )
        print( "timerMM = ", timerMM )
        print( "    Sum = ", timerBC + timerSX + timerVX + timerSV + timerVV + timerMM )
    
    def ApplyPositivityLimiter( self, BX, f_h ):
        
        if( not self.UsePositivityLimiter ):
            
            return
        
        m = 0.0 # --- Minimum Value Allowed ---
        
        # --- Evaluate Distribution in Positive Points ---
        
        f_PP = np.reshape( self.phi_PP @ np.reshape( f_h, ( self.nDOFz, BX.w[0] * BX.w[1] ), order='F' ), ( self.nPP, BX.w[0], BX.w[1] ), order='F' )
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                if( np.any( f_PP[:,i_v,i_x] < m ) ):
                    
                    # --- Cell Average ---
                    
                    f_K = np.sum( self.wLG_z[:] * f_h[:,i_v,i_x] )
                    
                    # --- Minimum Value ---
                    
                    f_m = np.min( f_PP[:,i_v,i_x] )

                    # --- Limiter Parameter ---
                    
                    Theta = min( abs( ( m - f_K ) / ( f_m - f_K ) ), 1.0 )
                    
                    # --- Limit Towards Cell Average ---
                    
                    f_h[:,i_v,i_x] = ( 1.0 - Theta ) * f_K + Theta * f_h[:,i_v,i_x]
    
    def CleanMicroDistribution( self, BX, g_h, ForceCleaning = False ):
        
        if( not self.UseMicroCleaning and not ForceCleaning ):
            
            return
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            g_q = np.reshape( g_h[:,BX.lo[0]:BX.hi[0]+1,i_x], ( self.nDOFv, self.nDOFx, BX.nx[0] ) , order = 'F' )
            g_q = np.reshape( np.swapaxes( g_q, 1, 2 )      , ( self.nDOFv * BX.nx[0], self.nDOFx ), order = 'F' )

            for q_x in range( self.nDOFx ):
                
                g_q[:,q_x] = CleanMicroDistribution_LeastSquares( g_q[:,q_x], self.wVec, self.eVec )
            
            g_q = np.swapaxes( np.reshape( g_q, ( self.nDOFv, BX.nx[0], self.nDOFx ), order = 'F' ), 1, 2 )
            g_h[:,BX.lo[0]:BX.hi[0]+1,i_x] = np.reshape( g_q, ( self.nDOFz, BX.nx[0] ), order = 'F' )
        
def CleanMicroDistribution_LeastSquares( g, wVec, eVec ):
    
    N = g.size
    
    A = np.identity( N, np.float64 ) * wVec
    c = np.reshape( wVec * g, (N,1), order = 'F' )
    
    B = eVec.copy()
    d = np.zeros( (3,1), np.float64, order = 'F' )
    
#    print( "Before: ", np.transpose( B.dot( c ) ) )
    
    t, r, res, x, info = sp.linalg.lapack.dgglse( A, B, c, d )
    
#    print( "After:  ", np.transpose( B.dot( x ) ) )
    
    g_cleaned = x
    
    return g_cleaned