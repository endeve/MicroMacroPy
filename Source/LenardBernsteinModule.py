
# --- Lenard-Bernstein Module ---

import numpy as np
import time as timer

from LagrangeModule import LagrangeP, dLagrangeP, ddLagrangeP

class LenardBernsteinSolver:
    
    def __init__( self ):
        
        self.iv = 0
        self.ix = 1
        
        self.CollisionFrequency = 1.0
    
    def Initialize( self, CollisionFrequency, BX, dv, v_c, nDOFv, nDOFx, n_v, n_x ):
        
        self.CollisionFrequency = CollisionFrequency
        
        self.nDOFv = nDOFv
        self.nDOFx = nDOFx
        self.nDOFz = nDOFv * nDOFx
        
        self.nDOFV = 2 * self.nDOFv
        self.nDOFZ = 2 * self.nDOFz
        
        # --- Lagrange Polynomials Basis (v) ---
        
        self.pLG_v, self.wLG_v = np.polynomial.legendre.leggauss( self.nDOFv )
        self.pLG_v = self.pLG_v / 2.0 # --- Unit Interval
        self.wLG_v = self.wLG_v / 2.0 # --- Unit Interval
        
        # --- Lagrange Polynomial Basis (x) ---
        
        self.pLG_x, self.wLG_x = np.polynomial.legendre.leggauss( self.nDOFx )
        self.pLG_x = self.pLG_x / 2.0 # --- Unit Interval
        self.wLG_x = self.wLG_x / 2.0 # --- Unit Interval
        
        # --- Recovery Polynomial Basis (v) ---
        
        self.pLG_V, self.wLG_V = np.polynomial.legendre.leggauss( self.nDOFV )
        self.pLG_V = self.pLG_V / 2.0 # --- Unit Interval
        self.wLG_V = self.wLG_V / 2.0 # --- Unit Interval
        
        # --- Quadrature (v) ---
        
        self.n_v = n_v
        
        self.pv_q, self.wv_q = np.polynomial.legendre.leggauss( self.n_v )
        self.pv_q = self.pv_q / 2.0 # --- Unit Interval
        self.wv_q = self.wv_q / 2.0 # --- Unit Interval
        
        # --- Quadrature (x) ---
        
        self.n_x = n_x
        
        self.px_q, self.wx_q = np.polynomial.legendre.leggauss( self.n_x )
        self.px_q = self.px_q / 2.0 # --- Unit Interval
        self.wx_q = self.wx_q / 2.0 # --- Unit Interval
        
        # --- Quadrature for Recovery Polynomial (v) ---
        
        self.n_V = 2 * self.n_v
        
        self.pV_q, self.wV_q = np.polynomial.legendre.leggauss( self.n_V )
        self.pV_q = self.pV_q / 2.0 # --- Unit Interval
        self.wV_q = self.wV_q / 2.0 # --- Unit Interval
        
        # --- Quadrature (z) ---
        
        self.n_z = self.n_v * self.n_x
        
        self.pz_q = np.empty( [ self.n_z, 2 ], np.float64, 'F' )
        self.wz_q = np.empty( [ self.n_z    ], np.float64, 'F' )
        
        i_z = 0
        for i_x in range( self.n_x ):
            for i_v in range( self.n_v ):
                
                self.pz_q[i_z,self.iv] = self.pv_q[i_v]
                self.pz_q[i_z,self.ix] = self.px_q[i_x]
                
                self.wz_q[i_z] = self.wv_q[i_v] * self.wx_q[i_x]
                
                i_z += 1
        
        # --- Mass Matrix (Diagonal) ---
        
        self.MassM = np.empty( [ self.nDOFz ], np.float64, 'F' )
        
        l_z = 0
        for l_x in range( self.nDOFx ):
            for l_v in range( self.nDOFv ):
                
                self.MassM[l_z] = self.wLG_v[l_v] * self.wLG_x[l_x]
                
                l_z += 1
        
        # --- phi_z, dphidv_z, ddphidv_z ---
        
        self.phi_z     = np.empty( [ self.n_z, self.nDOFz ], np.float64, 'F' )
        self.dphidv_z  = np.empty( [ self.n_z, self.nDOFz ], np.float64, 'F' )
        self.ddphidv_z = np.empty( [ self.n_z, self.nDOFz ], np.float64, 'F' )
        
        l_z = 0
        for l_x in range( self.nDOFx ):
            for l_v in range( self.nDOFv ):
                
                q_z = 0
                for q_x in range( self.n_x ):
                    for q_v in range( self.n_v ):
                        
                        self.phi_z[q_z,l_z] \
                        = LagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                        * LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                        
                        self.dphidv_z[q_z,l_z] \
                        = dLagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                        *  LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                        
                        self.ddphidv_z[q_z,l_z] \
                        = ddLagrangeP( [ self.pv_q[q_v] ], l_v, self.pLG_v, self.nDOFv ) \
                        *   LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                        
                        q_z += 1
                
                l_z += 1
        
        
        # --- phi_v_dn, phi_v_up, dphidv_v_dn, dphidv_v_up ---
        
        self.phi_v_dn    = np.empty( [ self.n_x, self.nDOFz ], np.float64, 'F' )
        self.phi_v_up    = np.empty( [ self.n_x, self.nDOFz ], np.float64, 'F' )
        self.dphidv_v_dn = np.empty( [ self.n_x, self.nDOFz ], np.float64, 'F' )
        self.dphidv_v_up = np.empty( [ self.n_x, self.nDOFz ], np.float64, 'F' )
        
        l_z = 0
        for l_x in range( self.nDOFx ):
            for l_v in range( self.nDOFv ):
                
                for q_x in range( self.n_x ):
                    
                    self.phi_v_dn[q_x,l_z] \
                    = LagrangeP( [ - 0.5 ]         , l_v, self.pLG_v, self.nDOFv ) \
                    * LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                    
                    self.phi_v_up[q_x,l_z] \
                    = LagrangeP( [ + 0.5 ]         , l_v, self.pLG_v, self.nDOFv ) \
                    * LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                    
                    self.dphidv_v_dn[q_x,l_z] \
                    = dLagrangeP( [ - 0.5 ]         , l_v, self.pLG_v, self.nDOFv ) \
                    *  LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                    
                    self.dphidv_v_up[q_x,l_z] \
                    = dLagrangeP( [ + 0.5 ]         , l_v, self.pLG_v, self.nDOFv ) \
                    *  LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
                
                l_z += 1
        
        # --- phi_x ---
        
        self.phi_x = np.empty( [ self.n_x, self.nDOFx ], np.float64, 'F' )
        
        for l_x in range( self.nDOFx ):
            for q_x in range( self.n_x ):
                
                self.phi_x[q_x,l_x] = LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x, self.nDOFx )
        
        # --- phi_x2z ---
        
        self.phi_x2z = np.empty( [ self.n_z, self.nDOFx ], np.float64, 'F' )
        
        for l_x in range( self.nDOFx ):
            for q_z in range( self.n_z ):
                
                self.phi_x2z[q_z,l_x] = LagrangeP( [ self.pz_q[q_z,self.ix] ], l_x, self.pLG_x, self.nDOFx )
        
        # --- Matrices for Recovery Polynomial ---
        
        self.m_Matrix = np.zeros( [ self.nDOFZ, self.nDOFZ ], np.float64, 'F' )
        self.M_Matrix = np.zeros( [ self.nDOFZ, self.nDOFZ ], np.float64, 'F' )
        
        l_z = 0
        l_o = self.nDOFz # --- Offset ---
        for l_x in range( self.nDOFx ):
            for l_v in range( self.nDOFv ):
                
                self.m_Matrix[    l_z,    l_z] = self.MassM[l_z]
                self.m_Matrix[l_o+l_z,l_o+l_z] = self.MassM[l_z]
                
                m_z = 0
                for m_x in range( self.nDOFx ):
                    for m_v in range( self.nDOFV ):
                        
                        for q_x in range( self.n_x ):
                            for q_V in range( self.n_V ):
                                
                                self.M_Matrix[    l_z,m_z] \
                                += self.wx_q[q_x] * self.wV_q[q_V] \
                                * LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x[:], self.nDOFx ) \
                                * LagrangeP( [ self.px_q[q_x] ], m_x, self.pLG_x[:], self.nDOFx ) \
                                * LagrangeP( [ self.pV_q[q_V] ], l_v, self.pLG_v[:], self.nDOFv ) \
                                * LagrangeP( [ - 0.25 * ( 1.0 - 2.0 * self.pV_q[q_V] ) ], m_v, self.pLG_V[:], self.nDOFV )
                                
                                self.M_Matrix[l_o+l_z,m_z] \
                                += self.wx_q[q_x] * self.wV_q[q_V] \
                                * LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x[:], self.nDOFx ) \
                                * LagrangeP( [ self.px_q[q_x] ], m_x, self.pLG_x[:], self.nDOFx ) \
                                * LagrangeP( [ self.pV_q[q_V] ], l_v, self.pLG_v[:], self.nDOFv ) \
                                * LagrangeP( [ + 0.25 * ( 1.0 + 2.0 * self.pV_q[q_V] ) ], m_v, self.pLG_V[:], self.nDOFV )
                        
                        m_z += 1
                
                l_z += 1
        
        self.R_Matrix = np.linalg.inv( self.M_Matrix ).dot( self.m_Matrix )
        
        self.Psi_Matrix  = np.zeros( [ self.n_x, self.nDOFZ ], np.float64, 'F' )
        self.dPsi_Matrix = np.zeros( [ self.n_x, self.nDOFZ ], np.float64, 'F' )
        
        l_z = 0
        for l_x in range( self.nDOFx ):
            for l_v in range( self.nDOFV ):
                for q_x in range( self.n_x ):
                    
                    self.Psi_Matrix[q_x,l_z] \
                    = LagrangeP( [ self.px_q[q_x] ], l_x, self.pLG_x[:], self.nDOFx ) \
                    * LagrangeP( [ 0.0 ]           , l_v, self.pLG_V[:], self.nDOFV )
                    
                    self.dPsi_Matrix[q_x,l_z] \
                    = LagrangeP ( [ self.px_q[q_x] ], l_x, self.pLG_x[:], self.nDOFx ) \
                    * dLagrangeP( [ 0.0 ]           , l_v, self.pLG_V[:], self.nDOFV )
                    
                l_z += 1
        
        self. C_Matrix = self. Psi_Matrix.dot( self.R_Matrix )
        self.dC_Matrix = self.dPsi_Matrix.dot( self.R_Matrix ) / 2.0 # --- Divide by to since "recovery element" is double the DG element
    
    def ComputeBLF( self, dt, BX, dv, v_c, dx, x_c, u_h, T_h, f_h, B_h, MultiplyInverseMass = False ):
        
        print( "-----------------------" )
        print( "--- ComputeBLF (LB) ---" )
        print( "-----------------------" )
        
        TimerAlloc     = 0.0
        TimerLoadB     = 0.0
        TimerLoadA     = 0.0
        TimerLoadA_Drf = 0.0
        TimerLoadA_Dif = 0.0
        TimerLoadM     = 0.0
        TimerSolve     = 0.0
        TimerFormBLF   = 0.0
        TimerUnpackBLF = 0.0
        TimerMM        = 0.0
        
        TimerT = timer.time()
        
        tau = dt * self.CollisionFrequency
        
        B_h[:,:,:] = 0.0

        timeStart = timer.time()

        # --- Drift Term Matrices ---
        
        c_Mat = np.zeros( [ self.nDOFz, self.nDOFz ], np.float64, 'F' )
        a_Mat = np.zeros( [ self.nDOFz, self.nDOFz ], np.float64, 'F' )
        b_Mat = np.zeros( [ self.nDOFz, self.nDOFz ], np.float64, 'F' )
        s_Mat = np.zeros( [ self.nDOFz, self.nDOFz ], np.float64, 'F' )
        
        # --- Diffusion Term Matrices ---
        
        dc_Up_Mat = np.zeros( [ self.nDOFz, self.nDOFZ ], np.float64, 'F' )
        dc_Dn_Mat = np.zeros( [ self.nDOFz, self.nDOFZ ], np.float64, 'F' )
        cc_Up_Mat = np.zeros( [ self.nDOFz, self.nDOFZ ], np.float64, 'F' )
        cc_Dn_Mat = np.zeros( [ self.nDOFz, self.nDOFZ ], np.float64, 'F' )
        
        # --- Global Matrix and Solution Vector ---
        
        self.A_Mat  = np.zeros( [ self.nDOFz * BX.nx[0], self.nDOFz * BX.nx[0], BX.w[1] ], np.float64, 'F' )
        self.M_Mat  = np.zeros( [ self.nDOFz * BX.nx[0], self.nDOFz * BX.nx[0], BX.w[1] ], np.float64, 'F' )
        self.Mb_Vec = np.zeros( [ self.nDOFz * BX.nx[0]                       , BX.w[1] ], np.float64, 'F' )
        self.b_Vec  = np.zeros( [ self.nDOFz * BX.nx[0]                       , BX.w[1] ], np.float64, 'F' )
        self.B_Vec  = np.zeros( [ self.nDOFz * BX.nx[0]                       , BX.w[1] ], np.float64, 'F' )
        
        TimerAlloc += timer.time() - timeStart
        
        # --- Load b_Vec ---
        
        timeStart = timer.time()
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                i_z_lo = (i_v-BX.ng[0]  ) * self.nDOFz
                i_z_hi = (i_v-BX.ng[0]+1) * self.nDOFz
                
                self.Mb_Vec[i_z_lo:i_z_hi,i_x] \
                = dv[i_v] * dx[i_x] * self.MassM[:] * f_h[:,i_v,i_x]
        
        TimerLoadB += timer.time() - timeStart
        
        # --- Load A_Mat ---
        
        timeStart = timer.time()
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                if i_v == BX.lo[0]:
                    weight_lo = 0.0 # To Enforce Zero Flux at Inner Boundary in v
                else:
                    weight_lo = 1.0
                
                if i_v == BX.hi[0]:
                    weight_hi = 0.0 # To Enforce Zero Flux at Outer Boundary in v
                else:
                    weight_hi = 1.0
                
                
                # --- Drift Term ---
                
                w_Dn = self.phi_x.dot( u_h[:,i_x] ) - ( v_c[i_v] - 0.5 * dv[i_v] )
                
                w_Dn_P = 0.5 * ( w_Dn + np.abs( w_Dn ) )
                w_Dn_M = 0.5 * ( w_Dn - np.abs( w_Dn ) )
                
                w_Up = self.phi_x.dot( u_h[:,i_x] ) - ( v_c[i_v] + 0.5 * dv[i_v] )
                
                w_Up_P = 0.5 * ( w_Up + np.abs( w_Up ) )
                w_Up_M = 0.5 * ( w_Up - np.abs( w_Up ) )
                
                w_h = self.phi_x2z.dot( u_h[:,i_x] ) - ( v_c[i_v] + dv[i_v] * self.pz_q[:,self.iv] )
                
                timeStart_1 = timer.time()

                c_Mat = - dx[i_x] * weight_lo * ( ( self.wx_q * w_Dn_P ).reshape(self.wx_q.size,1) * self.phi_v_dn ).T @ self.phi_v_up
                b_Mat =   dx[i_x] * weight_hi * ( ( self.wx_q * w_Up_M ).reshape(self.wx_q.size,1) * self.phi_v_up ).T @ self.phi_v_dn
                a_Mat =   dx[i_x] * weight_hi * ( ( self.wx_q * w_Up_P ).reshape(self.wx_q.size,1) * self.phi_v_up ).T @ self.phi_v_up \
                        - dx[i_x] * weight_lo * ( ( self.wx_q * w_Dn_M ).reshape(self.wx_q.size,1) * self.phi_v_dn ).T @ self.phi_v_dn
                s_Mat =   dx[i_x] * ( ( self.wz_q * w_h ).reshape(self.wz_q.size,1) * self.dphidv_z ).T @ self.phi_z
                
                i_z_lolo = (i_v-BX.ng[0]-1) * self.nDOFz
                i_z_lo   = (i_v-BX.ng[0]  ) * self.nDOFz
                i_z_hi   = (i_v-BX.ng[0]+1) * self.nDOFz
                i_z_hihi = (i_v-BX.ng[0]+2) * self.nDOFz
                
                TimerLoadA_Drf += timer.time() - timeStart_1
                
                if( i_v - BX.ng[0] > 0 ):
                    
                    self.A_Mat[i_z_lo:i_z_hi,i_z_lolo:i_z_lo,i_x] += c_Mat
                    
                self.A_Mat[i_z_lo:i_z_hi,i_z_lo:i_z_hi,i_x] += a_Mat - s_Mat
                
                if( i_v - BX.ng[0] + 1 < BX.nx[0] ):
                    
                    self.A_Mat[i_z_lo:i_z_hi,i_z_hi:i_z_hihi,i_x] += b_Mat
                
                # --- Diffusion Term ---
                
                timeStart_1 = timer.time()
                
                wT_q = self.wx_q * self.phi_x.dot( T_h[:,i_x] )
                dxdv = dx[i_x] / dv[i_v]
                
                dc_Up_Mat =   dxdv * ( ( wT_q ).reshape(self.n_x,1) * self.phi_v_up    ).T @ self.dC_Matrix
                dc_Dn_Mat = - dxdv * ( ( wT_q ).reshape(self.n_x,1) * self.phi_v_dn    ).T @ self.dC_Matrix
                cc_Up_Mat = - dxdv * ( ( wT_q ).reshape(self.n_x,1) * self.dphidv_v_up ).T @ self.C_Matrix
                cc_Dn_Mat =   dxdv * ( ( wT_q ).reshape(self.n_x,1) * self.dphidv_v_dn ).T @ self.C_Matrix
                
                wT_q = self.wz_q * self.phi_x2z.dot( T_h[:,i_x] )
                
                s_Mat = dxdv * ( ( wT_q ).reshape(self.n_z,1) * self.ddphidv_z ).T @ self.phi_z
                
                if( i_v - BX.ng[0] > 0 ):
                    
                    self.A_Mat[i_z_lo:i_z_hi,i_z_lolo:i_z_lo,i_x] \
                    -= ( dc_Dn_Mat[0:self.nDOFz,0:self.nDOFz] \
                        + cc_Dn_Mat[0:self.nDOFz,0:self.nDOFz] )
                
                self.A_Mat[i_z_lo:i_z_hi,i_z_lo:i_z_hi,i_x] \
                -= ( dc_Dn_Mat[0:self.nDOFz,self.nDOFz:self.nDOFZ] * weight_lo \
                    + cc_Dn_Mat[0:self.nDOFz,self.nDOFz:self.nDOFZ] * weight_lo \
                    + dc_Up_Mat[0:self.nDOFz,0         :self.nDOFz] * weight_hi \
                    + cc_Up_Mat[0:self.nDOFz,0         :self.nDOFz] * weight_hi \
                    + s_Mat )
                
                if( i_v - BX.ng[0] + 1 < BX.nx[0] ):
                    
                    self.A_Mat[i_z_lo:i_z_hi,i_z_hi:i_z_hihi,i_x] \
                    -= ( dc_Up_Mat[0:self.nDOFz,self.nDOFz:self.nDOFZ] \
                        + cc_Up_Mat[0:self.nDOFz,self.nDOFz:self.nDOFZ] )
                
                TimerLoadA_Dif += timer.time() - timeStart_1
                    
        TimerLoadA += timer.time() - timeStart
        
        # --- Load M_Mat ---
        
        timeStart = timer.time()
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                i_z_o = (i_v-BX.ng[0]) * self.nDOFz
                
                for i_z in range( self.nDOFz ):
                    
                    self.M_Mat[i_z_o+i_z,i_z_o+i_z,i_x] \
                    = dv[i_v] * dx[i_x] * self.MassM[i_z]
        
        TimerLoadM += timer.time() - timeStart
        
        # --- Implicit Solve ---
        
        timeStart = timer.time()
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            self.b_Vec[:,i_x] \
            = np.linalg.solve( self.M_Mat[:,:,i_x] + tau * self.A_Mat[:,:,i_x], self.Mb_Vec[:,i_x] )
        
        TimerSolve += timer.time() - timeStart
        
        # --- Bilinear Form ---
        
        timeStart = timer.time()
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            self.B_Vec[:,i_x] = self.A_Mat[:,:,i_x].dot( self.b_Vec[:,i_x] )
        
        TimerFormBLF += timer.time() - timeStart
        
        # --- Unpack Bilinear Form for Output ---
        
        timeStart = timer.time()
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                i_z_lo = (i_v-BX.ng[0]  ) * self.nDOFz
                i_z_hi = (i_v-BX.ng[0]+1) * self.nDOFz
                
                B_h[:,i_v,i_x] = self.B_Vec[i_z_lo:i_z_hi,i_x]
        
        TimerUnpackBLF += timer.time() - timeStart
        
        timeStart = timer.time()
        
        if( MultiplyInverseMass ):
            
            for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
                for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                    
                    B_h[:,i_v,i_x] = B_h[:,i_v,i_x] / ( dv[i_v] * dx[i_x] * self.MassM[:] )
        
        TimerMM += timer.time() - timeStart
        
        TimerSum = TimerLoadB + TimerLoadA + TimerLoadM + TimerSolve \
                    + TimerFormBLF + TimerUnpackBLF + TimerMM
        
        TimerT = timer.time() - TimerT
        
        print( "                 " )
        print( "TimerAlloc     = ", TimerAlloc )
        print( "TimerLoadB     = ", TimerLoadB )
        print( "TimerLoadA     = ", TimerLoadA )
        print( "TimerLoadA_Drf = ", TimerLoadA_Drf )
        print( "TimerLoadA_Dif = ", TimerLoadA_Dif )
        print( "TimerLoadM     = ", TimerLoadM )
        print( "TimerSolve     = ", TimerSolve )
        print( "TimerFormBLF   = ", TimerFormBLF )
        print( "TimerUnpackBLF = ", TimerUnpackBLF )
        print( "TimerMM        = ", TimerMM )
        print( "  Sum          = ", TimerSum )
        print( "TimerT         = ", TimerT )
        print( "                 " )