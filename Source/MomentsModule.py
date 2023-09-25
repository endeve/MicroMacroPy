
# --- VelocityMoments ---

import numpy as np

from LagrangeModule import LagrangeP

class VelocityMoments:
    
    def __init__( self ):
        
        self.iv = 0
        self.ix = 1
        
    def Initialize( self, BX_v, dv, v_c, nDOFv, nDOFx, n_v, n_x ):
        
        self.nDOFv = nDOFv
        self.nDOFx = nDOFx
        self.nDOFz = nDOFv * nDOFx
        
        self.pLG_v, dummy = np.polynomial.legendre.leggauss( self.nDOFv )
        self.pLG_v = self.pLG_v / 2.0 # --- Unit Interval
        
        self.pLG_x, self.wLG_x = np.polynomial.legendre.leggauss( self.nDOFx )
        self.pLG_x = self.pLG_x / 2.0 # --- Unit Interval
        self.wLG_x = self.wLG_x / 2.0 # --- Unit Interval
        
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
        
        self.w_phi_x = np.empty( [ self.n_z, self.nDOFx ], np.float64, 'F' )
        
        for l_x in range( self.nDOFx ):
            
            self.w_phi_x[:,l_x] \
            = self.wz_q[:] * LagrangeP( self.pz_q[:,self.ix], l_x, self.pLG_x, self.nDOFx )
        
        self.e1_q = np.empty( [ self.n_z, BX_v.w[0] ], np.float64, 'F' )
        self.e2_q = np.empty( [ self.n_z, BX_v.w[0] ], np.float64, 'F' )
        
        for i_v in range( BX_v.lo[0], BX_v.hi[0] + 1 ):
            
            self.e1_q[:,i_v] = ( v_c[i_v] + dv[i_v] * self.pz_q[:,self.iv] )
            self.e2_q[:,i_v] = ( v_c[i_v] + dv[i_v] * self.pz_q[:,self.iv] )**2

    def ComputeVelocityMoments( self, BX, dv, f_h, rho0_h, rho1_h, rho2_h ):

        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            rho0_h[:,i_x] = 0.0
            rho1_h[:,i_x] = 0.0
            rho2_h[:,i_x] = 0.0
            
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                f_q  = self.phi_z.dot( f_h[:,i_v,i_x] )
                
                rho0_h[:,i_x] += dv[i_v] * np.transpose( self.w_phi_x ).dot(                    f_q )
                rho1_h[:,i_x] += dv[i_v] * np.transpose( self.w_phi_x ).dot( self.e1_q[:,i_v] * f_q )
                rho2_h[:,i_x] += dv[i_v] * np.transpose( self.w_phi_x ).dot( self.e2_q[:,i_v] * f_q )
            
            rho0_h[:,i_x] = rho0_h[:,i_x] / self.wLG_x[:]
            rho1_h[:,i_x] = rho1_h[:,i_x] / self.wLG_x[:]
            rho2_h[:,i_x] = rho2_h[:,i_x] / self.wLG_x[:]
    
    def ComputeVelocityMoment_0( self, BX, dv, f_h, rho0_h ):
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            rho0_h[:,i_x] = 0.0
            
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                f_q = self.phi_z.dot( f_h[:,i_v,i_x] )
                
                rho0_h[:,i_x] += dv[i_v] * np.transpose( self.w_phi_x ).dot( f_q )
            
            rho0_h[:,i_x] = rho0_h[:,i_x] / self.wLG_x[:]
    
    def ComputeVelocityMoments_DUT( self, BX, dv, f_h, D_h, U_h, T_h ):
        
        rho0_h = np.empty_like( D_h )
        rho1_h = np.empty_like( D_h )
        rho2_h = np.empty_like( D_h )
        
        self.ComputeVelocityMoments( BX, dv, f_h, rho0_h, rho1_h, rho2_h )
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            D_h[:,i_x] = rho0_h[:,i_x]
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            U_h[:,i_x] = rho1_h[:,i_x] / D_h[:,i_x]
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            T_h[:,i_x] = ( rho2_h[:,i_x] - U_h[:,i_x] * rho1_h[:,i_x] ) / D_h[:,i_x]
    
    def ComputeVelocityMoments_DSG( self, BX, dv, f_h, D_h, S_h, G_h ):
        
        rho0_h = np.empty_like( D_h )
        rho1_h = np.empty_like( D_h )
        rho2_h = np.empty_like( D_h )
        
        self.ComputeVelocityMoments( BX, dv, f_h, rho0_h, rho1_h, rho2_h )
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            D_h[:,i_x] = rho0_h[:,i_x]
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            S_h[:,i_x] = rho1_h[:,i_x]
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            
            G_h[:,i_x] = 0.5 * rho2_h[:,i_x]