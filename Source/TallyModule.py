
# --- Tally Module ---

import numpy as np
import h5py  as h5

from LagrangeModule import LagrangeP

class Tally:
    
    def __init__( self ):
        
        print( "Initializing Tally" )
    
    def Initialize( self, BX, dv, v_c, dx, x_c, nDOFv, nDOFx, n_v, n_x ):
        
        self.nDOFv = nDOFv
        self.nDOFx = nDOFx
        self.nDOFz = nDOFv * nDOFx
        
        self.pLG_v, dummy = np.polynomial.legendre.leggauss( self.nDOFv )
        self.pLG_v = self.pLG_v / 2.0 # --- Unit Interval
        
        self.pLG_x, dummy = np.polynomial.legendre.leggauss( self.nDOFx )
        self.pLG_x = self.pLG_x / 2.0 # --- Unit Interval
        
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
        
        self.e_0 = np.ones ( ( self.n_z          ), np.float64, 'F' )
        self.e_1 = np.empty( ( self.n_z, BX.w[0] ), np.float64, 'F' )
        self.e_2 = np.empty( ( self.n_z, BX.w[0] ), np.float64, 'F' )

        for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
            
            q_z = 0
            for q_x in range( self.n_x ):
                for q_v in range( self.n_v ):
                    
                    v_q = v_c[i_v] + dv[i_v] * self.pz_q[q_v,0]
                    
                    self.e_1[q_z,i_v] = v_q
                    self.e_2[q_z,i_v] = 0.5 * v_q**2
                    
                    q_z += 1
        
        self.E_0 = np.empty( ( self.nDOFz, BX.w[0], BX.w[1] ), np.float64, 'F' )
        self.E_1 = np.empty( ( self.nDOFz, BX.w[0], BX.w[1] ), np.float64, 'F' )
        self.E_2 = np.empty( ( self.nDOFz, BX.w[0], BX.w[1] ), np.float64, 'F' )
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                for l_z in range( self.nDOFz ):
                    
                    self.E_0[l_z,i_v,i_x] = dv[i_v] * dx[i_x] * np.sum( self.wz_q[:] * self.phi_z[:,l_z] * self.e_0[:] )
                    self.E_1[l_z,i_v,i_x] = dv[i_v] * dx[i_x] * np.sum( self.wz_q[:] * self.phi_z[:,l_z] * self.e_1[:,i_v] )
                    self.E_2[l_z,i_v,i_x] = dv[i_v] * dx[i_x] * np.sum( self.wz_q[:] * self.phi_z[:,l_z] * self.e_2[:,i_v] )
        
        self.ArrayT = np.zeros( 0, np.float64, 'F' )
        self.ArrayD = np.zeros( 0, np.float64, 'F' )
        self.ArrayS = np.zeros( 0, np.float64, 'F' )
        self.ArrayG = np.zeros( 0, np.float64, 'F' )
        
    def Compute( self, t, BX, f ):
        
        D = 0.0
        S = 0.0
        G = 0.0
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                D += np.sum( self.E_0[:,i_v,i_x] * f[:,i_v,i_x] )
                S += np.sum( self.E_1[:,i_v,i_x] * f[:,i_v,i_x] )
                G += np.sum( self.E_2[:,i_v,i_x] * f[:,i_v,i_x] )
        
        self.ArrayT = np.append( self.ArrayT, [ t ] )
        self.ArrayD = np.append( self.ArrayD, [ D ] )
        self.ArrayS = np.append( self.ArrayS, [ S ] )
        self.ArrayG = np.append( self.ArrayG, [ G ] )
    
    def Write( self, Name ):
        
        FileName = Name + '_Tally.h5'
        
        fH5 = h5.File( FileName, 'w' )
        
        # --- Time ---
        
        dset = fH5.create_dataset( "Time", self.ArrayT.shape, 'double' )
        
        dset.write_direct( self.ArrayT )
        
        # --- Particle Number ---
        
        dset = fH5.create_dataset( "Particle Number"  , self.ArrayD.shape, 'double' )
        
        dset.write_direct( self.ArrayD )
        
        # --- Particle Momentum ---
        
        dset = fH5.create_dataset( "Particle Momentum", self.ArrayS.shape, 'double' )
        
        dset.write_direct( self.ArrayS )
        
        # --- Particle Energy ---
        
        dset = fH5.create_dataset( "Particle Energy"  , self.ArrayG.shape, 'double' )
        
        dset.write_direct( self.ArrayG )
        
        fH5.close()