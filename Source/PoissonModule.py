
# --- Poisson Module ---

import numpy as np

from LagrangeModule import LagrangeP

class PoissonSolver:
    
    def __init__( self ):
        
        self.iL = 0
        self.iR = 1
        
        self.nDOFx = []
        self.n_x   = []
        
    def Initialize( self, nDOFx, n_x ):
        
        self.nDOFx = nDOFx
        
        self.pLG_x, self.wLG_x = np.polynomial.legendre.leggauss( self.nDOFx )
        self.pLG_x = self.pLG_x / 2.0 # --- Unit Interval
        self.wLG_x = self.wLG_x / 2.0 # --- Unit Interval
        
        self.n_x = n_x
        
        self.px_q, self.wx_q = np.polynomial.legendre.leggauss( self.n_x )
        self.px_q = self.px_q / 2.0 # --- Unit Interval
        self.wx_q = self.wx_q / 2.0 # --- Unit Interval
        
        self.I_HatPhi = np.empty( [ self.nDOFx, 2 ], np.float64, 'F' )
        
        for l_x in range( self.nDOFx ):
            
            self.I_HatPhi[l_x,self.iL] \
            = np.sum( self.wx_q * ( 0.5 + self.px_q ) \
                     * LagrangeP( self.px_q, l_x, self.pLG_x, self.nDOFx ) )
            
            self.I_HatPhi[l_x,self.iR] \
            = np.sum( self.wx_q * ( 0.5 - self.px_q ) \
                     * LagrangeP( self.px_q, l_x, self.pLG_x, self.nDOFx ) )
        
    def Solve( self, BX, dx, S, Phi, Phi_BC = [ 0.0, 0.0 ] ):
                
        L_x = np.sum( dx[BX.lo[0]:BX.hi[0]+1] )
        
        I_S = 0.0
        if( BX.bc[0] == 1 ): # --- Periodic Domain
            for i_x in range( BX.lo[0], BX.hi[0] + 1 ):
                I_S += dx[i_x] * np.sum( self.wLG_x * S[:,i_x] ) / L_x
        
        b = np.zeros( [BX.nx[0]+1]           , np.float64, 'F' ) # --- RHS Vector
        A = np.zeros( [BX.nx[0]+1,BX.nx[0]+1], np.float64, 'F' ) # --- Matrix
        
        for i_x in range( BX.lo[0], BX.hi[0] ):
                        
            b[i_x] =  dx[i_x  ] * self.I_HatPhi[:,self.iL].dot( S[:,i_x  ] - I_S ) \
                    + dx[i_x+1] * self.I_HatPhi[:,self.iR].dot( S[:,i_x+1] - I_S )
            
            A[i_x,i_x  ] = ( 1.0 / dx[i_x] + 1.0 / dx[i_x+1] )
            
            A[i_x,i_x-1] = - 1.0 / dx[i_x]
            
            A[i_x,i_x+1] = - 1.0 / dx[i_x+1]
        
        # --- Boundary Conditions ---
        
        if( BX.bc[0] == 0 ):
            
            # --- Dirichlet ---
            
            b[0       ] = Phi_BC[0]
            b[BX.nx[0]] = Phi_BC[1]
        
            A[0       ,0       ] = 1.0
            A[BX.nx[0],BX.nx[0]] = 1.0
            
        elif( BX.bc[0] == 1 ):
            
            # --- Periodic ---
            
            b[0] = Phi_BC[0]
            
            A[0       ,0       ] =   1.0
            A[BX.nx[0],0       ] = - 1.0
            A[BX.nx[0],BX.nx[0]] =   1.0
            
        else:
            
            # --- Dirichlet (Zero) ---
            
            b[0       ] = 0.0
            b[BX.nx[0]] = 0.0
        
            A[0       ,0       ] = 1.0
            A[BX.nx[0],BX.nx[0]] = 1.0
                
        # --- Solve Linear System ---
        
        Phi[:] = np.linalg.solve( A, b )
    
    def ComputeElectricField( self, BX, dx, Phi, E ):
        
        for i_x in range( BX.lo[0], BX.hi[0] + 1 ):
            
            e_x = i_x - BX.ng[0]
            
            E[:,i_x] = - ( Phi[e_x+1] - Phi[e_x] ) / dx[i_x]