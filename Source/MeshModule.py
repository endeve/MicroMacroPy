
# --- Mesh Module ---

import numpy as np

class Mesh:
    
    def __init__( self ):

        self.nx  = [] # Number of Cells
        self.ng  = [] # Number of Ghost Cells
        self.lo  = [] # First Interior Cell
        self.hi  = [] # Last  Interior Cell
        self.x_l = [] # Left  Cell Edges
        self.x_r = [] # Right Cell Edges
        self.x_c = [] # Cell Centers
        self.dx  = [] # Cell Widths
        self.x_e = [] # Cell Edges

    def CreateMesh( self, nx, ng, x_lo, x_hi ):

        self.nx = nx
        self.ng = ng
        self.lo = ng
        self.hi = nx + ng

        w = nx + 2 * ng

        self.x_l = np.zeros( w,   np.float64 )
        self.x_r = np.zeros( w,   np.float64 )
        self.x_c = np.zeros( w,   np.float64 )
        self.dx  = np.zeros( w,   np.float64 )
        self.x_e = np.zeros( w+1, np.float64 )

        dx = ( x_hi - x_lo ) / nx

        for i in range( w ):
            self.x_l[i] = x_lo + (i-ng  ) * dx
            self.x_r[i] = x_lo + (i-ng+1) * dx
            self.x_c[i] = 0.5 * ( self.x_l[i] + self.x_r[i] )
            self.dx [i] = ( self.x_r[i] - self.x_l[i] )

        for i in range( w+1 ):
            self.x_e[i] = x_lo + (i-ng) * dx
