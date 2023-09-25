
# --- Box Module ---

import numpy as np

class Box:

    def __init__( self ):

        self.dim   = [] # Box Dimension
        self.nx    = [] # Cells Per Dimension
        self.ng    = [] # Ghost Cells Per Dimension
        self.bc    = [] # Boundary Conditions
        self.lo    = [] # Index of First Interior Cell
        self.hi    = [] # Index of Last  Interior Cell
        self.lo_gl = [] # Index of First Ghost Cell on Left
        self.hi_gl = [] # Index of Last  Ghost Cell on Left
        self.lo_gr = [] # Index of First Ghost Cell on Right
        self.hi_gr = [] # Index of Last  Ghost Cell on Right
        self.w     = [] # Width of Box (Includes Ghost Cells)

    def CreateBox( self, nx, ng, bc ):

        if( len( nx ) != len( ng ) ):
            raise SystemExit( "Error, CreateBox: len(nx) != len(ng)" )

        dim = len( nx )

        self.dim   = dim
        self.nx    = np.zeros( dim, np.uint32 )
        self.ng    = np.zeros( dim, np.uint32 )
        self.bc    = np.zeros( dim, np.uint32 )
        self.lo    = np.zeros( dim, np.uint32 )
        self.hi    = np.zeros( dim, np.uint32 )
        self.lo_gl = np.zeros( dim, np.uint32 )
        self.hi_gl = np.zeros( dim, np.uint32 )
        self.lo_gr = np.zeros( dim, np.uint32 )
        self.hi_gr = np.zeros( dim, np.uint32 )
        self.w     = np.zeros( dim, np.uint32 )

        for i in range( self.dim ):
            
            self.nx[i] = nx[i]
            self.ng[i] = ng[i]
            self.bc[i] = bc[i]
            
            self.lo_gl[i] = 0
            self.hi_gl[i] = self.lo_gl[i] + max( 0, ( self.ng[i] - 1 ) )
            self.lo   [i] = self.lo_gl[i] + self.ng[i]
            self.hi   [i] = self.lo[i] + ( self.nx[i] - 1 )
            self.lo_gr[i] = self.hi[i] + min( 1, self.ng[i] )
            self.hi_gr[i] = self.lo_gr[i] + max( 0, ( self.ng[i] - 1 ) )
            self.w    [i] = self.nx[i] + 2 * self.ng[i]
