
# --- ImexModule ---

import sys
import numpy as np

class IMEX:
    
    def __init__( self ):
        
        self.a_EX = []
        self.w_EX = []
        self.c_EX = []
    
        self.a_IM = []
        self.w_IM = []
        self.c_IM = []
    
        self.nStages = 0
    
    def AllocateButcherTables( self ):
        
        self.a_EX = np.zeros( [ self.nStages, self.nStages ], np.float64, 'F' )
        self.w_EX = np.zeros( [ self.nStages ]              , np.float64, 'F' )
        self.c_EX = np.zeros( [ self.nStages ]              , np.float64, 'F' )
        
        self.a_IM = np.zeros( [ self.nStages, self.nStages ], np.float64, 'F' )
        self.w_IM = np.zeros( [ self.nStages ]              , np.float64, 'F' )
        self.c_IM = np.zeros( [ self.nStages ]              , np.float64, 'F' )
        
    def Initialize( self, Scheme, Verbose = False ):
        
        if( Scheme == 'ARS_111' ):
            
            self.nStages = 2
            
            self.AllocateButcherTables()
            
            # --- Ascher et al. (1997), Appl. Num. Math., 25, 151-167 ---

            # --- Explicit Coefficients ---
            
            self.a_EX[1,0] = 1.0
            self.w_EX[0  ] = 1.0
            
            # --- Implicit Coefficients ---
            
            self.a_IM[1,1] = 1.0
            self.w_IM[1  ] = 1.0
        
        elif( Scheme == 'RKCB2' ):
            
            self.nStages = 3
            
            self.AllocateButcherTables()
            
            # --- Cavaglieri & Bewley (2015), JCP, 286, 172-193 ---

            # --- Explicit Coefficients ---
            
            self.a_EX[1,0] = 0.4
            self.a_EX[2,1] = 1.0
            
            self.w_EX[1  ] = 5.0 / 6.0
            self.w_EX[2  ] = 1.0 / 6.0
            
            # --- Implicit Coefficients ---
            
            self.a_IM[1,1] = 0.4
            self.a_IM[2,1] = 5.0 / 6.0
            self.a_IM[2,2] = 1.0 / 6.0
            
            self.w_IM[1  ] = 5.0 / 6.0
            self.w_IM[2  ] = 1.0 / 6.0
            
        elif( Scheme == 'PDARS' ):
            
            self.nStages = 3
            
            self.AllocateButcherTables()
            
            # --- Chu et al. (2019), JCP, 389, 62-93 ---

            # --- Explicit Coefficients ---
            
            self.a_EX[1,0] = 1.0
            self.a_EX[2,0] = 0.5
            self.a_EX[2,1] = 0.5
            
            self.w_EX[0  ] = 0.5
            self.w_EX[1  ] = 0.5
            
            # --- Implicit Coefficients ---
            
            self.a_IM[1,1] = 1.0
            self.a_IM[2,1] = 0.5
            self.a_IM[2,2] = 0.5
            
            self.w_IM[1  ] = 0.5
            self.w_IM[2  ] = 0.5
        
        elif( Scheme == 'SSPRK1' ):
            
            self.nStages = 1
            
            self.AllocateButcherTables()
            
            self.w_EX[0] = 1
            
        elif( Scheme == 'SSPRK2' ):
            
            self.nStages = 2
            
            self.AllocateButcherTables()
            
            self.a_EX[1,0] = 1.0
            self.w_EX[0  ] = 0.5
            self.w_EX[1  ] = 0.5
            
        elif( Scheme == 'SSPRK3' ):
            
            self.nStages = 3
            
            self.AllocateButcherTables()
            
            self.a_EX[1,0] = 1.0
            self.a_EX[2,0] = 0.25
            self.a_EX[2,1] = 0.25
            self.w_EX[0  ] = 1.0 / 6.0
            self.w_EX[1  ] = 1.0 / 6.0
            self.w_EX[2  ] = 2.0 / 3.0
            
        else:
            
            print( "Unknown IMEX Scheme.  ")
            print( "Available Options are: ARS_111, RKCB2, PDARS, SSPRK1, SSPRK2, SSPRK3")
            sys.exit()
        
        for i in range( self.nStages ):
            
            self.c_EX[i] = np.sum( self.a_EX[i,0:i  ] )
            self.c_IM[i] = np.sum( self.a_IM[i,0:i+1] )
        
        if( Verbose ):

            print( "                 " )
            print( "  Butcher Tables:" )
            print( "                 " )
            print( "  Explicit:      " )
            print( "                 " )
            for i in range( self.nStages ):
                print( "  ", self.c_EX[i], self.a_EX[i,:] )
            print( "    ", self.w_EX[0:self.nStages+1] )
            print( "                 " )
            print( "  Implicit:      " )
            print( "                 " )
            for i in range( self.nStages ):
                print( "  ", self.c_IM[i], self.a_IM[i,:] )
            print( "    ", self.w_IM[0:self.nStages+1] )
        