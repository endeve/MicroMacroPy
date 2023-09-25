import numpy as np

# --- Lagrange Module ---

def LagrangeP( x, i, xx, nn ):

    # --- ith Lagrange polynomial interpolating xx (nn points) in x ---

    Lag = np.ones( len( x ), np.float64, 'F' )
    for j in range( nn ):
        if( j != i ):
            Lag = Lag * ( x - xx[j] ) / ( xx[i] - xx[j] )

    return Lag

def dLagrangeP( x, i, xx, nn ):

    # --- Derivative of ---
    # --- ith Lagrange polynomial interpolating xx (nn points) in x ---

    Den = 1.0
    for j in range( nn ):
        if( j != i ):
            Den = Den * ( xx[i] - xx[j] )

    dLag = np.zeros( len( x ), np.float64, 'F' )
    for j in range( nn ):
        if( j != i ):
            Num = 1.0
            for k in range( nn ):
                if( k != i and k != j ):
                    Num = Num * ( x - xx[k] )
            dLag = dLag + Num / Den

    return dLag

def ddLagrangeP( x, i, xx, nn ):
    
    # --- Second Derivative of ---
    # --- ith Lagrange polynomial interpolating xx (nn points) in x ---
    
    Den = 1.0
    for j in range( nn ):
        if( j != i ):
            Den = Den * ( xx[i] - xx[j] )
    
    ddLag = np.zeros( len( x ), np.float64, 'F' )
    for j in range( nn ):
        if( j != i ):
            for k in range( nn ):
                if( k != i and k != j ):
                    Num = np.ones( len( x ), np.float64, 'F' )
                    for l in range( nn ):
                        if( l != i and l != j and l != k ):
                            Num = Num * ( x - xx[l] )
                    ddLag = ddLag + Num / Den
    
    return ddLag