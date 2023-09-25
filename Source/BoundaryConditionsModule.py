
# --- Boundary Conditions Module ---

def ApplyBC_Z( BX, u ):
    
    # --- v boundaries ---
        
    if( BX.bc[0] == 1 ): # Periodic
        
        # --- Inner Boundary ---
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.ng[0] ):
                
                u[:,BX.lo_gl[0]+i_v,i_x] = u[:,BX.hi[0]-(BX.ng[0]-1)+i_v,i_x]
        
        # --- Outer Boundary ---
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.ng[0] ):
                
                u[:,BX.lo_gr[0]+i_v,i_x] = u[:,BX.lo[0]+i_v,i_x]
    
    elif( BX.bc[0] == 2 ): # Homogeneous
        
        # --- Inner Boundary ---
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.ng[0] ):
                
                u[:,BX.lo_gl[0]+i_v,i_x] = u[:,BX.lo[0],i_x]
        
        # --- Outer Boundary ---
        
        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for i_v in range( BX.ng[0] ):
                
                u[:,BX.lo_gr[0]+i_v,i_x] = u[:,BX.hi[0],i_x]
    
    # --- x boundaries ---
    
    if( BX.bc[1] == 1 ): # Periodic
        
        # --- Inner Boundary ---
        
        for i_x in range( BX.ng[1] ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                u[:,i_v,BX.lo_gl[1]+i_x] = u[:,i_v,BX.hi[1]-(BX.ng[1]-1)+i_x]
        
        # --- Outer Boundary ---
        
        for i_x in range( BX.ng[1] ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                u[:,i_v,BX.lo_gr[1]+i_x] = u[:,i_v,BX.lo[1]+i_x]
    
    elif( BX.bc[1] == 2 ): # Homogeneous
        
        # --- Inner Boundary ---
        
        for i_x in range( BX.ng[1] ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                u[:,i_v,BX.lo_gl[1]+i_x] = u[:,i_v,BX.lo[1]]
        
        # --- Outer Boundary ---
        
        for i_x in range( BX.ng[1] ):
            for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
                
                u[:,i_v,BX.lo_gr[1]+i_x] = u[:,i_v,BX.hi[1]]
    
    
def ApplyBC_X( BX, u ):
    
    # --- x boundaries ---
    
    if( BX.bc[0] == 1 ): # Periodic
        
        # --- Inner Boundary ---
        
        for i_x in range( BX.ng[0] ):
            
            u[:,BX.lo_gl[0]+i_x] = u[:,BX.hi[0]-(BX.ng[0]-1)+i_x]
        
        # --- Outer Boundary ---
        
        for i_x in range( BX.ng[0] ):
            
            u[:,BX.lo_gr[0]+i_x] = u[:,BX.lo[0]+i_x]
    
    elif( BX.bc[0] == 2 ): # Homogeneous
        
        # --- Inner Boundary ---
        
        for i_x in range( BX.ng[0] ):
            
            u[:,BX.lo_gl[0]+i_x] = u[:,BX.lo[0]]
        
        # --- Outer Boundary ---
        
        for i_x in range( BX.ng[0] ):
            
            u[:,BX.lo_gr[0]+i_x] = u[:,BX.hi[0]]