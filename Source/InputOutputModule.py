
# Input Output Module

import h5py  as h5
import numpy as np

class InputOutput:
    
    DEFAULT = object()
    
    def __init__( self ):
        
        self.FileNumber = []
        
        self.nFields1D    = []
        self.FieldNames1D = \
        [ "Density", \
          "Velocity", \
          "Temperature", \
          "Momentum Density", \
          "Energy Density", \
          "Electric Field", \
          "Electrostatic Potential", \
          "0th Moment of g", \
          "1st Moment of g", \
          "2nd Moment of g" ]
        
        self.nFields2D    = []
        self.FieldNames2D = \
        [ "Distribution Function", \
          "Maxwellian Distribution", \
          "Maxwellian Deviation (g)" ]
    
    def Initialize( self, ApplicationName, nDOFv, nDOFx ):

        print( "Initializing IO")
        
        self.FileNumber = 0
        
        self.Name = ApplicationName
        
        self.nDOFv = nDOFv
        self.nDOFx = nDOFx
        self.nDOFz = nDOFv * nDOFx
        
        self.pQv, self.wQv = np.polynomial.legendre.leggauss( nDOFv )
        self.pQv = self.pQv / 2.0
        self.wQv = self.wQv / 2.0
        
        self.pQx, self.wQx = np.polynomial.legendre.leggauss( nDOFx )
        self.pQx = self.pQx / 2.0
        self.wQx = self.wQx / 2.0
        
    def WriteFields( self, BX, dv, v_c, dx, x_c, t, \
                    D = DEFAULT, U = DEFAULT, T = DEFAULT, S = DEFAULT, G = DEFAULT, \
                    E = DEFAULT, P = DEFAULT, f = DEFAULT, M = DEFAULT, g = DEFAULT, \
                    D_g = DEFAULT, S_g = DEFAULT, G_g = DEFAULT ):
        
        FileName = self.Name + '_' + str( self.FileNumber ).zfill(8) + '.h5'
        
        fH5 = h5.File( FileName, 'w' )
        
        # --- Write Coordinates ---
        
        dset = fH5.create_dataset( "time", (1,), 'double' )
        
        dset.write_direct( np.asarray( t ) )
        
        grp = fH5.create_group( "Coordinates" )
        
        v_N = np.zeros( [ (BX.hi[0]-BX.lo[0]+1)*self.nDOFv ], np.float64, 'F' )
        
        for i_v in range( BX.lo[0], BX.hi[0] + 1 ):
            for q_v in range( self.nDOFv ):
                
                i_V = (i_v-BX.lo[0]) * self.nDOFv + q_v
                
                v_N[i_V] = v_c[i_v] + dv[i_v] * self.pQv[q_v]
        
        dset = grp.create_dataset( "v", v_N.shape, 'double' )
        
        dset.write_direct( v_N )

        x_N = np.zeros( [ (BX.hi[1]-BX.lo[1]+1)*self.nDOFx ], np.float64, 'F' )

        for i_x in range( BX.lo[1], BX.hi[1] + 1 ):
            for q_x in range( self.nDOFx ):
                
                i_X = (i_x-BX.lo[1]) * self.nDOFx + q_x
                
                x_N[i_X] = x_c[i_x] + dx[i_x] * self.pQx[q_x]
        
        dset = grp.create_dataset( "x", x_N.shape, 'double' )
        
        dset.write_direct( x_N )
        
        # --- Write 1D Fields ---
        
        grp = fH5.create_group( "1D Fields" )
        
        if( D is not self.DEFAULT ):
            Dummy = self.MapData1D( BX.lo[1], BX.hi[1], D )
            dset = grp.create_dataset( self.FieldNames1D[0], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( U is not self.DEFAULT ):
            Dummy = self.MapData1D( BX.lo[1], BX.hi[1], U )
            dset = grp.create_dataset( self.FieldNames1D[1], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( T is not self.DEFAULT ):
            Dummy = self.MapData1D( BX.lo[1], BX.hi[1], T )
            dset = grp.create_dataset( self.FieldNames1D[2], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( S is not self.DEFAULT ):
            Dummy = self.MapData1D( BX.lo[1], BX.hi[1], S )
            dset = grp.create_dataset( self.FieldNames1D[3], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( G is not self.DEFAULT ):
            Dummy = self.MapData1D( BX.lo[1], BX.hi[1], G )
            dset = grp.create_dataset( self.FieldNames1D[4], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( E is not self.DEFAULT ):
            Dummy = self.MapData1D( BX.lo[1], BX.hi[1], E )
            dset = grp.create_dataset( self.FieldNames1D[5], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( P is not self.DEFAULT ):
            Dummy = self.MapData1D( BX.lo[1], BX.hi[1], P )
            dset = grp.create_dataset( self.FieldNames1D[6], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( D_g is not self.DEFAULT ):
            Dummy = self.MapData1D( BX.lo[1], BX.hi[1], D_g )
            dset = grp.create_dataset( self.FieldNames1D[7], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( S_g is not self.DEFAULT ):
            Dummy = self.MapData1D( BX.lo[1], BX.hi[1], S_g )
            dset = grp.create_dataset( self.FieldNames1D[8], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( G_g is not self.DEFAULT ):
            Dummy = self.MapData1D( BX.lo[1], BX.hi[1], G_g )
            dset = grp.create_dataset( self.FieldNames1D[9], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        # --- Write 2D Fields ---
        
        grp = fH5.create_group( "2D Fields" )
        
        if( f is not self.DEFAULT ):
            Dummy = self.MapData2D( BX.lo, BX.hi, f )
            dset = grp.create_dataset( self.FieldNames2D[0], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( M is not self.DEFAULT ):
            Dummy = self.MapData2D( BX.lo, BX.hi, M )
            dset = grp.create_dataset( self.FieldNames2D[1], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        if( g is not self.DEFAULT ):
            Dummy = self.MapData2D( BX.lo, BX.hi, g )
            dset = grp.create_dataset( self.FieldNames2D[2], Dummy.shape, 'double' )
            dset.write_direct( Dummy )
        
        fH5.close()
        
        self.FileNumber += 1
        
    def MapData1D( self, lo, hi, DataIn ):
        
        DataOut = np.zeros( [ (hi-lo+1)*self.nDOFx ], np.float64, 'F' )
        
        for i_x in range( lo, hi + 1 ):
            for l_x in range( self.nDOFx ):
                
                i_X = (i_x-lo) * self.nDOFx + l_x
                
                DataOut[i_X] = DataIn[l_x,i_x]
        
        return DataOut
    
    def MapData2D( self, lo, hi, DataIn ):
        
        DataOut = np.zeros( [ (hi[0]-lo[0]+1)*self.nDOFv, (hi[1]-lo[1]+1)*self.nDOFx ], np.float64, 'C' )
        
        for i_x in range( lo[1], hi[1] + 1 ):
            for i_v in range( lo[0], hi[0] + 1 ):
                
                l_z = 0
                for l_x in range( self.nDOFx ):
                    for l_v in range( self.nDOFv ):
                
                        i_X = (i_x-lo[1]) * self.nDOFx + l_x
                        i_V = (i_v-lo[0]) * self.nDOFv + l_v
                
                        DataOut[i_V,i_X] = DataIn[l_z,i_v,i_x]
                        
                        l_z += 1
        
        return DataOut