function [ t, v, x, D, U, T, S, G, E, P, f, M, g, D_g, S_g, G_g ]...
  = ReadData( ApplicationName, FileNumber, Directory )

  if( exist( 'Directory', 'var' ) )
    DirName = Directory;
  else
    DirName = '.';
  end
  
  FileName = [ DirName '/' ApplicationName '_' sprintf( '%08d', FileNumber ) '.h5' ];

  t = h5read( FileName, '/time' );
  
  v = h5read( FileName, '/Coordinates/v' );
  x = h5read( FileName, '/Coordinates/x' );

  % --- 1D Fields ---
  
  D   = 0.0;
  U   = 0.0;
  T   = 0.0;
  S   = 0.0;
  G   = 0.0;
  E   = 0.0;
  P   = 0.0;
  D_g = 0.0;
  S_g = 0.0;
  G_g = 0.0;
  
  info = h5info( FileName, '/1D Fields' );
  
  if( DatasetPresent( info, 'Density') )
    D = h5read( FileName, '/1D Fields/Density' );
  end
  
  if( DatasetPresent( info, 'Velocity' ) )
    U = h5read( FileName, '/1D Fields/Velocity' );
  end
  
  if( DatasetPresent( info, 'Temperature' ) )
    T = h5read( FileName, '/1D Fields/Temperature' );
  end
  
  if( DatasetPresent( info, 'Momentum Density' ) )
    S = h5read( FileName, '/1D Fields/Momentum Density' );
  end
  
  if( DatasetPresent( info, 'Energy Density' ) )
    G = h5read( FileName, '/1D Fields/Energy Density' );
  end
  
  if( DatasetPresent( info, 'Electric Field' ) )
    E = h5read( FileName, '/1D Fields/Electric Field' );
  end
  
  if( DatasetPresent( info, 'Electrostatic Potential' ) )
    P = h5read( FileName, '/1D Fields/Electrostatic Potential' );
  end
  
  if( DatasetPresent( info, '0th Moment of g' ) )
    D_g = h5read( FileName, '/1D Fields/0th Moment of g' );
  end
  
  if( DatasetPresent( info, '1st Moment of g' ) )
    S_g = h5read( FileName, '/1D Fields/1st Moment of g' );
  end
  
  if( DatasetPresent( info, '2nd Moment of g' ) )
    G_g = h5read( FileName, '/1D Fields/2nd Moment of g' );
  end
  
  % --- 2D Fields ---
  
  f = 0.0;
  M = 0.0;
  g = 0.0;
  
  info = h5info( FileName, '/2D Fields' );
  
  if( DatasetPresent( info, 'Distribution Function' ) )
    f = h5read( FileName, '/2D Fields/Distribution Function' );
    f = f';
  end
  
  if( DatasetPresent( info, 'Maxwellian Distribution' ) )
    M = h5read( FileName, '/2D Fields/Maxwellian Distribution' );
    M = M';
  end
  
  if( DatasetPresent( info, 'Maxwellian Deviation (g)' ) )
    g = h5read( FileName, '/2D Fields/Maxwellian Deviation (g)' );
    g = g';
  end
  
end