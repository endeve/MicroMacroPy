function [ IsPresent ] = DatasetPresent( info, dataset )

  IsPresent = 0;
  for i = 1 : size( info.Datasets, 1 )
    if( strcmp(info.Datasets(i).Name,dataset) )
        IsPresent = 1;
    end
  end
  
end

