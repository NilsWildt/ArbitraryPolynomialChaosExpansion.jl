function import_data(number_models) %Imports all the simulated data (i.e., riverbed elevations) from each of the 22 models in the surrogate model for the years 2005,2010 and 2013
for i=0:number_models-1             %Coupled with files .txt obtained from SMS and Hydro_FT-2D
    directory='C:\Users\Andres Heredia\Documents\Andres Heredia\MASTER THESIS UNIVERSITY OF STUTTGART\Simulations\Data_simulations\';
    structure=importdata(strcat(directory,'Output_elev_all_',num2str(i),'.txt'));
    values_elevations=getfield(structure, 'data');
    Elevations_2005(:,i+1)=values_elevations(:,1);
    Elevations_2010(:,i+1)=values_elevations(:,2);
    Elevations_2013(:,i+1)=values_elevations(:,3);
end
save('Imported_data_SMS.mat','Elevations_2005','Elevations_2010','Elevations_2013');
