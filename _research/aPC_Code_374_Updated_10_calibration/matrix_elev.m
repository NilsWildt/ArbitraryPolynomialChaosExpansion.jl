function matrix_elev(number_of_simulations,option) % Filters the nodes that are of interest for the analysis. 374 Nodes if the Nodes where measured data is available or 1812 Nodes if Nodes on the whole riverbed
 %---- Type in number_of_simulations: 22 for all the models in the aPC
 %surrogate model
                                                   


import_data(number_of_simulations);                 % Type "1" if cross-section nodes on riverbed (i.e. Measured data)
load('Imported_data_SMS.mat')                       % Type "2" if all the nodes on the riverbed 
for calibration_year=1:3
format short g
if option ==1
sortingM= xlsread('Model_Results.xlsx','NodesID','d2:d4717')'; %Imports the number of the nodes that I need (374)
elseif option==2
sortingM= xlsread('Model_Results.xlsx','NodesID','c2:c4717')'; %Imports the number of the nodes that I need (1812)
end
sortingM_M=zeros(1,14040);   
j=1;
for i=1:length(sortingM_M)
    if sortingM_M(1,i)==0 && sortingM(1,j)==i
        sortingM_M(1,i)=1;
        j=j+1;
        if j==length(sortingM)+1
            j=1;
        end
    end
end
sortingM_M=sortingM_M';
sortingM= sortingM_M;
Nodos_sort=zeros(sum(sortingM),number_of_simulations);
ii=1;
for i=1:length(sortingM)
    if sortingM(i,1)==1
       for j=1:number_of_simulations
        if calibration_year==1
           Nodos_sort(ii,j)=Elevations_2005(i,j);
           year=2005;
        elseif calibration_year==2
            Nodos_sort(ii,j)=Elevations_2010(i,j);
            year=2010;
        elseif calibration_year==3
            Nodos_sort(ii,j)=Elevations_2013(i,j);
            year=2013;
        end
       end
       ii=ii+1;
    end
end
matrix_elev=Nodos_sort;
fprintf('---> Importing elevation values from model %d \n',year);
filenameelevations= strcat('Matrix_elevations_',num2str(calibration_year),'.mat');
save(filenameelevations,'matrix_elev','sortingM');
end
%toc;
deviation_riverbed;
end

