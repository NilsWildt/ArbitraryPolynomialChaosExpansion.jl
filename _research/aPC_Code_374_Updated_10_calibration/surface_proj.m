%Represents graphically the erosion and deposition effects every 1000 m in
%the river channel
% Chainages Lower river Salzach
% 1.- 0 - 788.0 [m]
% 2.- 3184.4 - 4178.0 [m]
% 3.- 4186.62 - 5192.62 [m]
% 4.- 5192.62-6198.62 [m]
% 5.- 6198.62 - 7201.62 [m]
% 6.- 7201.62 - 8205.62 [m]
% 7.- 8205.62 - 9211.62 [m]
% 8.- 9211.62 - 10211.62 [m]
% 9.- 10211.62 - 10817.62 [m]




function surface_proj(model,reach) 
close all;
load('Imported_data_SMS.mat');
for calibration1=1:3
    clearvars -except calibration1 model reach Elevations_2005 Elevations_2010 Elevations_2013;
if calibration1==1
    calibration_year=2005;
end
if calibration1==2
        calibration_year=2010;
end
if calibration1==3
        calibration_year=2013;
end

[~,~,River_reach]=xlsread('Model_Results.xlsx','NodesID','bp1:bx1');
if reach==1
    River_reach=River_reach{1};
    sortingMatrix= xlsread('Model_Results.xlsx','NodesID','bp2:bp4717')';%Imports the number of the nodos that I need
elseif reach==2
        River_reach=River_reach{2};
        sortingMatrix= xlsread('Model_Results.xlsx','NodesID','bq2:bq4717')';
elseif reach==3
        River_reach=River_reach{3};
        sortingMatrix= xlsread('Model_Results.xlsx','NodesID','br2:br4717')';
elseif reach==4
        River_reach=River_reach{4};
        sortingMatrix= xlsread('Model_Results.xlsx','NodesID','bs2:bs4717')';
elseif reach==5
        River_reach=River_reach{5};
        sortingMatrix= xlsread('Model_Results.xlsx','NodesID','bt2:bt4717')';
elseif reach==6
        River_reach=River_reach{6};
        sortingMatrix= xlsread('Model_Results.xlsx','NodesID','bu2:bu4717')';
elseif reach==7
        River_reach=River_reach{7};
        sortingMatrix= xlsread('Model_Results.xlsx','NodesID','bv2:bv4717')';
elseif reach==8
        River_reach=River_reach{8};
        sortingMatrix= xlsread('Model_Results.xlsx','NodesID','bw2:bw4717')';
elseif reach==9
        River_reach=River_reach{9};
        sortingMatrix= xlsread('Model_Results.xlsx','NodesID','bx2:bx4717')';
end
 
t=length(sortingMatrix);
sortingM_M=zeros(1,14040);   
j=1;
format short g
for i=1:length(sortingM_M)
    if sortingM_M(1,i)==0 && sortingMatrix(1,j)==i
        sortingM_M(1,i)=1;
        j=j+1;
        if j==t+1
            j=1;
        end
    end
end
sortingM_M=sortingM_M';
sortingMatrix= sortingM_M;
sum(sortingMatrix);
[Node_ID,z_2002,x,y]=textread('2002_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');
[~,z_2005,~,~]=textread('2005_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');
[~,z_2010,~,~]=textread('2010_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');
[~,z_2013,~,~]=textread('2013_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');

Matrix_coord=horzcat(Node_ID,z_2002,x,y,z_2005,z_2010,z_2013);
Coord_sorted=zeros(sum(sortingMatrix),4);
ii=1;
for i=1:length(sortingMatrix)
    if sortingMatrix(i,1)==1
       for j=1:7
        Coord_sorted(ii,j)=Matrix_coord(i,j);
       end
       ii=ii+1;
       end
end
Coord_sorted;
if calibration_year==2005
matrix_elev=Elevations_2005(:,model+1);
elseif calibration_year==2010
matrix_elev=Elevations_2010(:,model+1);
elseif calibration_year==2013
matrix_elev=Elevations_2013(:,model+1);
end
Elevations_sorted_models=zeros(sum(sortingMatrix),1);
ii=1;
for i=1:length(sortingMatrix)
    if sortingMatrix(i,1)==1
       for j=1:1
        Elevations_sorted_models(ii,j)=matrix_elev(i,j);
       end
       ii=ii+1;
       end
end
Elevations_sorted_models;
length(Elevations_sorted_models);
Elevations_selected_model=Elevations_sorted_models(:,1);


xcoord=[Coord_sorted(:,3)];
ycoord=[Coord_sorted(:,4)];
zcoord_2002=[Coord_sorted(:,2)];
zcoord_2005=[Coord_sorted(:,5)];
zcoord_2010=[Coord_sorted(:,6)];
zcoord_2013=[Coord_sorted(:,7)];
dz_2005=zcoord_2005-zcoord_2002;
dz_2010=zcoord_2010-zcoord_2002;
dz_2013=zcoord_2013-zcoord_2002;
if calibration_year==2005
    dz_2005_model=Elevations_selected_model-zcoord_2002;
elseif calibration_year==2010
    dz_2010_model=Elevations_selected_model-zcoord_2002;
elseif calibration_year==2013
    dz_2013_model=Elevations_selected_model-zcoord_2002;
end
v_x=linspace(min(xcoord),max(xcoord),200);
v_y=linspace(min(ycoord),max(ycoord),200);
[xx,yy]=meshgrid(v_x,v_y);
zz_2005=griddata(xcoord,ycoord,dz_2005,xx,yy);
zz_2010=griddata(xcoord,ycoord,dz_2010,xx,yy);
zz_2013=griddata(xcoord,ycoord,dz_2013,xx,yy);
if calibration_year==2005
    zz_2005_model=griddata(xcoord,ycoord,dz_2005_model,xx,yy);
    figure;
    set(gcf,'color','w');
surf(xx,yy,zz_2005);
colormap jet;
caxis([-3 3]);
shading interp;
h=colorbar;
ylabel(h, 'Erosion / Deposition Values on the riverbed')
view(180,90);
title([strcat('Observed erosion and Deposition Effects - River Reach: ',River_reach ,' Calibration year: ',num2str(calibration_year))]);
 xlabel('X Coordinates (m)') % x-axis label
 ylabel('Y Coordinates (m)'); % y-axis label
figure;
set(gcf,'color','w');
surf(xx,yy,zz_2005_model);
colormap jet; % Invert the colors of Jet color bar ( to show erosion and deposition)
caxis([-3 3]);
shading interp;
h=colorbar;
ylabel(h, 'Erosion / Deposition Values on the riverbed')
view(180,90);
title([strcat('Simulated erosion and Deposition Effects - River Reach:   ',River_reach ,'Calibration year: ',num2str(calibration_year))]);
 xlabel('X Coordinates (m)') % x-axis label
 ylabel('Y Coordinates (m)'); % y-axis label
elseif calibration_year==2010
    zz_2010_model=griddata(xcoord,ycoord,dz_2010_model,xx,yy);
    figure;
    set(gcf,'color','w');
surf(xx,yy,zz_2010);
colormap jet; %  Jet color bar ( to show erosion and deposition)
caxis([-3 3]);
shading interp;
h=colorbar;
ylabel(h, 'Erosion / Deposition Values on the riverbed')
view(180,90);
title([strcat('Observed erosion and Deposition Effects - River Reach:   ',River_reach ,' Calibration year: ',num2str(calibration_year))]);
 xlabel('X Coordinates (m)') % x-axis label
 ylabel('Y Coordinates (m)'); % y-axis label
figure;
set(gcf,'color','w');
surf(xx,yy,zz_2010_model);
colormap jet; % Invert the colors of Jet color bar ( to show erosion and deposition)
shading interp;
h=colorbar;
caxis([-3 3]);
ylabel(h, 'Erosion / Deposition Values on the riverbed')
view(180,90);
title([strcat('Simulated erosion and Deposition Effects - River Reach:   ',River_reach ,' Calibration year: ',num2str(calibration_year))]);
 xlabel('X Coordinates (m)') % x-axis label
 ylabel('Y Coordinates (m)'); % y-axis label
 
 elseif calibration_year==2013
    zz_2013_model=griddata(xcoord,ycoord,dz_2013_model,xx,yy);
    figure;
    set(gcf,'color','w');
surf(xx,yy,zz_2013);
colormap jet; %  Jet color bar ( to show erosion and deposition)
caxis([-3 3]);
shading interp;
h=colorbar;
ylabel(h, 'Erosion / Deposition Values on the riverbed')
view(180,90);
title([strcat('Observed erosion and Deposition Effects - River Reach:   ',River_reach ,' Calibration year: ',num2str(calibration_year))]);
 xlabel('X Coordinates (m)') % x-axis label
 ylabel('Y Coordinates (m)'); % y-axis label
figure;
set(gcf,'color','w');
surf(xx,yy,zz_2013_model);
colormap jet; % Invert the colors of Jet color bar ( to show erosion and deposition)
caxis([-3 3]);
shading interp;
h=colorbar;
ylabel(h, 'Erosion / Deposition Values on the riverbed')
view(180,90);
title([strcat('Simulated erosion and Deposition Effects - River Reach:   ',River_reach ,' Calibration year: ',num2str(calibration_year))]);
 xlabel('X Coordinates (m)') % x-axis label
 ylabel('Y Coordinates (m)'); % y-axis label
end

end
