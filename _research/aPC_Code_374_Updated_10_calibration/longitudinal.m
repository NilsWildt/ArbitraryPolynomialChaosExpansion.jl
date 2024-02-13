function longitudinal(model,profile)     % Draws the longitudinal profiles: Choose 1.- If you want to visualize Right bank profiles
                                                          %2.- If you want to visualize Left bank profiles
close all;                                                  %3.-  If you want to visualize the middle riverbed profile
load('Imported_data_SMS.mat');
sheet=model;
g=1;
[ID_2002,z_2002,x_2002,y_2002]=textread('2002_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');
for calib_year=1:3
    clearvars -EXCEPT profile sheet calib_year Elevations_2005 Elevations_2010 Elevations_2013 matrix_acumulation_years_measured_values matrix_acumulation_years_simulated_values g ID_2002 z_2002 x_2002 y_2002
if profile==1 
filename = fullfile('C:\Users\Andres Heredia\Documents\Andres Heredia\MASTER THESIS UNIVERSITY OF STUTTGART\Simulations\aPC_Code','longitudinal_1_right.txt');
fileID = fopen(filename);
Nodes_longitudinal = textscan(fileID,'%f');
Nodes_longitudinal = cell2mat(Nodes_longitudinal);
side='Right';
end
if profile==2
filename = fullfile('C:\Users\Andres Heredia\Documents\Andres Heredia\MASTER THESIS UNIVERSITY OF STUTTGART\Simulations\aPC_Code','longitudinal_2_left.txt');
fileID = fopen(filename);
Nodes_longitudinal = textscan(fileID,'%f');
Nodes_longitudinal = cell2mat(Nodes_longitudinal);
side='Left';
end
if profile==3 
filename = fullfile('C:\Users\Andres Heredia\Documents\Andres Heredia\MASTER THESIS UNIVERSITY OF STUTTGART\Simulations\aPC_Code','longitudinal_3_middle.txt');
fileID = fopen(filename);
Nodes_longitudinal = textscan(fileID,'%f');
Nodes_longitudinal = cell2mat(Nodes_longitudinal);
side='Middle';
end
Nodes_longitudinal(numel(Nodes_longitudinal))=[];
Nodes_longitudinal=Nodes_longitudinal';
sortingM=sort(Nodes_longitudinal);
length(sortingM)
sortingM_M=zeros(1,14040);
j=1;
for i=1:length(sortingM_M)
    if sortingM_M(1,i)==0 && sortingM(1,j)==i
        sortingM_M(1,i)=1;
        j=j+1;
       if j==length(sortingM)+1
            break;
        end
    end
end
sortingM_M=sortingM_M';
sortingM= sortingM_M;
sum(sortingM);
if calib_year==1
[ID,z,x,y]=textread('2005_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');

year='2005';
elseif calib_year==2
    [ID,z,x,y]=textread('2010_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');
    
    year='2010';
elseif calib_year==3
[ID,z,x,y]=textread('2013_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');

year='2013';
end
Coord_z_x_y=horzcat(ID,z,x,y,ID_2002,z_2002,x_2002,y_2002);
ii=1;
for i=1:length(sortingM)
    if sortingM(i,1)==1
       for j=1:4
        Nodos_coord(ii,j)=Coord_z_x_y(i,j);
       end
       ii=ii+1;
       end
end

ii=1;
for i=1:length(sortingM)
    if sortingM(i,1)==1
       for j=5:8
        Nodos_coord_2002(ii,j-4)=Coord_z_x_y(i,j);
       end
       ii=ii+1;
       end
end
Nodos_coord_real=zeros(size(Nodos_coord));
Nodos_coord_real_2002=zeros(size(Nodos_coord_2002));
for tt=1:length(Nodes_longitudinal)
    for ll=1:length(Nodos_coord)
        if Nodes_longitudinal(1,tt)==Nodos_coord(ll,1)
           Nodos_coord_real(tt,:)=Nodos_coord(ll,:);
           break;
        end
     end
end

for tt=1:length(Nodes_longitudinal)
    for ll=1:length(Nodos_coord_2002)
        if Nodes_longitudinal(1,tt)==Nodos_coord_2002(ll,1)
           Nodos_coord_real_2002(tt,:)=Nodos_coord_2002(ll,:);
           break;
        end
     end
end
 Nodos_coord_real;
 Nodos_coord_real_2002; 
 cinitial=0;
 for jj=1:length(Nodos_coord_real)-1
 chainage(1,jj)=cinitial+((Nodos_coord_real(jj,3)-Nodos_coord_real(jj+1,3))^2+(Nodos_coord_real(jj,4)-Nodos_coord_real(jj+1,4))^2)^0.5;
 cinitial=chainage(1,jj);
 end
 initial_x=63.4;
 chainage=horzcat(0,chainage/1000);
 chainage=initial_x-chainage;
 elev_calib_year=Nodos_coord_real(:,2);
 plot(chainage,elev_calib_year,'--k.','LineWidth',0.6);
 title(['Longitudinal section (Measured - Simulation) String: ',side,'--Calibration year: ',year]);
 xlabel('Chainage (km)'); % x-axis label
 set(gca,'XDir','Reverse');
 ylabel('Elevation (m.a.s.l)'); % y-axis label
 hold on;
 if calib_year==1
 Neu_sohle_calib_year=Elevations_2005(:,sheet+1);
 elseif calib_year==2
     Neu_sohle_calib_year=Elevations_2010(:,sheet+1);
 elseif calib_year==3
     Neu_sohle_calib_year=Elevations_2013(:,sheet+1);
 end
 Node_ID=xlsread('Model_Results.xlsx',sheet,'a3:a14042');
 Neu_sohle_calib_year=horzcat(Node_ID,Neu_sohle_calib_year);
 ii=1;
 for i=1:length(sortingM)
    if sortingM(i,1)==1
       for j=1:2
        Nodos_elev_simulation(ii,j)=Neu_sohle_calib_year(i,j);
       end
       ii=ii+1;
       end
end
Nodos_elev_simulation_real=zeros(size(Nodos_elev_simulation));
for tt=1:length(Nodes_longitudinal)
    for ll=1:length(Nodos_elev_simulation)
        if Nodes_longitudinal(1,tt)==Nodos_elev_simulation(ll,1)
           Nodos_elev_simulation_real(tt,:)=Nodos_elev_simulation(ll,:);
           break;
        end
     end
end
elev_neu=Nodos_elev_simulation_real(:,2); % Valores de simulacion 
dif_elev_neu=Nodos_elev_simulation_real(:,2)-elev_calib_year; % Valores de simulacion vs valores reales 
standart_deviation=std(dif_elev_neu)
logic_nodes_out=abs(dif_elev_neu)>=(2*standart_deviation);
length(logic_nodes_out);
sum(logic_nodes_out);
percentage_in=100- sum(logic_nodes_out)/numel(logic_nodes_out)*100
 [max_dev,max_index]=max(abs(elev_neu-elev_calib_year));
 [X,~]=ind2sub(size(elev_calib_year),max_index);
 chainage_max_dev=chainage(1,X);
 x_dev=[chainage_max_dev chainage_max_dev];
 y_dev=[elev_neu(X,1) elev_calib_year(X,1)];
 plot(chainage,elev_neu,'--r.','LineWidth',0.6); hold on;
 plot(chainage,[elev_calib_year+2*standart_deviation],'-b','LineWidth',0.1); hold on;
 plot(chainage,[elev_calib_year-2*standart_deviation],'-b','LineWidth',0.1); hold on;
 plot(x_dev,y_dev,'-g','LineWidth',0.6); grid on;
 legend(strcat('Observed values ',year),strcat('Simulation ',year),'+2 Standart deviation','-2 Standart deviation',['Maximum deviation: ' num2str(max_dev) ' in node # ' num2str(Nodos_coord_real(X,1))] ,'Location','north');
 figure;
 %plot(chainage,dif_elev_neu,'--r.','LineWidth',0.6); 
 
if calib_year==1
 matrix_acumulation_years_measured_values(:,g)= elev_calib_year;
 matrix_acumulation_years_simulated_values(:,g)=elev_neu;
 g=g+1;
elseif calib_year==2
    matrix_acumulation_years_measured_values(:,g)= elev_calib_year;
 matrix_acumulation_years_simulated_values(:,g)=elev_neu;
 g=g+1;
 elseif calib_year==3
    matrix_acumulation_years_measured_values(:,g)= elev_calib_year;
 matrix_acumulation_years_simulated_values(:,g)=elev_neu; 
end
if calib_year==3
    size(matrix_acumulation_years_measured_values);
    size(matrix_acumulation_years_simulated_values);
    matrix_acumulation_years_measured_values=horzcat(Nodos_coord_real_2002(:,2),matrix_acumulation_years_measured_values);
    deviation_2005_measured= matrix_acumulation_years_measured_values(:,2)-matrix_acumulation_years_measured_values(:,1);
    deviation_2010_measured= matrix_acumulation_years_measured_values(:,3)-matrix_acumulation_years_measured_values(:,1);
    deviation_2013_measured= matrix_acumulation_years_measured_values(:,4)-matrix_acumulation_years_measured_values(:,1);
    deviation_2005_simulated=matrix_acumulation_years_simulated_values(:,1)-matrix_acumulation_years_measured_values(:,1);
    deviation_2010_simulated=matrix_acumulation_years_simulated_values(:,2)-matrix_acumulation_years_measured_values(:,1);
    deviation_2013_simulated=matrix_acumulation_years_simulated_values(:,3)-matrix_acumulation_years_measured_values(:,1);
    figure;
     plot(chainage,deviation_2005_measured,'--b.','LineWidth',0.6); hold on;
    plot(chainage,deviation_2005_simulated,'--g.','LineWidth',0.6); hold on;
    title(['Riverbed evolution (Measured vs Simulated) in the year 2005']);
	xlabel('Chainage (km)'); % x-axis label
     set(gca,'XDir','Reverse');
     ylabel('Variation from elevation 2002 (m)'); % y-axis label
     legend('Real Riverbed Evolution','Simulated Riverbed Evolution','Location','north');
     figure;
     plot(chainage,deviation_2010_measured,'--b.','LineWidth',0.6); hold on;
    plot(chainage,deviation_2010_simulated,'--g.','LineWidth',0.6); hold on;
    title(['Riverbed evolution (Measured vs Simulated) in the year 2010']);
	xlabel('Chainage (km)'); % x-axis label
     set(gca,'XDir','Reverse');
     ylabel('Variation from elevation 2002 (m)'); % y-axis label
     legend('Real Riverbed Evolution','Simulated Riverbed Evolution','Location','north');
     figure;
     plot(chainage,deviation_2013_measured,'--b.','LineWidth',0.6); hold on;
    plot(chainage,deviation_2013_simulated,'--g.','LineWidth',0.6); hold on;
    title(['Riverbed evolution (Measured vs Simulated) in the year 2013']);
	xlabel('Chainage (km)'); % x-axis label
     set(gca,'XDir','Reverse');
     ylabel('Variation from elevation 2002 (m)'); % y-axis label
     legend('Real Riverbed Evolution','Simulated Riverbed Evolution','Location','north');
     save('matrix_ch.mat','deviation_2010_simulated','chainage','Nodos_coord_real');
end
end
end
