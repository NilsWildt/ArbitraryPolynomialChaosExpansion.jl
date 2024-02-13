function validation(model,section) % Cross section at the specified nodestring and model number (measured vs simulation) 
load('Imported_data_SMS.mat');
ind=0;                             % Upper river: 1; Middle river: 2; Lower river: 3   
if ind==0;
    if section==1
        data=imread('Upper.png');
        imshow(data);
        string_vector=input('Choose Nodestring(s).. > ');
     ind=ind+1;
        close all;
        
    else if section==2
        data=imread('Middle.png');
     imshow(data);
        string_vector=input('Choose Nodestring(s).. > ');
        ind=ind+1;
        close all;
        
     else if section==3  
    data=imread('Lower.png');
    imshow(data);
    string_vector=input('Choose Nodestring(s).. > ');
    ind=ind+1;
    close all;
            
    end 
end

end
end
for iter=1:length(string_vector)
close all;
%% Simulation 2013
NS=string_vector(1,iter);
if ~exist('Nodestrings_names.mat', 'file') 
    Nodestrings_names=xlsread('Model_Results.xlsx','NodesID','f19:bj19');
    save('Nodestrings_names.mat','Nodestrings_names');
end
    if ~exist('Nodestrings_matrix.mat', 'file')
    sortingM_real_order=xlsread('Model_Results.xlsx','NodesID','f2:bj17')';
     save('Nodestrings_matrix.mat','sortingM_real_order');
    end
    
    load('Nodestrings_matrix.mat');
    load('Nodestrings_names.mat');
format short g
real_name_NS=NS;
logic_vector=Nodestrings_names==NS;
[~,J]=find(logic_vector);
NS=J;
sheet=model;
[r,c]=size(sortingM_real_order);
sortingM=sort(sortingM_real_order(NS,:));
length(sortingM);
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
[ID,z,x,y]=textread('2013_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');
Coord_z_x_y=horzcat(ID,z,x,y);
ii=1;
for i=1:length(sortingM)
    if sortingM(i,1)==1
       for j=1:4
        Nodos_coord(ii,j)=Coord_z_x_y(i,j);
       end
       ii=ii+1;
       end
end
Nodos_coord_real=zeros(size(Nodos_coord));
for tt=1:length(sortingM_real_order(NS,:))
    for ll=1:length(Nodos_coord)
        if sortingM_real_order(NS,tt)==Nodos_coord(ll,1)
           Nodos_coord_real(tt,:)=Nodos_coord(ll,:);
           break;
        end
     end
end
 Nodos_coord_real;
 cinitial=0;
 for jj=1:length(Nodos_coord_real)-1
 chainage(1,jj)=cinitial+((Nodos_coord_real(jj,3)-Nodos_coord_real(jj+1,3))^2+(Nodos_coord_real(jj,4)-Nodos_coord_real(jj+1,4))^2)^0.5;
 cinitial=chainage(1,jj);
 end
 chainage=horzcat(0,chainage);
 elev_valid_year_2013=Nodos_coord_real(:,2);
 plot(chainage,elev_valid_year_2013,'-k*','LineWidth',1.5); % Plot # 1: Elevation measured values 2005 
 title(['Cross-section (Measured - Simulation) Node String Number: ',num2str(real_name_NS)]);
 xlabel('Width (m)') % x-axis label
 ylabel('Elevation (m.a.s.l)'); % y-axis label
 hold on;
 Neu_sohle_valid_year=Elevations_2013(:,sheet+1);
 Node_ID=xlsread('Model_Results.xlsx',sheet,'a3:a14042');
 Neu_sohle_valid_year=horzcat(Node_ID,Neu_sohle_valid_year);
 ii=1;
 for i=1:length(sortingM)
    if sortingM(i,1)==1
       for j=1:2
        Nodos_elev_simulation(ii,j)=Neu_sohle_valid_year(i,j);
       end
       ii=ii+1;
       end
end
Nodos_elev_simulation_real=zeros(size(Nodos_elev_simulation));
for tt=1:length(sortingM_real_order(NS,:))
    for ll=1:length(Nodos_elev_simulation)
        if sortingM_real_order(NS,tt)==Nodos_elev_simulation(ll,1)
           Nodos_elev_simulation_real(tt,:)=Nodos_elev_simulation(ll,:);
           break;
        end
     end
end
 elev_neu=Nodos_elev_simulation_real(:,2);
 [max_dev,max_index]=max(abs(elev_neu-elev_valid_year_2013));
 [X,~]=ind2sub(size(elev_valid_year_2013),max_index);
 chainage_max_dev=chainage(1,X);
 x_dev=[chainage_max_dev chainage_max_dev];
 y_dev=[elev_neu(X,1) elev_valid_year_2013(X,1)];
 plot(chainage,elev_neu,'-r*','LineWidth',1.5);
 plot(x_dev,y_dev,'-g','LineWidth',1);
 legend('Observed values 2013','Simulation 2013',['Maximum deviation: ' num2str(max_dev) ' in node # ' num2str(Nodos_coord_real(X,1))]  ,'Location','north');
 filename1=strcat('','\Simulation 2013_',num2str(real_name_NS),'_Model_',num2str(model),'.fig');
 directory='C:\Users\Andres Heredia\Documents\Andres Heredia\MASTER THESIS UNIVERSITY OF STUTTGART\Simulations\aPC_Code_374_Updated_10_calibration\Results';
savefig(strcat(directory,filename1));
 %end



%% Erosion and deposition in 2013
figure; 
clearvars -EXCEPT NS sheet sortingM_real_order elev_valid_year_2013 real_name_NS directory string_vector model Elevations_2013
sortingM=sort(sortingM_real_order(NS,:));
length(sortingM);
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
[ID,z,x,y]=textread('2002_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');
Coord_z_x_y=horzcat(ID,z,x,y);
ii=1;
for i=1:length(sortingM)
    if sortingM(i,1)==1
       for j=1:4
        Nodos_coord(ii,j)=Coord_z_x_y(i,j);
       end
       ii=ii+1;
       end
end
Nodos_coord_real=zeros(size(Nodos_coord));
for tt=1:length(sortingM_real_order(NS,:))
    for ll=1:length(Nodos_coord)
        if sortingM_real_order(NS,tt)==Nodos_coord(ll,1)
           Nodos_coord_real(tt,:)=Nodos_coord(ll,:);
           break;
        end
     end
end
 Nodos_coord_real;
 cinitial=0;
 for jj=1:length(Nodos_coord_real)-1
 chainage(1,jj)=cinitial+((Nodos_coord_real(jj,3)-Nodos_coord_real(jj+1,3))^2+(Nodos_coord_real(jj,4)-Nodos_coord_real(jj+1,4))^2)^0.5;
 cinitial=chainage(1,jj);
 end
 chainage=horzcat(0,chainage);
 elev_calib_year=Nodos_coord_real(:,2);
 plot(chainage,elev_calib_year,'-k*','LineWidth',1.5);  % Plot # 1: Elevation year 2002 (measured values)
 title(['Erosion/ Deposition at Node String Number: ',num2str(real_name_NS)]);
 xlabel('Width (m)') % x-axis label
 ylabel('Elevation (m.a.s.l)'); % y-axis label
 hold on;
 Neu_sohle_valid_year=Elevations_2013(:,sheet+1); % Imports the values of elevation at each node from the simulation number:.... the one it is introduced as input value.... % column b: means year 2005 and column c: means year 2010
 Node_ID=xlsread('Model_Results.xlsx',sheet,'a3:a14042');                
 Neu_sohle_valid_year=horzcat(Node_ID,Neu_sohle_valid_year);
 ii=1;
 for i=1:length(sortingM)
    if sortingM(i,1)==1
       for j=1:2
        Nodos_elev_simulation(ii,j)=Neu_sohle_valid_year(i,j);
       end
       ii=ii+1;
       end
end
Nodos_elev_simulation_real=zeros(size(Nodos_elev_simulation));
for tt=1:length(sortingM_real_order(NS,:))
    for ll=1:length(Nodos_elev_simulation)
        if sortingM_real_order(NS,tt)==Nodos_elev_simulation(ll,1)
           Nodos_elev_simulation_real(tt,:)=Nodos_elev_simulation(ll,:);
           break;
        end
     end
end
 elev_neu=Nodos_elev_simulation_real(:,2);
 [max_dev,max_index]=max((elev_neu-elev_calib_year));
 [min_dev,min_index]=min(elev_neu-elev_calib_year);
 [X,~]=ind2sub(size(elev_calib_year),max_index);
 [X1,~]=ind2sub(size(elev_calib_year),min_index);
 chainage_max_dev=chainage(1,X);
 chainage_min_dev=chainage(1,X1);
 x_dev=[chainage_max_dev chainage_max_dev];
 y_dev=[elev_neu(X,1) elev_calib_year(X,1)];
 x1_dev=[chainage_min_dev chainage_min_dev];
 y1_dev=[elev_neu(X1,1) elev_calib_year(X1,1)];
 plot(chainage,elev_neu,'-r*','LineWidth',1.5); %Plot # 2: Elevation values Simulation 2010
 plot(x_dev,y_dev,'-g','LineWidth',1);%Plot # 4: Erosion values
 plot(x1_dev,y1_dev,'-m','LineWidth',1);%Plot # 3: Deposition values
 plot (chainage,elev_valid_year_2013,'-b.','LineWidth',1)% Plot # 5: Elevation year 2010 (mesured values)
 legend('Elevation 2002','Simulation 2013',['Deposition: ' num2str(max_dev) ' in node # ' num2str(Nodos_coord_real(X,1))],['Erosion: ' num2str(min_dev) ' in node # ' num2str(Nodos_coord_real(X1,1))],'Elevation 2013'  ,'Location','north');
 filename1=strcat('','\Erosion_Deposition 2013_',num2str(real_name_NS),'_Model_',num2str(model),'.fig');
savefig(strcat(directory,filename1));
 clearvars -EXCEPT iter directory NS string_vector model;
end
close all;
