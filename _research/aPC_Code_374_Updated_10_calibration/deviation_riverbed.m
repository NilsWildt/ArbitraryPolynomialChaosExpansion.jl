function deviation_riverbed % Deterministic comparison between models of the aPC surrogate model (Years: 2005,2010,2013)
number_of_simulations=22;   % Change the number of simulations according the number of simulations (including base simulation and final simulation)
format short g
for calibration_year=1:3
variance_table=zeros(4,number_of_simulations);
load(strcat('Matrix_elevations_',num2str(calibration_year),'.mat'));

if calibration_year==1
[Node_ID,Elevation_riverchannel_2005,x,y]=textread('2005_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');
Elevation_riverchannel=horzcat(Node_ID,Elevation_riverchannel_2005,x,y);
count=calibration_year;
calibration_year=2005;

end
if calibration_year==2
[Node_ID,Elevation_riverchannel_2010,x,y]=textread('2010_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');
Elevation_riverchannel=horzcat(Node_ID,Elevation_riverchannel_2010,x,y);
count=calibration_year;
calibration_year=2010;
end
if calibration_year==3
[Node_ID,Elevation_riverchannel_2013,x,y]=textread('2013_Nodes_elevation(x14040_nodes).txt','%f%f%f%f%*c');
Elevation_riverchannel=horzcat(Node_ID,Elevation_riverchannel_2013,x,y);
count=calibration_year;
calibration_year=2013;
end


ii=1;
for i=1:length(sortingM)
    if sortingM(i,1)==1
       for j=1:4
        Real_nodes_elev(ii,j)=Elevation_riverchannel(i,j);
       end
       ii=ii+1;
       end
end
Real_nodes_elev;
length(Real_nodes_elev);
Matrix_real_observed_2005_2010_2013(:,count)=Real_nodes_elev(:,2);
for model=0:number_of_simulations-1

    Mean_deviation=mean(matrix_elev(:,model+1)-Real_nodes_elev(:,2));
    Maximum_deviation=max(matrix_elev(:,model+1)-Real_nodes_elev(:,2));
    Minimum_deviation=min(matrix_elev(:,model+1)-Real_nodes_elev(:,2));
    [fila_max,~]=find(matrix_elev(:,model+1)==max(max(matrix_elev(:,model+1)))); 
    [fila_min,~]=find(matrix_elev(:,model+1)==min(min(matrix_elev(:,model+1)))); 
    Nodo_numero_maximum_deviation=matrix_elev(fila_max,1);
    Nodo_numero_minimum_deviation=matrix_elev(fila_min,1);
    Variance=var(matrix_elev(:,model+1)-Real_nodes_elev(:,2));
    Residual_sum_of_squares= sum((Real_nodes_elev(:,2)-matrix_elev(:,model+1)).^2);
    column=vertcat(model,Mean_deviation,Variance,Residual_sum_of_squares);
    variance_table(:,model+1)=column;

end

% -------Residuals sum of squares ranking for each model in the surrogate model

[~,rank_RSS]=sort(variance_table(end,:),'ascend'); % Ranking of the models. From the lowest to the highest RSS
r=1:length(variance_table);
r(rank_RSS)=r;
if calibration_year==2005
Observed_values_2005=Real_nodes_elev;
Simulated_values_2005=matrix_elev;
end
if calibration_year==2010
Observed_values_2010=Real_nodes_elev;
Simulated_values_2010=matrix_elev;
end
if calibration_year==2013
Observed_values_2013=Real_nodes_elev;
Simulated_values_2013=matrix_elev;
end
variance_table=vertcat(variance_table,r);
variance_table=variance_table';
Table_deviation=array2table(variance_table,'VariableNames',{'Models','Mean_deviation_m','Variance_deviation_m2','Residual_sum_of_squares','Model_ranking'});
if calibration_year==2005; variance_table_2005= variance_table; end
if calibration_year==2010; variance_table_2010= variance_table; end
if calibration_year==2013; variance_table_2013= variance_table; end
end
save('Matrix observed values 2005 2010 2013(x374Nodes)','Observed_values_2005','Observed_values_2010','Observed_values_2013');
save('Matrix simulated values 2005 2010 2013(x374Nodes)','Simulated_values_2005','Simulated_values_2010','Simulated_values_2013');
save ('Tables_models_surrogate model','variance_table_2005','variance_table_2010','variance_table_2013');

% ----Nash-Sutcliffe efficiency (NSE)----
load('Matrix observed values 2005 2010 2013(x374Nodes)'); % Matrix 374x4: Includes nodes coordenates. Column # 1 has the observed elevation values
load('Matrix simulated values 2005 2010 2013(x374Nodes)'); % Matrix 374 x 22: Includes the simulated elevations at each of the 374 nodes in the 20 Models and also in the base model and final simulation (Deterministic analysis)
load ('Tables_models_surrogate model');       
Observed_values_together= horzcat(Observed_values_2005(:,2),Observed_values_2010(:,2),Observed_values_2013(:,2));
% ----Nash-Sutcliffe efficiency (NSE)---- 2005
for jj=0:number_of_simulations-1
Numerator=sum((Observed_values_2005(:,2)-Simulated_values_2005(:,jj+1)).^2);
Denominator= sum((Observed_values_2005(:,2)-mean(Observed_values_together,2)).^2);
NSE=1-Numerator/Denominator;
NSE_models(jj+1,1)=NSE;
end
variance_table_2005=horzcat(variance_table_2005,NSE_models);
[~,rank_NSE]=sort(variance_table_2005(:,end),'descend'); % Ranking of the models. From the lowest to the highest RSS
t=1:length(variance_table_2005);
t(rank_NSE)=t;
variance_table_2005=horzcat(variance_table_2005,t');
Table_deviation_2005=array2table(variance_table_2005,'VariableNames',{'Models','Mean_deviation_m','Variance_deviation_m','Residual_sum_of_squares','Ranking_RSS','NSE','Ranking_NSE'})

% ----Nash-Sutcliffe efficiency (NSE)---- 2010
for jj=0:number_of_simulations-1
Numerator=sum((Observed_values_2010(:,2)-Simulated_values_2010(:,jj+1)).^2);
Denominator= sum((Observed_values_2010(:,2)-mean(Observed_values_together,2)).^2);
NSE=1-Numerator/Denominator;
NSE_models(jj+1,1)=NSE;
end
variance_table_2010=horzcat(variance_table_2010,NSE_models);
[~,rank_NSE]=sort(variance_table_2010(:,end),'descend'); % Ranking of the models. From the lowest to the highest RSS
t=1:length(variance_table_2010);
t(rank_NSE)=t;
variance_table_2010=horzcat(variance_table_2010,t');
Table_deviation_2010=array2table(variance_table_2010,'VariableNames',{'Models','Mean_deviation_m','Variance_deviation_m','Residual_sum_of_squares','Ranking_RSS','NSE','Ranking_NSE'})

% ----Nash-Sutcliffe efficiency (NSE)---- 2013
for jj=0:number_of_simulations-1
Numerator=sum((Observed_values_2013(:,2)-Simulated_values_2013(:,jj+1)).^2);
Denominator= sum((Observed_values_2013(:,2)-mean(Observed_values_together,2)).^2);
NSE=1-Numerator/Denominator;
NSE_models(jj+1,1)=NSE;
end
variance_table_2013=horzcat(variance_table_2013,NSE_models);
[~,rank_NSE]=sort(variance_table_2013(:,end),'descend'); % Ranking of the models. From the lowest to the highest RSS
t=1:length(variance_table_2013);
t(rank_NSE)=t;
variance_table_2013=horzcat(variance_table_2013,t');
Table_deviation_2013=array2table(variance_table_2013,'VariableNames',{'Models','Mean_deviation_m','Variance_deviation_m','Residual_sum_of_squares','Ranking_RSS','NSE','Ranking_NSE'})
