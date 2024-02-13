% ========================================================================
% Bayesian update of the morphodynamic model "River Salzach"
% Author: Andres Heredia 
% baherediah@gmail.com
% 374 Calibration Nodes
% ========================================================================

clear all;
close all;
load('Matrix observed values 2005 2010 2013(x374Nodes).mat')
load('Random_elevation_outputs.mat')
Input_distributions=Input_distributions';
for calibration_year=1:2
    if calibration_year==1
        Random_output_1= Random_output_2005(:,[2:end]);
        Observation_1=[Observed_values_2005(:,2)]';
        year1=2005;
    elseif calibration_year==2
        Random_output_2= Random_output_2010(:,[2:end]);
        Observation_2=[Observed_values_2010(:,2)]';
        year2=2010;
    end
end
Model_Output=horzcat(Random_output_1,Random_output_2);
[r,c]=size(Model_Output); % rows: 100000, columns= 748 (because there are 2 years)
%No time series in the test
NumberOfObservations=c; % Number of observations. In this case we have 374 Nodes where I have observed data
                           % Here I substract 1 because the first column is
                           % an indicator of the number of iterations (in this case 100000)
%Initialization of Synthetic Observations

Observation=horzcat(Observation_1,Observation_2); % 2 times the vector Observed values (1 x 748)

%Initialization of Prior

MCsize=r;
Prior_MC_Vector(1,:)=Input_distributions(1,:); % Shields parameter
Prior_MC_Vector(2,:)=Input_distributions(2,:); % Deviation in Grain roughness coefficient ks [mm]
Prior_MC_Vector(3,:)=Input_distributions(3,:); % Deviation Grain size distribution [mm]

%Initialization of Measurement Error 
MeasurementError=1.5; 
for i=1:NumberOfObservations
for j=1:NumberOfObservations
    if i==j Measurement_Error(i,j)=MeasurementError^2; end
    if i~=j Measurement_Error(i,j)=0; end
end
end       

% ========================================================================
% ====== Bayesian update
% ========================================================================
fprintf(strcat('\n---> Calculating deviations observed values vs model output for year\n',num2str(year1),'\n','and',num2str(year2)));
Deviation=zeros(MCsize,NumberOfObservations);
length(Observation);
for j=1:NumberOfObservations
    Deviation(:,j)=Observation(j)-Model_Output(:,j);
end
Deviation;
size(Deviation)
fprintf(strcat('\n---> Computation of Weight according observations for year\n',num2str(year1),'\n','and',num2str(year2)));
Weight=zeros(1,MCsize);
format long;
wb=waitbar(0,'Loading Weights...');
for i=1:1:MCsize 
    Weight(i)=exp(-0.5*Deviation(i,:)*pinv(Measurement_Error)*Deviation(i,:)');
    waitbar(i/MCsize);
    %fprintf(strcat('\n---> Realization number : ',num2str(i),'\n'));
end
delete(wb);
Weight=Weight/max(Weight);
BME=mean(Weight);
[max_weight,max_index]=max(Weight) % Finds the location of the max Weight
Prior_MC_Vector(:,max_index)

    shading interp;
    colormap jet;
    view(-45,25);
    grid on;

    

fprintf('\n---> Rejection sampling via Uniform distribution\n');

% ========================================================================
% ====== Rejection Sampling: filtration of prior distribution via uniform
% ========================================================================
Unif=random('unif',0,1,1,MCsize);
ii=0;
for i=1:MCsize
    if Weight(i)>Unif(i)
        ii=ii+1;
        Post_MC_Vector(:,ii)=Prior_MC_Vector(:,i);
    end
    
    %Progress report 
    if mod(100*i/MCsize,10)==0 
        fprintf('Rejection sampling: ');
        disp([datestr(now) ' - ' num2str(round(100*i/MCsize)) '% completed']);
   end
end
Post_MC_Vector;

bins=50;
%Prior Plot
figure;
subplot(2,3,1)
hist(Prior_MC_Vector(1,:),bins);
xlim([min(Prior_MC_Vector(1,:)) max(Prior_MC_Vector(1,:))]);
xlabel('Prior - Shields Parameter');
subplot(2,3,2)
hist(Prior_MC_Vector(2,:),bins);
xlim([min(Prior_MC_Vector(2,:)) max(Prior_MC_Vector(2,:))]);
xlabel('Prior - Deviation in Grain Roughness');
title('Prior and posterior distributions of the calibration parameters','FontWeight', 'bold');
subplot(2,3,3)
hist(Prior_MC_Vector(2,:),bins);
xlim([min(Prior_MC_Vector(3,:)) max(Prior_MC_Vector(3,:))]);
xlabel('Prior - Deviation-Grain Size Distribution curve [mm]');
%Posterior Plot
subplot(2,3,4)
hist(Post_MC_Vector(1,:),bins);
xlim([min(Prior_MC_Vector(1,:)) max(Prior_MC_Vector(1,:))]);
xlabel('Posterior - Shields Parameter');
subplot(2,3,5)
hist(Post_MC_Vector(2,:),bins);
xlim([min(Prior_MC_Vector(2,:)) max(Prior_MC_Vector(2,:))]);
xlabel('Posterior - Deviation in Grain Roughness [mm]');
subplot(2,3,6)
hist(Post_MC_Vector(3,:),bins);
xlim([min(Prior_MC_Vector(3,:)) max(Prior_MC_Vector(3,:))]);
xlabel('Posterior - Deviation-Grain Size Distribution curve [mm]');
%Plotting weigths
figure;

subplot(1,3,1)
plot(Prior_MC_Vector(1,:),Weight,'.')
xlabel('Shields Parameter')
%[histw, intervals] = histcounts(Prior_MC_Vector(1,:), Weight, 30); 
%bar(intervals, histw);

subplot(1,3,2)
plot(Prior_MC_Vector(2,:),Weight,'.')
xlabel('Deviation in Grain Roughness [mm]');
title('Plotting of Weights after Rejection Sampling','FontWeight', 'bold');

%[histw, intervals] = histcounts(Prior_MC_Vector(2,:), Weight, 30); 
%bar(intervals, histw);
subplot(1,3,3)
plot(Prior_MC_Vector(3,:),Weight,'.')
xlabel('Deviation-Grain Size Distribution curve [mm]');
%}
figure;
scatter3(Prior_MC_Vector(1,:),Prior_MC_Vector(2,:),Prior_MC_Vector(3,:),'.');
    colorbar;
   hold on;
    scatter3(Prior_MC_Vector(1,:),Prior_MC_Vector(2,:),Prior_MC_Vector(3,:),Weight./Weight+1,Weight./Weight+2,'o');
    title('MC combinations in the parameter space and most probable region','FontWeight', 'bold');
    xlabel('Shields Parameter (Meyer-Peter Mueller Eq.)')
    ylabel('Deviation in Grain Roughness [m^1/3/s]')
    zlabel('Deviation-Grain Size Distribution curve [mm]')
    colorbar;
save(strcat(('Calibration_data_'),num2str(year1),'_',num2str(year2),'.mat'),'Weight','Prior_MC_Vector','Post_MC_Vector')   

