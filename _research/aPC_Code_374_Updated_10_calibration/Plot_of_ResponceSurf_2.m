%==================================================================================================================================================
% Surrogate model construction of the morphodynamic model "River Salzach"
% Author: Andres Heredia based on the work of Oladyshkin, S. and W. Nowak. Data-driven uncertainty quantification using the arbitrary polynomial chaos expansion. Reliability Engineering & System Safety, Elsevier, V. 106, P. 179–190, 2012. DOI: 10.1016/j.ress.2012.05.002. 
% SRC SimTech,Universitaet Stuttgart, Pfaffenwaldring 61, 70569 Stuttgart
% E-mail: baherediah@gmail.com
% 374 Calibration nodes
%
% The current program is using definition of aPC which is presented in the following manuscript: 
% Oladyshkin, S. and W. Nowak. Data-driven uncertainty quantification using the arbitrary polynomial chaos expansion. Reliability Engineering & System Safety, Elsevier, V. 106, P. 179–190, 2012. DOI: 10.1016/j.ress.2012.05.002. 
%==================================================================================================================================================


clear all;
format short g
for calibration_year=1:2
    
        if calibration_year==1
        year=2005;
    elseif calibration_year==2
        year=2010;
    end;
load(strcat('ManagementData_',num2str(calibration_year),'.mat'));
clear Input_distributions
rng(49);
for i=1:N   
   if i==1
    Input_distributions(i,:)=random('unif',0.033,0.047,1,100000); % Generation of 100000 candidates from our prior distribution-- Param # 1
   elseif i==2
   Input_distributions(i,:)=random('unif',-4,4,1,100000);
   elseif i==3
   Input_distributions(i,:)=random('unif',-0.2,4,1,100000);
   end
end
Input_distributions=Input_distributions'; % Store candidates as a matrix 100000 x 3
Indicator=[1:length(Input_distributions)]'; % Number of iteration-- I will use this to check which combination has the heighest Weight
Input_distributions_indicator=horzcat(Indicator,Input_distributions);
fprintf('\n');
fprintf('______________________________________________________________________________________________________________\n');
fprintf(strcat('\n---> Post-Processing on Arbitraty Polynomial Chaos for year\n',num2str(year),'\n'));
fprintf('\n');
tic


%==================================================================================================================================================
%======     Post-Processing on Arbitraty Polynomial Chaos
%==================================================================================================================================================


%Here you can see an Example how to construct Respose Surface

for index=1:1:374    %NumberOfOutputs; -- NumberofOutputs: Number of nodes? 
   
    %Step 2: Pre Computation for d and N
    for j=1:1:P;
        for ii=1:1:N;     
            load(PolynomialBasisFileName{ii},'Polynomial');
            Pre_Poly(ii,j,:)=polyval(Polynomial(1+PolynomialDegree(j,ii),length(Polynomial):-1:1),Input_distributions(:,ii));
        end
    end
    
    %Step 3: Computation RS on polynomials
    for i=1:1:length(Input_distributions)
            %Respose Surfance OUTPUT
            RS_Output(i)=0;
            for j=1:1:P;
                 multi=1;
                 for ii=1:1:N;          
                     multi=multi*Pre_Poly(ii,j,i); %(,i1,i2) Quite esto de este paso. Solo asi corrio el codigo
                 end             
                 RS_Output(i)=RS_Output(i)+Matrix_coeffs(j,index)*multi;
            end
    end     


% Construction of the Random Output matrix for the two calibration years
if calibration_year==1
Random_output_2005(:,index)=RS_Output;
end
if calibration_year==2
Random_output_2010(:,index)=RS_Output;
end
end
end
    %============ PLOTTING RESPONSE SURFACE-- Plots the response surface
    %only for the last node (374) and time step 2010
    
    figure(index);

    LFS=8; %Litle Font Size
    BFS=10; %Big Font Size     
    hold on;
    
    scatter3(Input_distributions(:,1),Input_distributions(:,2),Input_distributions(:,3),Random_output_2005(:,index),Random_output_2005(:,index),'.');
    colorbar;
    hold on;
    plot3(CollocationPointsBase([1:10],1),CollocationPointsBase([1:10],2),CollocationPointsBase([1:10],3),'k*','LineWidth',6);
    hold on;
    plot3(CollocationPointsBase([11:end],1),CollocationPointsBase([11:end],2),CollocationPointsBase([11:end],3),'g*','LineWidth',6);
    shading interp;
    colormap jet;
    view(-45,25);
    grid on;

    xlabel('Shields Parameter (Meyer-Peter Mueller Eq.)')
    ylabel('Deviation in Grain Roughness [mm]')
    zlabel('Deviation-Grain Size Distribution curve [mm]')
    set(gca,'FontSize',9)
    set(gca,'FontSize',8)
    title('Respose Surface','FontWeight', 'bold');
    set(gca,'Color',[1 1 1])
    set(gcf,'Color',[1 1 1]);
    legend('Original Collocation points ','New Collocation points (After Iterative Bayesian Update)');
    %set(gcf,'Position',[550*(index-1)+100 200 500 400]);
    %box on;

%=========================================================================
Random_output_2005=horzcat(Indicator,Random_output_2005);
Random_output_2010=horzcat(Indicator,Random_output_2010);

save('Random_elevation_outputs.mat','Random_output_2005','Random_output_2010','Input_distributions'); % Outputs will save as a matrix for each time step
toc
fprintf('______________________________________________________________________________________________________________\n');
fprintf('______________________________________________________________________________________________________________\n');
%close all;
