%% aPC Matlab Toolbox
% Data-driven Arbitrary Polynomial Chaos Expansion
% Author: Sergey Oladyshkin
% Developed year: 2009
% Stuttgart Center for Simulation Science
% Department of Stochastic Simulation and Safety Research for Hydrosystems,
% Institute for Modelling Hydraulic and Environmental Systems
% University of Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart
% E-mail: Sergey.Oladyshkin@iws.uni-stuttgart.de
% Phone: +49-711-685-60116
% Fax: +49-711-685-51073
% http://www.iws.uni-stuttgart.de

% The current aPC Matlab Toolbox is using definition of aPC that is presented in the following manuscripts: 
% Oladyshkin S. and Nowak W. Data-driven uncertainty quantification using the arbitrary polynomial chaos expansion. Reliability Engineering & System Safety, Elsevier, V. 106, P. 179–190, 2012.
% Oladyshkin S. and Nowak W. Incomplete statistical information limits the utility of high-order polynomial chaos expansions. Reliability Engineering & System Safety, 169, 137-148, 2018.

close all;

load('aPC.mat');
%% Post-Processing on Data-Driven Arbitraty Polynomial Chaos: an Example how to construct the Respose Surface for 2 Parameters
for index=1:aPC.NumberOfOutputs; 
    
    %% Step 1: Construction of Meshgrid for Inp1 and Inp2
    for ii=1:aPC.NumberOfInputs; 
        minVal=min(TrainingInput(:,ii));
        maxVal=max(TrainingInput(:,ii));
        step=100;
        Inp(ii,:)=minVal:(maxVal-minVal)/step:maxVal; %basic points
    end
    [X,Y] = meshgrid(Inp(1,:),Inp(2,:)); %mesh construction
    RS_Grid(1,:,:)=X;
    RS_Grid(2,:,:)=Y;

    %% Step 2: Pre Computation for d and N
    for j=1:aPC.NumberOfTerms;
        for ii=1:aPC.NumberOfInputs;     
            Polynomial=aPC.OrthonormalBasis(:,:,ii);
            Pre_Poly(ii,j,:,:)=polyval(Polynomial(1+aPC.MultivariatePolynomialDegrees(j,ii),end:-1:1),RS_Grid(ii,:,:));
        end
    end

    %% Step 3: Computation of Respose Surfance on polynomials
    for i1=1:1:length(RS_Grid)
    for i2=1:1:length(RS_Grid)
            %Respose Surface 
            ResponseSurface(i1,i2)=0;
            for j=1:aPC.NumberOfTerms;
                 multi=1;
                 for ii=1:aPC.NumberOfInputs;          
                     multi=multi*Pre_Poly(ii,j,i1,i2);
                 end             
                 ResponseSurface(i1,i2)=ResponseSurface(i1,i2)+aPC.ExpansionCoefficients(j,index)*multi;
            end
    end     
    end


    %% Visualization
    figure(3);
    surf(X,Y, ResponseSurface);
    shading interp;
    colormap jet;
    view(-45,25);
    grid on;
    xlabel('Parameter 1')
    ylabel('Parameter 2')
    zlabel('Response Surface');
    set(gca,'Color',[1 1 1])
    set(gcf,'Color',[1 1 1]);
    set(gcf,'Position',[900 200 500 400]);
    axis([min(TrainingInput(:,1)) max(TrainingInput(:,1)) min(TrainingInput(:,2)) max(TrainingInput(:,2)) -8 2 min(ResponseSurface(:)) max(ResponseSurface(:))]);
    box on;

    Movie(index) = getframe(gcf);
end


