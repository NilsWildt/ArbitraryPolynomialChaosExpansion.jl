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
% Oladyshkin S., de Barros F. P. J. and Nowak W. Global sensitivity analysis: a flexible and efficient framework with an example from stochastic hydrogeology. Advances in Water Resources 37 (2012): 10-22.

clear all;
tic

%% Initialization of uncertain parameters
InputDistribution(1,:)=random('Beta',2,1,1,1000); %This is an example, you can load your data, etc.
InputDistribution(2,:)=random('Uniform',0,1,1,1000); %This is an example, you can load your data, etc.

%% Initialzation of Arbitrary Polynomial Chaos
aPC = ArbitraryPolynomialChaos; %Object Arbitrary Polynomial Chaos
aPC.InputDistribution = InputDistribution'; % Data Driven Input Distribution
aPC.NumberOfInputs = 2; %Number of input parameters
aPC.ExpansionDegree = 3; %Expansion Degree
aPC = Initialization(aPC); %Initialization of Arbitrary Polynomial Chaos

%% Construction of Input Training Data Set
TrainingInput = GaussianCollocaiton(aPC,'PCM'); %Selection of Gaussian  Training Points: 'FT'- Full Tensor Grid or 'PCM' - Probabilistic Collocation Method

%% Construction of Output Training Data Set - Evaluation of the original physical model 
NumberOfOutputs=100;
for i=1:aPC.NumberOfTerms;     
    TrainingOutput(i,:)=PhysicalModel(1:NumberOfOutputs,TrainingInput(i,:)); %Very simple example for non-linear dynamic system
end

%% Training of Arbitrary Polynomial Chaos
aPC=train(aPC,TrainingInput,TrainingOutput); 
        
%% Prediction using Arbitrary Polynomial Chaos
PredictionOutput=predict(aPC,aPC.InputDistribution);

%% Uncertainty Quantification using Arbitrary Polynomial Chaos
[OutputMean, OutputVar] = UQ(aPC);

%% Global sensitivity analysis using Arbitrary Polynomial Chaos
[SortSobolIndices, SobolTotal] = GSA(aPC);
      
%% Saving the results
save('aPC.mat');
aPC
fprintf('=> aPC Toolbox: Machine learning via Arbitrary Polynomial Chaos has been successfully completed. \n\n');
toc
