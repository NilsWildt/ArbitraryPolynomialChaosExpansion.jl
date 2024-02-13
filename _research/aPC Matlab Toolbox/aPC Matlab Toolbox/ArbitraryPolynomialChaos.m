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

classdef ArbitraryPolynomialChaos
  
   properties 
       InputDistribution % Data Driven Input Distribution
       NumberOfInputs %Number of input parameters     
       NumberOfOutputs % Number Of Outputs in Pycical Space X,Y;Z and t
       ExpansionDegree %Expansion Degree
       NumberOfTerms %Number of Expansion terms
       MultivariatePolynomialDegrees %Multivariate Polynomial Degrees 
       OrthonormalRepresentation = 'Yes' % Repesentation via Orthonormal Basis: Yes or No
       OrthonormalBasis %Orthonormal Data Driven Polynomial Basis
       ExpansionCoefficients %Expansion Coefficients
   end
   
   methods

       
      %% Initialization of Arbitrary Polynomial Chaos
      function obj = Initialization(obj) 
        % Construction of orthonormal aPC polynomial basis 
        for i=1:obj.NumberOfInputs
            fprintf('=> aPC Toolbox: Construction of data-driven polynomial basis for parameter %1i ... \n', i);
            obj.OrthonormalBasis(:,:,i)=aPC_OrthonormalBasis(obj.InputDistribution(:,i)', obj.ExpansionDegree+1);
        end
        %Number of Expansion terms
        obj.NumberOfTerms=factorial(obj.NumberOfInputs+obj.ExpansionDegree)/(factorial(obj.NumberOfInputs)*factorial(obj.ExpansionDegree));          
      end
      
      
      %% Construction of Gaussian Collocaiton Training Points
      function [TrainingInput, obj] = GaussianCollocaiton(obj,Strategy)   
        %Strategy - Selection of Gaussian Training Points: 'FT'- Full Tensor Grid or 'PCM' - Probabilistic Collocation Method

        %Construction of corresponding collocation Training points 
        AvailableCollocationPoints=zeros(obj.NumberOfInputs,obj.ExpansionDegree+1);
        for i=1:obj.NumberOfInputs
            Polynomial=obj.OrthonormalBasis(:,:,i);
            AvailableCollocationPoints(i,:)=roots(fliplr(Polynomial(obj.ExpansionDegree+2,:))); %Gaussian Roots of Degree+1 polynomials
        end
        
        %Unique combination of collocation points
        [List{obj.NumberOfInputs:-1:1}]=ndgrid(1:obj.ExpansionDegree+1);
        PointsVector=1:obj.ExpansionDegree+1;
        UniqueCombinations=PointsVector(reshape(cat(obj.NumberOfInputs+1,List{:}),[],obj.NumberOfInputs)); 
        if obj.NumberOfInputs==1    
            UniqueCombinations=UniqueCombinations';
        end
        DigitalPointsWeight=zeros(length(UniqueCombinations),1);
        for i=1:1:length(UniqueCombinations) 
            DigitalPointsWeight(i)=sum(UniqueCombinations(i,:));                 
        end
        %Sorting of weights
        [~, index_SDPW]=sort(DigitalPointsWeight);
        SortUniqueCombinations=UniqueCombinations(index_SDPW,:);
        %Full Tensor Grid 
        if strcmp(Strategy,'FT')
            TrainingInput=SortUniqueCombinations;
        end
        
        %Probabilistic Collocation Selection
        if strcmp(Strategy,'PCM')
            %PCM relative ranking via mean value 
            temp=zeros(obj.NumberOfInputs,size(AvailableCollocationPoints,2));
            for j=1:obj.NumberOfInputs    
                temp(j,:)=abs(AvailableCollocationPoints(j,:)-mean(obj.InputDistribution(:,j)'));
    
            end
            [~, index_CP]=sort(temp,2);
            SortAvailableCollocationPoints=zeros(obj.NumberOfInputs,size(AvailableCollocationPoints,2));
            for j=1:obj.NumberOfInputs    
                SortAvailableCollocationPoints(j,:)=AvailableCollocationPoints(j,index_CP(j,:));
            end
            %Mapping of unique combination to collocation points
            for i=1:1:length(SortUniqueCombinations) 
                for j=1:obj.NumberOfInputs
                    SortUniqueCombinations(i,j)=SortAvailableCollocationPoints(j,SortUniqueCombinations(i,j));
                end
            end
            %Optimal P-Collocation Points
            TrainingInput=SortUniqueCombinations(1:obj.NumberOfTerms,:);
        end       
      end
      
      
      %% Training of Arbitrary Polynomial Chaos
      function obj = train(obj,TrainingInput,TrainingOutput) 
        % TrainingInput - Training Input
        % TrainingOutput - Training Output
        
        fprintf('=> aPC Toolbox: Training of Arbitrary Polynomial Chaos ... \n');
        % Initialization of Multivariate Polynomial Degrees             
        obj.MultivariatePolynomialDegrees=aPC_MultivariatePolynomialDegrees(obj.NumberOfInputs, obj.ExpansionDegree); 
        
        % Setting up of space/time independent matrix of arbitrary polynomials 
        Psi=aPC_PsiPolynomialMatrix(obj.NumberOfInputs, obj.ExpansionDegree, obj.MultivariatePolynomialDegrees, obj.InputDistribution, TrainingInput, obj.OrthonormalRepresentation)';

        % Training of multi-dimensional expansion coefficients for arbitrary Polynomial Chaos
        obj.NumberOfOutputs=size(TrainingOutput,2);
        P_total=size(TrainingInput,1);   
        temp=zeros(1,P_total);
        for j=1:obj.NumberOfOutputs;
            for ii=1:P_total;
                temp(ii)=TrainingOutput(ii,j);
            end
            obj.ExpansionCoefficients(:,j) = pinv(Psi)*temp'; %Least-square optimization problem with Regularization                                    
        end
      end
      
      
      %% Prediction using Arbitrary Polynomial Chaos
      function PredictionOutput = predict(obj,PredictionInput) 
        % PredictionInput- Prediction Input
        % PredictionOutput - Prediction Output
        
        fprintf('=> aPC Toolbox: Prediction using Arbitrary Polynomial Chaos ... \n');
        % Setting up of space/time independent matrix of arbitrary polynomials 
        Psi=aPC_PsiPolynomialMatrix(obj.NumberOfInputs, obj.ExpansionDegree, obj.MultivariatePolynomialDegrees, obj.InputDistribution, PredictionInput, obj.OrthonormalRepresentation)';
        
        % Prediction using multi-dimensional expansion coefficients for arbitrary Polynomial Chaos
        PredictionOutput=zeros(size(PredictionInput,1),obj.NumberOfOutputs);
        for j=1:obj.NumberOfOutputs;
            PredictionOutput(:,j)=Psi*obj.ExpansionCoefficients(:,j);                    
        end       
      end
      
      %% Uncertainty Quantification using Arbitrary Polynomial Chaos
      function [OutputMean, OutputVar] = UQ(obj) 
        % OutputMean - Output Mean
        % OutputVar - Output Variance
      
        fprintf('=> aPC Toolbox: Uncertainty Quantification ... \n');
        % Computation of output statistics: mean and variance
        for j=1:obj.NumberOfOutputs    
            OutputMean(j)=obj.ExpansionCoefficients(1,j); %Analytical form of mean 
            OutputVar(j)=sum(obj.ExpansionCoefficients(2:end,j).^2); %Analytical form of variance
        end
      end

      
      %% Global sensitivity analysis using Arbitrary Polynomial Chaos
      function [SortSobolIndices, SobolTotal] = GSA(obj) 
        % SortSobolInices - Sorted Sobol Indices
        % SobolTotal - Sobol Total Indices
        
        fprintf('=> aPC Toolbox: Global Sensitivity Analysis ... \n');
        % Computation of Sobol Indices    
        for id=1:obj.NumberOfOutputs    

            %Computationnal of Location and Variance
            for j=2:1:obj.NumberOfTerms;
                location=find(obj.MultivariatePolynomialDegrees(j,:)~=0);
                LocationInfo(j,location)=location;    
                TermsVar(j)=obj.ExpansionCoefficients(j,id).^2/sum(obj.ExpansionCoefficients(2:end,id).^2);
            end
            %Initialization and Definitaion of Dublicates
            for j=2:1:obj.NumberOfTerms;    
                TempSobol(j)=TermsVar(j); %Initialization
                for i=j:1:obj.NumberOfTerms
                    if ((LocationInfo(j,:)==LocationInfo(i,:)) & (j~=i)) %Dublicates
                        TempSobol(j)=TermsVar(j)+TermsVar(i);            
                        DublicatesIndex(j)=i;
                    end        
                end
            end    
            %Final defenition of Indices
            i=0;
            for j=2:1:obj.NumberOfTerms;
               if  j~=DublicatesIndex
                   i=i+1;
                   Sobol(i)=TempSobol(j);
                   SobolInfo(i,:)=LocationInfo(j,:);
               end    
            SobolInices(i,1)=Sobol(i);
            SobolInices(i,2:obj.NumberOfInputs+1)=SobolInfo(i,:);
            end
            %Sorted Sobol Indecis
            [~,index]=sort(Sobol);
            SortSobolIndices(:,:,id)=SobolInices(index(end:-1:1),:); 

            % Computation of Total Sobol Indices
            for ii=1:obj.NumberOfInputs;
                location=find(obj.MultivariatePolynomialDegrees(:,ii)~=0);
                SobolTotal(ii,id)=sum(obj.ExpansionCoefficients(location,id).^2)/sum(obj.ExpansionCoefficients(2:end,id).^2);
            end
        end
      end
      
   end
   
end