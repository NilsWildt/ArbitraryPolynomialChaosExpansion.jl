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

tic;
clear all;
for calibration_year=1:2
    clearvars -EXCEPT calibration_year
if exist('simulation_CPoints.xlsx','file')
    delete('simulation_CPoints.xlsx');
else
end;
format short
fprintf('\n');
fprintf('______________________________________________________________________________________________________________\n');
fprintf('\n---> Massive aPC-based stochastic model reduction \n');
tic


%==================================================================================================================================================
%======     Main Initialization 
%==================================================================================================================================================

% Arbitratry Polynomial Initialization for Uncertanties and Design Parameters
fprintf('\n---> Initialization of Uncertanties and Design Parameters ...\n');
N=3; %Number of Uncertanties and Design Parameters
d=2; %d-order polynomial
P=factorial(N+d)/(factorial(N)*factorial(d)); %Total number of terms
number_of_simulations=P;
  %matrix_elev(number_of_simulations,iter_count);
  load(strcat('Matrix_elevations_',num2str(calibration_year),'.mat'));

%Initialization Uncertain and Design Parameters
%Example - Initialization of Uncertain Parameters
rng(8)
for i=1:N   
   if i==1
    Input_distributions(i,:)=random('unif',0.033,0.047,1,1000); % Shields parameter 
   elseif i==2
   Input_distributions(i,:)=random('unif',-4,4,1,1000); % Grain roughness
   elseif i==3
   Input_distributions(i,:)=random('unif',-0.2,4,1,1000); % Grain size distribution
   end
end
rst=Input_distributions;
%==================================================================================================================================================
%======     Construction of orthonormal aPC polynomial basis
%==================================================================================================================================================

%Computation of Arbitratry Polynomials
fprintf('\n---> Construction of orhtonormal aPC polynomial basis ...\n');
for i=1:N
    PolynomialBasisFileName(i)=strcat({'PolynomialBasis_'},num2str(i),{'.mat'});    
    aPoly_Construction(Input_distributions(i,:), d, PolynomialBasisFileName{i});
end

%Initialization of Collocation points 

for i=1:N
   load(PolynomialBasisFileName{i},'Roots');
    Cpoints(i,:)=Roots;
end



%==================================================================================================================================================
%======     Construction of optimal integration points
%==================================================================================================================================================

%Digital set of Collocation points combination - Manual 
fprintf('\n---> Construction of optimal integration points ...\n');
%---------- Digital set of Collocation points combination 
switch logical(true)
    case N==1
     DigitalUniqueCombinations=allcomb(1:d+1);
    case N==2
     DigitalUniqueCombinations=allcomb(1:d+1,1:d+1);
    case N==3
     DigitalUniqueCombinations=allcomb(1:d+1,1:d+1,1:d+1);
    case N==4
     DigitalUniqueCombinations=allcomb(1:d+1,1:d+1,1:d+1,1:d+1);
    case N==5
     DigitalUniqueCombinations=allcomb(1:d+1,1:d+1,1:d+1,1:d+1,1:d+1);
    case N==6
      DigitalUniqueCombinations=allcomb(1:d+1,1:d+1,1:d+1,1:d+1,1:d+1,1:d+1);
   %otherwise
   %   disp('---> Fatal Error: Number of input parameters is too big.'),
   %   disp('---> Hasta la vista, baby !'),
   %   break;
end

for i=1:1:length(DigitalUniqueCombinations) 
    DigitalPointsWeight(i)=sum(DigitalUniqueCombinations(i,:));                 
end
%Sorting of Posible Digital Points Weight
[SortDigitalPointsWeight, index_SDPW]=sort(DigitalPointsWeight); %[B,I] = sort(A,...) also returns a sort index I which specifies how the elements of A were rearranged to obtain the sorted output B. If A is a vector, then B = A(I)
SortDigitalUniqueCombinations=DigitalUniqueCombinations(index_SDPW,:);
%Ranking relatively mean (max as well possible too)
for j=1:N    
        temp(j,:)=abs(Cpoints(j,:)-mean(Input_distributions(j,:)));
end
[temp_sort, index_CP]=sort(temp,2);
for j=1:N    
    SortCpoints(j,:)=Cpoints(j,index_CP(j,:));
end
%Mapping of Digital Combination to Cpoint Combination
for i=1:1:length(SortDigitalUniqueCombinations) 
    for j=1:N
        SortUniqueCombinations(i,j)=SortCpoints(j,SortDigitalUniqueCombinations(i,j));
    end
end
CollocationPointsBase=SortUniqueCombinations(1:P,:); % Truncate of the polynomial. Order of expansion P 
NewCollocationPoint1=[0.038,-0.38,3.97];
NewCollocationPoint2=[0.047,0.02,1.77];
NewCollocationPoint3=[0.040,-0.52,3.52];
NewCollocationPoint4=[0.042,-0.62,3.30];
NewCollocationPoint5=[0.041,-0.82,3.10];
NewCollocationPoint6=[0.042,-0.97,2.72];
NewCollocationPoint7=[0.043,-1.22,2.19];
NewCollocationPoint8=[0.043,-1.35,2.12];
NewCollocationPoint9=[0.044,-1.39,1.81];
NewCollocationPoint10=[0.044,-1.42,2.0];
CollocationPointsBase(P+1,:)=NewCollocationPoint1;
CollocationPointsBase(P+2,:)=NewCollocationPoint2;
CollocationPointsBase(P+3,:)=NewCollocationPoint3;
CollocationPointsBase(P+4,:)=NewCollocationPoint4;
CollocationPointsBase(P+5,:)=NewCollocationPoint5;
CollocationPointsBase(P+6,:)=NewCollocationPoint6;
CollocationPointsBase(P+7,:)=NewCollocationPoint7;
CollocationPointsBase(P+8,:)=NewCollocationPoint8;
CollocationPointsBase(P+9,:)=NewCollocationPoint9;
CollocationPointsBase(P+10,:)=NewCollocationPoint10;
%Polynomail Degree Computation
PosibleDegree=0:1:length(Cpoints(1,:))-1;
for i=2:1:N
    PosibleDegree=[PosibleDegree,0:1:length(Cpoints(i,:))-1];    % Concatenation of 2 vectors (which have the possible degrees of the polynomials) In this case 2 Polynomials degree 2.
end
control_nchoosek=nchoosek(PosibleDegree,N); % Control para ver que hace nchoosek: nchoosek(V,K) where V is a vector of length N, produces a matrix with N!/K!(N-K)! rows and K columns. Each row of the result has K ofthe elements in the vector V.
UniqueDegreeCombinations=unique(nchoosek(PosibleDegree,N),'rows'); %  C = unique(A,'rows') for the matrix A returns the unique rows of A. The rows of the matrix C will be in sorted order.
%Posible Degree Weight Computation
for i=1:1:length(UniqueDegreeCombinations) 
    DegreeWeight(i)=0;
    for j=1:1:N
        DegreeWeight(i)=DegreeWeight(i)+UniqueDegreeCombinations(i,j);
    end
end
%Sorting of Posible Degree Weight
[SortDegreeWeight, i]=sort(DegreeWeight);
SortDegreeCombinations=UniqueDegreeCombinations(i,:);
%Set of MultiDim Collocation Points 
PolynomialDegree=SortDegreeCombinations(1:P,:);

%Storring the optimal integration Points
OptimalCollocationPointsBase=CollocationPointsBase;
cr_shields=OptimalCollocationPointsBase(:,1);
dev_ks=OptimalCollocationPointsBase(:,2);
dev_dm=OptimalCollocationPointsBase(:,3);
simulation_Parameters=table(cr_shields,dev_ks,dev_dm)
%filename = 'simulation_CPoints.xlsx';
%writetable(simulation_Parameters,filename,'Sheet','Hoja1','Range','A1');
%Run your model for "OptimalCollocationPointsBase"
%save output here

%==================================================================================================================================================
%======     Evaluation of the original physical model and reading of corresponding model outputs
%==================================================================================================================================================
fprintf('\n---> Runing the original phasical model -> model outputs ...\n');
%Very simple example for input/output data
  %Output=OptimalCollocationPointsBase(:,1)+OptimalCollocationPointsBase(:,2).^2
  for rr=1:374 % These loop indicates the number of nodes. In this case there are 374 nodes on the main river bed which own measured data
    clearvars Psi inv_PsiPsi; 
  Output= matrix_elev(rr,(2:end))'; % Data output obtained from the morphodynamic model. It is a matrix that has all the output values of all the models and all the nodes
Output_elevations=Output;


%CollocationPointsBase=input_s'; %Replacing optimal integration poins by expisting input data
s=size(Output);
NumberOfOutputs=s(2);
P_total=length(CollocationPointsBase); %Updating values of P



%==================================================================================================================================================
%======     Setting up of space/time independent matrix of arbitrary polynomials
%==================================================================================================================================================

fprintf('\n---> Setting up of space/time independent matrix of arbitrary polynomials ...\n');
%Initialization of matrix Psi for P-polynomials in P_total collocation points: space/time independent matrix PP*P_total (of polynomials)
for i=1:1:P;        
    for j=1:1:P_total;        
        Psi(i,j)=1;
        for ii=1:1:N;             
            load(PolynomialBasisFileName{ii},'Polynomial');
            Psi(i,j)=Psi(i,j)*polyval(Polynomial(1+PolynomialDegree(i,ii),length(Polynomial):-1:1),CollocationPointsBase(j,ii));
            % Evaluation in each integration point            
        end        
    end
end


%==================================================================================================================================================
%======     Computation of the expansion coefficients for arbitrary Polynomial Chaos
%==================================================================================================================================================

fprintf('\n---> Computation of the expansion coefficients ...\n');
%Equi-weighted projection, leading to least square fit of coefficients
Psi=Psi';
clear temp;
inv_PsiPsi=pinv(Psi);
%Multi-dimensional expansion coeffs:
for j=1:1:NumberOfOutputs;
    for ii=1:1:P_total;
        temp(ii)=Output(ii,j);
    end
    Coeffs(:,j) = inv_PsiPsi*temp'     ;                 
end
Matrix_coeffs(:,rr)=Coeffs;


%==================================================================================================================================================
%======     Computation of output statistics: mean and variance
%==================================================================================================================================================

fprintf('\n---> Computation of output statistics: mean and variance...\n');
%Analytical form of mean and variance
for j=1:1:NumberOfOutputs;    
    OutputMean(j)=Coeffs(1,j);
    OutputVar(j)=sum(Coeffs(2:P,j).^2);
end

fprintf('\n---> Mean value of model outputs: %d:\n');
disp(OutputMean);
fprintf('---> Variance of model outputs: %d:\n');
disp(OutputVar);
Variance_vector(rr,1)=OutputVar;


save(strcat('ManagementData_',num2str(calibration_year),'.mat'));
fprintf('\n');
fprintf('---> Calculations are successfully completed \n\n');
%toc
fprintf('\n');
fprintf('______________________________________________________________________________________________________________\n');
  end
filenamecoeffs=strcat('Matrix_coeffs_',num2str(calibration_year),'.mat');  
save(filenamecoeffs,'Matrix_coeffs');
end
toc;