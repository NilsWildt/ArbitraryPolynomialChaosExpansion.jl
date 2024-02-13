%==================================================================================================================================================
% Construction of Data-driven Orthonormal Polynomial Basis
% Author: Dr.-Ing. Sergey Oladyshkin
% SRC SimTech,Universitaet Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart
% E-mail: Sergey.Oladyshkin@iws.uni-stuttgart.de
%
% The current program is using definition of aPC which is presented in the following manuscript:
% Oladyshkin, S. and W. Nowak. Data-driven uncertainty quantification using the arbitrary polynomial chaos expansion. Reliability Engineering & System Safety, Elsevier, V. 106, P. 179�190, 2012. DOI: 10.1016/j.ress.2012.05.002.
%==================================================================================================================================================

%surrogate model for SC MICP model


clear all;
fprintf('\n');
fprintf('______________________________________________________________________________________________________________\n');
fprintf('\n---> massive aPC-based stochastic model reduction \n');

tic

%==================================================================================================================================================
%======     main initialization
%==================================================================================================================================================

% arbitrary polynomial initialization for uncertanties and design parameters

fprintf('\n---> initialization of uncertanties and design parameters ...\n');
N = 2; %number of incertanties and design parameters
d = 2; %polynomial order
P = factorial(N+d)/(factorial(N)*factorial(d)); %total number of terms
F = 1; %factor for number of collocation points
numberOfPointsSpace = 57; %number of spacial points
numberOfPointsTime = 35; %number of points in time


%initialization uncertain and design parameters

%Ishigami
%Input_distributions = random('unif',-3.14,3.14,N,1000);
%dlmwrite('Input_distributions.txt',Input_distributions)

%MICP
Input_distributions = zeros(N,1000);
Input_distributions(1,:) = random('unif',1,15,1,1000); %rhoBiofilm,3
Input_distributions(2,:) = random('unif',10^-5,5*10^-4,1,1000); %kub,4

%==================================================================================================================================================
%======     construction of orthonormal aPC polynomial basis
%==================================================================================================================================================

%computation of arbitrary polynomials

fprintf('\n---> construction of orthonormal aPC polynomial basis ...\n');
for i=1:N
    PolynomialBasisFileName(i)=strcat({'PolynomialBasis_'},num2str(i),{'.mat'});
    aPoly_Construction(Input_distributions(i,:), d, PolynomialBasisFileName{i});
    %dlmwrite('PolynomialBasis.txt',kk)
end


%initialization of collocation points

for i=1:N
    load(PolynomialBasisFileName{i},'Roots')
    Cpoints(i,:)= Roots;
end


%==================================================================================================================================================
%==================================================================================================================================================
%======     computation of the original surrogate model
%==================================================================================================================================================
%==================================================================================================================================================


%==================================================================================================================================================
%======     construction of optimal integration points
%==================================================================================================================================================

%digital set of collocation points combination - manual

fprintf('\n---> construction of optimal integration points ...\n');
%---------- Digital set of collocation points combination
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


%sorting of possible digital points weight

[SortDigitalPointsWeight, index_SDPW]=sort(DigitalPointsWeight);
SortDigitalUniqueCombinations=DigitalUniqueCombinations(index_SDPW,:);


%ranking relatively mean (max as well possible too)

for j=1:N
    temp(j,:)=abs(Cpoints(j,:)-mean(Input_distributions(j,:)));
end
[temp_sort, index_CP]=sort(temp,2);
for j=1:N
    SortCpoints(j,:)=Cpoints(j,index_CP(j,:));
end


%mapping of digital combination to collocation point combination
for i=1:1:length(SortDigitalUniqueCombinations)
    for j=1:N
        SortUniqueCombinations(i,j)=SortCpoints(j,SortDigitalUniqueCombinations(i,j));
    end
end
CollocationPointsBase=SortUniqueCombinations(1:round(F*P),:);


%polynomial degree computation
PosibleDegree=0:1:length(Cpoints(1,:))-1;
for i=2:1:N
    PosibleDegree=[PosibleDegree,0:1:length(Cpoints(i,:))-1];
end
UniqueDegreeCombinations=unique(nchoosek(PosibleDegree,N),'rows');

%possible degree weight computation
for i=1:1:length(UniqueDegreeCombinations)
    DegreeWeight(i)=0;
    for j=1:1:N
        DegreeWeight(i)=DegreeWeight(i)+UniqueDegreeCombinations(i,j);
    end
end


%sorting of possible degree weight

[SortDegreeWeight, i] = sort(DegreeWeight);
SortDegreeCombinations = UniqueDegreeCombinations(i,:);


%set of multi-dim collocation points
PolynomialDegree = SortDegreeCombinations(1:P,:);


%storring the optimal integration points
OptimalCollocationPointsBase = CollocationPointsBase;
dlmwrite('OptimalCollocationPointsBase.txt',OptimalCollocationPointsBase)



%==================================================================================================================================================
%======     evaluation of the original physical model and reading of corresponding model outputs
%==================================================================================================================================================

fprintf('\n---> running the original phasical model -> model outputs ...\n');

%------------
%Ishigami
%------------

%Output=ishigami(OptimalCollocationPointsBase, 7, 0.1);

%------------
%Random
%------------
% OutputCt = random('unif',0,1,P,numberOfPointsSpace);
% for i = 1:P
%     OutputCa(:,:,i) = random('unif',0,1,numberOfPointsTime,numberOfPointsSpace);
% end
% save OutputCt.mat OutputCt;
% save OutputCa.mat OutputCa;


%------------
%MICP
%------------

OutputCt = ones(P,numberOfPointsSpace);
OutputCa = ones(numberOfPointsTime,numberOfPointsSpace,P);

save OutputCt.mat OutputCt;
save OutputCa.mat OutputCa;

%TODO: adapt to inputfile of model
%initial values
rhoBiofilm_old = 'rhoBiofilm = 6.9 #10 # 			//[kg/m3] 		kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';
kub_old = 'kub = 3.81e-4 # 			// [kg_urease/kg_bio] 		calculated from kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';


%folder for each model run
for inp = 1:F*P
    new_folder = strcat('mkdir Simulation',num2str(inp));
    system(new_folder)
end


%evaluating the model, moving the resulting files
for inp = 1:F*P
    rhoBiofilm = strcat('rhoBiofilm = ',num2str(OptimalCollocationPointsBase(inp,1)),' #');
    kub = strcat('kub = ',num2str(OptimalCollocationPointsBase(inp,2)),' #');
    
    %change corresponding input values in input file
    change_Inputs('column8_nosuspendedbiomass.input', rhoBiofilm_old, rhoBiofilm);
    change_Inputs('column8_nosuspendedbiomass.input', kub_old, kub);
    
    %copy necessary files
    move_input = strcat('cp column8_nosuspendedbiomass.input Simulation',num2str(inp));
    system(move_input);
    move_exec = strcat('cp micp_no_susp_bio Simulation',num2str(inp));
    system(move_exec);
    move_inj = strcat('cp Column8InjNoSuspBio.dat Simulation',num2str(inp));
    system(move_inj);
    move_paraRef = strcat('cp paraRef.m Simulation',num2str(inp));
    system(move_paraRef);
    
    %to be setted values in next run
    rhoBiofilm_old = rhoBiofilm;
    kub_old = kub;  
end

    %save intermediate variables
    save intermediate.mat;