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
N = 4; %number of incertanties and design parameters
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
Input_distributions(1,:) = random('unif',10^-10,10^-7,1,1000); %ca1,1
Input_distributions(2,:) = random('unif',10^-10,10^-6,1,1000); %ca2,2
Input_distributions(3,:) = random('unif',1,15,1,1000); %rhoBiofilm,3
Input_distributions(4,:) = random('unif',10^-5,5*10^-4,1,1000); %kub,4

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
ca1_old = 'ca1 = 8.3753e-8 #0.04434 # 		// [1/s] 		kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';
ca2_old = 'ca2 = 8.5114e-7 #9.187e-4 # 	// [1/s] 		kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';
rhoBiofilm_old = 'rhoBiofilm = 6.9 #10 # 			//[kg/m3] 		kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';
kub_old = 'kub = 3.81e-4 # 			// [kg_urease/kg_bio] 		calculated from kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';


%folder for each model run
for inp = 1:F*P
    new_folder = strcat('mkdir Simulation',num2str(inp));
    system(new_folder)
end


%evaluating the model, moving the resulting files
for inp = 1:F*P
    ca1 = strcat('ca1 = ',num2str(OptimalCollocationPointsBase(inp,1)),' #');
    ca2 = strcat('ca2 = ',num2str(OptimalCollocationPointsBase(inp,2)),' #');
    rhoBiofilm = strcat('rhoBiofilm = ',num2str(OptimalCollocationPointsBase(inp,3)),' #');
    kub = strcat('kub = ',num2str(OptimalCollocationPointsBase(inp,4)),' #');
    
    %change corresponding input values in input file
    change_Inputs('column8.input', ca1_old, ca1);
    change_Inputs('column8.input', ca2_old, ca2);
    change_Inputs('column8.input', rhoBiofilm_old, rhoBiofilm);
    change_Inputs('column8.input', kub_old, kub);
    
    %to be setted values in next run
    ca1_old = ca1;
    ca2_old = ca2;
    rhoBiofilm_old = rhoBiofilm;
    kub_old = kub;
    
    %save intermediate variables
    save intermediate.mat;
    
    %run model
    system('./micp_simple_chemistry column8.input');
    
    run paraRef.m;
    load intermediate.mat;
    load MICPResult.mat;
    Calcite = ref_volFracCalcite;
    Calcium = ref_molarityCa2_w;
    
    %put needed files to corresponding folder
    get_to_folder_vtp = strcat('mv *.vtp Simulation', num2str(inp),'/');
    system(get_to_folder_vtp);
    get_to_folder_pvd = strcat('mv *.pvd Simulation', num2str(inp),'/');
    system(get_to_folder_pvd);
    get_to_folder_MICP = strcat('mv MICPResult.mat Simulation', num2str(inp),'/');
    system(get_to_folder_MICP);
    
    load intermediate.mat;
end

%change input values to initial values
ca1 = 'ca1 = 8.3753e-8 #0.04434 # 		// [1/s] 		kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';
ca2 = 'ca2 = 8.5114e-7 #9.187e-4 # 	// [1/s] 		kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';
rhoBiofilm = 'rhoBiofilm = 6.9 #10 # 			//[kg/m3] 		kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';
kub = 'kub = 3.81e-4 # 			// [kg_urease/kg_bio] 		calculated from kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';

change_Inputs('column8.input', ca1_old, ca1);
change_Inputs('column8.input', ca2_old, ca2);
change_Inputs('column8.input', rhoBiofilm_old, rhoBiofilm);
change_Inputs('column8.input', kub_old, kub);

%save outputs in one output file per poi
for inp = 1:F*P
    copy_Output_down = strcat('cp OutputCt.mat Simulation',num2str(inp),'/');
    system(copy_Output_down);
    copy_Output_down = strcat('cp OutputCa.mat Simulation',num2str(inp),'/');
    system(copy_Output_down);
    get_to_folder = strcat('Simulation', num2str(inp),'/');
    cd(get_to_folder)
    load OutputCt.mat OutputCt;
    load OutputCa.mat OutputCa;
    load 'MICPResult.mat';
    
    %output row (over spacial points)
    OutputCt(inp,:) = ref_volFracCalcite(end,:); %volFracCalcite
    ref_Calcium = ref_molarityCa2_w; %molarityCa2_w
    OutputCa(:,:,inp) = [ref_Calcium(2:23,:);ref_Calcium(25:37,:)];
    
    save OutputCt.mat OutputCt;
    save OutputCa.mat OutputCa;
    copy_Output_up = strcat('cp OutputCt.mat ..');
    system(copy_Output_up);
    copy_Output_up = strcat('cp OutputCa.mat ..');
    system(copy_Output_up);
    cd ..;
end

%replacing optimal integration points by existing input data
sCt = size(OutputCt);
sCa = size(OutputCa);

NumberOfOutputsCt = sCt(2);
NumberOfOutputsCa = sCa(2);

%updating values of P
P_total=length(CollocationPointsBase);


%==================================================================================================================================================
%======     setting up of space/time independent matrix of arbitrary polynomials
%==================================================================================================================================================

fprintf('\n---> setting up of space/time independent matrix of arbitrary polynomials ...\n');

%initialization of matrix Psi for P-polynomials in P_total collocation points: space/time independent matrix PP*P_total (of polynomials)

for i=1:1:P
    for j=1:1:P_total
        Psi(i,j)=1;
        for ii=1:1:N
            load(PolynomialBasisFileName{ii},'Polynomial');
            Psi(i,j)=Psi(i,j)*polyval(Polynomial(1+PolynomialDegree(i,ii),length(Polynomial):-1:1),CollocationPointsBase(j,ii)); % evaluation in each integration point
        end
    end
end

%==================================================================================================================================================
%======     computation of the expansion coefficients for aPC
%==================================================================================================================================================

fprintf('\n---> Computation of the expansion coefficients ...\n');

%equi-weighted projection, leading to least square fit of coefficients

Psi=Psi';
clear temp;
inv_PsiPsi=pinv(Psi'*Psi);


%------------
%multi-dimensional expansion coeffs
%------------

%Calcite
for j=1:1:NumberOfOutputsCt
    for ii=1:1:P_total
        temp(ii)=OutputCt(ii,j);
    end
    CoeffsCt(:,j) = inv_PsiPsi*Psi'*temp';
    %disp(CoeffsCt(:,j))
end

%Calcium
for i = 1:1:numberOfPointsTime
    for j=1:1:NumberOfOutputsCa
        for ii=1:1:P_total
            temp(ii)=OutputCa(i,j,ii);
        end
        CoeffsCa(i,j,:) = inv_PsiPsi*Psi'*temp';
        %disp(CoeffsCa(i,j,:))
    end
    
end
Psi = Psi';

%Psi evaluated on collocation points
save Psi.mat Psi;

%coeffs needed to evaluate surrogate model on other points
save CoeffsCt.mat CoeffsCt;
save CoeffsCa.mat CoeffsCa;
save intermediate.mat;


