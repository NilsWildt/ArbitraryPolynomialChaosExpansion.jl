%after evaluating the model

load intermediate.mat;


%change input values to initial values
rhoBiofilm_old = 'rhoBiofilm = 6.9 #10 # 			//[kg/m3] 		kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';
kub_old = 'kub = 3.81e-4 # 			// [kg_urease/kg_bio] 		calculated from kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';


%save outputs in one output file per poi
for inp = 1:F*P
    save intermediate.mat;
    copy_intermediate_down = strcat('cp intermediate.mat Simulation',num2str(inp),'/');
    system(copy_intermediate_down);
    copy_Output_down = strcat('cp OutputCt.mat Simulation',num2str(inp),'/');
    system(copy_Output_down);
    copy_Output_down = strcat('cp OutputCa.mat Simulation',num2str(inp),'/');
    system(copy_Output_down);
    get_to_folder = strcat('Simulation', num2str(inp),'/');
    cd(get_to_folder)
    
    run paraRef.m;
    load OutputCt.mat OutputCt;
    load OutputCa.mat OutputCa;
    load intermediate.mat;
    load MICPResult.mat;
    
    
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


%calcite
for j=1:1:NumberOfOutputsCt
    for ii=1:1:P_total
        temp(ii)=OutputCt(ii,j);
    end
    CoeffsCt(:,j) = inv_PsiPsi*Psi'*temp';
    %disp(CoeffsCt(:,j))
end


%calcium
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



%==================================================================================================================================================
%==================================================================================================================================================
%======     computation of the new collocation points to improve model
%==================================================================================================================================================
%==================================================================================================================================================



%==================================================================================================================================================
%======     compute LOOCV evolution
%==================================================================================================================================================

%------------
%LOOCV evolution
%------------

%initialize
loocvCt = zeros(6,numberOfPointsSpace);
loocvCa = zeros(6,numberOfPointsSpace,numberOfPointsTime);

loocvCt(1,:) = loocv(Psi, OutputCt);
for i = 1:numberOfPointsTime
    loocvCa(1,:,i) = loocv(Psi, reshape(OutputCa(i,:,:),numberOfPointsSpace,P)');
end


%------------
%surrogate data  evaluated on first collocation points
%------------
surrogateCt_allCPoints = Psi'*CoeffsCt;

for i = 1:numberOfPointsTime
    surrogateCa_allCPoints(i,:,:) = (Psi'*reshape(CoeffsCa(i,:,:),numberOfPointsSpace,P)')';
end


%surrogate data only on measured points, matrices (CP)
surrogateCt_measPoints = [surrogateCt_allCPoints(:,4),surrogateCt_allCPoints(:,10),surrogateCt_allCPoints(:,16),...
    surrogateCt_allCPoints(:,22),surrogateCt_allCPoints(:,28),surrogateCt_allCPoints(:,34),...
    surrogateCt_allCPoints(:,40),surrogateCt_allCPoints(:,46)];
surrogateCa_measPoints = [surrogateCa_allCPoints(:,9,:),surrogateCa_allCPoints(:,16,:),surrogateCa_allCPoints(:,24,:),...
    surrogateCa_allCPoints(:,31,:),surrogateCa_allCPoints(:,39,:)];


%model data only on measured points, matrices (CP)
modelCt_measPoints = [OutputCt(:,4),OutputCt(:,10),OutputCt(:,16),OutputCt(:,22),OutputCt(:,28),...
    OutputCt(:,34),OutputCt(:,40),OutputCt(:,46)];
modelCa_measPoints = [OutputCa(:,9,:),OutputCa(:,16,:),OutputCa(:,24,:),...
    OutputCa(:,31,:),OutputCa(:,39,:)];


%------------
%MC points to evaluate surrogate model
%------------


%initialize
MC_points = zeros(N,1000);
MC_points(1,:) = random('unif',1,15,1,1000); %rhoBiofilm,3
MC_points(2,:) = random('unif',10^-5,5*10^-4,1,1000); %kub,4


%evaulate MC Points on basis of surrogate model
MC_points = MC_points';
for i=1:1:P
    for j=1:1:length(MC_points)
        evaluatedBasis(i,j)=1;
        for ii=1:1:N
            load(PolynomialBasisFileName{ii},'Polynomial');
            evaluatedBasis(i,j)=evaluatedBasis(i,j)*polyval(Polynomial(1+PolynomialDegree(i,ii),length(Polynomial):-1:1),MC_points(j,ii)); % evaluation in each integration point
        end
    end
end


%surrogate data  evaluated on MC points
surrogateCt_MCPoints = evaluatedBasis'*CoeffsCt;
for i = 1:numberOfPointsTime
    surrogateCa_MCPoints(i,:,:) = (evaluatedBasis'*reshape(CoeffsCa(i,:,:),numberOfPointsSpace,P)')';
end


%surrogate data only on measured points, matrices (MC)
surrogateCt_measMCPoints = [surrogateCt_MCPoints(:,4),surrogateCt_MCPoints(:,10),surrogateCt_MCPoints(:,16),...
    surrogateCt_MCPoints(:,22),surrogateCt_MCPoints(:,28),surrogateCt_MCPoints(:,34),...
    surrogateCt_MCPoints(:,40),surrogateCt_MCPoints(:,46)];
surrogateCa_measMCPoints = [surrogateCa_MCPoints(:,9,:),surrogateCa_MCPoints(:,16,:),surrogateCa_MCPoints(:,24,:),...
    surrogateCa_MCPoints(:,31,:),surrogateCa_MCPoints(:,39,:)];


%==================================================================================================================================================
%======     computation of the maximum likelihood
%=================================================================================================================================================


%observed data
load observedCt.mat;
load measuredCa.mat;
load StdCt.mat;
observedCa = reshape(measuredCa',1,numel(measuredCa));
observedData = [observedCt, observedCa];


%R
RCt = diag((0.2.*observedCt).^2);
RCa = diag((0.2 .* observedCa).^2);
R = diag([(0.2.*observedCt).^2,(0.2 .* observedCa).^2]);

for P_total = P+1:P+10
    
    %------------
    %find point with max likelihood
    %------------
    maxlikelihood = 10^1000;
    new_point = [random('unif',1,15,1,1),random('unif',10^-5,5*10^-4,1,1)];
    
    for pointnumber = 1:length(MC_points)
        point = MC_points(pointnumber,:);
        
        
        %surrogate data only on measured points, row vectors
        surrogateCt = surrogateCt_measMCPoints(pointnumber,:);
        surrogateCa = reshape(surrogateCa_measMCPoints(:,:,pointnumber)',1,numel(surrogateCa_measMCPoints(:,:,pointnumber)));
        surrogateData = [surrogateCt,surrogateCa];
        
        new_maxlikelihood = (observedData-surrogateData)*pinv(R)*(observedData-surrogateData)';
        
        if new_maxlikelihood < maxlikelihood && ismember(point,CollocationPointsBase,'rows') == 0
            maxlikelihood = new_maxlikelihood;
            new_point = point;
        end
    end
    
    CollocationPointsBase = [CollocationPointsBase; new_point];
    
    
    
    %==================================================================================================================================================
    %======     evaluate original model at new collocation point
    %==================================================================================================================================================
    
    fprintf('\n---> running the original phasical model -> model outputs ...\n');
    
    %------------
    %MICP
    %------------
    
    %TODO: adapt to inputfile of model
    %change input values to initial values
    %     rhoBiofilm_old = 'rhoBiofilm = 6.9 #10 #                  //[kg/m3]               kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';
    %     kub_old = 'kub = 3.81e-4 # 			// [kg_urease/kg_bio] 		calculated from kurease fit 2014_10_2, fittet parameter!!!!!!!!!!!!';
    
    
    %folder for new model run
    new_folder = strcat('mkdir Simulation',num2str(P_total));
    system(new_folder)
    
    
    %evaluating the model, moving the resulting files
    rhoBiofilm = strcat('rhoBiofilm = ',num2str(new_point(1)),' #');
    kub = strcat('kub = ',num2str(new_point(2)),' #');
    
    
    %change corresponding input values in input file
    change_Inputs('column8_nosuspendedbiomass.input', rhoBiofilm_old, rhoBiofilm);
    change_Inputs('column8_nosuspendedbiomass.input', kub_old, kub);
    
    
    %save intermediate variables
    save intermediate.mat;
    
    
    %run model
    system('./micp_no_susp_bio column8_nosuspendedbiomass.input');
    
    run paraRef.m;
    load intermediate.mat;
    load MICPResult.mat;
    Calcite = ref_volFracCalcite;
    Calcium = ref_molarityCa2_w;
    
    
    %put needed files to corresponding folder
    get_to_folder_vtp = strcat('mv *.vtp Simulation', num2str(P_total),'/');
    system(get_to_folder_vtp);
    get_to_folder_pvd = strcat('mv *.pvd Simulation', num2str(P_total),'/');
    system(get_to_folder_pvd);
    get_to_folder_MICP = strcat('mv MICPResult.mat Simulation', num2str(P_total),'/');
    system(get_to_folder_MICP);
    
    load intermediate.mat;
    
    
    %change input values to initial values
    change_Inputs('column8_nosuspendedbiomass.input', rhoBiofilm, rhoBiofilm_old);
    change_Inputs('column8_nosuspendedbiomass.input', kub, kub_old);
    
    
    %save outputs in one output file per qoi
    copy_Output_down = strcat('cp OutputCt.mat Simulation',num2str(P_total),'/');
    system(copy_Output_down);
    copy_Output_down = strcat('cp OutputCa.mat Simulation',num2str(P_total),'/');
    system(copy_Output_down);
    get_to_folder = strcat('Simulation', num2str(P_total),'/');
    cd(get_to_folder)
    load OutputCt.mat OutputCt;
    load OutputCa.mat OutputCa;
    load 'MICPResult.mat';
    
    
    %output row (over spacial points)
    OutputCt(P_total,:) = ref_volFracCalcite(end,:); %volFracCalcite
    ref_Calcium = ref_molarityCa2_w; %molarityCa2_w
    OutputCa(:,:,P_total) = [ref_Calcium(2:23,:);ref_Calcium(25:37,:)];
    
    save OutputCt.mat OutputCt;
    save OutputCa.mat OutputCa;
    copy_Output_up = strcat('cp OutputCt.mat ..');
    system(copy_Output_up);
    copy_Output_up = strcat('cp OutputCa.mat ..');
    system(copy_Output_up);
    cd ..;
    
    
    %replacing optimal integration points by existing input data
    sCt = size(OutputCt);
    sCa = size(OutputCa);
    
    NumberOfOutputsCt = sCt(2);
    NumberOfOutputsCa = sCa(2);
    
    %==================================================================================================================================================
    %======     evaluate Psi at new collocation point, compute new coeffs
    %==================================================================================================================================================
    
    %Psi
    fprintf('\n---> setting up of new space/time independent matrix of arbitrary polynomials ...\n');
    
    new_row = ones(1,P);
    for i=1:1:P
        for ii = 1:1:N
            load(PolynomialBasisFileName{ii},'Polynomial');
            new_row(i)=new_row(i) * polyval(Polynomial(1+PolynomialDegree(i,ii),length(Polynomial):-1:1),new_point(ii));
        end
    end
    
    Psi = [Psi, new_row'];
    
    
    %coefficients
    fprintf('\n---> computation of the new expansion coefficients ...\n');
    
    
    %equi-weighted projection, leading to least square fit of coefficients
    Psi=Psi';
    clear temp;
    inv_PsiPsi=pinv(Psi'*Psi);
    
    
    %------------
    %multi-dimensional expansion coeffs
    %------------
    
    %calcite
    for j=1:1:NumberOfOutputsCt
        for ii=1:1:P_total
            temp(ii)=OutputCt(ii,j);
        end
        CoeffsCt(:,j) = inv_PsiPsi*Psi'*temp';
        %disp(CoeffsCt(:,j))
    end
    
    
    %calcium
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
    
    %------------
    %LOOCV evolution
    %------------
    Psi = Psi';
    loocvCt(P_total-P+1,:) = loocv(Psi, OutputCt);
    for i = 1:numberOfPointsTime
        loocvCa(P_total-P+1,:,i) = loocv(Psi, reshape(OutputCa(i,:,:),numberOfPointsSpace,P_total)');
    end
    Psi = Psi';
    
    
    %------------
    %surrogate data evaluated on collocation points (new coefficients)
    %------------
    surrogateCt_allCPoints = Psi'*CoeffsCt;
    clear surrogateCa_allCPoints; %to fit new data in old 3D matrix
    for i = 1:numberOfPointsTime
        surrogateCa_allCPoints(i,:,:) = (Psi'*reshape(CoeffsCa(i,:,:),numberOfPointsSpace,P)')';
    end
    
    
    %surrogate data only on measured points, matrices (CP)
    surrogateCt_measPoints = [surrogateCt_allCPoints(:,4),surrogateCt_allCPoints(:,10),surrogateCt_allCPoints(:,16),...
        surrogateCt_allCPoints(:,22),surrogateCt_allCPoints(:,28),surrogateCt_allCPoints(:,34),...
        surrogateCt_allCPoints(:,40),surrogateCt_allCPoints(:,46)];
    surrogateCa_measPoints = [surrogateCa_allCPoints(:,9,:),surrogateCa_allCPoints(:,16,:),surrogateCa_allCPoints(:,24,:),...
        surrogateCa_allCPoints(:,31,:),surrogateCa_allCPoints(:,39,:)];
    
    
    %model data only on measured points, matrices (CP) (new model evaluation)
    modelCt_measPoints = [OutputCt(:,4),OutputCt(:,10),OutputCt(:,16),OutputCt(:,22),OutputCt(:,28),...
        OutputCt(:,34),OutputCt(:,40),OutputCt(:,46)];
    modelCa_measPoints = [OutputCa(:,9,:),OutputCa(:,16,:),OutputCa(:,24,:),...
        OutputCa(:,31,:),OutputCa(:,39,:)];
    
    
    %------------
    %surrogate data evaluated on MC points (new coefficients)
    %------------
    
    
    %surrogate data  evaluated on MC points
    surrogateCt_MCPoints = evaluatedBasis'*CoeffsCt;
    for i = 1:numberOfPointsTime
        surrogateCa_MCPoints(i,:,:) = (evaluatedBasis'*reshape(CoeffsCa(i,:,:),numberOfPointsSpace,P)')';
    end
    
    
    %surrogate data only on measured points, matrices (MC)
    surrogateCt_measMCPoints = [surrogateCt_MCPoints(:,4),surrogateCt_MCPoints(:,10),surrogateCt_MCPoints(:,16),...
        surrogateCt_MCPoints(:,22),surrogateCt_MCPoints(:,28),surrogateCt_MCPoints(:,34),...
        surrogateCt_MCPoints(:,40),surrogateCt_MCPoints(:,46)];
    surrogateCa_measMCPoints = [surrogateCa_MCPoints(:,9,:),surrogateCa_MCPoints(:,16,:),surrogateCa_MCPoints(:,24,:),...
        surrogateCa_MCPoints(:,31,:),surrogateCa_MCPoints(:,39,:)];
    
    
end

%==================================================================================================================================================
%======     computation of output statistics: mean and variance
%==================================================================================================================================================

fprintf('\n---> computation of output statistics: mean and variance...\n');

%------------
%analytical form of mean and variance
%------------


%calcite
for j=1:1:NumberOfOutputsCt
    OutputMeanCt(j)=CoeffsCt(1,j);
    OutputVarCt(j)=sum(CoeffsCt(2:P,j).^2);
end

fprintf('\n---> mean value of model outputs of Calcite: %d:\n');
%disp(OutputMeanCt);
fprintf('---> variance of model outputs of Calcite: %d:\n');
%disp(OutputVarCt);


%calcium
for j=1:1:NumberOfOutputsCa
    OutputMeanCa(j)=CoeffsCa(1,j);
    for i = 1:numberOfPointsTime
        OutputVarCa(i,j)=sum(CoeffsCa(i,j,2:P).^2);
    end
end

fprintf('\n---> mean value of model outputs of Calcium: %d:\n');
%disp(OutputMeanCa);
fprintf('---> variance of model outputs of Calcium: %d:\n');
%disp(OutputVarCa);

%==================================================================================================================================================
%======    save
%==================================================================================================================================================


%Psi evaluated at collocation points
save Psi.mat Psi;

%collocation points
save CollocationPointsBase.mat CollocationPointsBase;

%output
save OutputCt.mat OutputCt;
save OutputCa.mat OutputCa;

%coeffs needed to evaluate surrogate model on other points
save CoeffsCt.mat CoeffsCt;
save CoeffsCa.mat CoeffsCa;

%loocv needed to plot
save loocvCt.mat loocvCt;
save loocvCa.mat loocvCa;

dlmwrite('CPointsBase.txt',CollocationPointsBase)
dlmwrite('OutputMeanCt.txt',OutputMeanCt);
%dlmwrite('OutputVarCt.txt',OutputVarCt);
dlmwrite('OutputMeanCa.txt',OutputMeanCa);
%dlmwrite('OutputVarCa.txt',OutputVarCa);

save OutputMeanCt.mat OutputMeanCt;
save OutputMeanCa.mat OutputMeanCa;
save OutputVarCt.mat OutputVarCt;
save OutputVarCa.mat OutputVarCa;

save('ManagementData.mat');




fprintf('\n');
fprintf('---> calculations are successfully completed \n\n');
toc
fprintf('\n');
fprintf('______________________________________________________________________________________________________________\n');
