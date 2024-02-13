% ========================================================================
% Construction of Data-driven Orthonormal Polynomial Basis
% Author: Dr.-Ing. habil. Sergey Oladyshkin
% Department of Stochastic Simulation and Safety Research for Hydrosystems
% Institute for Modelling Hydraulic and Environmental Systems
% Universitaet Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart
% E-mail: Sergey.Oladyshkin@iws.uni-stuttgart.de
% http://www.iws-ls3.uni-stuttgart.de
% The current program is using definition of arbitrary polynomial chaos expansion (aPC) which is presented in the following manuscript: 
% Oladyshkin, S. and W. Nowak. Data-driven uncertainty quantification using the arbitrary polynomial chaos expansion. Reliability Engineering & System Safety, Elsevier, V. 106, P. 179�190, 2012. DOI: 10.1016/j.ress.2012.05.002. 
% ========================================================================

function P = aPoly_Construction(Data, Degree, Poly_FileName)
% Input parameters:
% Data - raw data array
% Degree - higher degree of the orthonormal polynomial basis
% Poly_FileName - file name in which the orthonormal polynomial basis
% should be storred

%----- Initialization
d=Degree; %Degree of polinomial expansion
dd=d+1; %Degree of polinomial for roots defenition
L_norm=1; % L-norm for polnomial normalization
NumberOfDataPoints=length(Data);

fprintf('\n');
fprintf('---> Construction of Arbitrary Polynomial Basis \n');

%Forward linear transformation (Avoiding of numerical problem):
MeanOfData=mean(Data);
VarOfData=var(Data);
Data=Data/MeanOfData;

%---------------------------------------------------------------------------------------------- 
%----- Moments computation
%----------------------------------------------------------------------------------------------
%Moments Computation for Input Data
for i=0:(2*dd+1)
    m(i+1)=sum(Data.^i)/length(Data); %Raw Moments
end

%---------------------------------------------------------------------------------------------- 
%----- Main Loop for Polinomial with degree up to dd
%---------------------------------------------------------------------------------------------- 
for degree=0:dd
    fprintf('degree =%1i',degree)
    %Definition of Moments Matrix Mm
    for i=0:degree
        for j=0:degree                   
            if (i<degree) 
                Mm(i+1,j+1)=m(i+j+1); 
            end            
            if (i==degree) && (j<degree)
                Mm(i+1,j+1)=0;
            end
            if (i==degree) && (j==degree)
                Mm(i+1,j+1)=1;  
            end        
        end
        %Numerical Optimization for Matrix Solver
        Mm(i+1,:)=Mm(i+1,:)/max(abs(Mm(i+1,:)));
    end
    %Definition of Right Hand side ortogonality conditions: Vc
    for i=0:degree;
        if (i<degree) 
             Vc(i+1)=0; 
        end            
        if (i==degree)
            Vc(i+1)=1;
        end
    end
     
    
    %Solution: Coefficients of Non-Normal Orthogonal Polynomial: Vp
    inv_Mm=pinv(Mm);
    Vp=Mm\Vc'; %p in Paper
    PolyCoeff_NonNorm(degree+1,1:degree+1)=Vp';   
 
    
    fprintf('---> Computational Error for Polynomial of degree %1i is: %5f percents',degree, 100*abs(sum(abs(Mm*PolyCoeff_NonNorm(degree+1,1:degree+1)'))-sum(abs(Vc)))); 
    if 100*abs(sum(abs(Mm*PolyCoeff_NonNorm(degree+1,1:degree+1)'))-sum(abs(Vc)))>0.5
        fprintf('\n---> Attention: Computational Error too hi !');    
        fprintf('\n---> Problem: Convergence of Linear Solver');  
    end
    fprintf('\n');

    %Original Numerical Normalization of Coefficients with Norm and Ortho-normal Basis computation
    %Matrix Sorrage Notice: Polynomial(i,j) correspond to coefficient number "j-1" of polinomial degree "i-1"
    P_norm=0;
    for i=1:NumberOfDataPoints;        
        Poly=0;
        for k=0:degree;
            Poly=Poly+PolyCoeff_NonNorm(degree+1,k+1)*Data(i)^k;     
        end
       
        P_norm=P_norm+Poly^2/NumberOfDataPoints;        
    end
    P_norm=sqrt(P_norm);
    
    for k=0:degree;
       
        Polynomial(degree+1,k+1)=PolyCoeff_NonNorm(degree+1,k+1)/P_norm;
    end
end

%Backward linear transformation to the real data space
Data=Data*MeanOfData;
for k=1:length(Polynomial);
    Polynomial(:,k)=Polynomial(:,k)/MeanOfData^(k-1);
end

%---------------------------------------------------------------------------------------------- 
%----- Roots of polinomial and Plot
%----------------------------------------------------------------------------------------------

%Roots of polynomial with degree dd=d+1
Roots=roots(fliplr(Polynomial(dd+1,:)));
save(Poly_FileName);
