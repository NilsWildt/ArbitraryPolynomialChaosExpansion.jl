%Contaminated Site simulator
clear all
%1. Permeability field generator (intrinsicY)
%2. CSA generater (SequentialGrowth)
%3. Flow calculator (FEMq)
%4. Dissolution and transport calulator (CSARWPTschoen)
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(1000*clock)));

% addpath /data/homes/koch/Documents/MATLAB/Source_IDSIM_3D/RWPT/
% addpath /data/homes/koch/Documents/MATLAB/Source_IDSIM_3D/Input_Data/
% addpath /data/homes/koch/Documents/MATLAB/Source_IDSIM_3D/FEM_3D/
% addpath /data/homes/koch/Documents/MATLAB/Source_IDSIM_3D/FEM_3D/INIT
% addpath /data/homes/koch/Documents/MATLAB/Source_IDSIM_3D/RWPT/figtree/matlab
% addpath /data/homes/koch/Documents/MATLAB/Source_IDSIM_3D/RWPT/gpuml/matlab


addpath RWPT/
addpath Input_Data/
addpath FEM_3D/
addpath FEM_3D/INIT
addpath RWPT/figtree/matlab
addpath RWPT/gpuml/
% Discretisation
grid.n_pts = [100 100 75];
grid.d_pts = [0.4 0.4 0.2];
lam =grid.d_pts;
N = grid.n_pts;

grid              = ndgrid_setup(grid);
% -------------------------------------------------------------
% O T H E R   F E M   I N I T S
% -------------------------------------------------------------
grid = init_FORWARD(grid);            % stupid interface to older method
grid = init_FEM(grid);                % completes FEM grid

%parameters
Z = 1; %number of realizations

vx = zeros(prod(N+1),Z);%vxcsa = zeros(2000,Z);
vy = zeros(prod(N+1),Z);%vycsa = zeros(2000,Z);
vz = zeros(prod(N+1),Z);%vzcsa = zeros(2000,Z);

vxn = zeros(prod(N+1),Z);%vxncsa = zeros(2000,Z);
vyn = zeros(prod(N+1),Z);%vyncsa = zeros(2000,Z);
vzn = zeros(prod(N+1),Z);%vzncsa = zeros(2000,Z);

vxr = zeros(prod(N+1),Z);%vxrcsa = zeros(2000,Z);
vyr = zeros(prod(N+1),Z);%vyrcsa = zeros(2000,Z);
vzr = zeros(prod(N+1),Z);%vzrcsa = zeros(2000,Z);

Y= zeros(prod(N),Z);Yr = zeros(prod(N),Z); % Yn = zeros(2000,Z); Yrn = zeros(2000,Z); 
Sn =  zeros(prod(N),Z); kappa = zeros(1,Z);
ctrl.ne    = 0.3;
ne = 0.3;
% if Z>1
% matlabpool 5
% end

for z = 1:Z
%% ------------------------------------------------------------------------
%1. Permeability field generator (intrinsicY)------------------------------
%--------------------------------------------------------------------------

[YY,Kappa] = intrinsicY(grid);
kappa(z) = Kappa;
Y(:,z) = YY;
%% ------------------------------------------------------------------------
%2. CSA generater (SequentialGrowth)---------------------------------------
%--------------------------------------------------------------------------

%how to get from intrinsic permeability (K_i) over median grain diameter (d_med) 
%to pore radius and throat radius (r_p and r_t)
dKrelation  = 'BeringerPoiseuille'; %or 'Dullien'

brooks      = 2;   % broks corey parameter usualy called lambda

% [XX,SSn,S_rn,S_rw] = SequentialGrowth(dKrelation,brooks,YY,grid,TM,ctrl);
[XX,SSn,S_rn,S_rw,NF] = BlockWiseGrowth(dKrelation,brooks,YY,N,lam,ne);
X{z}    = XX;
Sn(:,z) = SSn;
Yn{z} = YY(SSn>0);
% sourrounding nodes of occupied boxes
idx1            = sub2ind(N+1,XX(:,1),XX(:,2),XX(:,3));
idx2            = sub2ind(N+1,XX(:,1)+1,XX(:,2),XX(:,3));
idx3            = sub2ind(N+1,XX(:,1),XX(:,2)+1,XX(:,3));
idx4            = sub2ind(N+1,XX(:,1),XX(:,2),XX(:,3)+1);
idx5            = sub2ind(N+1,XX(:,1)+1,XX(:,2)+1,XX(:,3));
idx6            = sub2ind(N+1,XX(:,1),XX(:,2)+1,XX(:,3)+1);
idx7            = sub2ind(N+1,XX(:,1)+1,XX(:,2),XX(:,3)+1);
idx8            = sub2ind(N+1,XX(:,1)+1,XX(:,2)+1,XX(:,3)+1);

%% ------------------------------------------------------------------------
%3. Flow calculator (FEMq)-------------------------------------------------
%--------------------------------------------------------------------------
[qqx,qqy,qqz] = FEMq(grid,YY);

% seepage velocity
VX = qqx./ne;VY=qqy./ne; VZ = qqz./ne;
vx(:,z) = VX;vy(:,z)=VY; vz(:,z) = VZ;

% the uneffected steady state seepage velocity at occupied boxes 
vxcsa{z}      = (VX(idx1)+VX(idx2)+VX(idx3)+VX(idx4)+VX(idx5)+VX(idx6)+VX(idx7)+VX(idx8))./8;
vycsa{z}      = (VY(idx1)+VY(idx2)+VY(idx3)+VY(idx4)+VY(idx5)+VY(idx6)+VY(idx7)+VY(idx8))./8;
vzcsa{z}      = (VZ(idx1)+VZ(idx2)+VZ(idx3)+VZ(idx4)+VZ(idx5)+VZ(idx6)+VZ(idx7)+VZ(idx8))./8;

% updating Y with relative Permeability
Model = 'VanGenuchten';%or 'BrooksCorey';
[YY] = relativeK(YY,SSn,Model,S_rw);
Yr(:,z) = YY;
Yrn{z} = YY(SSn>0);
% updated flow field
[qqxn,qqyn,qqzn] = FEMq(grid,YY);
Vxr = qqxn./ne;Vyr = qqyn./ne;Vzr = qqzn./ne;
vxr(:,z) = Vxr;vyr(:,z) = Vxr;vzr(:,z) = Vxr;
% with relative permeability updated flow field
vxrcsa{z}      = (Vxr(idx1)+Vxr(idx2)+Vxr(idx3)+Vxr(idx4)+Vxr(idx5)+Vxr(idx6)+Vxr(idx7)+Vxr(idx8))./8;
vyrcsa{z}      = (Vyr(idx1)+Vyr(idx2)+Vyr(idx3)+Vyr(idx4)+Vyr(idx5)+Vyr(idx6)+Vyr(idx7)+Vyr(idx8))./8;
vzrcsa{z}      = (Vzr(idx1)+Vzr(idx2)+Vzr(idx3)+Vzr(idx4)+Vzr(idx5)+Vzr(idx6)+Vzr(idx7)+Vzr(idx8))./8;

% updated porosity
nen = ne.*(1-SSn);
% seepage velocity
[qqxn,qqyn,qqzn,nenq] = darcy2seepage(qqxn,qqyn,qqzn,nen,N);
vxn(:,z) = qqxn;vyn(:,z) = qqyn;vzn(:,z) = qqzn;

vxncsa{z}         = (qqxn(idx1)+qqxn(idx2)+qqxn(idx3)+qqxn(idx4)+qqxn(idx5)+qqxn(idx6)+qqxn(idx7)+qqxn(idx8))./8;
vyncsa{z}         = (qqyn(idx1)+qqyn(idx2)+qqyn(idx3)+qqyn(idx4)+qqyn(idx5)+qqyn(idx6)+qqyn(idx7)+qqyn(idx8))./8;
vzncsa{z}         = (qqzn(idx1)+qqzn(idx2)+qqzn(idx3)+qqzn(idx4)+qqzn(idx5)+qqzn(idx6)+qqzn(idx7)+qqzn(idx8))./8;


%% ------------------------------------------------------------------------
%4. Dissolution and transport calulator (CSARWPTschoen)--------------------
%--------------------------------------------------------------------------
 XX(:,1)     = (XX(:,1)-0.5).*lam(1);
 XX(:,2)     = (XX(:,2)-0.5).*lam(2);
 XX(:,3)     = (XX(:,3)-0.5).*lam(3);

 Disp = 1;
 readx      = N(2)*lam(2)-lam(2)*2;
 [dsa idx] = sort(XX(:,1));
 XX = XX(idx,:);
 idx = sub2ind(N,XX(:,1)./lam(1)+0.5,XX(:,2)./lam(2)+0.5,XX(:,3)./lam(3)+0.5);
 ctrl.ne_em = nen(round(idx));
 [Cij,M,MAPcsa,C] = CSARWPTschoen(qqxn,qqyn,qqzn,grid,XX,readx,Disp,ctrl,nen);
 [Cij,M,MAPcsa,C] = CSARWPTschoen(Vx,Vy,Vz,grid,XX,readx,Disp,ctrl,nen);

%% ------------------------------------------------------------------------
%4. Depletion calulator (-)------------------------------------------------
%--------------------------------------------------------------------------



end

% if Z>1
% matlabpool close
% end
v               = sqrt(vx(:).^2+vy(:).^2+vz(:).^2);              % whole flow field (seepage) withou any NAPL effect
vn              = sqrt(vxn(:).^2+vyn(:).^2+vzn(:).^2);           % whole flow field (seapage) after all napl effects

%cell2mat
vxcsa = cell2mat(vxcsa(:));vycsa = cell2mat(vycsa(:));vzcsa = cell2mat(vzcsa(:));
vxrcsa = cell2mat(vxrcsa(:));vyrcsa = cell2mat(vyrcsa(:));vzrcsa = cell2mat(vzrcsa(:));
vxncsa = cell2mat(vxncsa(:));vyncsa = cell2mat(vyncsa(:));vzncsa = cell2mat(vzncsa(:));

vcsa            = sqrt(vxcsa.^2+vycsa.^2+vzcsa.^2);     % the uneffected steady state seepage velocity at occupied boxes
vrcsa           = sqrt(vxrcsa.^2+vyrcsa.^2+vzrcsa.^2);  % with relative permeability updated flow field 
vncsa           = sqrt(vxncsa.^2+vyncsa.^2+vzncsa.^2);  % with relative permeabilit and porosity due to 
                                                        %immobile napl update flofiel  
                                                        
%Y original log(K) field, Yr: with relative permeability updated log(K) field
Y = Y(:);Yr = Yr(:);   
%Yn original log(K) field occupied by dnapl; Yrn updated log(K) field occupied by dnapl
Yn = cell2mat(Yn(:));Yrn = cell2mat(Yrn(:));
 
%estimate pdfs
xv = min(log(v)):(max(log(v))-min(log(v)))/100:max(log(v));

[pdfv] = kde_figtree(log(v'),xv,1e-2);
[pdfvcsa] = kde_figtree(log(vcsa'),xv,1e-2);
[pdfvrcsa] = kde_figtree(log(vrcsa'),xv,1e-2);
[pdfvncsa] = kde_figtree(log(vncsa'),xv,1e-2);
[pdfvn] = kde_figtree(log(vn'),xv,1e-2);

xy = min(Y):(max(Y)-min(Y))/100:max(Y);
[pdfY] = kde_figtree(Y',xy,1e-2);
[pdfYn] = kde_figtree(Yn',xy,1e-2);
[pdfYr] = kde_figtree(Yr',xy,1e-2);
[pdfYrn] = kde_figtree(Yrn',xy,1e-2);
save(['flowstat' datestr(now,30)],'pdfvn','pdfv','pdfvcsa','pdfvrcsa','pdfvncsa','pdfY','pdfYn',...
                                  'pdfYrn','-v7.3');
save(['flow' datestr(now,30)],'vn','v','vcsa','vrcsa','vncsa','Y','Yn','Yrn','kappa','-v7.3');

