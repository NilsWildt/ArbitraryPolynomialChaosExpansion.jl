function [dmk,Mflux,out,Cij] = CSARWPTconstdLananosave(qx,qy,qz,grid,X0,readx,Disp,ne_em,nen,mod,test)
%
% RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));

% used fruits: apricot, cherry, grape, grape, mango, melon, papaya, peach, pear,

%mod    = 1 --> only mass fluxes are called
%mod    = 2 --> mass fluxes and n controll planes, per default n = 1
%mod    = 3 --> mass fluxes and plume

 
kdefun = @kde_figtree;%kde_gpuml; %
%% Model Preamble
trilinear = 1;
% -------------------------------------------------------------------------
% Discretizing-------------------------------------------------------------
% -------------------------------------------------------------------------
Dim     = 3;
lambda  = grid.d_pts;               % Box size in (y,x,z) direction
N       = grid.n_pts;               % Number of nodes in (y,x,z) direction
Domain  = lambda.*N;                % Domain size [y,x,z]

vx = reshape(qx,N+1);
vy = reshape(qy,N+1);
vz = reshape(qz,N+1);
ne                = max(nen);
% ne_em             = ne_em;
if mod == 3
    fac     = [2 2 2];
else
    fac     = [2 2 2];
end
bwe     = lambda./(fac);                % emitter bandwidth, the actual one
% bwe(:) = bwe(2);
bwk     = bwe;%.*sqrt(2);%IS DONE IN kde_figtree                 % kde corrected bandwidth, since figtree calculates e^((x-<x>) bw⁻² (x-<x>)'). the factor 1/2 is missing


%normailization factor, integral of the boundary kernel evaluation
P       = (2*pi())^(Dim/2)*prod(bwe);   % single kernel integral
P2      = (pi())^(Dim/2)*prod(bwe);     % kernel^2, none is yet normalized
Ppri    = 2^(-Dim/2) *P^-1;             % kernel^2, both kernels are already normalized
V       = P2/P^2;
%V = P2/P Kl normalized Kk not --> V = (1/NP).*kde_ij;
%V = P2/P^2 both Kl and Kk are normalized --> V = (1/NP).*kde_ij/P (with normalized kde);
%V = P2 both Kl and Kk are not normalized --> V = (P/NP).*kde_ij;
absv    = sqrt(vx.^2 + vy.^2+vz.^2); % absolute velocity
cs      = 1;                        % solubility concentration
gdt     = bwe(2)/(test(1)*mean(absv(:)));   % constant time steps in sec.
dL      = bwe(2)/(test(2));             % constant sapce step in [m]




NP      = test(3);         % same Number of particles with same mass at each emitter - unit mass release
NE      = size(X0,1);  % number of emitters
NPT     = NP*NE;       % total number of particles per time step
%% Dispersions Tensor
%--------------------------------------------------------------------------
% Dispersion---------------------------------------------------------------
% D = (at |u| + Dm) I + (al - at) u u' / |u|-------------------------------
%--------------------------------------------------------------------------

at                = 0;%0.02*lambda(1);          % transverse dispersivity
al                = at*5;                 % longitudinal dispersivity
Dm                = 0;%2e-9;                 % molecular diffusion coefficient

if Disp == 1
    % D  =  Diffusion + Dispersions Tensor
    Dispersion
else
    % D  =  Diffusion
    % D  = [Dm 0 0; 0 Dm 0; 0 0 Dm];
    B = diag(ones(3,1).*sqrt(Dm));
    
    ddDyy = 0;ddDyx = 0;ddDyz = 0;
    ddDxy = 0;ddDxx = 0;ddDxz = 0;
    ddDzy = 0;ddDzx = 0;ddDzz = 0;
end
%% predefinitions

X       = zeros(NPT,Dim);   % Particle Matrix, tracking particle locations
Xi_1    = zeros(NPT,Dim);   % Particle Matrix, tracking particle locations at i-1 time step
subX    = [X0(:,1)./lambda(1)+0.5,X0(:,2)./lambda(2)+0.5,X0(:,3)./lambda(3)+0.5];
idnX    = round(sub2ind(N,subX(:,1),subX(:,2),subX(:,3)));

Cij         = zeros(NE);    % Sensitivity matrix to controll boundary condition and calculate allowed mass releases
read        = readx;
npe         = ones(NE,1)*NP;

%% --sample points for density estimation are defined ------------------------------
x_PDF = X0'; % boundary condition sample points or emitter midpoints, locations where source term > 0
if mod == 3
    % concentration plume
    a = grid.x_pts{2}(:) >= min(X0(:,2))-10*lambda(2);
    %    b = grid.x_pts{2}(:) < max(X0(:,2))+10*lambda(2);
    %    a = a.*b == 1;
    %    a = ones(prod(N),1) == 1;%repmat([1;0],prod(N)/2,1) == 1;
    %     x_pdf       = [grid.x_pts{1}(a)+lambda(1)/2,grid.x_pts{2}(a)+lambda(2)/2,grid.x_pts{3}(a)+lambda(3)/2];
    x_pdf       = [grid.x_pts{1}(a),grid.x_pts{2}(a),grid.x_pts{3}(a)];
    C           = ones(sum(a),NE)-1;       % concentration field, plume
    nen         = nen(a);
    idnXsmall   = round(sub2ind([ 100,size(C,1)/(100*75),75 ],subX(:,1),...
        subX(:,2)- round((min(X0(:,2))-11*lambda(2))/lambda(2) + 0.5),subX(:,3)));
elseif mod == 2
    [a b]       = meshgrid(lambda(1)/2:lambda(1):lambda(1)*N(1)-lambda(1)/2,lambda(3)/2:lambda(3):lambda(3)*N(3)-lambda(3)/2);
    xpdf        = [a(:) b(:)];
    N_planes = 1;
    locplane = [round((max(X0(:,2))+lambda(2)/2)*1e5)/1e5,readx-5,readx-10];
    cpro = zeros(N(1)*N(3),NE,N_planes);
    mapCSAc = zeros(N(1)*N(3),N_planes);
    x_pdfpro = zeros(N(1)*N(3),Dim,N_planes);
    ne_map  = zeros(N(1)*N(3),1);
    for plane = 1:N_planes
        node = ceil((locplane(plane)-1e-5)./lambda(2));
        x_pdfpro(:,:,plane) = [xpdf(:,1),ones(N(1)*N(3),1).*locplane(plane),xpdf(:,2)];
        idx                 = sub2ind(N,round(x_pdfpro(:,1,plane)./lambda(1)+0.5),... % without + 0.5 because ne is at midpoint
            round(ones(N(1)*N(3),1).*node),...
            round(x_pdfpro(:,3,plane)./lambda(3)+0.5));
        
        ne_map(:,plane)     = nen(round(idx));
        %         idx2                 = sub2ind(N,round(x_pdfpro(:,1,plane)./lambda(1)+0.5),... % without + 0.5 because ne is at midpoint
        %             round(x_pdfpro(:,2,plane)./lambda(2)+1),...
        %             round(x_pdfpro(:,3,plane)./lambda(3)+0.5));
        
        %         ne_map(:,plane)     = (nen(round(idx))+nen(round(idx2)))/2;
    end
    %     save derTester x_PDF x_pdfpro locplane xpdf read node
    for plane = 1:N_planes
        x_PDF = [x_PDF x_pdfpro(:,:,plane)'];
    end
end
%% particle release

for cherry =1:NE
    
    r                        =  [randn(npe(cherry),1).*(bwe(1)),...
        randn(npe(cherry),1).*(bwe(2)),...
        randn(npe(cherry),1).*(bwe(3))];
    
    Xy                       = ones(npe(cherry),1)*X0(cherry,1)+r(:,1);
    Xx                       = ones(npe(cherry),1)*X0(cherry,2)+r(:,2);
    Xz                       = ones(npe(cherry),1)*X0(cherry,3)+r(:,3);
    
    counte                   = sum(npe(1:cherry));
    counta                   = sum(npe(1:cherry-1));
    X(1+counta:counte,:)     = [Xy,Xx,Xz];
    look(1+counta:counte)    = cherry;
end

NPT = sum(npe);
compromise = test(4);
% disp([ num2str(compromise) '_' datestr(now)])
plume = 0;
Xallt       = zeros(compromise*NPT,Dim);
% if mod ==3,
%     Yallt{1}    = Xallt;
% end
point       = zeros(compromise*NPT,1);
% recorded    = 0;
Tp      = zeros(NPT,1);
T       = zeros(NPT,1);

sub1x      = zeros(NPT,1);sub2x = sub1x;sub1y = sub1x;
sub2y      = sub1x;sub1z = sub1x;sub2z = sub1x;subD = sub1x;
vx_linear  = zeros(NPT,1);vy_linear  = vx_linear;vz_linear  = vx_linear;

dDD     = zeros(NPT,Dim);A = dDD; Rand = A; BZ = A;
node1   = A; node2 = A; node3 = A; X1 = A; fun = A;
yxzabsv = zeros(NPT,1); yxabsv = yxzabsv; Wal = yxabsv; Wat = Wal;
Byy = Wal; Byx = Byy; Byz = Byy; Bxy = Byy; Bxx = Byy; Bxz = Byy;
Bzy = Byy; Bzx = Byy; Bzz = Byy;

ind     = ones(size(X,1),1) == 1;
movep   = zeros(1,size(X,1));
td      = zeros(1,size(X,1));
dt      = zeros(size(X,1),3);
steps   = 0;
stoppa = 0;
% Xallt(1:NPT,:)  = X;
% point(1:NPT)    = look;
count           = 1;%NPT+1;


% calculate mass matrix, äquvalent to normalization factor
% q = 1/NP;
% for lime = 1:size(X0,1)
%
%
%     %                     if any(point==lime)
%
%     x = Xallt(point==lime,:)';
%
%     %                         [kde]       = figtree(x,bw(2),q1,x_PDF,1e-2);%[kde]           = figtree(x,bw(2),q1,[X0' x_pdfpro3' x_pdfpro2' x_pdfpro1'],1e-3);
%
%
%     [kde]       = kdefun(x,x_PDF(:,1:NE),1e-3,ones(size(x,2),1).*q,bwk',1,[0;0;0],[N.*lambda]'); % kdefun calls either figtree or gpuml
%     kde(lime)   = kde(lime);
%     %                         Cij(:,lime) = Cij(:,lime) + kde(1:size(X0,1)) .* dt .* size(x,2) ./ (ne*NP);
%     %                         this is the actual calculation however all factors are constant
%     %                         thus we can make the calcualtion afterward and NP
%     %                         is anyway canceling out since it is also in
%     %                         size(x,2) ; size(x,2) = compromise*NP
%
%     Mij(:,lime) = kde; % if kde normal else 1/NP *kde
%
% end

% [mgy] = allcomb(X0(:,1),X0(:,1));
% dy = diff(mgy');
% [mgx] = allcomb(X0(:,2),X0(:,2));
% dx = diff(mgx');
% [mgz] = allcomb(X0(:,3),X0(:,3));
% dz = diff(mgz');
% Dis = reshape(sqrt(dx.^2 + dy.^2 + dz.^2),size(X0,1),size(X0,1));

[mgy] = meshgrid(X0(:,1),X0(:,1));
[mgx] = meshgrid(X0(:,2),X0(:,2));
[mgz] = meshgrid(X0(:,3),X0(:,3));
mgx = mgx-mgx';
mgy = mgy-mgy';
mgz = mgz-mgz';
R = 1./bwe.^2;
dist = (mgx(:).^2.*R(2).^2 + mgy(:).^2.*R(1).^2 + mgz(:).^2.*R(3).^2);
Mij = reshape(exp(-1/4.*dist).*V,NE,NE);

%------------------------------------------------------------------
% BOUNCING BOUNDARIES----------------------------------------------
%------------------------------------------------------------------
% keeping the particles within the domain,
% angle of incidence is equal to the angle of reflection
B1 = X(ind,1);B2 = X(ind,2);B3 = X(ind,3);

while any(B1>(N(1)*lambda(1))) ||  any(B1<0) || any(B2>(N(2)*lambda(2))) || ...
        any(B2<0) ||  any(B3>(N(3)*lambda(3))) || any(B3<0)
    
    upper         = B1>(N(1)*lambda(1));
    B1(upper)     = (N(1)*lambda(1)) + ((N(1)*lambda(1))-B1(upper));%(N(1)*lambda(1)-lambda(1)) + ((N(1)*lambda(1)-lambda(1))-B1(upper));
    B1(B1<0)      = -1.*B1(B1<0);
    
    upper         = B2>(N(2)*lambda(2));
    B2(upper)     = (N(2)*lambda(2)) + ((N(2)*lambda(2))-B2(upper));
    B2(B2<0)      = B2(B2<0).*-1;
    
    upper         = B3>(N(3)*lambda(3));
    B3(upper)     = (N(3)*lambda(3)) + ((N(3)*lambda(3))-B3(upper));
    B3(B3<0)      = -1.*B3(B3<0);
end
X(ind,:)      = [B1,B2,B3];

while sum(ind) > 0; %min(X(:,2)) < max(readx) - 0.05
    steps=steps+1;
    % + (steps <= compromise(hip)) * (100-gdt)*(-(hip-2));
    
    
    
    %------------------------------------------------------------------
    % INTERPOLATION OF FLOW FIELD--------------------------------------
    %------------------------------------------------------------------
    
    % find current particle positions - nodes/element
    % ceil(X/lambda) gives the lower node of the current particle location,
    % since X = [0,0,0] at node (1,1,1 )! round(X/lambda+1) gives the closest node
    node1(ind,:)   = [floor(X(ind,1)./lambda(1))+1,floor(X(ind,2)./lambda(2))+1,floor(X(ind,3)./lambda(3))+1];
    node2(ind,:)   = node1(ind,:) + 1;
    node3(ind,:)   = [round(X(ind,1)./lambda(1)+1),round(X(ind,2)./lambda(2)+1),round(X(ind,3)./lambda(3)+1)];
    % left side node in physical cooridinates
    X1(ind,:)      = [node1(ind,1).*lambda(1)-lambda(1),node1(ind,2).*lambda(2)-lambda(2),node1(ind,3).*lambda(3)-lambda(3)];
    % distance of particle positon to left side node
    fun(ind,:)     = [(X(ind,1)-X1(ind,1))/lambda(1),(X(ind,2)-X1(ind,2))/lambda(2),(X(ind,3)-X1(ind,3))/lambda(3)];
    
    %         node1(node1<=0)=1;
    
    %% advection
    %linear interpolation
    sub1x(ind)       = sub2ind(size(vx),node3(ind,1),node1(ind,2),node3(ind,3));
    sub2x(ind)       = sub2ind(size(vx),node3(ind,1),node2(ind,2),node3(ind,3));
    sub1y(ind)       = sub2ind(size(vy),node1(ind,1),node3(ind,2),node3(ind,3));
    sub2y(ind)       = sub2ind(size(vy),node2(ind,1),node3(ind,2),node3(ind,3));
    sub1z(ind)       = sub2ind(size(vz),node3(ind,1),node3(ind,2),node1(ind,3));
    sub2z(ind)       = sub2ind(size(vz),node3(ind,1),node3(ind,2),node2(ind,3));
    
    vx_linear(ind)   = ((1-fun(ind,2)) .* qx(sub1x(ind)) + fun((ind),2) .* qx(sub2x((ind))));
    vy_linear(ind)   = ((1-fun((ind),1)) .* qy(sub1y((ind))) + fun((ind),1) .* qy(sub2y((ind))));
    vz_linear(ind)   = ((1-fun((ind),3)) .* qz(sub1z((ind))) + fun((ind),3) .* qz(sub2z((ind))));
    
    
    %% diffusion and dispersion
    Rand(ind,:)    = randn(size(X(ind,:)));
    if Disp
        subD(ind)        = sub2ind(size(vz),node3(ind,1),node3(ind,2),node3(ind,3));
        dDD(ind,:)       = [ddDyy(subD(ind)) + ddDxy(subD(ind)) + ddDzy(subD(ind)),...
            ddDyx(subD(ind)) + ddDxx(subD(ind)) + ddDzx(subD(ind)),...
            ddDyz(subD(ind)) + ddDxz(subD(ind)) + ddDzz(subD(ind))];
        
        
        if ~trilinear
            % absolute local velocities yxz and yx
            %---------only linear interpolation--------------------------------
            yxzabsv(ind) = sqrt(vy_linear(ind).^2 + vx_linear(ind).^2 + vz_linear(ind).^2);
            yxabsv(ind)  = sqrt(vy_linear(ind).^2 + vx_linear(ind).^2);
            %------------------------------------------------------------------
            Wal(ind) = sqrt(2.*(al.*yxzabsv(ind) + Dm));
            Wat(ind) = sqrt(2.*(at.*yxzabsv(ind) + Dm));
            %---------dispersion term only linear interpolation----------------
            %                 Byy(ind) = Wal(ind).*vy_linear(ind)./yxzabsv(ind); Byx(ind) = -Wat(ind).*vy_linear(ind).*vz_linear(ind)./(yxzabsv(ind).*yxabsv(ind)); Byz(ind) = -Wat(ind).*vx_linear(ind)./(yxabsv(ind));
            %                 Bxy(ind) = Wal(ind).*vx_linear(ind)./yxzabsv(ind); Bxx(ind) = -Wat(ind).*vx_linear(ind).*vz_linear(ind)./(yxzabsv(ind).*yxabsv(ind)); Bxz(ind) =  Wat(ind).*vy_linear(ind)./(yxabsv(ind));
            %                 Bzy(ind) = Wal(ind).*vz_linear(ind)./yxzabsv(ind); Bzx(ind) =  Wat(ind).* yxabsv(ind)./yxzabsv(ind); Bzz(:) = 0;
            %
            Bxx(ind) = Wal(ind).*vx_linear(ind)./yxzabsv(ind); Bxy(ind) = -Wat(ind).*vx_linear(ind).*vz_linear(ind)./(yxzabsv(ind).*yxabsv(ind)); Bxz(ind) = -Wat(ind).*vy_linear(ind)./(yxabsv(ind));
            Byx(ind) = Wal(ind).*vy_linear(ind)./yxzabsv(ind); Byy(ind) = -Wat(ind).*vy_linear(ind).*vz_linear(ind)./(yxzabsv(ind).*yxabsv(ind)); Byz(ind) =  Wat(ind).*vx_linear(ind)./(yxabsv(ind));
            Bzx(ind) = Wal(ind).*vz_linear(ind)./yxzabsv(ind); Bzy(ind) =  Wat(ind).* yxabsv(ind)./yxzabsv(ind); Bzz(:) = 0;
            
            %------------------------------------------------------------------
            
        else
            IntTrilinear2
        end
        %               BZ(ind,:)      = [(Byy(ind).*Rand(ind,1) + Byx(ind).*Rand(ind,2) + Byz(ind).*Rand(ind,3)),...
        %                 (Bxy(ind).*Rand(ind,1) + Bxx(ind).*Rand(ind,2) + Bxz(ind).*Rand(ind,3)),...
        %                 (Bzy(ind).*Rand(ind,1) + Bzx(ind).*Rand(ind,2) + Bzz(ind).*Rand(ind,3))];
        
        BZ(ind,:)      = [(Byx(ind).*Rand(ind,1) + Byy(ind).*Rand(ind,2) + Byz(ind).*Rand(ind,3)),...
            (Bxx(ind).*Rand(ind,1) + Bxy(ind).*Rand(ind,2) + Bxz(ind).*Rand(ind,3)),...
            (Bzx(ind).*Rand(ind,1) + Bzy(ind).*Rand(ind,2) + Bzz(ind).*Rand(ind,3))];
    else
        BZ(ind,:) = [B(1,1).*Rand(ind,1),B(2,2).*Rand(ind,2),B(3,3).*Rand(ind,3)];
    end
    %%  particle displacement algorithm
    
    A(ind,:)    = [vy_linear(ind),vx_linear(ind),vz_linear(ind)] + dDD(ind,:);
    
    %individual time step
    %     movep(ind)  = sqrt(A(ind,1).^2 + A(ind,2).^2 + A(ind,3).^2);
    movep(ind)  = sqrt(vy_linear(ind).^2 + vx_linear(ind).^2 + vz_linear(ind).^2);
    td(ind)     = dL./movep(ind); % time step such that advective disp. == dL
    %     td((Tp+td')>(T+2*gdt)) = 2*gdt; % Tp should have at least one entry within each time intervat T - (T + gdt)
    td(td>2*gdt) = 2*gdt;
    dt(ind,:)    = repmat(td(ind)',1,3); % same time step for each dimension
    
%     plo = (log(movep(ind)) - min(log(absv(:))))./(max(log(absv(:))) - min(log(absv(:)))); 
%     asd = linspace(25,25,ceil(readx));
%     scatter3(X(ind,2),X(ind,1),X(ind,3),asd(ceil(X(ind,2))),plo,'Marker','.');

    Xi_1(ind,:) = X(ind,:);         % Particle position at tp - delta_tp for interpolation of the particle position at T
    
    X(ind,:)    = X(ind,:) + A(ind,:) .*dt(ind,:) + sqrt(dt(ind,:)).* BZ(ind,:);
    
    %----------------------------------------------------------------------
    % BOUNCING BOUNDARIES--------------------------------------------------
    %----------------------------------------------------------------------
    % keeping the particles within the domain,
    % angle of incidence is equal to the angle of reflection
    B1 = X(ind,1);B2 = X(ind,2);B3 = X(ind,3);
    
    while any(B1>(N(1)*lambda(1))) ||  any(B1<0) || any(B2>(N(2)*lambda(2))) || ...
            any(B2<0) ||  any(B3>(N(3)*lambda(3))) || any(B3<0)
        
        upper         = B1>(N(1)*lambda(1));
        B1(upper)     = (N(1)*lambda(1)) + ((N(1)*lambda(1))-B1(upper));%(N(1)*lambda(1)-lambda(1)) + ((N(1)*lambda(1)-lambda(1))-B1(upper));
        B1(B1<0)      = -1.*B1(B1<0);
        
        upper         = B2>(N(2)*lambda(2));
        B2(upper)     = (N(2)*lambda(2)) + ((N(2)*lambda(2))-B2(upper));
        B2(B2<0)      = B2(B2<0).*-1;
        
        upper         = B3>(N(3)*lambda(3));
        B3(upper)     = (N(3)*lambda(3)) + ((N(3)*lambda(3))-B3(upper));
        B3(B3<0)      = -1.*B3(B3<0);
    end
    X(ind,:)      = [B1,B2,B3];
    %----------------------------------------------------------------------
    
    
    checkt          = 1 == round( (Tp < (T + gdt)) .* ((T + gdt) <= (Tp + dt(:,1))));
    %Tp is still the time of the previous time step (not yet updated)
    %T is the time of the last notation of the particle position in Xallt
    %T is only updated when a particle position is stored in Xallt thus
    %when Tp crossed within the current sub time step (variable) a constant
    %time step T_i (here T+gdt) Thus there are two condition for checkt to be 1:
    %1. the old Tp (not yet updated) was smaller as the new T (which is
    %T(new) = T(old) +gdt) T(old) is also T(current) since T(old) is always
    %valid until T(new)
    %2.the new Tp (which is Tp(new) = Tp(old) + dt) is >= T(new) (which is
    %T(new) = T(old) + gdt)
    
    
    Tp(ind)         = Tp(ind) + dt(ind,1);
    T(checkt)       = T(checkt) + gdt;
    
    l               = movep(checkt)' .* (Tp(checkt) -T(checkt));
    l               = repmat(l,1,3);
    L               = movep(checkt)' .* dt(checkt,1);
    L               = repmat(L,1,3);
    Xallt(count:count+sum(checkt)-1,:)    = Xi_1(checkt,:) .* l./L + X(checkt,:) .* (1-l./L); %Xallt(look(checkt)+round(T(checkt)/gdt).*size(X,1),:) = Xi_1(checkt,:) .* l./dL + X(checkt,:) .* (dL-l)./dL;
    
    %     l               = (Tp(checkt) -T(checkt));
    %     l               = repmat(l,1,3);
    %
    %     Xallt(count:count+sum(checkt)-1,:)    = Xi_1(checkt,:) .* l./dt(checkt,:) + X(checkt,:) .* (1-l./dt(checkt,:)); %Xallt(look(checkt)+round(T(checkt)/gdt).*size(X,1),:) = Xi_1(checkt,:) .* l./dL + X(checkt,:) .* (dL-l)./dL;
    
    
    point(count:count+sum(checkt)-1)      = look(checkt);
    count                                 = count +sum(checkt);
    
    while any(1 == round( ((Tp-dt(:,1)) < (T + gdt)) .* ((T + gdt) <= (Tp))));
        
        checkt(checkt)  = 1 == round( ((Tp(checkt)-dt(checkt,1)) < (T(checkt) + gdt)) .* ((T(checkt) + gdt) <= (Tp(checkt))));
        T(checkt)       = T(checkt) + gdt;
        
        l               = movep(checkt)' .* (Tp(checkt) - T(checkt));
        l               = repmat(l,1,3);
        L               = movep(checkt)' .* dt(checkt,1);
        L               = repmat(L,1,3);
        Xallt(count:count+sum(checkt)-1,:)    = Xi_1(checkt,:) .* l./L + X(checkt,:) .* (1-l./L); %Xallt(look(checkt)+round(T(checkt)/gdt).*size(X,1),:) = Xi_1(checkt,:) .* l./dL + X(checkt,:) .* (dL-l)./dL;
        point(count:count+sum(checkt)-1)      = look(checkt);
        count                                 = count +sum(checkt);
    end
    
    ind         = X(:,2) < read;
    X(~ind,2)   = X(~ind,2)+1e3;
    dt(~ind,:)  = 0;
    
   %%  intermediate particle density estimation  
    if size(Xallt,1) > compromise*NPT
        plume = plume +1;
        recorded = 1 - sum(ind)/length(ind);
%         disp([ num2str(recorded) ' particles arrived' datestr(now)])
        q = 1/NP;%(NPT)^-1;
        
        
        for lime = 1:size(X0,1)
            
            x = Xallt(point==lime,:)';
            
            if any(point==lime)
                
                if mod<3
                    
                    [kde]       = kdefun(x,x_PDF,1e-3,ones(1,size(x,2)).*q,bwk',1,[0;0;0],[N.*lambda]');%#ok % kdefun calls either figtree or gpuml
                    
                    kde(lime)   = kde(lime)*2;
                    
                    Cij(:,lime) = Cij(:,lime) + kde(1:size(X0,1));%P is canceled times P because of Kk and divided by P because of Kl
                    
                    if mod == 2
                        for plane = 1:N_planes
                            cpro(:,lime,plane)   = cpro(:,lime,plane) + kde(size(X0,1)+1+(plane-1)*N(1)*N(3):size(X0,1)+N(1)*N(3)*plane);
                        end
                    end
                    
                else
                    
                    [kde]       = kdefun(x,x_pdf',1e-3,ones(1,size(x,2)).*q,bwk',1,[0;0;0],[N.*lambda]'); %#ok % kdefun calls either figtree or gpuml
                    
                    C(:,lime)               = C(:,lime) + kde;
                    kde(idnXsmall(lime))    = kde(idnXsmall(lime))*2;
                    Cij(:,lime)             = Cij(:,lime) + kde(idnXsmall);
                    
                end
            end
        end
        
        Xallt       = zeros(compromise*NPT,Dim);
        point       = zeros(compromise*NPT,1);
        count       = 1;
        steps       = 0;
        stoppa      = recorded>0.99;
    end
    
    recorded = 1 - sum(ind)/length(ind);   
    if  recorded > 0.99 && stoppa == 1
        disp(['break at' num2str(1-sum(ind)/size(ind,1))  '% arrived particles'])
        break
    end
end
recorded = 1 - sum(ind)/length(ind);
% disp([ num2str(recorded) ' particles arrived' datestr(now)])
% q is mp_0 the unit mass releas 1 [kg/s], NP number of particle are released in gdt time
% disp([ num2str(1 - sum(ind)/length(ind)) ' particles arrived' datestr(now)])
q = 1/NP;%(NPT)^-1;

for lime = 1:size(X0,1)
    
    if any(point==lime)
        
        x = Xallt(point==lime,:)';
        
        if mod<3
            
            [kde]       = kdefun(x,x_PDF,1e-3,ones(1,size(x,2)).*q,bwk',1,[0;0;0],[N.*lambda]');%#ok % kdefun calls either figtree or gpuml
            
            kde(lime)   = kde(lime)*2;
            
            
            Cij(:,lime) = Cij(:,lime) + kde(1:size(X0,1));%P is canceled times P because of Kk and divided by P because of Kl
            
            if mod == 2
                for plane = 1:N_planes
                    cpro(:,lime,plane)   = cpro(:,lime,plane) + kde(size(X0,1)+1+(plane-1)*N(1)*N(3):size(X0,1)+N(1)*N(3)*plane);
                end
            end
            
        else
            
            [kde]       = kdefun(x,x_pdf',1e-3,ones(1,size(x,2)).*q,bwk',1,[0;0;0],[N.*lambda]'); %#ok % kdefun calls either figtree or gpuml
            
            C(:,lime)               = C(:,lime) + kde;
            kde(idnXsmall(lime))    = kde(idnXsmall(lime))*2;
            Cij(:,lime)             = Cij(:,lime) + kde(idnXsmall);
            
        end
    end
end
clear Xallt kde
cs = ones(size(X0,1),1).*cs *P;

% disp('solving Matrix equation')

neem        = repmat(ne_em,1,size(X0,1));
% Cij     = Cij./P;
Cij         = (Cij+ Mij + diag(diag(Mij)))./neem .*gdt;%[s/m^3]%
epsi        = Cij\cs;
% epsi     = (cs'*(Mij.*neem))/Cij;%Cij\(Mij.*neem)*cs;


% [a b c]     = meshgrid(-3:3,-3:3,-3:3);
% neig        = [a(:) b(:) c(:)];
%
% lneig       = size(neig,1);
% neig        = repmat(neig,size(subX,1),1);       % neig for neighbors
% inha        = sortrows(repmat(subX,lneig,1));     % inha for inhabitants
% xx          = inha + neig;
% xx          = unique(xx,'rows');
%
% xx(:,1)     = (xx(:,1)-0.5).*lambda(1);
% xx(:,2)     = (xx(:,2)-0.5).*lambda(2);
% xx(:,3)     = (xx(:,3)-0.5).*lambda(3);
%
% [kde]       = kde_figtree(X0',xx',1e-3,epsi,[lambda*0.7],1,[min(xx)],[max(xx)]');%#ok
% DMK         = kde.*prod(lambda);
%
% for po=1:size(xx,1), [kd2(:,po)]   = kde_figtree(xx(po,:)',X0',1e-3,DMK(po),[lambda*0.7],1,[min(xx)],[max(xx)]');end %#ok
% for po=1:size(kd2,2),
%     if ~sum(kd2(:,po))==0
%         kd2(:,po) = kd2(:,po)./sum(kd2(:,po));
%     end,
% end
%
% [a b c]     = meshgrid(-2:2,-2:2,-2:2);
% neig        = [a(:) b(:) c(:)];
% q           = zeros(size(neig,1),1);
% p           = ((neig(:,1)==0).*(neig(:,2)==0).*(neig(:,3)==0)) ==1;
% q(p)        = 1;
% [kernel]    = kde_figtree(neig',neig',1e-3,q,[2 2 2],1,[min(neig)],[max(neig)]);
% kernel      = kernel./sum(kernel);
% kernel      = reshape(kernel,5,5,5);
% %
% %
% [a b c]         = meshgrid(([-2:122]-.5).*lambda(2),([-2:102]-.5).*lambda(1),([-2:77]-.5).*lambda(3));
% outt            = zeros(prod(N+5),1);
% idnXneu         = round(sub2ind(N+5,subX(:,1)+3,subX(:,2)+3,subX(:,3)+3));
% outt(idnXneu)   = epsi;
% outt            = reshape(outt,105,125,80);
%
% z               = convn(outt,kernel,'same');
%
% zz              = z(:);
% clear z
% asd             = zz==0;
% DMK             = zz(~asd);
% clear zz
%
% xxx(:,1) = b(~asd);
% xxx(:,2) = a(~asd);
% xxx(:,3) = c(~asd);
%
% for po=1:size(xxx,1),
%     [kd2(:,po)]   = kde_figtree(xxx(po,:)',X0',1e-3,DMK(po),[lambda*2],1,[min(xxx)],[max(xxx)]');
% end
%
% for po=1:size(kd2,2),
%     if ~sum(kd2(:,po))==0
%         kd2(:,po) = kd2(:,po)./sum(kd2(:,po));
%     end
% end
%
% dmk = kd2*DMK;%epsi;%
dmk = epsi;
% because P == m0 and dmk/m0 == epsi
Mflux    = sum(dmk);
% check if any epsi is negative, actually that means that another epsi
% is depleting for it
% mineps = min(dmk);
% if any(dmk<0)
%     disp(['fuck ' num2str(sum(dmk<0)/length(dmk)*100) ' % negative epsilon with worst' num2str(mineps) 'and mean' num2str(mean(epsi))])
% end

% projection on control plane calculation
if mod == 2
    for plane = 1:N_planes
        [mgy] = allcomb(X0(:,1),x_pdfpro(:,1,plane));
        dy = diff(mgy');
        [mgx] = allcomb(X0(:,2),x_pdfpro(:,2,plane));
        dx = diff(mgx');
        [mgz] = allcomb(X0(:,3),x_pdfpro(:,3,plane));
        dz = diff(mgz');
        
        
        dist = (dx.^2.*R(2).^2 + dy.^2.*R(1).^2 + dz.^2.*R(3).^2);
        Mij  = reshape(exp(-1/4.*dist).*V,N(1)*N(3),NE);
        
        mapCSAc(:,plane) = (cpro(:,:,plane)+Mij) * epsi .* gdt ./ (ne_map.*P);
    end
    out     = mapCSAc(:);
    
end

% plume calculation
if mod == 3;
    C = C * dmk;
    C(idnXsmall) = C(idnXsmall) + sum(Mij,2).*dmk;
    out     = zeros(prod(N),1);
    a = grid.x_pts{2}(:) >= min(X0(:,2))-10*lambda(2);
    out(a) = C.*gdt./(nen.*P);
end

if mod==1
    out=0;
end

end




