function [Cij,Mflux,mapCSAc,C] = CSARWPTschoen(qx,qy,qz,grid,X0,readx,Disp,ctrl,nen)
%
% RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));

% used fruits: apricot, cherry, grape, grape, mango, melon, papaya, peach, pear,
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
fac     = [5 5 5];
bwe     = lambda./(fac);            % emitter bandwidth
% bwe(:)  = bwe(3);
bwk     = bwe.*sqrt(2);             % kde corrected bandwidth 
P       = (2*pi())^(Dim/2)*prod(bwe); % kernel integral
CorrectSquarInt = 2^(Dim/2) *P;     % correction factor a kernel for particle 
%release and for emitter volume is used, it comes from the integral over p^2
cs      = 1;                        % solubility concentration
gdt     = lambda(2)/(4*mean(qx));  % constant time steps in sec.
% dL      = lambda(2)/100;          % constant sapce step [m]

vx = reshape(qx,grid.n_pts+1);
vy = reshape(qy,grid.n_pts+1);
vz = reshape(qz,grid.n_pts+1);
ne                = ctrl.ne;
ne_em             = ctrl.ne_em;
%Same Number of particles with same mass everywere
NP      = 1000;

%% Dispersions Tensor
%--------------------------------------------------------------------------
% Dispersion---------------------------------------------------------------
% D = (at |u| + Dm) I + (al - at) u u' / |u|-------------------------------
%--------------------------------------------------------------------------

    at                = 0.02*grid.d_pts(1);   % transverse dispersivity
    al                = at*5;                 % longitudinal dispersivity
    Dm                = 1e-9;                 % molecular diffusion coefficient
 
    absv    = sqrt(vx.^2 + vy.^2+vz.^2); % absolute velocity
   
 if Disp == 1
    % D  =  Diffusion + Dispersions Tensor
    D_diag = at.*absv+Dm;   
    D{1}   = ((vy.*vy.*(al-at))./absv)+D_diag;    % Dyy
    D{2}   = ((vy.*vx.*(al-at))./absv);           % Dyx
    D{3}   = ((vy.*vz.*(al-at))./absv);           % Dyz
    D{4}   = ((vx.*vy.*(al-at))./absv);           % Dxy = Dyx
    D{5}   = ((vx.*vx.*(al-at))./absv)+D_diag;    % Dxx
    D{6}   = ((vx.*vz.*(al-at))./absv);           % Dxz
    D{7}   = ((vz.*vy.*(al-at))./absv);           % Dzy = Dyz
    D{8}   = ((vz.*vx.*(al-at))./absv);           % Dzx = Dxz
    D{9}   = ((vz.*vz.*(al-at))./absv)+D_diag;    % Dzz
    
 
    % Dispersions Gradient
    dDxx = zeros(Domain./ lambda + 1);
    dDyx = dDxx; dDzx = dDxx; dDxy = dDxx; dDyy = dDxx; dDzy = dDxx;
    dDxz = dDxx; dDyz = dDxx; dDzz = dDxx;
    %delta()/delta x
    for apricot = 1:N(3)+1
        for kl=2:N(2)%(Domain(2)-1)/lambda(2)
            %   vorwärts Differenzen
            %     dDxx(:,kl) = D{1}(kl+1,:) - D{1}(kl,:)/lambda(2);
            %   rückwärts Differenzen
            %     dDxx(:,kl)  = D{1}(:,kl) - D{1}(:,kl-1)/lambda(2);
            %   zentrale Differenzen 1. Ableitung
            dDxx(:,kl,apricot) = (D{5}(:,kl+1,apricot) - D{5}(:,kl-1,apricot))./(2*lambda(2));
            dDxy(:,kl,apricot) = (D{4}(:,kl+1,apricot) - D{4}(:,kl-1,apricot))./(2*lambda(2));
            dDxz(:,kl,apricot) = (D{6}(:,kl+1,apricot) - D{6}(:,kl-1,apricot))./(2*lambda(2));
            %   zentrale Differenzen 2. Ableitung
            %     dDxx(:,kl)  = D{1}(:,kl+1) - 2*D{1}(:,kl) + D{1}(:,kl-1)/lambda(2)^2;
        end
    end
    
    %delta()/delta y
    for papaya = 1:N(3)+1
        for kl=2:N(1);%(Domain(1)-1)/lambda(1)
            %   vorwärts Differenzen
            %     dDyy(kl,:) = D{4}(kl+1,:) - D{4}(kl,:)/lambda(1);
            %   rückwärts Differenzen
            %     dDyy(kl,:)  = D{4}(kl,:) - D{4}(kl-1,:)/lambda(1);
            %   zentrale Differenzen 1. Ableitung
            dDyx(kl,:,papaya)  = (D{2}(kl+1,:,papaya) - D{2}(kl-1,:,papaya))./(2*lambda(1));
            dDyy(kl,:,papaya)  = (D{1}(kl+1,:,papaya) - D{1}(kl-1,:,papaya))./(2*lambda(1));
            dDyz(kl,:,papaya)  = (D{3}(kl+1,:,papaya) - D{3}(kl-1,:,papaya))./(2*lambda(1));
            %   zentrale Differenzen 2. Ableitung
            %     dDyy(kl,:)  = D{4}(kl+1,:) - 2*D{4}(kl,:) + D{4}(kl-1,:)/lambda(1)^2;
        end
    end
    
    %delta()/delta z
    for grape = 1:N(2)+1
        for kl=2:N(3)%(Domain(3)-1)/lambda(3)
            dDzx(:,grape,kl)  = (D{8}(:,grape,kl+1) - D{8}(:,grape,kl-1))./(2*lambda(3));
            dDzy(:,grape,kl)  = (D{7}(:,grape,kl+1) - D{7}(:,grape,kl-1))./(2*lambda(3));
            dDzz(:,grape,kl)  = (D{9}(:,grape,kl+1) - D{9}(:,grape,kl-1))./(2*lambda(3));
        end
    end
    
    ddDyy = dDyy(:);ddDyx = dDyx(:);ddDyz = dDzy(:);
    ddDxy = dDxy(:);ddDxx = dDxx(:);ddDxz = dDxz(:);
    ddDzy = dDzy(:);ddDzx = dDzx(:);ddDzz = dDzz(:);
 else
% D  =  Diffusion
%     D      = [Dm 0 0; 0 Dm 0; 0 0 Dm];
    B      = chol([Dm 0 0; 0 Dm 0; 0 0 Dm]);

    ddDyy = 0;ddDyx = 0;ddDyz = 0;
    ddDxy = 0;ddDxx = 0;ddDxz = 0;
    ddDzy = 0;ddDzx = 0;ddDzz = 0;
 end     
%% predefinitions
NPT     = NP*length(X0(:,1));       % number of particles per position
XX0     = zeros(NPT,Dim);

cutXat      = NPT;%50000; % up to know only one core, this is only possible for small CSA
Core        = ceil(NP/cutXat);
compromise  = [10,10];
red         = 0;

Cij         = zeros(size(X0,1));
C           = zeros(prod(N),1);

% hdf         = (max(ceil(X0(:,2)))+10*al+4*lambda(1)) * (max(ceil(X0(:,2)))+10*al+4*lambda(1)<readx) +...
%     readx *  (max(ceil(X0(:,2)))+10*al+4*lambda(1)>readx);
% read        = [hdf;readx];
read        = [max(X0(:,2))+5*lambda(2),readx];
npe         = ones(size(X0,1),1)*NP;

% bw = sqrt(bwe.^2 + bwk.^2);
% Vs = (2*pi())^(3/2)*(prod(bw))^(1/2)*ne;%/(2*sqrt(2));%only with normal particle release and not uniform

% --smaple points for density estimation are defined ------------------------------
% x_pdf       = [grid.x_pts{1}(:)+lambda(1)/2,grid.x_pts{2}(:)+lambda(2)/2,grid.x_pts{3}(:)+lambda(3)/2]; 
x_pdf       = [grid.x_pts{1}(:),grid.x_pts{2}(:),grid.x_pts{3}(:)]; 
% x_pdfpro = x_pdf(x_pdf(:,2) == readx-1,:);
[a b]       = meshgrid(lambda(1)/2:lambda(1):lambda(1)*N(1)-lambda(1)/2,lambda(3)/2:lambda(3):lambda(3)*N(3)-lambda(3)/2);
xpdf        = [a(:) b(:)];
N_planes = 1;
locplane = [max(X0(:,2))+lambda/2,readx-10,readx-5]; 
cpro = zeros(N(1)*N(3),size(X0,1),N_planes);
mapCSAc = zeros(N(1)*N(3),N_planes);
x_pdfpro = zeros(size(xpdf,1),Dim,N_planes);
ne_map  = zeros(size(xpdf,1),1);
for plane = 1:N_planes
    x_pdfpro(:,:,plane) = [xpdf(:,1),ones(N(1)*N(3),1).*locplane(plane),xpdf(:,2)];
    idx                 = sub2ind(N,round(x_pdfpro(:,2,plane)./lambda(2)),... % without + 0.5 because ne is at midpoint
                                    round(x_pdfpro(:,1,plane)./lambda(1)+0.5),...
                                    round(x_pdfpro(:,3,plane)./lambda(3)+0.5));
    
    idx2                 = sub2ind(N,round(x_pdfpro(:,2,plane)./lambda(2)+1),... % without + 0.5 because ne is at midpoint
                                    round(x_pdfpro(:,1,plane)./lambda(1)+0.5),...
                                    round(x_pdfpro(:,3,plane)./lambda(3)+0.5));
                            
    ne_map(:,plane)     = (nen(round(idx))+nen(round(idx2)))/2;
end

x_PDF = X0';
for plane = 1:N_planes
    x_PDF = [x_PDF x_pdfpro(:,:,plane)'];
end
%% --------------------------------------------------------------------------
dt = gdt;
for hip=1:1 %if you are going for the whole plume calculation hip must go to 2
%   stream = RandStream.getGlobalStream;
%   reset(stream);
for cherry =1:size(X0,1)
%         fact                     = repmat(relra,ceil(npe(cherry)/length(relra)),1); fact(npe(cherry)+1:end) = [];
%         Xy                       = ones(npe(cherry),1)*X0(cherry,1);%+(rand(npe(cherry),1)-0.5).*(bw(1)./fact);
%         Xx                       = ones(npe(cherry),1)*X0(cherry,2);%+(rand(npe(cherry),1)-0.5).*(bw(2)./fact);
%         Xz                       = ones(npe(cherry),1)*X0(cherry,3);%+(rand(npe(cherry),1)-0.5).*(bw(3)./fact);
        r                        =  [randn(npe(cherry),1).*(bwe(1)),...  
                                     randn(npe(cherry),1).*(bwe(2)),...
                                     randn(npe(cherry),1).*(bwe(3))];
%         while any(r(:,1)>lambda(1)/2) || any(r(:,1)<-lambda(1)/2) || ...
%               any(r(:,2)>lambda(2)/2) || any(r(:,2)<-lambda(2)/2) || ...
%               any(r(:,3)>lambda(3)/2) || any(r(:,3)<-lambda(3)/2)
%         r(r(:,1)>lambda(1)/2,1)         =  lambda(1)./2 + lambda(1)./2 - r(r(:,1)>lambda(1)/2,1);
%         r(r(:,1)<-lambda(1)/2,1)        = -lambda(1)./2 - lambda(1)./2 - r(r(:,1)<-lambda(1)/2,1);
%         r(r(:,2)>lambda(2)/2,2)         =  lambda(2)./2 + lambda(2)./2 - r(r(:,2)>lambda(2)/2,2);
%         r(r(:,2)<-lambda(2)/2,2)        = -lambda(2)./2 - lambda(2)./2 - r(r(:,2)<-lambda(2)/2,2);
%         r(r(:,3)>lambda(3)/2,3)         =  lambda(3)./2 + lambda(3)./2 - r(r(:,3)>lambda(3)/2,3);
%         r(r(:,3)<-lambda(3)/2,3)        = -lambda(3)./2 - lambda(3)./2 - r(r(:,3)<-lambda(3)/2,3);
%         end
%         
%         r                        =  [(rand(npe(cherry),1)-0.5).*(lambda(1)),...
%                                      (rand(npe(cherry),1)-0.5).*(lambda(2)),...
%                                      (rand(npe(cherry),1)-0.5).*(lambda(3))];
        Xy                       = ones(npe(cherry),1)*X0(cherry,1)+r(:,1);      
        Xx                       = ones(npe(cherry),1)*X0(cherry,2)+r(:,2);
        Xz                       = ones(npe(cherry),1)*X0(cherry,3)+r(:,3);
        
        counte                   = sum(npe(1:cherry));
        counta                   = sum(npe(1:cherry-1));
        XX0(1+counta:counte,:)   = [Xy,Xx,Xz];
end
    
    Xallt       = zeros(compromise(hip)*NP*size(X0,1),3);

    
    sub1x      = zeros(cutXat,1);sub2x = sub1x;sub1y = sub1x;
    sub2y      = sub1x;sub1z = sub1x;sub2z = sub1x;subD = sub1x;
    vx_linear  = zeros(cutXat,1);vy_linear  = vx_linear;vz_linear  = vx_linear;
    
    dDD     = zeros(cutXat,Dim);A = dDD; Rand = A; BZ = A;
    node1   = A; node2 = A; node3 = A; X1 = A; fun = A;
    yxzabsv = zeros(cutXat,1); yxabsv = yxzabsv; Wal = yxabsv; Wat = Wal;
    Byy = Wal; Byx = Byy; Byz = Byy; Bxy = Byy; Bxx = Byy; Bxz = Byy;
    Bzy = Byy; Bzx = Byy; Bzz = Byy;
    
    %%
    % tic
    for kiwi=1:Core;
        %     k
        if kiwi < Core
            X       = XX0((cutXat*(kiwi-1)+1):(cutXat*kiwi),:);
        else
            X       = XX0((cutXat*(kiwi-1)+1):end,:);
            sub1x      = zeros(length(X),1);sub2x = sub1x;sub1y = sub1x;
            sub2y      = sub1x;sub1z = sub1x;sub2z = sub1x;subD = sub1x;
            vx_linear  = zeros(length(X),1);vy_linear  = vx_linear;vz_linear  = vx_linear;
            
            dDD     = zeros(length(X),3);A = dDD; Rand = A; BZ = A;
            node1   = A; node2 = A; node3 = A;
            yxzabsv = zeros(length(X),1); yxabsv = yxzabsv; Wal = yxabsv; Wat = Wal;
            Byy = Wal; Byx = Byy; Byz = Byy; Bxy = Byy; Bxx = Byy; Bxz = Byy;
            Bzy = Byy; Bzx = Byy; Bzz = Byy;
        end
        
        ind = ones(length(X),length(readx)) == 1;
        steps=0;

        while sum(ind) > 0; %min(X(:,2)) < max(readx) - 0.05
            steps=steps+1;
            % + (steps <= compromise(hip)) * (100-gdt)*(-(hip-2));
            
            % keeping the particles within the domain, bouncing boundaries
            % angle of incidence is equal to the angle of reflection
            
            B1 = X(:,1);
            upper = B1>(N(1)*lambda(1));
            lower = B1<0;
            B1(upper)     = (N(1)*lambda(1)) + ((N(1)*lambda(1))-B1(upper));%(N(1)*lambda(1)-lambda(1)) + ((N(1)*lambda(1)-lambda(1))-B1(upper));
            B1(lower)     = -1.*B1(lower);
            
            B2                  = X(:,2);
            B2(B2<0)            = B2(B2<0).*-1;
            B3                  =X(:,3);
            upper = B3>(N(3)*lambda(3));
            lower = B3<0;
            B3(upper)     = (N(3)*lambda(3)) + ((N(3)*lambda(3))-B3(upper));
            B3(lower)     = -1.*B3(lower);
            
            X                  = [B1,B2,B3];
            
            Xallt(1+(steps-red-1)*length(X):(steps-red)*length(X),:) = X;
            
            %         % upper bonds
            %         B1(B1>N(1)*lambda(1)+(lambda(1)/2-0.000001)-1)  = N(1)*lambda(1)+(lambda(1)/2-0.001)-1;
            %         % B2(B2>N(2)*lambda(2)+(lambda(2)/2-0.000001)-1)  = N(2)*lambda(2)+(lambda(2)/2-0.001)-1;
            %         B3(B3>N(3)*lambda(3)+(lambda(3)/2-0.000001)-1)  = N(3)*lambda(3)+(lambda(3)/2-0.001)-1;
            %
            %         % Lower bounds
            %         X(X<=0)  = 1e-16;
            
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
            
            A(ind,:)         = [vy_linear(ind),vx_linear(ind),vz_linear(ind)] + dDD(ind,:);
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
             
            %--------------------------------------------------------------------------
            % particle displacement algorithm------------------------------------------
            %--------------------------------------------------------------------------

            X(ind,:) = X(ind,:) + A(ind,:) .*dt + sqrt(dt).* BZ(ind,:);

            ind         = X(:,2) < read(hip);
            X(~ind,2)   = X(~ind,2)+1e3;
            
%             figure(1), plot3(X(ind,2),X(ind,1),X(ind,3),'.','Color','k'),
%             set(gca,'Xlim',[0 N(1)*lambda(1)],'Ylim',[0 N(2)*lambda(2)],'Zlim',[0 N(3)*lambda(3)]), view(30,-15),box on,hold off
            
            if ~mod(steps,compromise(hip))
                red = red +compromise(hip);
                %         tic
                
                if hip == 2
                         
                    x       = [Xallt(:,1)';Xallt(:,2)';Xallt(:,3)'];
                    
%                     q       = ones(1,size(x,2)).*q1(1);
%                     [kde]   = figtree(x,bw(2),q,x_pdf',1e-2);
%                     [kde]   = kde_figtree(x,x_pdf',1e-2,[],bw);
                    [kde]   = kdefun(x,x_pdf',1e-2,[],bwk',[0;0;0],[N.*lambda]'); % kdefun calls either figtree or gpuml
%                     C       = C + kde.* dt .* size(x,2) .* mdach ./ (ne);
                    C      = C + kde;%.*mdach.*size(x,2)./Vs .* dt;
%easy controlleti----------------------------------------------------------
%                     [kde] = figtree(x,bw(2),q,X0',1e-3);
%                     ccchecker = ccchecker + kde.*mdach./Vs .* dt;
%--------------------------------------------------------------------------
                else

                    
                    for lime = 1:size(X0,1)
                        for jk = 1:compromise(hip)
                            Xi(1+NP*(jk-1):NP*jk,:) = Xallt(1+NP*(lime-1)+length(X)*(jk-1):length(X)*(jk-1)+NP*lime,:);
                        end
       
                        x = [Xi(:,1)';Xi(:,2)';Xi(:,3)'];

%                         [kde]       = figtree(x,bw(2),q1,x_PDF,1e-2);%                         [kde]           = figtree(x,bw(2),q1,[X0' x_pdfpro3' x_pdfpro2' x_pdfpro1'],1e-1);
%                         [kde]       = kde_figtree(x,x_PDF,1e-2,[],bw);
                        [kde]       = kdefun(x,x_PDF,1e-2,[],bwk',[0;0;0],[N.*lambda]'); % kdefun calls either figtree or gpuml
                        kde(lime)   = kde(lime)*2;
%                         Cij(:,lime) = Cij(:,lime) + kde(1:size(X0,1)) .* dt .* size(x,2) ./ (ne*NP); 
%                         this is the actual calculation however all factoras are constant
%                         thus we can make the calcualtion afterward and NP
%                         is anyway canceling out since it is also in
%                         size(x,2) ; size(x,2) = compromise*NP
                        
                        Cij(:,lime) = Cij(:,lime) + kde(1:size(X0,1));
                       
                        for plane = 1:N_planes
%                         cpro(:,lime,plane)   = cpro(:,lime,plane) + kde(size(X0,1)+1+(plane-1)*N(1)*N(3):size(X0,1)+N(1)*N(3)*plane)./Vs .* dt;
                        cpro(:,lime,plane)   = cpro(:,lime,plane) + kde(size(X0,1)+1+(plane-1)*N(1)*N(3):size(X0,1)+N(1)*N(3)*plane);
                        end
                                           
                    end
                end
                %         toc, disp(['Time of KDE at time step' num2str(steps) '_im' num2str(hip) '.Lauf'])
                %         min(X(ind,2))
            end
            
        end
        
    end
    % tic
    red =0;
    if hip == 1;
       
        cs = ones(size(X0,1),1).*cs;
            
        % Matrix system to ensure local equlibrium at emitter locations
        % ATTENTION: 1. IF Cij IS (CLOSED TO)  DIAGONAL MATRIX
        % ATTENTION: 2. IF EPSILON HAS NEGATIVE ENTRIES
        % case 1: only epsilon_i is able to reduce c_j=i --> overall to much
        % reduction of emissions
        % 2. DNAPL emitteres can not absorbe thus negative epsilons = 0 and emission
        % is to high wont lead to cs at any emitter
        % For both cases higher dispersion helps, also larger or smaller bandwidth of kde
        % migth help. Of course, increasing number of particles helps!
        disp('solving Matrix equation')
   
%         qind = sub2ind(N+1,X0(:,1)./lambda(1)+1,X0(:,2)./lambda(2)+1,X0(:,3)./lambda(3)+1);
%         qem = qx(qind);
%         msat = qem.*cs.*(Vs/bw(1));
%         epsi = Cij\msat;
%         mh = cs.*Vs./dt;
        neem = repmat(ne_em,1,size(X0,1));
        Cij = Cij .* compromise(1) .* dt ./ neem .* CorrectSquarInt; %[s/m^3]
        epsi = Cij\cs;
        mineps = min(epsi);%/max(epsi);
        epsi(epsi<0) = 0;
        if any(epsi==0)
            disp(['fuck ' num2str(sum(epsi==0)/length(epsi)*100) ' % negative epsilon with worst' num2str(mineps) 'and mean' num2str(mean(epsi))])
        end

        
%         dC          = diag(diag(Cij));
%         depsi       = dC\cs;
%         aktiveEm    = ((Cij*cs)'*epsi)/((dC*cs)'*depsi);%/(sum(1./diag(Cij)));
%         effMCSA     = sum(epsi)/sum(1./diag(dC));
        
        % nepsi = epsi./sum(epsi);
        
        % SCij = Cij*S;
        % Sepsi = SCij\cs;
        % epsi  = S*Sepsi;
        
        % invSCij         = pinv(SCij);
        % Sepsi           = invSCij*cs;
        % epsi            = S*Sepsi;
        %
        % invCij          = pinv(Cij);
        % epsilon(:,hip)  = invCij*cs;
        % epsilon(:,hip)     = (ones(length(X0),1)*cs)\Cij;
        
        % toc,
        % variable number of particles release where all have the same mass mdach;
%         mdach = sum(epsi)/(length(X0).*NP);%mdach = mass of each single particle
%         npe   = round(epsi./mdach);% npe number of released particles at each emitter location
        mdach       = mean(epsi./NP);%sum(epsi)/(length(X0)); %eigt wie oben aber dann vorher Cij = Cij./NP, deswegen kürzt sich NP raus
        npe         = round(epsi./mdach); %
  
%         for melon =1:length(X0)
%             mp(1+NP*(melon-1):NP*melon) = ones(NP,1).*epsi(melon,1);
%         end
%         % mpi = mp;
%         % for mango = 1:compromise(hip)
%         %        mpi = [mpi,mp];
%         % end
%         mpi = repmat(mp,1,compromise(hip));

        for plane = 1:N_planes
            mapCSAc(:,plane) = cpro(:,:,plane)*epsi;
        end
        
        %         Mflux = sum(mp)/gdt;

        Mflux(1)    = sum(epsi);


        mapCSAc     = mapCSAc(:);
        mapCSAc = mapCSAc .*dt.*compromise(1)./ne_map .*CorrectSquarInt;
        %% 
        %zwischen resident und fluss concentration springen nicht so
%         %einfach, diffusiver fluss fehlt!!
%         x_pdf = allcomb(1:grid.n_pts(1),1:grid.n_pts(3));
%         x_pdf = (x_pdf-1)*grid.d_pts(1);
%         xmap = [mapCSA(:,1),mapCSA(:,3)];
%         qmap = ones(1,size(xmap,1));
%         bwmap=lambda.*0.25;
%         Akernel = bwmap(1)*bwmap(3)*ne;
%         [mapCSAc]  = figtree(xmap',bwmap(2),qmap,x_pdf',1e-3);
% %         mapCSAc   = mapCSAc./sum(mapCSAc).* (mdach*NPT/gdt); % mass flux per support volume --> denisty/concentration/t
%         mapCSAc   = mapCSAc .* mdach;
%         
%         qx_pro      = qx(:);
%         inq         = allcomb(1:N(1)+1,1:N(3)+1);
%         inqx_pro    = sub2ind(N+1,inq(:,1),ones(prod([N(1)+1,N(3)+1]),1).*(readx(1)./lambda(2)),inq(:,2));
%         qx_pro      = qx_pro(inqx_pro);
%         [qxpro]     = fieldmapping(qx_pro,N(1),N(3));
%         mapCSAc     = (mapCSAc)./(qxpro.*Akernel); % das geht so nicht!! qxpro is ja rein advective noch nicht mal korrigiert um grad D

        % V  = zeros(N);
        % for kl=1:length(readx)
        %    V(:,readx(kl)/lambda(2),:)               = reshape(cc(:,kl),N(1),N(3))';
        % end
    else
    C = C.*mdach.*dt.*compromise(2).*NP.*size(X0,1)./ne .* CorrectSquarInt; 
    end
end


%%Post calculation

% for i = 1:length(X0)
%     asd(i) = sub2ind(N,X0(i,1)./lambda(1)+1,X0(i,2)./lambda(2)+1,X0(i,3)./lambda(3)+1);
% end
%% Plotting
% C1 = C;
% C1(C1>cs(1)) = cs(1);
% CC = reshape(C,N);

% % % % % % % % % % % % % % % % delete(s(3));
% % % % % % % % % % % % % % % figure,
% % % % % % % % % % % % % % % plotter_nd(grid.x_pts{2},grid.x_pts{1},grid.x_pts{3},CC,grid.d_tot,'',[1 1 1],grid.n_pts,0,max(C),[],[],lambda,round(readx)-lambda(2)*2)
% % % % % % % % % % % % % % % % set(gca,'ytick',[],'xtick',[],'ztick',[],'zlim',[0 80],'xlim',[0 max(readx)-1],'ylim',[0 50])
% % % % % % % % % % % % % % % saveas(gcf,['CSApro' datestr(now) '.fig']);







end



