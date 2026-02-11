function [L] = VBDM_kappa(x,c,k,k2,operator,epsilon,dim)
%%% Variable Bandwidth DM: estimate the operator L:= div(c \grad )
%%% Inputs
    %%% x       - N-by-n data set with N data points in R^n
    %%% c         - diffusion coefficient
    %%% k       - number of nearest neighbors to use
    %%% k2      - number of nearest neighbors to use to determine the "epsilon"
    %%%             parameter
    %%% operator- 1 - Laplace-Beltrami operator, 2 - generator of grad system
    %%% dim     - optionally prescribe the intrinsic dimension of the manifold lying inside R^n
    %%% epsilon - optionally choose an arbitrary "global" epsilon
    
%%% Outputs
    %%% L      - matrix approximation to div(c \grad )  
    
    %%% Theory requires c2 = 1/2 - 2*alpha + 2*dim*alpha + dim*beta/2 + beta < 0 
    %%% The resulting operator will have c1 = 2 - 2*alpha + dim*beta + 2*beta
    %%% Thus beta = (c1/2 - 1 + alpha)/(dim/2+1), since we want beta<0,
    %%% natural choices are beta=-1/2 or beta = -1/(dim/2+1)

    N = size(x,1); %number of points
    
    [d,inds] = knnCPU(x,x,k);

    %%% Build ad hoc bandwidth function by autotuning epsilon for each pt.
    epss = 2.^(-30:.1:10);
    rho0 = sqrt(mean(d(:,2:k2).^2,2));
    
    %%% Pre-kernel used with ad hoc bandwidth only for estimating dimension
    %%% and sampling density
    dt = d.^2./(repmat(rho0,1,k).*rho0(inds));
    
    %%% Tune epsilon on the pre-kernel
    dpreGlobal=zeros(1,length(epss));
    for i=1:length(epss)
        dpreGlobal(i) = sum(sum(exp(-dt./(2*epss(i)))))/(N*k);       
    end
   
    [maxval,maxind] = max(diff(log(dpreGlobal))./diff(log(epss)));
       
    if (nargin < 8)
        dim=2*maxval;
    end
    
    %%% Use ad hoc bandwidth function, rho0, to estimate the density
    dt = exp(-dt./(2*epss(maxind)))/((2*pi*epss(maxind))^(dim/2));
    dt = sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(dt'),N*k,1),N,N,N*k)';
    dt = (dt+dt')/2;

    % sampling density estimate for bandwidth function
    qest = (sum(dt,2))./(N*rho0.^(dim)); 
    
    clear dt;
    
    if (operator == 1)
        %%% Laplace-Beltrami, c1 = 0
        beta = -1/2;
        alpha = -dim/4 + 1/2;
    elseif (operator == 2)
        %%% Kolmogorov backward operator, c1 = 1
        beta = -1/2;
        alpha = -dim/4;     
    end

    c1 = 2 - 2*alpha + dim*beta + 2*beta;
    c2=.5-2*alpha+2*dim*alpha+dim*beta/2+beta;
    
    d = d.^2;

    %%% bandwidth function rho(x) from the sampling density estimate
    rho = qest.^(beta).*exp(log(c)/(dim+2));
    rhomu = mean(rho);
    rho = rho/rhomu;

    %% construct the exponent of K^S_epsilon
    d = d./repmat((rho),1,k);  % divide row j by rho(j)
    d = d./rho(inds);
    
    %%% Tune epsilon for the final kernel
    if (nargin<7)
        for i=1:length(epss)
            s(i) = sum(sum(exp(-d./(4*epss(i))),2))/(N*k);
        end
        [~,maxind] = max(diff(log(s))./diff(log(epss)));    
        epsilon = epss(maxind);
    end

    %%% K^S_epsilon with final choice of epsilon
    d = exp(-d./(4*epsilon));
    d = sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(d'),N*k,1),N,N,N*k)';
    clear inds;
    d = (d+d')/2;   %%% symmetrize since this is the symmetric formulation

    %%% q^S_epsilon (this is the sampling density estimate q(x) obtained from VB kernel)
    qest = full((sum(d,2)./(rho.^dim)));
    %qest = 1./rho0.^dim;

    Dinv1 = spdiags(qest.^(-alpha),0,N,N);

    %%% K = K^S_{epsilon,alpha}
    d = Dinv1*d*Dinv1; % the "right" normalization
    
    %%% S^2 =P*D, where P = diag(rho), D = q^S_{epsilon,alpha}
    Ssquare = full((rho.^2).*(sum(d,2)));

    %%% S^{-1}
    Sinv = spdiags(Ssquare.^(-1/2),0,N,N);
    
    %%% matrix approximation to c^{-1} div(c \grad )
    L = (Sinv*Sinv*d - spdiags(rho.^(-2),0,N,N))/epsilon;
    
    %%% matrix approximation to div(c \grad )
    L=diag(c)*L;
    
end
