function [qest,epsilon,dim] = qestfind(x2,k,epsilon,dim)
% input
% x2        == data that lies on the Manifold, gridpoints
% k         == knn parameter
% c         == diffusion coefficient
% epsilon   == kernel bandwidth parameter 

% output
% L         == matrix approximation to either c^{-1} div(c \grad )

% automated pre-tuning of epsilon

N = size(x2,1);
[d2,inds] = knnCPU(x2,x2,k);

if (nargin<3)
    epss = 2.^(-30:.1:10);
    dpreGlobal=zeros(1,length(epss));


    for ll=1:length(epss)
        dpreGlobal(ll) = sum(sum(exp(-d2.^2./(4*epss(ll)))))/(N*k);
    end
    halfdim = diff(log(dpreGlobal))./diff(log(epss));
    [maxval,maxind] = max(halfdim);

%     figure(10);
%     semilogx(epss(2:end),halfdim,'b'); 
%     hold on
%     semilogx(epss(maxind),1/2,'o');
    dim = 2*maxval;
    epsilon = epss(maxind);
    
end

% Construct the matrix approximation
W = exp(-d2.^2/(4*epsilon));
W = sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(W'),N*k,1),N,N,N*k)';
W = (W+W')/2;
clear inds;


% right normalization
qest = full(sum(W,2));

qest = qest/(N*(4*pi*epsilon)^(dim/2));



end