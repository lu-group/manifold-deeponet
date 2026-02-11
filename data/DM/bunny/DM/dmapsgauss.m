function [L] = dmapsgauss(x,k,c,epsilon)
%%% Diffusion Maps:  estimate the operator L:= div(c \grad )
%%% Input
%%% x         - gridpoints
%%% k         - knn parameter
%%% c         - diffusion coefficient
%%% epsilon   - kernel bandwidth parameter 

%%% Output
%%% L         - matrix approximation to div(c \grad )

%%% automated pre-tuning of epsilon

N = size(x,1);
[d,inds] = knnCPU(x,x,k);

% %%% Construct the matrix approximation
% W = exp(-d.^2/(4*epsilon));
% W = sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(W'),N*k,1),N,N,N*k)';
% W = (W+W')/2;
% clear inds;
% 
% %%% right normalization
% q = full(sum(W,2));
% cw = diag(c.^(1/2)./q);
% W = W*cw;
% 
% %%% left normalization
% D = full(sum(W,2));
% Dinv = spdiags(D.^(-1),0,N,N);
% 
% %%% matrix approximation to c^{-1} div(c \grad )
% L = (Dinv*W-eye(N))/epsilon;



% Construct the matrix approximation
W = exp(-d.^2/(4*epsilon));
W = sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(W'),N*k,1),N,N,N*k)';
W = (W+W')/2;
clear inds;

% right normalization
q = full(sum(W,2));
cw = diag(c./q);
D1 = sum(W*cw,2);
Dinv1 = spdiags(D1.^(-1/2),0,N,N);
W = W*cw*Dinv1;


% left normalization
D = full(sum(W,2));
Dinv = spdiags(D.^(-1),0,N,N);
L = (Dinv*W-eye(N))/epsilon;

%%% matrix approximation to div(c \grad )
L = diag(c)*L;

end