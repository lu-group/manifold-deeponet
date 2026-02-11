function [u, X, kappa] = generate_torus_data(N, X, f, a1,a2,a3, a4)
%%% Input      
%%% Output
%%% X           - N by 3 data set of points on torus
%%% THET, PHI   - intrinsic coordinate theta and phi
s=0.3; % shape parameter in RBF
kerlflag = 2; % 1-=Gaussian kernel, 2=IMQ kernel

%%% piecewise constant kappa 
kappa=kappa_u_xyz_v2(X, a1,a2,a3,a4);
%% estimate projection matrix
[P0_tilde] = geod_mean_normal_svd(X,40,2,1:N);
   
%% RBF solution 
%%% Here, we use P0_tilde as projection matrix. One can also use P0, the true one.
[~,Gx,Gy,Gz]=Compute_Lmatrix(X,s,P0_tilde,kerlflag,10^-6); 

%%% Solve div(kappa \nabla u)-u=f weakly, i.e., 
%%% -\int kappa \nabla u\nabla v-\int uv =\int fv, \forall v
%LRBF=-(Gx'*(kappa.*Gx)+Gy'*(kappa.*Gy)+Gz'*(kappa.*Gz))-eye(N);
LRBF=-((Gx*kappa).*Gx+(Gy*kappa).*Gy+(Gz*kappa).*Gz+kappa.*(Gx*Gx+Gy*Gy+Gz*Gz))+eye(N);
uRBF=LRBF\f;

kappa = reshape(kappa, [1, N]);
u = reshape(uRBF, [1, N]);

end