function [u, X, kappa] = generate_torus_data(N, X, f, exNormal, T0, a1,a2,a3, a4)
%%% Input      
%%% Output
%%% X           - N by 3 data set of points on torus
%%% THET, PHI   - intrinsic coordinate theta and phi

%%% kappa 

d=3; % dimension of ambient space
l=3;  % degree of polynomial used 
k_RBF=51;
kappa=kappa_u_xyz_sin(X, a1,a2);

P0 = zeros(N,d,d);
for i = 1:N
   P0(i,:,:)=eye(d)-exNormal(:,i)*exNormal(:,i)';
end
[Lap,index1,Gx,Gy,Gz]=local_poly_3d_inv_linear(X,l,N,P0,T0,k_RBF);

index_B=index1; % detected points which are close to bounary.
index_I=1:N; index_I(index_B)=[];
L=-((Gx*kappa).*Gx+(Gy*kappa).*Gy+(Gz*kappa).*Gz+kappa.*Lap)+eye(N);
%LRBF=-((Gx*kappa).*Gx+(Gy*kappa).*Gy+(Gz*kappa).*Gz+kappa.*(Gx*Gx+Gy*Gy+Gz*Gz))+eye(N);

L_in=L(index_I,index_I);
u_poly=zeros(N,1);
u_poly(index_I)=L_in\f(index_I);


kappa = reshape(kappa, [1, N]);
u = reshape(u_poly, [1, N]);

end