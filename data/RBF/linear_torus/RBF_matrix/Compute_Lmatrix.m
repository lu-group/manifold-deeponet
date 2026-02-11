function [LRBF,Gx,Gy,Gz]=Compute_Lmatrix(X,s,P,kerlflag,sval)
%%% Inputs
     %%% X          - N by [n_R] data set with N data points in R^n_R
     %%% s          - shape parameter of RBF
     %%% P          - projection matrix
     %%% kerlflag   - 1 == gaussian kernel, 2 == IMQ kernel 
     %%% sval       - threshold for calculating pinv of RBF matrix
%%% Outputs 
     %%% LRBF       - estimated Laplacian operator 
     %%% Gx         - differential matrix Gx 
     %%% Gx         - differential matrix Gy 
     %%% Gx         - differential matrix Gz

n=size(X,1);
if (kerlflag==1)
    phi=gax();
else
    phi=iqx();
end

[r, rx,ry,rz] =phi.distanceMatrix3d(X(:,1),X(:,2),X(:,3)); % r distance matrix

B = phi.rbf(r,s); % RBF system matrix
Fx = phi.D1(r,s,rx);
Fy = phi.D1(r,s,ry);
Fz = phi.D1(r,s,rz);

Bx=zeros(n,n);By=zeros(n,n); Bz=zeros(n,n);
for i=1:n
  for j=1:n
   Bx(i,j)=P(i,1,1)*Fx(i,j)+P(i,1,2)*Fy(i,j)+P(i,1,3)*Fz(i,j);
   By(i,j)=P(i,2,1)*Fx(i,j)+P(i,2,2)*Fy(i,j)+P(i,2,3)*Fz(i,j);
   Bz(i,j)=P(i,3,1)*Fx(i,j)+P(i,3,2)*Fy(i,j)+P(i,3,3)*Fz(i,j);
  end
end

Binv=pinv(B,sval);

Gx=Bx*Binv;
Gy=By*Binv;
Gz=Bz*Binv;

LRBF=Gx*Gx+Gy*Gy+Gz*Gz;

end