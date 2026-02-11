function [LRBFn,index1,Gx,Gy,Gz]=local_poly_3d_inv_linear(X,l,n,P,T,k_RBF)
%%% PHS r^s+ polynomial of degree up to l with s odd
%%% P projection matrix
%%% T tangent vector
LRBFn=zeros(n,n);Gx=zeros(n,n);Gy=zeros(n,n);Gz=zeros(n,n);
index1=[];
[~,inds] = knnCPU(X,X,k_RBF);
for i=1:n
    X_local=X(inds(i,:),:); P_local=P(inds(i,:),:,:);T_local=T(:,inds(i,:),:);
 % p number of polynomial terms 
 [B,B_x,B_y,B_z,p]=generate_intrinsic_polynomial_3d(X_local,X_local(1,:),l,T_local(:,1,:));

 Fx=B_x;Fy=B_y;Fz=B_z;
 D_x=zeros(k_RBF,p);D_y=zeros(k_RBF,p);D_z=zeros(k_RBF,p);
for ll=1:k_RBF
  for jj=1:p
  D_x(ll,jj)=P_local(ll,1,1)*Fx(ll,jj)+P_local(ll,1,2)*Fy(ll,jj)+P_local(ll,1,3)*Fz(ll,jj);
   D_y(ll,jj)=P_local(ll,2,2)*Fy(ll,jj)+P_local(ll,2,1)*Fx(ll,jj)+P_local(ll,2,3)*Fz(ll,jj);
   D_z(ll,jj)=P_local(ll,3,2)*Fy(ll,jj)+P_local(ll,3,1)*Fx(ll,jj)++P_local(ll,3,3)*Fz(ll,jj);
  end
end

tic
Dinv=(B'*B)\B';
t1_temp(i)=toc;
%Dinv=pinv(B'*B,1e-7)*B';
%Dinv=(B'*B+1e-12*eye(p)/n)\B';

Gx_local=D_x*Dinv;
Gy_local=D_y*Dinv;
Gz_local=D_z*Dinv;
LRBFn_local=Gx_local*Gx_local+Gy_local*Gy_local+Gz_local*Gz_local;

v=LRBFn_local(1,:)';
if v(1)>0
    index1=[index1,i];
end    
%% min C with constraint B'u=B'v, u(1)<0,u\geq 0 else where
%%% no needs for c>=0
%b=zeros(k_RBF+1,1);b(end)=abs(min(v))+1;
%A=-eye(k_RBF+1);A(1,1)=1;A(end,end)=1; A(2:end-1,end)=-1;
%%% with constraint c>=0
b=zeros(k_RBF+2,1);b(end)=abs(min(v))+1;
A=-eye(k_RBF+2,k_RBF+1);A(1,1)=1; A(2:k_RBF,end)=-1;A(k_RBF+2,k_RBF+1)=1;
options = optimset('Display', 'off');
f=zeros(k_RBF+1,1);f(end)=1;
x = linprog(f,A,b,[B',zeros(p,1)],B'*v,[],[],options);

u=x(1:k_RBF);
LRBFn(i,inds(i,1:k_RBF))=u';
Gx(i,inds(i,1:k_RBF))=Gx_local(1,:)';
Gy(i,inds(i,1:k_RBF))=Gy_local(1,:)';
Gz(i,inds(i,1:k_RBF))=Gz_local(1,:)';
%% min 1/2|u|^2 with constraint B'u=B'v, u(1)<0,u\geq 0 else where
% b=zeros(k_RBF,1);
% A=-eye(k_RBF);A(1,1)=1; 
% options = optimset('Display', 'off');
% H=eye(k_RBF);
% u2=quadprog(H,zeros(k_RBF,1),A,b,B',B'*v,[],[],[],options);


end

end