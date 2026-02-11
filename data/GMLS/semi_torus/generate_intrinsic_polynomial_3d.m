function [B,B_x,B_y,B_z,p]=generate_intrinsic_polynomial_3d(x,x0,l,T)
%%% x0 taylor at point x0
%%% T -- N x 1 x d 
T=squeeze(T); %% N x d, each columns is tangent vector

N=size(x,1);
index=generatemultiindex(l,2);
p=size(index,2); % number of poylnomial terms

B=ones(N,p);B_x=zeros(N,p);B_y=zeros(N,p);B_z=zeros(N,p);
delta=zeros(N,2);
xnorm=std(sqrt(sum(x.^2,2)));
delta(:,1)=sum(T(:,1)'.*(x-x0)/xnorm,2); % p_k\dot (x-x0)
delta(:,2)=sum(T(:,2)'.*(x-x0)/xnorm,2); % p_k\dot (x-x0)

for ll=1:p
    for rr=1:2
     B(:,ll)=B(:,ll).*delta(:,rr).^(index(rr,ll));

     if index(rr,ll)==0
         D_x=zeros(N,1);D_y=zeros(N,1);D_z=zeros(N,1);
     else
     D_x=ones(N,1);D_y=ones(N,1);D_z=ones(N,1);
         for tt=1:2
             if tt==rr
             D_x=D_x.*index(tt,ll)*T(1,tt)./xnorm.*delta(:,tt).^(index(tt,ll)-1);
             D_y=D_y.*index(tt,ll)*T(2,tt)./xnorm.*delta(:,tt).^(index(tt,ll)-1);
             D_z=D_z.*index(tt,ll)*T(3,tt)./xnorm.*delta(:,tt).^(index(tt,ll)-1);
             else
             D_x=D_x.*delta(:,tt).^(index(tt,ll));
             D_y=D_y.*delta(:,tt).^(index(tt,ll));
             D_z=D_z.*delta(:,tt).^(index(tt,ll));
             end
         end
         
     end
     
     B_x(:,ll)=B_x(:,ll)+D_x;
     B_y(:,ll)=B_y(:,ll)+D_y;
     B_z(:,ll)=B_z(:,ll)+D_z;
     
    end
end


end