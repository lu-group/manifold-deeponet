function [X,P0,THET,PHI]=torus_manifold(N,R,r,flag)
%%% generate the data set on torus 
%%% embedding function: [(R+r*cos(THET)).*cos(PHI), (R+r*cos(THET)).*sin(PHI), r*sin(THET)];

%%% Input      
%%% N           - Number of points lie on torus
%%% R           - radius of larger circle 
%%% r           - radius of smaller circle
%%% flag        - ==1 well sampled data, ==2 uniformly random data

%%% Output
%%% X           - N by 3 data set of points on torus
%%% P0          - N x 3 x 3 tensor, P0(i,:,:) is projection matrix at x_i
%%% THET, PHI   - intrinsic coordinate theta and phi

if flag==1
       Ntheta = N;
    Nphi = N;
    dtheta = 2*pi/(Ntheta);
    theta = [0:dtheta:2*pi-dtheta]';
    dphi = 2*pi/(Nphi);
    phi = [0:dphi:2*pi-dphi]';
    [THET,PHI]=meshgrid(theta,phi);
    THET = reshape(THET,Ntheta*Nphi,1);
    PHI = reshape(PHI,Ntheta*Nphi,1);
elseif flag==2
   test = rand(N,2);
   THET = 2*pi*(test(:,1));
   PHI = 2*pi*(test(:,2));
end

X = [(R+r*cos(THET)).*cos(PHI), (R+r*cos(THET)).*sin(PHI), r*sin(THET)];

%% exact tangent and normal vector
x2 = -r*sin(THET).*cos(PHI);
y2 = -r*sin(THET).*sin(PHI);
z2 = r*cos(THET);
t2=[x2 y2 z2];

x3 = (R+r*cos(THET)).*(-sin(PHI));
y3 = (R+r*cos(THET)).*cos(PHI);
z3 = cos(THET)*0;
t3 = [x3 y3 z3];

t4=cross(t2,t3);

exT1=zeros(3,N);
exT2=zeros(3,N);
exNormal=zeros(3,N);
for i = 1:1:N
    exT1(:,i)= t2(i,:)'/norm(t2(i,:),2);
    exT2(:,i)= t3(i,:)'/norm(t3(i,:),2);
    exNormal(:,i) = t4(i,:)'/norm(t4(i,:),2);
end

P0=zeros(N,3,3);
for i=1:1:N
P0(i,:,:)=eye(3)-exNormal(:,i)*exNormal(:,i)';
end

end