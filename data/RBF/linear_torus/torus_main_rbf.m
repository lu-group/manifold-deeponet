clear all, close all
addpath ./RBF_matrix     % RBF differential matrix 
addpath ./Estimate_Projection_matrix  % esitmate porjection matrix needed in RBF method 

% kappa = ax + by + c (should be positive for all x,y,z on torus), Nx
% defines the resolution of X, Nk defines the number of k

N = 2500;
Nx = 26*26*9;% 26*26
Nk = 1000;

d=3; % dimension of ambient space
R=2;r=1; % R radius of larger circle, r radius of smaller circle
flag=2; % flag==1 well-sampled data, flag==2, uniformly random samples 

% sample X
Nxx = 25;
Nyy = 25;
Nzz = floor((r/(R+r))*(25));
dx = 2*(R+r)/(Nxx);
x = [-(R+r):dx:(R+r)]';
dy = 2*(R+r)/(Nyy);
y = [-(R+r):dy:(R+r)]';
dz = 2*(r)/(Nzz);
z = [-(r):dz:(r)]';
[xx,yy,zz]=meshgrid(x,y,z);
xx = reshape(xx,(Nxx+1)*(Nyy+1)*(Nzz+1),1);
yy = reshape(yy,(Nxx+1)*(Nyy+1)*(Nzz+1),1);
zz = reshape(zz,(Nxx+1)*(Nyy+1)*(Nzz+1),1);
X_grid = [xx, yy, zz];
N_sample = (Nxx+1)*(Nyy+1)*(Nzz+1);

% get f
[X_sample,P0,THET,PHI]=torus_manifold(N,R,r,flag); % for f
[f, kappa, u] = fcompute(THET,PHI,2);% theta and phi does not matter


% initialize a matrix kappa with shape (Nk, Nx)
us = zeros(Nk, N);
kappas_torus = zeros(Nk, N);
kappas = zeros(Nk, Nx);
for i = 1:Nk
    i
    a1 = rand*10;
    a2 = rand*10;
    a3 = rand*10;
    a4 = rand*10;
    [u, X_torus, kappa_torus] = generate_torus_data(N,X_sample,f, a1,a2,a3,a4); 
    kappa = kappa_u_xyz_v2(X_grid,a1, a2, a3, a4);
    us(i, :) = u; 
    kappas(i, :) = kappa;
    kappas_torus(i, :) = kappa_torus;
end

%%
% writematrix(X_torus, "data/Xt_train.txt")
% writematrix(kappas, "data/kappas_train.txt")
% writematrix(kappas_torus, "data/kappast_train.txt")
% writematrix(us, "data/us_train.txt")

%% test data
N = 2500;
Nx = 26*26*9;% 26*26
Nk = 1000;

% initialize a matrix kappa with shape (Nk, Nx)
us_t = zeros(Nk, N);
kappas_torus_t = zeros(Nk, N);
kappas_t = zeros(Nk, Nx);
for i = 1:Nk
    i
    a1 = rand*10;
    a2 = rand*10;
    a3 = rand*10;
    a4 = rand*10;
    [u, X_torus, kappa_torus] = generate_torus_data(N,X_sample,f, a1,a2,a3,a4);
    kappa = kappa_u_xyz_v2(X_grid,a1, a2, a3, a4);
    kappas_t(i, :) = kappa;
    us_t(i, :) = u;   
    kappas_torus_t(i, :) = kappa_torus;
end

% save the data
% writematrix(X_torus, "data/Xt_test.txt")
% writematrix(kappas_t, "data/kappas_test.txt")
% writematrix(kappas_torus_t, "data/kappast_test.txt")
% writematrix(us_t, "data/us_test.txt")