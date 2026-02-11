clear all, close all
%addpath /MATLAB Drive/Manifolds/semi_torus % Diffusion Maps code

% kappa = ax + by + c (should be positive for all x,y,z on torus), Nx
% defines the resolution of X, Nk defines the number of k

d=3; % dimension of ambient space
a=2;% 
flag=2; % flag==1 well-sampled data, flag==2, uniformly random samples 
l=3;  % degree of polynomial used 
k_RBF=51;
R=2;r=1; % R radius of larger circle, r radius of smaller circle

N=2500;
Nx = 26*26*9; %sensors for kappa
Nk = 200;

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

[X_sample,exNormal,T0,THETA,PHI] = semi_torus_manifold_random_v2(N,2,1);

writematrix(PHI, "data_exp/phit.txt");
writematrix(THETA, "data_exp/thetat.txt");

[f] = fcompute(THETA,PHI,2);

% initialize a matrix kappa with shape (Nk, Nx)
us = zeros(Nk, N);
kappas_torus = zeros(Nk, N);
kappas = zeros(Nk, Nx);
for i = 1:Nk
    a1 = rand;
    a2 = rand;
    b1 = rand;
    b2 = rand;
    disp([i, a1, a2, b1, b2])
    kappa = kappa_u_xyz_v5(X_grid,a1, a2, b1);
    [u, X_torus, kappa_torus] = generate_torus_data(N,X_sample,f, exNormal, T0, a1, a2, b1, b2);
    us(i, :) = u;  
    kappas(i, :) = kappa; 
    kappas_torus(i, :) = kappa_torus; % kappa values on the torus
end


writematrix(X_grid, "data_exp/X_train.txt") % X - Grid data
writematrix(X_torus, "data_exp/Xt_train.txt") % X - on torus
writematrix(kappas, "data_exp/kappas_train.txt") % kappa values on the grid
writematrix(kappas_torus, "data_exp/kappast_train.txt") % kappa values on the torus
writematrix(us, "data_exp/us_train.txt") % solutions


%% test data

Nk = 100;

% initialize a matrix kappa with shape (Nk, Nx)
us_t = zeros(Nk, N);
kappas_torus_t = zeros(Nk, N);
kappas_t = zeros(Nk, Nx);
for i = 1:Nk
    a1 = rand;
    a2 = rand;
    b1 = rand;
    b2 = rand;
    disp([i, a1, a2]);
    kappa = kappa_u_xyz_v5(X_grid,a1, a2, b1);
    [u, X_torus, kappa_torus] = generate_torus_data(N,X_sample,f, exNormal, T0, a1, a2, b1, b2);
    us_t(i, :) = u;   
    kappas_t(i, :) = kappa;
    kappas_torus_t(i, :) = kappa_torus;
end

% save the data
writematrix(X_grid, "data_exp/X_test.txt")
writematrix(X_torus, "data_exp/Xt_test.txt")
writematrix(kappas_t, "data_exp/kappas_test.txt")
writematrix(kappas_torus_t, "data_exp/kappast_test.txt")
writematrix(us_t, "data_exp/us_test.txt")

