clear all, close all
addpath DM            % Diffusion Maps code

isTest = true;
Nk = 100;
X = readmatrix("data_8k/Xt_8k.txt");
TRI = readmatrix("data_8k/TRI_8k.txt");
x = X(:,1); y = X(:,2); z = X(:,3);
k = 40;
N = size(X,1);
f = 0.1 * (x + y + z).^2;
[~, epsilon] = qestfind(X, k);

Nxx = 25;
Nyy = 25;
Nx = (Nxx + 1) * (Nyy + 1);
dx = 3.5 / Nxx;
xx = linspace(-2, 1.5, Nxx + 1)';
yy = linspace(0.5, 4, Nyy + 1)';
[xx, yy] = meshgrid(xx, yy);
xx = reshape(xx, [], 1);
yy = reshape(yy, [], 1);
writematrix([xx, yy], "data_8k/X.txt"); % X - Grid data

us = zeros(Nk, N);
kappas_torus = zeros(Nk, N);
kappas = zeros(Nk, Nx);
coefs = []; % To store i, a, b

for i = 1:Nk
    a1 = rand*2;
    b1 = rand;
    a2 = rand;
    b2 = rand*2;
    c = rand;
    disp([i, a1, b1, a2, b2, c]);
    % kappa = (a * xx + 2 * b * yy + 3 + c);
    % kappa_torus = (a * x + 2 * b * y + 3 + c);
    kappa = a1*xx.^2 + b1*yy.^2 + a2*xx + b2*yy + 3 + c;
    kappa_torus = a1*x.^2 + b1*y.^2 + a2*x + b2*y + 3 + c;
    L_DM = dmapsgauss(X, k, kappa_torus, epsilon); 
    L = -L_DM + eye(N);
    u = L \ f;
    
    us(i, :) = u;  
    kappas(i, :) = kappa; 
    kappas_torus(i, :) = kappa_torus;
    coefs = [coefs; i, a1, b1, a2, b2, c];
end

if ~isTest
    writematrix(kappas, "data_8k/kappas_train.txt"); % kappa values on the grid
    writematrix(kappas_torus, "data_8k/kappast_train.txt"); % kappa values on the torus
    writematrix(us, "data_8k/us_train.txt"); % solutions
    writematrix(coefs, "data_8k/coefs_train.txt"); % Saving iteration data for training
else
    writematrix(kappas, "data_8k/kappas_test.txt");
    writematrix(kappas_torus, "data_8k/kappast_test.txt");
    writematrix(us, "data_8k/us_test.txt");
    writematrix(coefs, "data_8k/coefs_test.txt"); % Saving iteration data for testing
end
