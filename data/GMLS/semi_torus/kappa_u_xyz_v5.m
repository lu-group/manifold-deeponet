function [C] = kappa_u_xyz_v5(X, a1, b1, c1)
%%% Input      
%%% X           - ambient data

%%% Output
%%% f           - force = -div(c*grad u)+u
%%% c           - diffusion coefficients c
%%% u           - true solution

syms x y z           % x,y,z cordinates in R^3
%% define c and u in ambient coordiante x,y,z
% analytic diffusion coefficients c on X
c(x,y,z)=a1*x+b1*y+ 10; % different expression of c could be chosen, but make sure c is positive for all x,y,z on torus
cH=matlabFunction(c);   % convert symbolic expression to function handle
C=cH(X(:,1),X(:,2),X(:,3));


end